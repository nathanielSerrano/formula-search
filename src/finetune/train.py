"""
Supervised fine-tuning of a pretrained formula encoder (single GPU).

Starts from a pretraining checkpoint (model weights and temperature), trains on
query / positive / hard-negative triples from the training split's judgments
(src.finetune.data), and selects checkpoints with the quick dev check on ARQMath-2.
The check also runs once before the first step, so the log shows where fine-tuning
starts from.

    <out_dir>/best.pt      best quick-dev nDCG′ (step 0 = the pretrained model)
    <out_dir>/latest.pt    last evaluation point
    <out_dir>/log.jsonl    training and evaluation metrics

Usage
-----
    python -m src.finetune.train --config configs/finetune.yaml
    python -m src.finetune.train --config configs/finetune.yaml --max-steps 100 --eval-every 25
    python -m src.model.retrieve --checkpoint checkpoints/finetune/best.pt --split dev
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.formula_graph import load_vocabs
from src.data.paths import REPO_ROOT, resolve_year
from src.data.qrels import load_qrels
from src.data.topics import load_topics
from src.finetune.data import FinetuneTriples, TopicBatchSampler, TripleCollator, build_examples
from src.model.encoder import batch_to
from src.model.loss import InfoNCE
from src.model.quick_dev import QuickDev
from src.model.retrieve import load_checkpoint
from src.pretrain.graph_store import STORE_DIR, GraphStore
from src.pretrain.train import lr_lambda, param_groups, save_checkpoint


def load_train_qrels(spec: str, split: str):
    """'official' / 'all' for the split's qrels, or a path to a TREC qrels file keyed by visual id."""
    if spec in ("official", "all"):
        return load_qrels(resolve_year(split), spec)
    qrels = {}
    for line in Path(spec).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            qrels.setdefault(parts[0], {})[parts[2]] = int(float(parts[3]))
    return qrels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/finetune.yaml")
    parser.add_argument("--init", type=Path, help="pretrained checkpoint (overrides init_checkpoint)")
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-quick-dev", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    t = config["train"]
    for key in ("max_steps", "eval_every", "lr"):
        if getattr(args, key) is not None:
            t[key] = getattr(args, key)
    init = args.init or REPO_ROOT / config["init_checkpoint"]
    out_dir = args.out_dir or REPO_ROOT / config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])

    model, ckpt = load_checkpoint(init, device)
    model.train()
    vocab_path = Path(ckpt["vocab_path"])
    vocabs = load_vocabs(vocab_path)
    loss_fn = InfoNCE().to(device)
    loss_fn.load_state_dict(ckpt["loss"])  # keep the pretrained temperature
    print(f"initialised from {init} (pretraining step {ckpt.get('step')}); temperature {loss_fn.temperature:.4f}",
          flush=True)

    store = GraphStore(args.store)
    split = config["data"]["split"]
    qrels = load_train_qrels(str(config["data"]["qrels"]), split)
    examples, stats = build_examples(store, vocabs, qrels, load_topics(split))
    print(f"training data ({split}, {config['data']['qrels']} qrels): {json.dumps(stats)}", flush=True)
    if not examples:
        raise SystemExit("no usable training topics")

    dataset = FinetuneTriples(store, examples, seed=t["seed"])
    sampler = TopicBatchSampler(len(dataset), t["batch_topics"], t["max_steps"], seed=t["seed"])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=t["num_workers"],
                        collate_fn=TripleCollator(vocabs), pin_memory=device.type == "cuda")

    optimizer = torch.optim.AdamW(param_groups([model, loss_fn], t["weight_decay"]), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(t["warmup_steps"], t["max_steps"]))
    use_amp = t["amp"] and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp and not torch.cuda.is_bf16_supported() else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler() if use_amp and amp_dtype == torch.float16 else None

    quick_dev: Optional[QuickDev] = None
    if not args.no_quick_dev:
        quick_dev = QuickDev.for_split(store, vocabs, config["quick_dev"]["split"],
                                       n_distractors=config["quick_dev"]["distractors"], seed=t["seed"])

    log = (out_dir / "log.jsonl").open("a")
    best = -1.0

    def checkpoint(step: int) -> None:
        nonlocal best
        ndcg = -1.0
        if quick_dev is not None:
            result = quick_dev.run(model, device)
            ndcg = result.mean["ndcg"]
            print(f"  quick dev @ {step}: {result.summary()}", flush=True)
            log.write(json.dumps({"step": step, "quick_dev": result.mean}) + "\n")
            log.flush()
        save_checkpoint(out_dir / "latest.pt", model, loss_fn, optimizer, scheduler, scaler,
                        step, 0, max(best, ndcg), config, vocab_path)
        if ndcg > best:
            best = ndcg
            save_checkpoint(out_dir / "best.pt", model, loss_fn, optimizer, scheduler, scaler,
                            step, 0, best, config, vocab_path)
            print(f"  new best quick-dev nDCG′ {best:.4f} (step {step}) → {out_dir / 'best.pt'}", flush=True)

    checkpoint(0)
    window = {"loss": 0.0, "accuracy": 0.0, "n": 0}
    t0 = time.time()
    step = 0
    for batch in loader:
        q, d = batch_to(batch["query"], device), batch_to(batch["docs"], device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = loss_fn.query_to_docs(model(q), model(d))
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
        scheduler.step()
        step += 1
        window["loss"] += float(out["loss"])
        window["accuracy"] += float(out["accuracy"])
        window["n"] += 1
        if step % t["log_every"] == 0:
            entry = {"step": step, "loss": window["loss"] / window["n"], "accuracy": window["accuracy"] / window["n"],
                     "temperature": loss_fn.temperature, "lr": scheduler.get_last_lr()[0],
                     "seconds": time.time() - t0}
            print(f"step {step:>5}  loss {entry['loss']:.4f}  acc {entry['accuracy']:.3f}  τ {entry['temperature']:.4f}  "
                  f"lr {entry['lr']:.2e}", flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            window = {"loss": 0.0, "accuracy": 0.0, "n": 0}
        if step % t["eval_every"] == 0 or step == t["max_steps"]:
            checkpoint(step)
    log.close()
    print(f"done: {step} steps; best quick-dev nDCG′ {best:.4f}")


if __name__ == "__main__":
    main()
