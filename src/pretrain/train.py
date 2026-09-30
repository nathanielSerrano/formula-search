"""
Contrastive pretraining of the formula encoder).

Reads the graph store (src.pretrain.graph_store) and vocabularies, trains the
dual-branch encoder with symmetric InfoNCE on pairs of augmented views, and every
`eval_every` steps runs the quick dev check (src.model.quick_dev) and saves
checkpoints:

    <out_dir>/latest.pt    last evaluation point (use --resume to continue)
    <out_dir>/best.pt      best quick-dev nDCG′ so far
    <out_dir>/log.jsonl    training and evaluation metrics

Usage
-----
    python -m src.pretrain.train --config configs/pretrain.yaml
    python -m src.pretrain.train --config configs/pretrain.yaml --max-steps 200 --eval-every 100   # smoke test
    python -m src.pretrain.train --config configs/pretrain.yaml --resume checkpoints/pretrain/latest.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.formula_graph import load_vocabs
from src.data.paths import PROCESSED_DIR, REPO_ROOT
from src.model.encoder import EncoderConfig, FormulaEncoder, batch_to
from src.model.loss import InfoNCE
from src.model.quick_dev import QuickDev
from src.pretrain.augment import AugmentConfig, Augmenter
from src.pretrain.dataset import PairCollator, PretrainPairs
from src.pretrain.graph_store import STORE_DIR, GraphStore

VOCAB_PATH = PROCESSED_DIR / "graph_vocab.json"


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


def param_groups(modules, weight_decay: float):
    """Weight decay on matrices only (not on biases, norms or the temperature)."""
    decay, no_decay = [], []
    for module in modules:
        for p in module.parameters():
            if p.requires_grad:
                (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def save_checkpoint(path: Path, model, loss_fn, optimizer, scheduler, scaler, step: int, epoch: int,
                    best: float, config: dict, vocab_path: Path) -> None:
    torch.save({"model_config": model.config_dict(), "model": model.state_dict(), "loss": loss_fn.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "step": step, "epoch": epoch, "best_ndcg": best, "config": config,
                "vocab_path": str(vocab_path)}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pretrain.yaml")
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-quick-dev", action="store_true", help="skip the dev check (e.g. without qrels)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    t = config["train"]
    for key in ("max_steps", "eval_every", "batch_size"):
        if getattr(args, key) is not None:
            t[key] = getattr(args, key)
    out_dir = args.out_dir or REPO_ROOT / config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])

    vocabs = load_vocabs(args.vocab)
    store = GraphStore(args.store)
    dataset = PretrainPairs(store, Augmenter(vocabs, AugmentConfig(**config["augment"])),
                            min_nodes=t["min_nodes"], seed=t["seed"])
    loader_kwargs = dict(batch_size=t["batch_size"], shuffle=True, drop_last=True, num_workers=t["num_workers"],
                         collate_fn=PairCollator(vocabs), pin_memory=device.type == "cuda")
    print(f"{len(dataset):,} anchors of {len(store):,} formulas; {len(dataset) // t['batch_size']:,} steps per epoch",
          flush=True)

    model = FormulaEncoder(EncoderConfig.from_vocabs(vocabs, **config["model"])).to(device)
    loss_fn = InfoNCE(init_temperature=t["init_temperature"]).to(device)
    optimizer = torch.optim.AdamW(param_groups([model, loss_fn], t["weight_decay"]), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(t["warmup_steps"], t["max_steps"]))
    use_amp = t["amp"] and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp and not torch.cuda.is_bf16_supported() else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler() if use_amp and amp_dtype == torch.float16 else None
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters; device {device}; "
          f"amp {amp_dtype if use_amp else 'off'}", flush=True)

    step, epoch, best = 0, 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)  # our own checkpoint
        model.load_state_dict(ckpt["model"])
        loss_fn.load_state_dict(ckpt["loss"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        step, epoch, best = ckpt["step"], ckpt["epoch"], ckpt["best_ndcg"]
        print(f"resumed from {args.resume} at step {step}", flush=True)

    quick_dev: Optional[QuickDev] = None
    if not args.no_quick_dev:
        quick_dev = QuickDev.for_split(store, vocabs, config["quick_dev"]["split"],
                                       n_distractors=config["quick_dev"]["distractors"], seed=t["seed"])
        print(f"quick dev: {len(quick_dev.topic_ids)} topics, {len(quick_dev.candidates):,} candidates, "
              f"{quick_dev.judged_coverage:.1%} of judged visual ids in the store", flush=True)

    log = (out_dir / "log.jsonl").open("a")
    window = {"loss": 0.0, "accuracy": 0.0, "n": 0}
    t0 = time.time()
    model.train()
    while step < t["max_steps"]:
        dataset.set_epoch(epoch)  # workers are re-created each epoch, so they see the new epoch
        for batch in DataLoader(dataset, **loader_kwargs):
            a, b = batch_to(batch["a"], device), batch_to(batch["b"], device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = loss_fn(model(a), model(b))
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
                elapsed = time.time() - t0
                entry = {"step": step, "epoch": epoch, "loss": window["loss"] / window["n"],
                         "accuracy": window["accuracy"] / window["n"], "temperature": loss_fn.temperature,
                         "lr": scheduler.get_last_lr()[0], "formulas_per_s": window["n"] * t["batch_size"] / elapsed}
                print(f"step {step:>6}  loss {entry['loss']:.4f}  acc {entry['accuracy']:.3f}  "
                      f"τ {entry['temperature']:.4f}  lr {entry['lr']:.2e}  {entry['formulas_per_s']:,.0f} formulas/s",
                      flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                window = {"loss": 0.0, "accuracy": 0.0, "n": 0}
                t0 = time.time()

            if step % t["eval_every"] == 0 or step == t["max_steps"]:
                ndcg = -1.0
                if quick_dev is not None:
                    result = quick_dev.run(model, device)
                    ndcg = result.mean["ndcg"]
                    print(f"  quick dev @ {step}: {result.summary()}", flush=True)
                    log.write(json.dumps({"step": step, "quick_dev": result.mean}) + "\n")
                    log.flush()
                save_checkpoint(out_dir / "latest.pt", model, loss_fn, optimizer, scheduler, scaler,
                                step, epoch, max(best, ndcg), config, args.vocab)
                if ndcg > best:
                    best = ndcg
                    save_checkpoint(out_dir / "best.pt", model, loss_fn, optimizer, scheduler, scaler,
                                    step, epoch, best, config, args.vocab)
                    print(f"  new best quick-dev nDCG′ {best:.4f} → {out_dir / 'best.pt'}", flush=True)
                t0 = time.time()

            if step >= t["max_steps"]:
                break
        epoch += 1
    log.close()
    print(f"done: {step} steps; best quick-dev nDCG′ {best:.4f}")


if __name__ == "__main__":
    main()
