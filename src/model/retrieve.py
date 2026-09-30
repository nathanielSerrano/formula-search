"""
Full-corpus retrieval with a trained encoder: embed every visual id in the graph store,
embed the topics of a split, and write the top 1,000 per topic as a run file for
src.eval.evaluate.

Corpus embeddings are cached next to the checkpoint (float16, ~4.3 GB for 8.4M
formulas) and reused for other splits:
    <checkpoint_dir>/<checkpoint_stem>_corpus.npy  (+ .json with checkpoint step and store size)

Search is exact inner product, computed in chunks on the GPU (vectors are unit length,
so this is cosine similarity).

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python -m src.model.retrieve --checkpoint checkpoints/pretrain/best.pt --split dev
    python -m src.eval.evaluate runs/gnn_best_dev.tsv --split dev
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.formula_graph import load_vocabs
from src.data.paths import REPO_ROOT, resolve_year
from src.data.topics import load_topics
from src.eval.metrics import MAX_DEPTH, Run
from src.eval.runs import write_run
from src.model.encoder import FormulaEncoder, batch_to
from src.model.quick_dev import encode_views, topic_views
from src.pretrain.dataset import collate_views, edge_type_counts
from src.pretrain.graph_store import REPS, STORE_DIR, GraphStore


class StoreViews(Dataset):
    """Un-augmented views of every formula in the store, in store order."""

    def __init__(self, store: GraphStore):
        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, i: int) -> dict:
        return {rep: self.store.get(i, rep) for rep in REPS}


def load_checkpoint(path: Path, device) -> Tuple[FormulaEncoder, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own checkpoint
    model = FormulaEncoder.from_config_dict(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def embed_corpus(model: FormulaEncoder, store: GraphStore, n_types: Dict[str, int], path: Path, device,
                 batch_size: int = 4096, workers: int = 16) -> np.ndarray:
    emb = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(store), model.cfg.out_dim))
    loader = DataLoader(StoreViews(store), batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=functools.partial(collate_views, n_types=n_types))
    t0, done = time.time(), 0
    for batch in loader:
        vectors = model(batch_to(batch, device)).half().cpu().numpy()
        emb[done:done + len(vectors)] = vectors
        done += len(vectors)
        if (done // batch_size) % 100 == 0 or done == len(store):
            print(f"  embedded {done:,}/{len(store):,} ({done / (time.time() - t0):,.0f} formulas/s)", flush=True)
    emb.flush()
    return emb


@torch.no_grad()
def search(corpus: np.ndarray, queries: torch.Tensor, k: int, device, chunk: int = 1_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Exact top-k inner product: (scores [Q, k], corpus row indices [Q, k])."""
    q = queries.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    best_scores = torch.full((len(q), 0), float("-inf"), device=device)
    best_index = torch.zeros((len(q), 0), dtype=torch.long, device=device)
    for start in range(0, len(corpus), chunk):
        block = torch.from_numpy(np.asarray(corpus[start:start + chunk])).to(device=device, dtype=q.dtype)
        scores = (q @ block.t()).float()
        top = scores.topk(min(k, scores.shape[1]), dim=1)
        best_scores = torch.cat([best_scores, top.values], dim=1)
        best_index = torch.cat([best_index, top.indices + start], dim=1)
        keep = best_scores.topk(min(k, best_scores.shape[1]), dim=1)
        best_scores, best_index = keep.values, best_index.gather(1, keep.indices)
    return best_scores.cpu().numpy(), best_index.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", required=True, help="train/dev/test or arqmath1/2/3")
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    parser.add_argument("--run-id", help="default gnn_<checkpoint stem>")
    parser.add_argument("--out", type=Path, help="run file (default runs/<run-id>_<split>.tsv)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    vocabs = load_vocabs(Path(ckpt["vocab_path"]))
    n_types = edge_type_counts(vocabs)
    store = GraphStore(args.store)
    print(f"checkpoint {args.checkpoint} (step {ckpt.get('step')}); {len(store):,} formulas", flush=True)

    cache = args.checkpoint.with_name(f"{args.checkpoint.stem}_corpus.npy")
    marker = cache.with_suffix(".json")
    expected = {"checkpoint_step": ckpt.get("step"), "n_formulas": len(store), "out_dim": model.cfg.out_dim}
    if cache.exists() and marker.exists() and json.loads(marker.read_text()) == expected:
        corpus = np.load(cache, mmap_mode="r")
        print(f"using cached corpus embeddings {cache}", flush=True)
    else:
        corpus = embed_corpus(model, store, n_types, cache, device, args.batch_size, args.workers)
        marker.write_text(json.dumps(expected))

    year = resolve_year(args.split)
    topics = load_topics(year)
    topic_ids = sorted(topics)
    views = topic_views(topics, vocabs)
    queries = encode_views(model, [views[t] for t in topic_ids], n_types, device)
    no_graph = [t for t in topic_ids if all(views[t][rep] is None for rep in REPS)]
    if no_graph:
        print(f"warning: {len(no_graph)} topics have no query graph: {no_graph[:5]}", flush=True)

    scores, index = search(corpus, queries, MAX_DEPTH, device)
    ids = store.visual_ids
    run: Run = {t: {ids[j]: float(s) for s, j in zip(scores[q], index[q])} for q, t in enumerate(topic_ids)}
    run_id = args.run_id or f"gnn_{args.checkpoint.stem}"
    out = args.out or REPO_ROOT / "runs" / f"{run_id}_{args.split}.tsv"
    write_run(run, out, run_id)
    print(f"Wrote {out} ({len(run)} topics)\nEvaluate: python -m src.eval.evaluate {out} --split {args.split}")


if __name__ == "__main__":
    main()
