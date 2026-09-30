"""
Fuse retrieval runs (e.g. BM25 and the GNN) into one ranking.

Methods
  linear   per topic, each run's scores are min-max normalised to [0, 1] and combined as
           Σ wᵢ·scoreᵢ; a formula missing from a run contributes 0 for that run.
  rrf      reciprocal rank fusion: Σ 1 / (k + rankᵢ), k = 60; needs no weights.

Tuning (two runs, linear): the weight w of the second run (the first gets 1 − w) is
chosen on the given split by grid search on nDCG′. Because the weight is tuned on the
same topics it is scored on, the tuned score is optimistic; --cv reports an honest
estimate by k-fold cross-validation over topics (tune on k−1 folds, score the held-out
fold). Tune on dev, then apply the chosen weight unchanged to the test runs.

Usage
-----
    python -m src.eval.fuse runs/bm25_dev.tsv runs/gnn_dev.tsv --split dev --tune --cv 5
    python -m src.eval.fuse runs/bm25_test.tsv runs/gnn_test.tsv --weights 0.4 0.6 --out runs/fused_test.tsv
    python -m src.eval.fuse runs/bm25_dev.tsv runs/gnn_dev.tsv --method rrf --out runs/rrf_dev.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.paths import resolve_year
from src.data.qrels import Qrels, load_qrels
from src.eval.metrics import MAX_DEPTH, Run, evaluate, ranked, truncate
from src.eval.runs import read_run, write_run

RRF_K = 60
GRID = [round(float(w), 2) for w in np.linspace(0, 1, 21)]


def _minmax(docs: Dict[str, float]) -> Dict[str, float]:
    if not docs:
        return {}
    lo, hi = min(docs.values()), max(docs.values())
    if hi == lo:
        return {d: 1.0 for d in docs}
    return {d: (s - lo) / (hi - lo) for d, s in docs.items()}


def fuse_linear(runs: Sequence[Run], weights: Sequence[float], depth: int = MAX_DEPTH) -> Run:
    fused: Run = {}
    for topic in sorted(set().union(*runs)):
        scores: Dict[str, float] = {}
        for run, w in zip(runs, weights):
            for doc, s in _minmax(run.get(topic, {})).items():
                scores[doc] = scores.get(doc, 0.0) + w * s
        fused[topic] = scores
    return truncate(fused, depth)


def fuse_rrf(runs: Sequence[Run], k: int = RRF_K, depth: int = MAX_DEPTH) -> Run:
    fused: Run = {}
    for topic in sorted(set().union(*runs)):
        scores: Dict[str, float] = {}
        for run in runs:
            for rank, doc in enumerate(ranked(run.get(topic, {})), 1):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
        fused[topic] = scores
    return truncate(fused, depth)


def _subset(qrels: Qrels, topics: Sequence[str]) -> Qrels:
    return {t: qrels[t] for t in topics}


def tune_weight(run_a: Run, run_b: Run, qrels: Qrels, measure: str = "ndcg") -> Tuple[float, List[Tuple[float, float]]]:
    """Best weight for run_b (run_a gets 1 − w) and the whole grid [(w, score)]."""
    grid = [(w, evaluate(fuse_linear([run_a, run_b], [1 - w, w]), qrels).mean[measure]) for w in GRID]
    best = max(grid, key=lambda ws: (ws[1], -abs(ws[0] - 0.5)))  # ties → the more balanced weight
    return best[0], grid


def cross_validate(run_a: Run, run_b: Run, qrels: Qrels, folds: int = 5, seed: int = 0,
                   measure: str = "ndcg") -> Tuple[Dict[str, float], List[float]]:
    """Mean metrics over held-out folds, each fused with the weight tuned on the other folds."""
    topics = sorted(qrels)
    order = np.random.default_rng(seed).permutation(len(topics))
    fold_of = {topics[i]: k % folds for k, i in enumerate(order)}
    per_topic: Dict[str, Dict[str, float]] = {}
    chosen = []
    for f in range(folds):
        train = [t for t in topics if fold_of[t] != f]
        held = [t for t in topics if fold_of[t] == f]
        w, _ = tune_weight(run_a, run_b, _subset(qrels, train), measure)
        chosen.append(w)
        per_topic.update(evaluate(fuse_linear([run_a, run_b], [1 - w, w]), _subset(qrels, held)).per_topic)
    measures = next(iter(per_topic.values())).keys()
    return {m: float(np.mean([v[m] for v in per_topic.values()])) for m in measures}, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--method", choices=("linear", "rrf"), default="linear")
    parser.add_argument("--weights", type=float, nargs="+", help="one weight per run (linear)")
    parser.add_argument("--split", help="split whose qrels are used for --tune/--cv and for scoring the result")
    parser.add_argument("--tune", action="store_true", help="grid-search the weight of the second run (two runs)")
    parser.add_argument("--cv", type=int, default=0, help="k-fold cross-validated estimate of the tuned fusion")
    parser.add_argument("--run-id", default="fused")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    runs = [read_run(p)[0] for p in args.runs]
    qrels: Optional[Qrels] = load_qrels(resolve_year(args.split)) if args.split else None
    if (args.tune or args.cv) and (qrels is None or len(runs) != 2 or args.method != "linear"):
        parser.error("--tune/--cv need --split, exactly two runs and --method linear")

    if args.method == "rrf":
        fused = fuse_rrf(runs)
        label = f"RRF (k={RRF_K})"
    else:
        weights = args.weights
        if args.tune:
            w, grid = tune_weight(runs[0], runs[1], qrels)
            print(f"weight of {args.runs[1].name} (vs {args.runs[0].name}) → nDCG′ on {args.split}:")
            print("  " + "  ".join(f"{g:.2f}:{s:.4f}" for g, s in grid))
            weights = [1 - w, w]
            print(f"  best weight {w:.2f} (tuned on the same topics: optimistic)")
        if args.cv:
            cv, chosen = cross_validate(runs[0], runs[1], qrels, args.cv)
            print(f"{args.cv}-fold cross-validated: nDCG′ {cv['ndcg']:.4f}  MAP′ {cv['map']:.4f}  "
                  f"P′@10 {cv['P_10']:.4f}  judged@10 {cv['judged_10']:.4f}  (fold weights {chosen})")
        if weights is None:
            weights = [1.0 / len(runs)] * len(runs)
        if len(weights) != len(runs):
            parser.error("give one weight per run")
        fused = fuse_linear(runs, weights)
        label = "linear, weights " + ", ".join(f"{w:.2f}" for w in weights)

    if qrels is not None:
        print(f"fused ({label}) on {args.split}: {evaluate(fused, qrels).summary()}")
    if args.out:
        write_run(fused, args.out, args.run_id)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
