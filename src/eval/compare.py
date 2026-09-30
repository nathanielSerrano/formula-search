"""
Compare runs topic by topic: mean differences, win/loss counts and significance.

The first run is the baseline; every other run is compared with it on the same topics
(all topics in the qrels, scored as in src.eval.metrics). For each measure:

  Δ          mean per-topic difference (run − baseline)
  95% CI     paired bootstrap over topics (percentile interval)
  p          two-sided paired randomization test: under the null hypothesis the sign of
             each topic's difference is arbitrary, so the observed |mean Δ| is compared
             with the means obtained by flipping signs at random (Smucker et al., 2007)
  W/L/T      topics where the run is better / worse / equal

With several runs or measures, some small p-values are expected by chance; treat
p < 0.05 on one of many comparisons with care.

Usage
-----
    python -m src.eval.compare runs/bm25_dev.tsv runs/gnn_best_step2500_dev.tsv --split dev
    python -m src.eval.compare runs/bm25_dev.tsv runs/gnn_dev.tsv --split dev --topics 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.data.paths import resolve_year
from src.data.qrels import load_qrels
from src.eval.metrics import LABELS, MEASURES, EvalResult, evaluate
from src.eval.runs import read_run

COMPARED = (*MEASURES, "judged_10")


@dataclass
class Comparison:
    measure: str
    baseline_mean: float
    run_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    losses: int
    ties: int


def paired_randomization(diffs: np.ndarray, n_permutations: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(diffs) ≠ 0 by random sign flips."""
    if not len(diffs) or not np.any(diffs):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(diffs)))
    permuted = np.abs((signs * diffs).mean(axis=1))
    observed = abs(diffs.mean())
    return float((np.sum(permuted >= observed - 1e-12) + 1) / (n_permutations + 1))


def bootstrap_ci(diffs: np.ndarray, n_samples: int = 10_000, seed: int = 0, level: float = 0.95):
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n_samples, len(diffs)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def compare(baseline: EvalResult, run: EvalResult, measures: Sequence[str] = COMPARED,
            n_permutations: int = 10_000, seed: int = 0) -> List[Comparison]:
    topics = sorted(baseline.per_topic)
    if set(topics) != set(run.per_topic):
        raise ValueError("runs were evaluated on different topic sets")
    out = []
    for m in measures:
        b = np.array([baseline.per_topic[t][m] for t in topics])
        r = np.array([run.per_topic[t][m] for t in topics])
        d = r - b
        low, high = bootstrap_ci(d, seed=seed)
        out.append(Comparison(m, float(b.mean()), float(r.mean()), float(d.mean()), low, high,
                              paired_randomization(d, n_permutations, seed),
                              int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((np.abs(d) <= 1e-9).sum())))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="baseline run first, then the runs to compare with it")
    parser.add_argument("--split", required=True)
    parser.add_argument("--qrels", choices=("official", "all"), default="official")
    parser.add_argument("--topics", type=int, default=0, help="also list the N topics with the largest nDCG′ gains and losses")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("give a baseline run and at least one run to compare")

    qrels = load_qrels(resolve_year(args.split), args.qrels)
    results: Dict[str, EvalResult] = {}
    for path in args.runs:
        run, run_id = read_run(path)
        results[run_id or path.stem] = evaluate(run, qrels)
    names = list(results)
    base = names[0]
    print(f"{len(qrels)} topics ({resolve_year(args.split)}, {args.qrels} qrels); baseline: {base}")

    for name in names[1:]:
        print(f"\n{name} vs {base}")
        print(f"  {'measure':<10}{'baseline':>9}{'run':>8}{'Δ':>9}   {'95% CI':<18}{'p':>7}   W/L/T")
        for c in compare(results[base], results[name], n_permutations=args.permutations):
            flag = " *" if c.p_value < 0.05 else ""
            print(f"  {LABELS[c.measure]:<10}{c.baseline_mean:>9.4f}{c.run_mean:>8.4f}{c.delta:>+9.4f}   "
                  f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]{c.p_value:>7.3f}{flag:<2} {c.wins}/{c.losses}/{c.ties}")
        if args.topics:
            deltas = sorted(((results[name].per_topic[t]["ndcg"] - results[base].per_topic[t]["ndcg"], t)
                             for t in results[base].per_topic))
            fmt = lambda items: ", ".join(f"{t} {d:+.3f}" for d, t in items)
            print(f"  largest nDCG′ gains:  {fmt(deltas[::-1][:args.topics])}")
            print(f"  largest nDCG′ losses: {fmt(deltas[:args.topics])}")
    print("\n* p < 0.05 (paired randomization test, two-sided)")


if __name__ == "__main__":
    main()
