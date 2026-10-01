"""
Additional trec_eval measures beyond the official nDCG′ / MAP′ / P′@10: bpref, and nDCG,
MAP and P at cutoffs 5, 10, 100 and 1000, each computed two ways:

  prime      unjudged visual ids removed first (the ARQMath protocol: comparable with
             the official numbers, and fair to runs that were not in the pools)
  standard   unjudged visual ids count as non-relevant (penalises runs that were not
             pooled, so not comparable between pooled and post-hoc runs)

nDCG uses graded relevance; MAP, P and bpref count grades 2–3 as relevant, like the
official MAP′ and P′@10. bpref ignores unjudged documents by definition, so it is the
same both ways. MAP@k is trec_eval's map_cut: precision at each relevant hit in the top
k, divided by *all* of the topic's relevant formulas. Every qrels topic counts (a topic
with no results scores 0), as in src.eval.metrics. Since runs hold at most 1000
results, prime nDCG@1000, MAP@1000 and P@10 equal the official nDCG′, MAP′ and P′@10.

Usage
-----
    python -m src.eval.extra_metrics runs/reranked_final_test.tsv --split test
    python -m src.eval.extra_metrics runs/bm25_test.tsv runs/reranked_final_test.tsv --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pytrec_eval

from src.data.paths import resolve_year
from src.data.qrels import RELEVANT_GRADE, Qrels, load_qrels
from src.eval.metrics import Run, prime, truncate
from src.eval.runs import read_run

CUTOFFS = (5, 10, 100, 1000)
MEASURES: List[Tuple[str, str]] = [("bpref", "bpref")] + [
    (f"{label}@{k}", f"{key}_{k}") for label, key in (("nDCG", "ndcg_cut"), ("MAP", "map_cut"), ("P", "P"))
    for k in CUTOFFS]
MODES = ("prime", "standard")


def extra_metrics(run: Run, qrels: Qrels) -> Dict[str, Dict[str, float]]:
    """{mode: {measure label: mean over all qrels topics}}."""
    run = truncate(run)
    graded = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut"}, relevance_level=1)
    binary = pytrec_eval.RelevanceEvaluator(qrels, {"map_cut", "P", "bpref"}, relevance_level=RELEVANT_GRADE)
    out: Dict[str, Dict[str, float]] = {}
    for mode in MODES:
        docs = prime(run, qrels) if mode == "prime" else {t: run.get(t, {}) for t in qrels}
        scored = {t: d for t, d in docs.items() if d}
        results = {t: dict(v) for t, v in graded.evaluate(scored).items()}
        for t, v in binary.evaluate(scored).items():
            results.setdefault(t, {}).update(v)
        out[mode] = {label: sum(results.get(t, {}).get(key, 0.0) for t in qrels) / len(qrels)
                     for label, key in MEASURES}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True, help="train/dev/test or arqmath1/2/3")
    parser.add_argument("--qrels", choices=("official", "all"), default="official")
    args = parser.parse_args()

    qrels = load_qrels(resolve_year(args.split), args.qrels)
    for path in args.runs:
        run, run_id = read_run(path)
        m = extra_metrics(run, qrels)
        print(f"\n{run_id or path.stem} ({resolve_year(args.split)}, {args.qrels} qrels, {len(qrels)} topics)")
        print(f"  {'measure':<11}{'prime':>8}{'standard':>10}")
        for label, _ in MEASURES:
            print(f"  {label:<11}{m['prime'][label]:>8.4f}{m['standard'][label]:>10.4f}")


if __name__ == "__main__":
    main()
