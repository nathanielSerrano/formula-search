"""
Score a visual-id level run file on one ARQMath year.

Usage
-----
    python -m src.eval.evaluate runs/bm25_dev.tsv --split dev
    python -m src.eval.evaluate runs/bm25_dev.tsv --split dev --qrels all --per-topic out.json
    python -m src.eval.evaluate runs/bm25_test.tsv --split test --official --trec-eval ~/trec_eval/trec_eval

--official also scores the run with the organizers' scripts and prints both results
side by side; they should agree whenever no topic is missing from the run.
The test split (ARQMath-3) is for final numbers only; tune on dev.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.paths import resolve_year
from src.data.qrels import load_qrels
from src.eval.metrics import LABELS, MAX_DEPTH, MEASURES, evaluate
from src.eval.runs import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="visual-id level TREC run file")
    parser.add_argument("--split", required=True, help="train/dev/test or arqmath1/2/3")
    parser.add_argument("--qrels", choices=("official", "all"), default="official")
    parser.add_argument("--per-topic", type=Path, help="write per-topic metrics as JSON")
    parser.add_argument("--official", action="store_true", help="also score with the official ARQMath scripts")
    parser.add_argument("--trec-eval", type=Path, help="path to the trec_eval binary (for --official)")
    parser.add_argument("--depth", type=int, default=MAX_DEPTH,
                        help=f"results scored per topic (default {MAX_DEPTH}, the ARQMath limit; 0 = all, "
                             "which is what the official scripts do with over-long runs)")
    args = parser.parse_args()
    if args.official and not args.trec_eval:
        parser.error("--official needs --trec-eval")

    year = resolve_year(args.split)
    qrels = load_qrels(year, args.qrels)
    run, run_id = read_run(args.run)

    unknown = sorted(set(run) - set(qrels))
    result = evaluate(run, qrels, depth=args.depth)
    print(f"{run_id} on {year} ({args.qrels} qrels)")
    print(f"  {result.summary()}")
    if result.missing_topics:
        print(f"  topics with no judged results: {', '.join(result.missing_topics)}")
    if unknown:
        print(f"  note: {len(unknown)} run topics have no judgments and are ignored")

    if args.per_topic:
        args.per_topic.parent.mkdir(parents=True, exist_ok=True)
        args.per_topic.write_text(json.dumps({"run_id": run_id, "year": year, "qrels": args.qrels,
                                              "mean": result.mean, "missing_topics": result.missing_topics,
                                              "per_topic": result.per_topic}, indent=2))
        print(f"  per-topic metrics → {args.per_topic}")

    if args.official:
        from src.data.visual_index import load_representatives
        from src.eval.official import run_official

        reps = load_representatives({vid for docs in run.values() for vid in docs})
        official = run_official(run, run_id, year, args.qrels, reps, args.trec_eval)
        print("\n  measure    ours      official")
        for m in MEASURES:
            print(f"  {LABELS[m]:<9}  {result.mean[m]:.4f}    {official[m]:.4f}")
        if result.missing_topics:
            print(f"  (both average over all {len(qrels)} topics; trec_eval runs with -c so the "
                  f"{len(result.missing_topics)} topic(s) without judged results score 0 in both)")


if __name__ == "__main__":
    main()
