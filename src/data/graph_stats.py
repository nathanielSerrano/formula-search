"""
Build the SLT/OPT graph vocabularies from the retrieval corpus and report graph statistics.

Every visual id in data/processed/visual_index is converted to both graphs
(src.data.formula_graph), once per distinct formula. The script counts tags, symbols
and edge types, keeps symbols seen at least --min-count times, and reports graph sizes
so the encoder's node limit can be chosen from data.

Outputs
  data/processed/graph_vocab.json        vocabularies (load with formula_graph.load_vocabs)
  data/processed/reports/graph_stats.json  counts, size percentiles, failures

Usage (on the server, after `python -m src.data.visual_index`):
    python -m src.data.graph_stats
    python -m src.data.graph_stats --sample 0.05      # quick look at ~5% of the corpus
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.data.formula_graph import GraphVocab, opt_graph, save_vocabs, slt_graph
from src.data.paths import PROCESSED_DIR, VISUAL_INDEX_DIR
from src.data.visual_index import iter_batches

VOCAB_PATH = PROCESSED_DIR / "graph_vocab.json"
REPORT_PATH = PROCESSED_DIR / "reports/graph_stats.json"
NODE_LIMITS = (32, 64, 128, 256, 512)
_BUILDERS = {"slt": slt_graph, "opt": opt_graph}


def _count(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, Counter]]:
    """Counters for one batch of (slt, opt) strings."""
    out = {}
    for i, rep in enumerate(("slt", "opt")):
        c = {"tags": Counter(), "symbols": Counter(), "edges": Counter(), "sizes": Counter(), "status": Counter()}
        for pair in pairs:
            xml = pair[i]
            if not xml:
                c["status"]["missing"] += 1
                continue
            graph = _BUILDERS[rep](xml)
            if graph is None:
                c["status"]["unparseable_or_empty"] += 1
                continue
            c["status"]["ok"] += 1
            c["sizes"][graph.num_nodes] += 1
            c["tags"].update(graph.tags)
            c["symbols"].update(s for s in graph.symbols if s)
            c["edges"].update(e for _, _, e in graph.edges)
        out[rep] = c
    return out


def _percentile(sizes: Counter, q: float) -> int:
    total, seen = sum(sizes.values()), 0
    for size in sorted(sizes):
        seen += sizes[size]
        if seen >= q * total:
            return size
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-count", type=int, default=5, help="minimum corpus count for a symbol or tag")
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of batches to process")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--visual-index-dir", type=Path, default=VISUAL_INDEX_DIR)
    parser.add_argument("--vocab-out", type=Path, default=VOCAB_PATH)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    every = max(1, round(1 / args.sample)) if args.sample < 1 else 1

    def batches():
        for k, batch in enumerate(iter_batches(["slt", "opt"], batch_size=args.batch_size,
                                               out_dir=args.visual_index_dir)):
            if k % every == 0:
                yield list(zip(batch.column("slt").to_pylist(), batch.column("opt").to_pylist()))

    totals = {rep: {k: Counter() for k in ("tags", "symbols", "edges", "sizes", "status")} for rep in _BUILDERS}
    t0, n = time.time(), 0
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(_count, batches()):
            for rep, counters in result.items():
                for key, counter in counters.items():
                    totals[rep][key].update(counter)
            n = sum(totals["slt"]["status"].values())
            print(f"  {n:,} formulas ({time.time() - t0:.0f}s)", flush=True)

    vocabs = {rep: GraphVocab.build(rep, totals[rep]["tags"], totals[rep]["symbols"], args.min_count)
              for rep in _BUILDERS}
    save_vocabs(vocabs, args.vocab_out)

    report: Dict[str, dict] = {"formulas": n, "min_count": args.min_count, "sample": args.sample}
    print(f"\n{n:,} formulas in {time.time() - t0:.0f}s (sample {args.sample:g})")
    for rep in _BUILDERS:
        t = totals[rep]
        ok = t["status"]["ok"]
        symbol_total = sum(t["symbols"].values())
        covered = sum(c for s, c in t["symbols"].items() if s in vocabs[rep].symbols)
        over = {limit: sum(c for size, c in t["sizes"].items() if size > limit) for limit in NODE_LIMITS}
        percentiles = {q: _percentile(t["sizes"], q) for q in (0.5, 0.9, 0.99, 0.999)}
        report[rep] = {
            "status": dict(t["status"]),
            "size_percentiles": {str(q): v for q, v in percentiles.items()},
            "max_size": max(t["sizes"]) if t["sizes"] else 0,
            "graphs_over_limit": {str(k): v for k, v in over.items()},
            "edge_types": dict(t["edges"].most_common()),
            "tags": len(vocabs[rep].tags),
            "distinct_symbols": len(t["symbols"]),
            "kept_symbols": sum(1 for sym in t["symbols"] if sym in vocabs[rep].symbols),
            "vocab_size": len(vocabs[rep].symbols),  # kept symbols + <pad>, <unk> and number buckets
            "symbol_occurrences_covered": covered / symbol_total if symbol_total else 0.0,
            "top_symbols": t["symbols"].most_common(50),
            "unknown_tags": sorted(tag for tag, c in t["tags"].items() if c < args.min_count),
        }
        r = report[rep]
        print(f"\n{rep.upper()}: {ok:,} graphs; {dict(t['status'])}")
        print(f"  nodes: median {percentiles[0.5]}, p90 {percentiles[0.9]}, p99 {percentiles[0.99]}, "
              f"p99.9 {percentiles[0.999]}, max {r['max_size']}")
        print("  graphs over node limit: " + ", ".join(f">{k}: {v:,} ({v / ok:.3%})" for k, v in over.items()))
        print(f"  vocabulary: {r['tags']} tags; keeps {r['kept_symbols']:,} of {r['distinct_symbols']:,} distinct symbols "
              f"(min count {args.min_count}), covering {r['symbol_occurrences_covered']:.3%} of symbol occurrences")
        print(f"  edge types: {r['edge_types']}")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nVocabularies → {args.vocab_out}\nReport → {args.report_out}")


if __name__ == "__main__":
    main()
