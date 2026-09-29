"""
Why do some corpus formulas yield no SLT/OPT graph? Sorts failures by cause and shows examples.

For a sample of the visual index, each SLT/OPT string that src.data.formula_graph cannot
turn into a graph is classified as
  parse:<XML error>   not well-formed after src.data.mathml.sanitize (message with the
                      position stripped, so identical causes group together)
  empty_graph         parses, but contains no visible symbol
and a few examples per class are printed with the text around the error position.

Usage
-----
    python scripts/inspect_graph_failures.py                  # 5% sample
    python scripts/inspect_graph_failures.py --sample 1 --examples 8
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.formula_graph import opt_graph, slt_graph  # noqa: E402
from src.data.mathml import sanitize  # noqa: E402
from src.data.visual_index import iter_batches  # noqa: E402

_POSITION_RE = re.compile(r":? line \d+, column \d+")


def classify(xml: str, build) -> str:
    try:
        ET.fromstring(sanitize(xml))
    except ET.ParseError as e:
        return "parse:" + _POSITION_RE.sub("", str(e))
    return "ok" if build(xml) is not None else "empty_graph"


def context(xml: str, width: int = 70) -> str:
    """Text around the parse error position (or the start, for empty graphs)."""
    s = sanitize(xml)
    try:
        ET.fromstring(s)
        return s[:2 * width]
    except ET.ParseError as e:
        col = e.position[1]
        return ("…" if col > width else "") + s[max(0, col - width):col + width]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=float, default=0.05, help="fraction of batches to inspect")
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    every = max(1, round(1 / args.sample)) if args.sample < 1 else 1

    counts = {"slt": Counter(), "opt": Counter()}
    examples = defaultdict(list)
    builders = {"slt": slt_graph, "opt": opt_graph}
    seen = 0
    for k, batch in enumerate(iter_batches(["visual_id", "latex", "slt", "opt"], batch_size=20_000)):
        if k % every:
            continue
        cols = {c: batch.column(c).to_pylist() for c in ("visual_id", "latex", "slt", "opt")}
        for i in range(batch.num_rows):
            seen += 1
            for rep in ("slt", "opt"):
                xml = cols[rep][i]
                if not xml:
                    counts[rep]["missing"] += 1
                    continue
                label = classify(xml, builders[rep])
                counts[rep][label] += 1
                if label != "ok" and len(examples[(rep, label)]) < args.examples:
                    examples[(rep, label)].append((cols["visual_id"][i], cols["latex"][i], context(xml)))

    print(f"Inspected {seen:,} visual ids (sample {args.sample:g})")
    for rep in ("slt", "opt"):
        total = sum(counts[rep].values())
        print(f"\n{rep.upper()}:")
        for label, n in counts[rep].most_common():
            print(f"  {n:>9,}  {n / total:7.3%}  {label}")
    for (rep, label), exs in sorted(examples.items()):
        print(f"\n--- {rep} {label}")
        for vid, latex, ctx in exs:
            print(f"  visual_id={vid}  latex: {' '.join((latex or '').split())[:100]}")
            print(f"    {ctx}")


if __name__ == "__main__":
    main()
