"""
Check whether 'v'-flagged formulas carry SLT that belongs to a different formula.

The index build found that a 'v' row's SLT alttext matches its LaTeX less often
than for clean rows (83.5% vs 93.1%). Some mismatches are only formatting
(LaTeXML re-spaces, drops \\left/\\right, …), so this script sorts every
mismatch into buckets using progressively looser normalisation. Only the
'different' bucket suggests a wrong formula. Buckets are compared between
flagged rows and a baseline sample of clean rows, and examples are printed.

Usage
-----
    python scripts/inspect_flagged_xml.py
    python scripts/inspect_flagged_xml.py --examples 20 --clean-per-shard 500
"""

from __future__ import annotations

import argparse
import html
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _find_repo_root() -> Path:
    """First directory holding data/processed: the cwd or an ancestor of it or of this file."""
    here = Path(__file__).resolve().parent
    for start in (Path.cwd().resolve(), here):
        for candidate in (start, *start.parents):
            if (candidate / "data/processed").is_dir():
                return candidate
    return here.parent


REPO_ROOT = _find_repo_root()
FLAGGED = ("v", "dv")
_ALTTEXT_RE = re.compile(r'alttext="([^"]*)"')

# Formatting-only LaTeX that LaTeXML may add, drop or rewrite.
_COSMETIC_RE = re.compile(
    r"\\(left|right|big|Big|bigg|Bigg)(?![a-zA-Z])"
    r"|\\(displaystyle|textstyle|limits|nolimits|mathrm|operatorname|rm|text|mbox)(?![a-zA-Z])"
    r"|\\[,;:! ]|\\q?quad(?![a-zA-Z])|~|&"
)


def _prep(s: str) -> str:
    """Unescape and drop LaTeXML's '%'-newline line wraps; whitespace is kept so commands stay delimited."""
    return re.sub(r"(?<!\\)%\s*", "", html.unescape(s))


def _strict(s: str) -> str:
    """Same normalisation as the index build check."""
    return re.sub(r"[\s{}]+", "", _prep(s))


def _loose(s: str) -> str:
    s = re.sub(r"[\s{}]+", "", _COSMETIC_RE.sub(" ", _prep(s)))
    return s.rstrip(".,;:")


_SYNONYMS = {
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq", r"\to": r"\rightarrow",
    r"\gets": r"\leftarrow", r"\dots": r"\ldots", r"\cdots": r"\ldots",
}


def _symbols(s: str) -> list:
    """Commands, letters and digits in order (synonyms unified); brackets, scripts and punctuation ignored."""
    tokens = re.findall(r"\\[a-zA-Z]+|[A-Za-z0-9]", _COSMETIC_RE.sub(" ", _prep(s)))
    return [_SYNONYMS.get(t, t) for t in tokens]


def classify(latex: str, slt: str) -> str:
    m = _ALTTEXT_RE.search(slt or "")
    if not m:
        return "no_alttext"
    alt = m.group(1)
    if _strict(alt) == _strict(latex):
        return "match"
    if _loose(alt) == _loose(latex):
        return "cosmetic"
    if _symbols(alt) == _symbols(latex):
        return "same_symbols"
    return "different"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", type=Path, default=REPO_ROOT / "data/processed/formula_index_v2")
    parser.add_argument("--clean-per-shard", type=int, default=300, help="Clean rows sampled per shard as baseline")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shards = sorted(args.index_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards in {args.index_dir}")

    counts = {g: Counter() for g in ("clean",) + FLAGGED}
    examples = defaultdict(list)  # (group, bucket) → [(id, visual_id, latex, alttext)]
    cols = ["id", "visual_id", "issue", "latex", "slt"]
    flagged_set = pa.array(list(FLAGGED))

    for n, shard in enumerate(shards, 1):
        table = pq.read_table(shard, columns=cols)
        flagged = table.filter(pc.is_in(table.column("issue"), value_set=flagged_set))
        clean = table.filter(pc.is_null(table.column("issue")))
        take = sorted(rng.sample(range(clean.num_rows), min(args.clean_per_shard, clean.num_rows)))
        clean = clean.take(pa.array(take, type=pa.int64()))

        for group_table, fixed_group in ((flagged, None), (clean, "clean")):
            for row in group_table.to_pylist():
                if row["slt"] is None:
                    continue
                group = fixed_group or row["issue"]
                bucket = classify(row["latex"], row["slt"])
                counts[group][bucket] += 1
                ex = examples[(group, bucket)]
                if bucket != "match" and len(ex) < args.examples:
                    alt = _ALTTEXT_RE.search(row["slt"])
                    ex.append((row["id"], row["visual_id"], row["latex"], html.unescape(alt.group(1)) if alt else None))
        if n % 20 == 0 or n == len(shards):
            print(f"  scanned {n}/{len(shards)} shards", flush=True)

    buckets = ("match", "cosmetic", "same_symbols", "different", "no_alttext")
    print(f"\n{'group':<8}{'rows':>10}  " + "".join(f"{b:>15}" for b in buckets))
    for group, c in counts.items():
        total = sum(c.values())
        if total:
            print(f"{group:<8}{total:>10,}  " + "".join(f"{c[b] / total:>15.2%}" for b in buckets))

    print("\nBuckets: match = identical after whitespace/brace removal; cosmetic = identical after "
          "dropping spacing/sizing macros; same_symbols = same commands, letters and digits in order but different brackets or "
          "scripts (e.g. x^2 vs x_2, so may still be a real difference); "
          "different = likely a different formula.")

    for group in ("v", "clean"):
        for bucket in ("different", "same_symbols"):
            ex = examples.get((group, bucket))
            if not ex:
                continue
            print(f"\n--- {group} rows, bucket '{bucket}' ---")
            for fid, vid, latex, alt in ex:
                print(f"  id={fid} visual_id={vid}")
                print(f"    latex:   {' '.join(latex.split())[:110]}")
                print(f"    alttext: {' '.join((alt or '').split())[:110]}")


if __name__ == "__main__":
    main()
