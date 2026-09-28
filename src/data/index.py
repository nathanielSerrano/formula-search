"""
Build the formula index Parquet shards from the three ARQMath TSV families.

The three TSV families (latex_representation_v3, opt_representation_v3,
slt_representation_v3) share identical row IDs and matching filenames
(1.tsv through 101.tsv).  This script streams them in lockstep, joins each
row on the shared `id` field, and writes one Parquet shard per file triplet.

Issue flags
-----------
ARQMath marks problem formulas in the `issue` column (README_formulas_V3.0.md):
  'd'  the formula is missing from the post XML and must not be returned for Task 2
  'v'  the formula's old visual id was wrong (corrected in `visual_id`)
  'dv' both
The flag is stored as-is and every formula keeps its OPT/SLT. Retrieval code must
drop rows with `retrievable == False` ('d' flags). An earlier version blanked the
XML of every flagged row, which hid ~4.1M formulas, including relevant ARQMath-3
formulas whose visual id had been corrected.

Output
------
data/processed/formula_index_v2/shard_001.parquet  …  shard_101.parquet

Usage
-----
    python -m src.data.index [--force] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMULAS_DIR = REPO_ROOT / "data/raw/arqmath/formulas"
LATEX_DIR = FORMULAS_DIR / "latex_representation_v3"
OPT_DIR   = FORMULAS_DIR / "opt_representation_v3"
SLT_DIR   = FORMULAS_DIR / "slt_representation_v3"
OUT_DIR   = REPO_ROOT / "data/processed/formula_index_v2"

# Every Nth unflagged row is used as the baseline for the alttext consistency check.
CLEAN_SAMPLE_EVERY = 100

SCHEMA = pa.schema([
    pa.field("id",            pa.int64()),
    pa.field("post_id",       pa.int64()),
    pa.field("thread_id",     pa.int64()),
    pa.field("type",          pa.string()),
    pa.field("comment_id",    pa.string()),
    pa.field("old_visual_id", pa.string()),
    pa.field("visual_id",     pa.string()),
    pa.field("issue",         pa.string()),
    pa.field("retrievable",   pa.bool_()),
    pa.field("latex",         pa.string()),
    pa.field("opt",           pa.large_string()),
    pa.field("slt",           pa.large_string()),
])

_ALTTEXT_RE = re.compile(r'alttext="([^"]*)"')


def _sorted_tsv_files(directory: Path) -> list:
    """Return TSV files sorted numerically by stem (1 < 2 < … < 101)."""
    return sorted(directory.glob("*.tsv"), key=lambda p: int(p.stem))


def _formula_value(row: list):
    """Return the formula string, or None if the field is empty."""
    formula = row[8].strip() if len(row) > 8 else ""
    return formula or None


def _latex_key(s: str) -> str:
    """Loose LaTeX form for comparing a row's LaTeX with its SLT alttext.

    LaTeXML adds braces and wraps long alttext with '%'-newline line breaks.
    """
    s = re.sub(r"(?<!\\)%\s*", "", html.unescape(s))
    return re.sub(r"[\s{}]+", "", s)


def _slt_matches_latex(slt: str, latex: str) -> bool:
    m = _ALTTEXT_RE.search(slt)
    return bool(m) and _latex_key(m.group(1)) == _latex_key(latex)


def build_shard(
    latex_path: Path,
    opt_path: Path,
    slt_path: Path,
    out_path: Path,
) -> Counter:
    """Stream one TSV file triplet in lockstep and write a Parquet shard. Returns build stats."""
    cols = {name: [] for name in SCHEMA.names}
    stats: Counter = Counter()

    f_l = latex_path.open(newline="", encoding="utf-8")
    f_o = opt_path.open(newline="", encoding="utf-8")
    f_s = slt_path.open(newline="", encoding="utf-8")
    try:
        reader_l = csv.reader(f_l, delimiter="\t")
        reader_o = csv.reader(f_o, delimiter="\t")
        reader_s = csv.reader(f_s, delimiter="\t")
        next(reader_l)
        next(reader_o)
        next(reader_s)

        for row_l, row_o, row_s in zip(reader_l, reader_o, reader_s):
            lid, oid, sid = int(row_l[0]), int(row_o[0]), int(row_s[0])
            if lid != oid or lid != sid:
                raise ValueError(f"Row ID mismatch in {latex_path.name}: latex={lid}, opt={oid}, slt={sid}")

            flags = {r[7].strip() for r in (row_l, row_o, row_s)}
            if len(flags) > 1:
                stats["issue_disagreements"] += 1
            issue = "".join(sorted(set("".join(flags))))  # union of flags, e.g. "dv"

            latex = row_l[8].strip()
            opt, slt = _formula_value(row_o), _formula_value(row_s)

            stats["rows"] += 1
            stats[f"issue={issue or '-'}"] += 1
            if opt is None or slt is None:
                stats["rows_without_xml"] += 1

            # Is the stored SLT really this formula's? Compare flagged rows against a clean baseline.
            group = f"issue={issue}" if issue else ("clean_sample" if lid % CLEAN_SAMPLE_EVERY == 0 else None)
            if group and slt:
                stats[f"alttext_checked[{group}]"] += 1
                stats[f"alttext_match[{group}]"] += _slt_matches_latex(slt, latex)

            cols["id"].append(lid)
            cols["post_id"].append(int(row_l[1]))
            cols["thread_id"].append(int(row_l[2]))
            cols["type"].append(row_l[3])
            cols["comment_id"].append(row_l[4].strip() or None)
            cols["old_visual_id"].append(row_l[5])
            cols["visual_id"].append(row_l[6])
            cols["issue"].append(issue or None)
            cols["retrievable"].append("d" not in issue)
            cols["latex"].append(latex)
            cols["opt"].append(opt)
            cols["slt"].append(slt)
    finally:
        f_l.close()
        f_o.close()
        f_s.close()

    table = pa.table({name: pa.array(cols[name], type=SCHEMA.field(name).type) for name in SCHEMA.names},
                     schema=SCHEMA)
    pq.write_table(table, out_path, compression="snappy")
    return stats


def _print_stats(stats: Counter) -> None:
    print("\nBuild statistics (shards built in this run):")
    print(f"  rows:                  {stats['rows']:,}")
    print(f"  rows without OPT/SLT:  {stats['rows_without_xml']:,}")
    print(f"  issue-flag disagreements across the three TSVs: {stats['issue_disagreements']:,}")
    for key in sorted(k for k in stats if k.startswith("issue=")):
        print(f"  {key:<22} {stats[key]:,}")
    print("  SLT alttext matches LaTeX (should be similar across groups; low for flagged rows = suspect XML):")
    for key in sorted(k for k in stats if k.startswith("alttext_checked[")):
        group = key[len("alttext_checked["):-1]
        checked, matched = stats[key], stats[f"alttext_match[{group}]"]
        print(f"    {group:<14} {matched:,}/{checked:,} ({matched / checked:.1%})")


def main(force: bool = False, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    latex_files = _sorted_tsv_files(LATEX_DIR)
    opt_files = _sorted_tsv_files(OPT_DIR)
    slt_files = _sorted_tsv_files(SLT_DIR)

    if not latex_files:
        print(f"ERROR: No TSV files found in {LATEX_DIR}")
        print("Run: bash scripts/setup.sh")
        sys.exit(1)

    if len(latex_files) != len(opt_files) or len(latex_files) != len(slt_files):
        print(f"WARNING: TSV file count mismatch — latex={len(latex_files)}, opt={len(opt_files)}, slt={len(slt_files)}")

    n_files = min(len(latex_files), len(opt_files), len(slt_files))
    total_rows, skipped = 0, 0
    stats: Counter = Counter()

    for latex_f, opt_f, slt_f in tqdm(
        zip(latex_files[:n_files], opt_files[:n_files], slt_files[:n_files]),
        total=n_files,
        desc="Building formula index",
        unit="shard",
    ):
        stem = int(latex_f.stem)
        out_path = out_dir / f"shard_{stem:03d}.parquet"
        if out_path.exists() and not force:
            total_rows += pq.read_metadata(out_path).num_rows
            skipped += 1
            continue
        shard_stats = build_shard(latex_f, opt_f, slt_f, out_path)
        stats.update(shard_stats)
        total_rows += shard_stats["rows"]

    print(f"\nFormula index complete.\n  Shards written: {n_files - skipped}\n  Shards skipped: {skipped}\n  Total rows: {total_rows:,}\n  Output: {out_dir}")
    if stats["rows"]:
        _print_stats(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build formula index Parquet shards from ARQMath TSVs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing shards")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    main(force=args.force, out_dir=args.out_dir)
