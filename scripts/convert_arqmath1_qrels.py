"""
Re-key the ARQMath-1 Task 2 qrels from formula ids to ARQMath-3 (v3) visual ids.

The ARQMath-1 visual-id qrels use ARQMath-1's own visual ids, which do not exist
in the v3 formula index (see scripts/check_qrel_ids.py). The formula-id versions
(qrel_task2_2020_formula_id.tsv, qrel_task2_2020_all_formula_id.tsv) do match the
index's `id` column, so each judged formula can be mapped to its v3 `visual_id`.

ARQMath-1 grouped formulas into visual ids differently, so several judged
formulas can land on one v3 visual id, sometimes with different grades. Such a
(topic, visual_id) pair gets the maximum grade: the formula appears in the
collection exactly as a formula the assessor graded that high. Every conflict is
written to a side file so the choice can be audited.

Outputs (same 4-column TREC format as the ARQMath-2/3 qrels):
    data/processed/qrels/task2/arqmath1/qrel_task2_2020_official.tsv
    data/processed/qrels/task2/arqmath1/qrel_task2_2020_all.tsv
    …plus a *_conflicts.tsv next to each.

Usage
-----
    python scripts/convert_arqmath1_qrels.py
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _find_repo_root() -> Path:
    """First directory holding data/raw/arqmath: the cwd or an ancestor of it or of this file."""
    here = Path(__file__).resolve().parent
    for start in (Path.cwd().resolve(), here):
        for candidate in (start, *start.parents):
            if (candidate / "data/raw/arqmath").is_dir():
                return candidate
    return here.parent


REPO_ROOT = _find_repo_root()

# input name (under qrels/task2/arqmath1/) → output name
CONVERSIONS = {
    "qrel_task2_2020_formula_id.tsv": "qrel_task2_2020_official.tsv",
    "qrel_task2_2020_all_formula_id.tsv": "qrel_task2_2020_all.tsv",
}

Judgment = Tuple[str, int, int]  # (topic, formula_id, grade)


def load_formula_qrels(path: Path) -> List[Judgment]:
    judgments = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                judgments.append((parts[0], int(parts[2]), int(float(parts[3]))))
            except ValueError:
                continue  # header or malformed line
    return judgments


def map_formula_ids(index_dir: Path, formula_ids: set) -> Dict[int, str]:
    """formula id → v3 visual_id, filtering each batch in Arrow so only matches reach Python."""
    shards = sorted(index_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards in {index_dir}")

    wanted = pa.array(sorted(formula_ids), type=pa.int64())
    mapping: Dict[int, str] = {}
    t0 = time.time()
    for n, shard in enumerate(shards, 1):
        for batch in pq.ParquetFile(shard).iter_batches(columns=["id", "visual_id"], batch_size=500_000):
            hit = batch.filter(pc.is_in(batch.column("id"), value_set=wanted))
            mapping.update(zip(hit.column("id").to_pylist(), hit.column("visual_id").to_pylist()))
        if n % 20 == 0 or n == len(shards):
            print(f"  scanned {n}/{len(shards)} shards, mapped {len(mapping):,}/{len(formula_ids):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return mapping


def convert(judgments: List[Judgment], mapping: Dict[int, str], out_path: Path) -> dict:
    grades: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)  # (topic, vid) → [(fid, grade)]
    unmapped = []
    for topic, fid, grade in judgments:
        vid = mapping.get(fid)
        if vid is None:
            unmapped.append((topic, fid, grade))
        else:
            grades[(topic, vid)].append((fid, grade))

    conflicts = {key: fg for key, fg in grades.items() if len({g for _, g in fg}) > 1}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _sort_key(key):
        topic, vid = key
        return topic, int(vid) if vid.isdigit() else vid

    with out_path.open("w", encoding="utf-8") as f:
        for key in sorted(grades, key=_sort_key):
            f.write(f"{key[0]}\t0\t{key[1]}\t{max(g for _, g in grades[key])}\n")

    conflict_path = out_path.with_name(out_path.stem + "_conflicts.tsv")
    with conflict_path.open("w", encoding="utf-8") as f:
        f.write("topic\tvisual_id\tassigned_grade\tformula_id:grade,…\n")
        for key in sorted(conflicts, key=_sort_key):
            fg = conflicts[key]
            pairs = ",".join(f"{fid}:{g}" for fid, g in sorted(fg))
            f.write(f"{key[0]}\t{key[1]}\t{max(g for _, g in fg)}\t{pairs}\n")

    spreads = Counter(max(g for _, g in fg) - min(g for _, g in fg) for fg in conflicts.values())
    return {
        "formula_judgments": len(judgments),
        "unmapped": len(unmapped),
        "unmapped_examples": unmapped[:5],
        "topics": len({t for t, _ in grades}),
        "visual_id_judgments": len(grades),
        "grades_before": dict(sorted(Counter(g for _, _, g in judgments).items())),
        "grades_after": dict(sorted(Counter(max(g for _, g in fg) for fg in grades.values()).items())),
        "conflicts": len(conflicts),
        "conflict_spread": dict(sorted(spreads.items())),
        "conflicts_crossing_relevance": sum(
            1 for fg in conflicts.values() if min(g for _, g in fg) < 2 <= max(g for _, g in fg)),
        "conflict_file": conflict_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qrel-dir", type=Path, default=REPO_ROOT / "data/raw/arqmath/qrels/task2/arqmath1")
    parser.add_argument("--index-dir", type=Path, default=REPO_ROOT / "data/processed/formula_index_v2")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/processed/qrels/task2/arqmath1")
    args = parser.parse_args()

    inputs = {}
    for src_name in CONVERSIONS:
        path = args.qrel_dir / src_name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/setup.sh to download it")
        inputs[src_name] = load_formula_qrels(path)
        print(f"loaded {src_name}: {len(inputs[src_name]):,} formula judgments")

    formula_ids = {fid for judgments in inputs.values() for _, fid, _ in judgments}
    print(f"\nMapping {len(formula_ids):,} formula ids to v3 visual ids …")
    mapping = map_formula_ids(args.index_dir, formula_ids)

    for src_name, dst_name in CONVERSIONS.items():
        out_path = args.out_dir / dst_name
        s = convert(inputs[src_name], mapping, out_path)
        print(f"\n{'=' * 70}\n{src_name} → {out_path}\n{'=' * 70}")
        print(f"formula judgments:   {s['formula_judgments']:,}  (unmapped: {s['unmapped']:,}"
              + (f", e.g. {s['unmapped_examples']}" if s["unmapped"] else "") + ")")
        print(f"visual-id judgments: {s['visual_id_judgments']:,} over {s['topics']} topics")
        print(f"grades before (formula level):   {s['grades_before']}")
        print(f"grades after (visual-id level):  {s['grades_after']}")
        print(f"grade conflicts: {s['conflicts']:,} (topic, visual_id) pairs; spread {s['conflict_spread']}; "
              f"{s['conflicts_crossing_relevance']:,} straddle the relevant/non-relevant line (grade 2)")
        print(f"conflict details: {s['conflict_file']}")


if __name__ == "__main__":
    main()
