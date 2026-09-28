"""
Convert official ARQMath Task 2 run files (formula-instance level) into our
visual-id run files, so published systems can be scored with src.eval.evaluate.

Official format (tab separated):  topic_id  formula_id  post_id  rank  score  run_id

Formula ids are mapped to v3 visual ids through data/processed/formula_index_v2 and
deduplicated the way de_duplicate_2022.py does: per topic, results are sorted by score
(ties keep file order) and only the first formula of each visual id is kept, with its
score. Formula ids missing from the index are dropped and counted.

Scoring the published ARQMath-3 runs (scripts/setup.sh --with-runs downloads them)
and comparing with Table 4 of the ARQMath-3 overview checks the whole evaluation
path on real system output.

Usage
-----
    python -m src.eval.import_run data/raw/arqmath/runs/arqmath3/task2/*.tsv
    python -m src.eval.evaluate runs/published/DPRL-Task2-CFT-auto-math-A.tsv --split test
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.paths import FORMULA_INDEX_DIR, REPO_ROOT
from src.eval.metrics import Run
from src.eval.runs import write_run

Submission = Dict[str, List[Tuple[int, float]]]  # topic → [(formula_id, score)] in file order


def read_submission(path: Path) -> Submission:
    sub: Submission = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                formula_id, score = int(parts[1]), float(parts[4])
            except ValueError:
                continue  # header line
            sub.setdefault(parts[0], []).append((formula_id, score))
    return sub


def formula_to_visual(formula_ids: set, index_dir: Path = FORMULA_INDEX_DIR) -> Dict[int, str]:
    shards = sorted(index_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards in {index_dir}; run `python -m src.data.index`")
    wanted = pa.array(sorted(formula_ids), type=pa.int64())
    mapping: Dict[int, str] = {}
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(columns=["id", "visual_id"], batch_size=1_000_000):
            hit = batch.filter(pc.is_in(batch.column("id"), value_set=wanted))
            mapping.update(zip(hit.column("id").to_pylist(), hit.column("visual_id").to_pylist()))
    return mapping


def to_visual_run(sub: Submission, mapping: Dict[int, str]) -> Tuple[Run, int]:
    """(visual-id run, number of results whose formula id is not in the index)."""
    run: Run = {}
    unmapped = 0
    for topic, results in sub.items():
        docs: Dict[str, float] = {}
        for formula_id, score in sorted(results, key=lambda r: r[1], reverse=True):  # stable: ties keep file order
            vid = mapping.get(formula_id)
            if vid is None:
                unmapped += 1
            elif vid not in docs:
                docs[vid] = score
        run[topic] = docs
    return run, unmapped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="official Task 2 run files")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/published")
    parser.add_argument("--index-dir", type=Path, default=FORMULA_INDEX_DIR)
    args = parser.parse_args()

    subs = {path: read_submission(path) for path in args.files}
    formula_ids = {fid for sub in subs.values() for results in sub.values() for fid, _ in results}
    print(f"Mapping {len(formula_ids):,} formula ids from {len(subs)} run files to visual ids …", flush=True)
    mapping = formula_to_visual(formula_ids, args.index_dir)

    for path, sub in subs.items():
        run, unmapped = to_visual_run(sub, mapping)
        run_id = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)
        out = args.out_dir / f"{run_id}.tsv"
        write_run(run, out, run_id, depth=0)  # keep over-long runs whole; evaluate --depth decides
        n = sum(len(v) for v in sub.values())
        print(f"  {path.name}: {n:,} results → {sum(len(d) for d in run.values()):,} visual ids over {len(run)} topics"
              + (f" ({unmapped:,} unmapped formula ids dropped)" if unmapped else "") + f" → {out}")


if __name__ == "__main__":
    main()
