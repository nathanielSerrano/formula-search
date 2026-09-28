"""
The retrieval corpus: one entry per retrievable visual id.

ARQMath Task 2 judges and deduplicates results by visual id, so systems retrieve
visual ids, not the 28.3M formula instances. This module picks one representative
instance per visual id from data/processed/formula_index_v2 and writes
data/processed/visual_index/visual_index.parquet with columns

    visual_id, formula_id, post_id, type, n_instances, latex, slt, opt

Only instances with `retrievable == True` are used ('d'-flagged formulas must not be
returned). The representative is chosen by, in order: has both SLT and OPT, is not in
a comment, lowest formula id. Its formula_id/post_id are what official submission
files report for that visual id. SLT/OPT are stored as in the index; pass them through
src.data.mathml.canonical before use.

Build (on the server, after `python -m src.data.index`):
    python -m src.data.visual_index
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.paths import FORMULA_INDEX_DIR, VISUAL_INDEX_DIR, YEARS

VISUAL_INDEX_FILE = "visual_index.parquet"
_ID_BITS = 32  # formula ids are < 2^32; the priority is packed above them

SCHEMA = pa.schema([
    pa.field("visual_id",   pa.string()),
    pa.field("formula_id",  pa.int64()),
    pa.field("post_id",     pa.int64()),
    pa.field("type",        pa.string()),
    pa.field("n_instances", pa.int64()),
    pa.field("latex",       pa.string()),
    pa.field("slt",         pa.large_string()),
    pa.field("opt",         pa.large_string()),
])


def _shards(index_dir: Path) -> List[Path]:
    shards = sorted(index_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards in {index_dir}; run `python -m src.data.index`")
    return shards


def _choose_representatives(shards: List[Path]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Pass 1: (sorted representative formula ids, instance count per representative, stats)."""
    keyed, all_vids = [], []
    stats = {"rows": 0, "retrievable_rows": 0}
    for n, shard in enumerate(shards, 1):
        t = pq.read_table(shard, columns=["id", "visual_id", "type", "retrievable", "opt", "slt"])
        stats["rows"] += t.num_rows
        all_vids.append(pc.unique(t.column("visual_id")))
        t = t.filter(t.column("retrievable"))
        stats["retrievable_rows"] += t.num_rows

        no_xml = pc.invert(pc.and_(pc.is_valid(t.column("opt")), pc.is_valid(t.column("slt"))))
        is_comment = pc.equal(t.column("type"), "comment")
        priority = pc.add(pc.multiply(pc.cast(no_xml, pa.int64()), 2), pc.cast(is_comment, pa.int64()))
        key = pc.add(pc.shift_left(priority, _ID_BITS), t.column("id"))
        keyed.append(pa.table({"visual_id": t.column("visual_id"), "key": key}))
        if n % 20 == 0 or n == len(shards):
            print(f"  pass 1: {n}/{len(shards)} shards", flush=True)

    grouped = pa.concat_tables(keyed).group_by("visual_id").aggregate([("key", "min"), ("key", "count")])
    keys = grouped.column("key_min").to_numpy()
    counts = grouped.column("key_count").to_numpy()
    formula_ids = keys & ((1 << _ID_BITS) - 1)
    order = np.argsort(formula_ids)

    stats["visual_ids"] = len(pc.unique(pa.chunked_array(all_vids).combine_chunks()))
    stats["retrievable_visual_ids"] = grouped.num_rows
    stats["representatives_without_xml"] = int(((keys >> _ID_BITS) >= 2).sum())
    return formula_ids[order], counts[order], stats


def build(index_dir: Path = FORMULA_INDEX_DIR, out_dir: Path = VISUAL_INDEX_DIR) -> dict:
    t0 = time.time()
    shards = _shards(index_dir)
    rep_ids, rep_counts, stats = _choose_representatives(shards)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / VISUAL_INDEX_FILE
    written = 0
    with pq.ParquetWriter(out_path, SCHEMA, compression="snappy") as writer:
        for n, shard in enumerate(shards, 1):
            t = pq.read_table(shard, columns=["id", "visual_id", "post_id", "type", "latex", "slt", "opt"])
            ids = t.column("id").to_numpy()
            # Representatives whose id falls in this shard's id range, then an exact membership test.
            lo = np.searchsorted(rep_ids, ids.min(), side="left")
            hi = np.searchsorted(rep_ids, ids.max(), side="right")
            local = rep_ids[lo:hi]
            mask = np.isin(ids, local, assume_unique=True)
            t = t.filter(pa.array(mask))
            chosen = t.column("id").to_numpy()
            n_instances = rep_counts[lo:hi][np.searchsorted(local, chosen)]
            writer.write_table(pa.table({
                "visual_id": t.column("visual_id"),
                "formula_id": t.column("id"),
                "post_id": t.column("post_id"),
                "type": t.column("type"),
                "n_instances": pa.array(n_instances, type=pa.int64()),
                "latex": t.column("latex"),
                "slt": t.column("slt"),
                "opt": t.column("opt"),
            }, schema=SCHEMA))
            written += t.num_rows
            if n % 20 == 0 or n == len(shards):
                print(f"  pass 2: {n}/{len(shards)} shards ({written:,} visual ids written)", flush=True)

    stats["written"] = written
    stats["seconds"] = round(time.time() - t0)
    stats["path"] = str(out_path)
    return stats


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def visual_index_path(out_dir: Path = VISUAL_INDEX_DIR) -> Path:
    path = out_dir / VISUAL_INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.data.visual_index`")
    return path


def iter_batches(columns: List[str], batch_size: int = 100_000,
                 out_dir: Path = VISUAL_INDEX_DIR) -> Iterator[pa.RecordBatch]:
    """Stream the visual index, e.g. to build a retrieval index."""
    yield from pq.ParquetFile(visual_index_path(out_dir)).iter_batches(columns=columns, batch_size=batch_size)


def load_representatives(visual_ids: Optional[Iterable[str]] = None,
                         out_dir: Path = VISUAL_INDEX_DIR) -> Dict[str, Tuple[int, int]]:
    """{visual_id: (formula_id, post_id)}, optionally restricted to `visual_ids`."""
    wanted = pa.array(sorted(set(visual_ids)), type=pa.string()) if visual_ids is not None else None
    reps: Dict[str, Tuple[int, int]] = {}
    for batch in iter_batches(["visual_id", "formula_id", "post_id"], batch_size=500_000, out_dir=out_dir):
        if wanted is not None:
            batch = batch.filter(pc.is_in(batch.column("visual_id"), value_set=wanted))
        reps.update(zip(batch.column("visual_id").to_pylist(),
                        zip(batch.column("formula_id").to_pylist(), batch.column("post_id").to_pylist())))
    return reps


def _report_qrel_coverage(out_dir: Path) -> None:
    """Relevant visual ids that are not retrievable cap every system's recall."""
    from src.data.qrels import load_qrels

    indexed = set()
    for batch in iter_batches(["visual_id"], batch_size=1_000_000, out_dir=out_dir):
        indexed.update(batch.column("visual_id").to_pylist())
    print("\nRelevant (grade ≥ 2) visual ids missing from the retrieval corpus:")
    for year in YEARS:
        for kind in ("official", "all"):
            try:
                qrels = load_qrels(year, kind)
            except FileNotFoundError as e:
                print(f"  {year} {kind}: skipped ({e})")
                continue
            rel = {(t, v) for t, docs in qrels.items() for v, g in docs.items() if g >= 2}
            missing = sorted(v for _, v in rel if v not in indexed)
            print(f"  {year} {kind:<8} {len(missing)}/{len(rel)}" + (f"  e.g. {missing[:5]}" if missing else ""))


def main():
    parser = argparse.ArgumentParser(description="Build the visual-id retrieval corpus")
    parser.add_argument("--index-dir", type=Path, default=FORMULA_INDEX_DIR)
    parser.add_argument("--out-dir", type=Path, default=VISUAL_INDEX_DIR)
    parser.add_argument("--skip-qrel-check", action="store_true")
    args = parser.parse_args()

    stats = build(args.index_dir, args.out_dir)
    print(f"\nVisual index written to {stats['path']} in {stats['seconds']}s")
    print(f"  formula instances:                {stats['rows']:,}")
    print(f"  retrievable instances:            {stats['retrievable_rows']:,}")
    print(f"  visual ids (all):                 {stats['visual_ids']:,}")
    print(f"  visual ids (retrievable):         {stats['retrievable_visual_ids']:,}  (written: {stats['written']:,})")
    print(f"  representatives without SLT/OPT:  {stats['representatives_without_xml']:,}")
    if not args.skip_qrel_check:
        _report_qrel_coverage(args.out_dir)


if __name__ == "__main__":
    main()
