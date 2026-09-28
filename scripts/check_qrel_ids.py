"""
Check which formula-index ID column each ARQMath Task 2 qrel file is keyed on.

ARQMath-3 renumbered the visual IDs, and the v3 formula TSVs carry both the new
`visual_id` and the earlier `old_visual_id` (plus the per-instance formula
`id`). Joining a qrel file against the wrong column still "works" when the two
numbering schemes overlap; it just pairs topics with the wrong formulas. For
every Task 2 qrel file this script reports:

  1. Coverage: the fraction of the file's IDs that exist in each column.
  2. Agreement: for each column, how similar the matched formulas' LaTeX is to
     the topic formula, split by relevance grade. Under the right column,
     high-grade formulas look much more like the query than grade-0 ones;
     under a wrong column the grades are indistinguishable.
  3. Samples: a few topics with their top-graded formulas under each column,
     for eyeballing.

Only the qrel IDs are held in memory, so one pass over the full index is cheap
on RAM (it takes a few minutes of CPU for ~28M rows).

Usage
-----
    python scripts/check_qrel_ids.py
    python scripts/check_qrel_ids.py --max-shards 5 --samples 2
    python scripts/check_qrel_ids.py --json-out qrel_id_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

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

ID_COLUMNS = ("visual_id", "old_visual_id", "id")
LATEX_PER_ID = 3            # formula instances kept per matched ID
MIN_SEPARATION = 0.05       # relevant-minus-nonrelevant similarity needed to trust a column
SEPARATION_TIE = 0.005      # separations closer than this are treated as equal
TRUNC = 90                  # characters of LaTeX shown in samples


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """{topic_id: {doc_id: grade}}. Skips headers and malformed lines."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                grade = int(float(parts[3]))
            except ValueError:
                continue
            qrels[parts[0]][parts[2]] = grade
    return dict(qrels)


def load_topics(paths: List[Path]) -> Dict[str, str]:
    """{topic_id: query LaTeX} merged over all topic files for one year."""
    topics: Dict[str, str] = {}
    for path in paths:
        for topic in ET.parse(path).getroot():
            num = topic.get("number")
            latex = topic.findtext("Latex")
            if num and latex:
                topics[num] = latex.strip()
    return topics


def _clean_id(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in {"none", "nan", "null"} else None


def scan_index(index_dir: Path, wanted: Set[str], max_shards: Optional[int]) -> dict:
    """
    Stream the formula index once. For every ID column, record LaTeX samples and
    post types for rows whose value is a qrel ID, plus global stats on how the
    old and new visual-ID numbering schemes relate.
    """
    shards = sorted(index_dir.glob("*.parquet"))
    if max_shards:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"No Parquet shards in {index_dir}")

    hits = {col: defaultdict(lambda: {"latex": [], "types": Counter(), "has_xml": False})
            for col in ID_COLUMNS}
    stats = {
        "shards_scanned": len(shards),
        "rows": 0,
        "rows_without_xml": 0,
        "rows_old_eq_new": 0,
        "rows_old_missing": 0,
        "ranges": {col: [None, None] for col in ("visual_id", "old_visual_id")},
    }

    t0 = time.time()
    for n, shard in enumerate(shards, 1):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["id", "type", "old_visual_id", "visual_id", "latex", "opt", "slt"],
                                     batch_size=200_000):
            # OPT/SLT are only checked for nulls (formulas LaTeXML could not convert),
            # so they are never converted to Python strings.
            has_xml = pc.and_(pc.is_valid(batch.column("opt")), pc.is_valid(batch.column("slt"))).to_pylist()
            cols = {name: batch.column(name).to_pylist()
                    for name in ("id", "type", "old_visual_id", "visual_id", "latex")}
            for fid, typ, old, new, latex, xml_ok in zip(cols["id"], cols["type"], cols["old_visual_id"],
                                                          cols["visual_id"], cols["latex"], has_xml):
                stats["rows"] += 1
                if not xml_ok:
                    stats["rows_without_xml"] += 1
                row = {"id": _clean_id(fid), "old_visual_id": _clean_id(old), "visual_id": _clean_id(new)}

                if row["old_visual_id"] is None:
                    stats["rows_old_missing"] += 1
                elif row["old_visual_id"] == row["visual_id"]:
                    stats["rows_old_eq_new"] += 1

                for col in ("visual_id", "old_visual_id"):
                    v = row[col]
                    if v is not None and v.isdigit():
                        lo, hi = stats["ranges"][col]
                        iv = int(v)
                        stats["ranges"][col] = [iv if lo is None else min(lo, iv),
                                                iv if hi is None else max(hi, iv)]

                for col in ID_COLUMNS:
                    v = row[col]
                    if v is not None and v in wanted:
                        entry = hits[col][v]
                        entry["types"][typ] += 1
                        entry["has_xml"] |= xml_ok
                        if latex and len(entry["latex"]) < LATEX_PER_ID:
                            entry["latex"].append(latex)
        print(f"  scanned {n}/{len(shards)} shards ({stats['rows']:,} rows, {time.time() - t0:.0f}s)",
              flush=True)

    stats["hits"] = hits
    return stats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize_latex(s: str) -> str:
    s = re.sub(r"\\(left|right)(?![a-zA-Z])", "", s)
    return re.sub(r"[\s$]+", "", s)


def _trigrams(s: str) -> Set[str]:
    s = _normalize_latex(s)
    return {s[i:i + 3] for i in range(max(1, len(s) - 2))}


def latex_similarity(a: str, b: str) -> float:
    """Character-trigram Jaccard. Crude, but enough to separate right from wrong joins."""
    ta, tb = _trigrams(a), _trigrams(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def analyse_file(qrels: Dict[str, Dict[str, int]], topics: Dict[str, str], hits: dict) -> dict:
    ids = {doc for judged in qrels.values() for doc in judged}
    report = {
        "topics": len(qrels),
        "topics_without_latex": sorted(t for t in qrels if t not in topics),
        "unique_ids": len(ids),
        "grades": dict(sorted(Counter(g for judged in qrels.values() for g in judged.values()).items())),
        "columns": {},
    }

    relevant_ids = {doc for judged in qrels.values() for doc, g in judged.items() if g >= 2}
    found_in = {col: {i for i in ids if i in hits[col]} for col in ID_COLUMNS}
    report["ids_in_both_visual_columns"] = len(found_in["visual_id"] & found_in["old_visual_id"])

    for col in ID_COLUMNS:
        by_grade: Dict[int, List[float]] = defaultdict(list)
        for topic, judged in qrels.items():
            query = topics.get(topic)
            if not query:
                continue
            for doc, grade in judged.items():
                lats = hits[col][doc]["latex"] if doc in hits[col] else None
                if lats:
                    by_grade[grade].append(max(latex_similarity(query, l) for l in lats))

        relevant = [s for g, v in by_grade.items() if g >= 2 for s in v]
        nonrel = by_grade.get(0, [])
        types = Counter()
        for doc in found_in[col]:
            types.update(hits[col][doc]["types"])
        relevant_found = relevant_ids & found_in[col]

        report["columns"][col] = {
            "coverage": len(found_in[col]) / len(ids) if ids else 0.0,
            "found": len(found_in[col]),
            "relevant_found": len(relevant_found),
            # Relevant formulas with no instance carrying OPT+SLT: unreachable by tree-based
            # retrieval, so they cap recall.
            "relevant_without_xml": sorted(d for d in relevant_found if not hits[col][d]["has_xml"]),
            "mean_sim_by_grade": {g: round(statistics.mean(v), 4) for g, v in sorted(by_grade.items())},
            "n_by_grade": {g: len(v) for g, v in sorted(by_grade.items())},
            "separation": (round(statistics.mean(relevant) - statistics.mean(nonrel), 4)
                           if relevant and nonrel else None),
            "post_types": dict(types.most_common()),
        }

    # A column is plausible only if relevant formulas actually resemble the query. Among
    # plausible columns, coverage decides: when IDs mostly coincide (visual_id vs
    # old_visual_id), the separations are equal and only the corrected IDs differ.
    cols = report["columns"]
    agreeing = [c for c in ID_COLUMNS if (cols[c]["separation"] or 0.0) >= MIN_SEPARATION]
    agreeing.sort(key=lambda c: (cols[c]["coverage"], cols[c]["separation"]), reverse=True)
    report["verdict"] = agreeing[0] if agreeing else None
    report["indistinguishable_from"] = [
        c for c in agreeing[1:]
        if abs(cols[c]["coverage"] - cols[agreeing[0]]["coverage"]) < 1e-9
        and abs(cols[c]["separation"] - cols[agreeing[0]]["separation"]) < SEPARATION_TIE
    ]
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _trunc(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= TRUNC else s[:TRUNC - 1] + "…"


def print_report(name: str, rep: dict, qrels, topics, hits, n_samples: int):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"topics: {rep['topics']}   unique IDs: {rep['unique_ids']:,}   grades: {rep['grades']}")
    if rep["topics_without_latex"]:
        print(f"topics missing from topic XML: {len(rep['topics_without_latex'])} "
              f"(e.g. {rep['topics_without_latex'][:5]})")
    print(f"IDs present in BOTH visual_id and old_visual_id: {rep['ids_in_both_visual_columns']:,}"
          "  (a wrong join would silently succeed for these)")

    print(f"\n{'column':<15}{'coverage':>10}{'separation':>12}   mean LaTeX similarity by grade (n)")
    for col, c in rep["columns"].items():
        by_grade = "  ".join(f"{g}:{m:.3f}({c['n_by_grade'][g]})" for g, m in c["mean_sim_by_grade"].items())
        sep = f"{c['separation']:+.3f}" if c["separation"] is not None else "n/a"
        print(f"{col:<15}{c['coverage']:>10.1%}{sep:>12}   {by_grade}")

    v = rep["verdict"]
    if v is None:
        print("\n→ NO column agrees with the topic formulas: these IDs come from a different ID space "
              "and cannot be joined against this index.")
    else:
        tied = rep["indistinguishable_from"]
        print(f"\n→ likely keyed on: {v}" + (f"  (indistinguishable from {', '.join(tied)})" if tied else ""))
        print(f"  post types of all instances of matched IDs under {v}: {rep['columns'][v]['post_types']}")
        _print_no_xml(rep, v)
    _print_samples(qrels, topics, hits, n_samples)


def _print_no_xml(rep: dict, v: str):
    no_xml = rep["columns"][v]["relevant_without_xml"]
    print(f"  relevant (grade ≥ 2) IDs with no OPT/SLT in any instance: "
          f"{len(no_xml)}/{rep['columns'][v]['relevant_found']}" + (f"  e.g. {no_xml[:5]}" if no_xml else ""))


def _print_samples(qrels, topics, hits, n_samples: int):
    shown = 0
    for topic in sorted(qrels):
        if shown >= n_samples or topic not in topics:
            continue
        shown += 1
        print(f"\n  [{topic}] query: {_trunc(topics[topic])}")
        top = sorted(qrels[topic].items(), key=lambda kv: -kv[1])[:3]
        for doc, grade in top:
            print(f"    id={doc} grade={grade}")
            for col in ID_COLUMNS:
                lats = hits[col][doc]["latex"] if doc in hits[col] else []
                print(f"      {col:<14} {_trunc(lats[0]) if lats else '(not found)'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arqmath-root", type=Path, default=REPO_ROOT / "data/raw/arqmath")
    parser.add_argument("--index-dir", type=Path, default=REPO_ROOT / "data/processed/formula_index_v2")
    parser.add_argument("--max-shards", type=int, default=None, help="Scan only the first N shards (coverage will be partial)")
    parser.add_argument("--samples", type=int, default=3, help="Topics printed per qrel file")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    qrel_files = sorted((args.arqmath_root / "qrels/task2").glob("*/*.tsv"))
    if not qrel_files:
        raise FileNotFoundError(
            f"No Task 2 qrel files matching {args.arqmath_root / 'qrels/task2'}/*/*.tsv. "
            "Run from the repo root or pass --arqmath-root and --index-dir.")

    qrels_by_file, topics_by_year = {}, {}
    for path in qrel_files:
        qrels_by_file[path] = load_qrels(path)
        year = path.parent.name
        if year not in topics_by_year:
            topic_paths = sorted((args.arqmath_root / "topics/task2" / year).glob("*.xml"))
            topics_by_year[year] = load_topics(topic_paths)
            if not topic_paths:
                print(f"WARNING: no topic XML for {year}; agreement scores will be empty")
        print(f"loaded {path.relative_to(args.arqmath_root)}: {len(qrels_by_file[path])} topics")

    wanted = {doc for q in qrels_by_file.values() for judged in q.values() for doc in judged}
    print(f"\nScanning formula index for {len(wanted):,} distinct qrel IDs …")
    stats = scan_index(args.index_dir, wanted, args.max_shards)
    hits = stats.pop("hits")

    print(f"\nIndex: {stats['rows']:,} rows in {stats['shards_scanned']} shards")
    print(f"  rows with no OPT/SLT:                 {stats['rows_without_xml']:,}")
    print(f"  rows with old_visual_id == visual_id: {stats['rows_old_eq_new']:,}")
    print(f"  rows with no old_visual_id:           {stats['rows_old_missing']:,}")
    for col, (lo, hi) in stats["ranges"].items():
        print(f"  {col:<14} numeric range: {lo} … {hi}")
    if args.max_shards:
        print("  NOTE: --max-shards is set, so coverage figures are lower bounds.")

    reports = {}
    for path, qrels in qrels_by_file.items():
        name = str(path.relative_to(args.arqmath_root))
        reports[name] = analyse_file(qrels, topics_by_year[path.parent.name], hits)
        print_report(name, reports[name], qrels, topics_by_year[path.parent.name], hits, args.samples)

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for name, rep in reports.items():
        v = rep["verdict"]
        if v is None:
            print(f"{name:<55} → NONE (different ID space; not joinable)")
            continue
        c = rep["columns"][v]
        tied = f"  (= {', '.join(rep['indistinguishable_from'])})" if rep["indistinguishable_from"] else ""
        print(f"{name:<55} → {v:<14} coverage {c['coverage']:.1%}  separation {c['separation']:+.3f}{tied}")

    if args.json_out:
        args.json_out.write_text(json.dumps({"index": stats, "files": reports}, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
