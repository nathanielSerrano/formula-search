"""
Score a run with the official ARQMath-3 Task 2 scripts, to confirm that
src.eval.metrics reproduces them.

Pipeline (as the organizers ran it):
  de_duplicate_2022.py  formula ids → visual ids via the LaTeX TSVs, keep the first
                        hit per visual id, drop unjudged visual ids ("prime" run)
  task2_get_results.py  trec_eval: nDCG′ (-m ndcg), MAP′ (-l2 -m map), P′@10 (-l2 -m P)

Requires the trec_eval binary (https://github.com/usnistgov/trec_eval) and the
LaTeX TSVs from scripts/setup.sh. The dedup script loads all 28M formula ids,
so expect a few minutes and a few GB of memory.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.data.paths import EVAL_SCRIPTS_DIR, LATEX_TSV_DIR, qrels_path
from src.eval.metrics import Run
from src.eval.runs import write_submission


def run_official(run: Run, run_id: str, year: str, kind: str, representatives: Dict[str, Tuple[int, int]],
                 trec_eval: Path, work_dir: Optional[Path] = None) -> Dict[str, float]:
    """Official nDCG′ / MAP′ / P′@10 for `run`, as {'ndcg', 'map', 'P_10'}."""
    for script in ("de_duplicate_2022.py", "task2_get_results.py"):
        if not (EVAL_SCRIPTS_DIR / script).exists():
            raise FileNotFoundError(f"{EVAL_SCRIPTS_DIR / script} not found; run scripts/setup.sh")
    if not LATEX_TSV_DIR.is_dir():
        raise FileNotFoundError(f"{LATEX_TSV_DIR} not found; run scripts/setup.sh")

    qrels = qrels_path(year, kind)
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="arqmath_official_"))
    sub_dir, prime_dir = work / "submission", work / "prime"
    sub_dir.mkdir(parents=True, exist_ok=True)
    prime_dir.mkdir(parents=True, exist_ok=True)

    dropped = write_submission(run, sub_dir / f"{run_id}.tsv", run_id, representatives)
    if dropped:
        print(f"  warning: {dropped} results had no representative formula and were left out", flush=True)

    subprocess.run([sys.executable, str(EVAL_SCRIPTS_DIR / "de_duplicate_2022.py"),
                    "-qre", str(qrels), "-tsv", str(LATEX_TSV_DIR),
                    "-sub", str(sub_dir), "-pri", str(prime_dir)], check=True)
    results = work / "results.tsv"
    subprocess.run([sys.executable, str(EVAL_SCRIPTS_DIR / "task2_get_results.py"),
                    "-eva", str(trec_eval), "-qre", str(qrels),
                    "-pri", str(prime_dir), "-res", str(results)], check=True)

    with results.open(encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    # Header: System, nDCG', mAP', p@10
    for row in rows[1:]:
        if row and row[0] == run_id:
            return {"ndcg": float(row[1]), "map": float(row[2]), "P_10": float(row[3])}
    raise RuntimeError(f"run {run_id!r} not found in {results}")
