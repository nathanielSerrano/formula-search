"""
Reading and writing retrieval runs.

Two on-disk formats:

  run file (ours)      TREC format at visual-id level, space separated:
                           topic_id  Q0  visual_id  rank  score  run_id
                       Every system writes this; src.eval.evaluate scores it.

  ARQMath submission   the official Task 2 format, tab separated:
                           topic_id  formula_id  post_id  rank  score  run_id
                       Needed only to run the official scripts. Each visual id is
                       written as its representative formula instance from the
                       visual index, so the official deduplication maps it back
                       to the same visual id.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

from src.eval.metrics import MAX_DEPTH, Run, ranked

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_run_id(run_id: str) -> None:
    # The official scripts derive the run name from the file name, splitting on '.'.
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"run_id {run_id!r} may only contain letters, digits, '_' and '-'")


def write_run(run: Run, path: Path, run_id: str, depth: int = MAX_DEPTH) -> None:
    _check_run_id(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for topic in sorted(run):
            for rank, vid in enumerate(ranked(run[topic])[:depth], 1):
                f.write(f"{topic} Q0 {vid} {rank} {run[topic][vid]:.6f} {run_id}\n")


def read_run(path: Path) -> Tuple[Run, str]:
    """(run, run_id) from a visual-id level TREC run file."""
    run: Run = {}
    run_id = ""
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"{path}:{line_no}: expected 6 columns (topic Q0 visual_id rank score run_id)")
            topic, _, vid, _, score, run_id = parts
            run.setdefault(topic, {})[vid] = float(score)
    return run, run_id


def write_submission(run: Run, path: Path, run_id: str, representatives: Dict[str, Tuple[int, int]],
                     depth: int = MAX_DEPTH) -> int:
    """
    Write the official ARQMath Task 2 format. `representatives` maps visual_id →
    (formula_id, post_id), see src.data.visual_index.load_representatives.
    Returns the number of results dropped because their visual id has no representative.
    """
    _check_run_id(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    dropped = 0
    with path.open("w", encoding="utf-8") as f:
        for topic in sorted(run):
            rank = 0
            for vid in ranked(run[topic])[:depth]:
                if vid not in representatives:
                    dropped += 1
                    continue
                rank += 1
                formula_id, post_id = representatives[vid]
                f.write(f"{topic}\t{formula_id}\t{post_id}\t{rank}\t{run[topic][vid]:.6f}\t{run_id}\n")
    return dropped
