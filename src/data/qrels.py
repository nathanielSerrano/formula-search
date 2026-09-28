"""
ARQMath Task 2 relevance judgments, keyed by v3 visual_id for every year.

Grades run 0–3. ARQMath counts grades 2–3 as relevant for the binary measures
(MAP′, P′@10); nDCG′ uses the grades directly.

Usage
-----
    from src.data.qrels import load_qrels
    qrels = load_qrels("test")                 # official ARQMath-3 judgments
    qrels = load_qrels("train", kind="all")    # all ARQMath-1 judgments
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from src.data.paths import qrels_path, resolve_year

RELEVANT_GRADE = 2

Qrels = Dict[str, Dict[str, int]]  # topic_id → {visual_id: grade}


@lru_cache(maxsize=None)
def load_qrels(year: str, kind: str = "official") -> Qrels:
    path = qrels_path(year, kind)
    if not path.exists():
        hint = ("run scripts/convert_arqmath1_qrels.py" if resolve_year(year) == "arqmath1"
                else "run scripts/setup.sh")
        raise FileNotFoundError(f"{path} not found; {hint}")

    qrels: Qrels = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                grade = int(float(parts[3]))
            except ValueError:
                continue  # header or malformed line
            qrels.setdefault(parts[0], {})[parts[2]] = grade
    return qrels


def relevant(qrels: Qrels, topic_id: str) -> Dict[str, int]:
    """The judgments of one topic with grade ≥ RELEVANT_GRADE."""
    return {vid: g for vid, g in qrels.get(topic_id, {}).items() if g >= RELEVANT_GRADE}
