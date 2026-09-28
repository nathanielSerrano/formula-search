"""
ARQMath Task 2 topics (queries) with their official formula representations.

Each topic names one query formula by `Formula_Id` (e.g. "q_6"). Its LaTeX comes
from the topic XML; its SLT and OPT come from the organizers' topic-formula TSVs
(data/raw/arqmath/topics/task2/<year>/formulas/), canonicalised with
src.data.mathml so they match the corpus format.

Usage
-----
    from src.data.topics import load_topics
    topics = load_topics("dev")          # or "arqmath2"
    topics["B.201"].slt
"""

from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from src.data.mathml import canonical
from src.data.paths import resolve_year, topics_dir

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class Topic:
    topic_id: str        # "B.301"
    year: str            # "arqmath3"
    formula_id: str      # "q_6"
    latex: str
    slt: Optional[str]   # canonical Presentation MathML, None if unavailable/unparseable
    opt: Optional[str]   # canonical Content MathML, None if unavailable/unparseable


def _formula_file(year_dir: Path, rep: str) -> Path:
    matches = [p for p in (year_dir / "formulas").glob("*.tsv") if rep in p.name.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {rep} topic-formula TSV in {year_dir / 'formulas'}, found {matches}; "
                                "run scripts/setup.sh")
    return matches[0]


def _read_formulas(path: Path) -> Dict[str, str]:
    """{formula_id: representation}. Columns: id, topic_id, thread_id, type, formula."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        return {row[0]: row[4] for row in reader if len(row) > 4}


@lru_cache(maxsize=None)
def load_topics(year: str) -> Dict[str, Topic]:
    """All topics of one year ('arqmath1'/'train', 'arqmath2'/'dev', 'arqmath3'/'test')."""
    year = resolve_year(year)
    year_dir = topics_dir(year)
    xml_files = sorted(year_dir.glob("Topics*.xml"))
    if len(xml_files) != 1:
        raise FileNotFoundError(f"Expected one Topics*.xml in {year_dir}, found {xml_files}; run scripts/setup.sh")

    slt = _read_formulas(_formula_file(year_dir, "slt"))
    opt = _read_formulas(_formula_file(year_dir, "opt"))

    topics: Dict[str, Topic] = {}
    for elem in ET.parse(xml_files[0]).getroot():
        topic_id = elem.get("number")
        formula_id = (elem.findtext("Formula_Id") or "").strip()
        if not topic_id or not formula_id:
            continue
        topics[topic_id] = Topic(
            topic_id=topic_id,
            year=year,
            formula_id=formula_id,
            latex=(elem.findtext("Latex") or "").strip(),
            slt=canonical(slt.get(formula_id)),
            opt=canonical(opt.get(formula_id)),
        )
    return topics
