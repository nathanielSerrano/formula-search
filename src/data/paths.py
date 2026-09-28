"""
Locations of the ARQMath data inside the repository, and the train/dev/test years.

All paths are relative to the repository root, so code works from any checkout
location as long as data/ has been populated by scripts/setup.sh and the build
steps in README.md.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "data/raw/arqmath"
PROCESSED_DIR = REPO_ROOT / "data/processed"

FORMULA_INDEX_DIR = PROCESSED_DIR / "formula_index_v2"
VISUAL_INDEX_DIR = PROCESSED_DIR / "visual_index"
LATEX_TSV_DIR = RAW_DIR / "formulas/latex_representation_v3"
EVAL_SCRIPTS_DIR = RAW_DIR / "eval_scripts/arqmath3"

YEARS = ("arqmath1", "arqmath2", "arqmath3")
SPLITS = {"train": "arqmath1", "dev": "arqmath2", "test": "arqmath3"}

_QREL_YEAR_TAG = {"arqmath1": "2020", "arqmath2": "2021", "arqmath3": "2022"}


def resolve_year(year_or_split: str) -> str:
    """Accept 'arqmath2' or 'dev' and return the ARQMath year name."""
    year = SPLITS.get(year_or_split, year_or_split)
    if year not in YEARS:
        raise ValueError(f"Unknown year or split {year_or_split!r}; expected one of {YEARS + tuple(SPLITS)}")
    return year


def topics_dir(year: str) -> Path:
    return RAW_DIR / "topics/task2" / resolve_year(year)


def qrels_path(year: str, kind: str = "official") -> Path:
    """
    Qrels keyed by v3 visual_id. ARQMath-1 is read from its converted copy, because
    the raw ARQMath-1 files use formula ids (see scripts/convert_arqmath1_qrels.py).
    """
    if kind not in ("official", "all"):
        raise ValueError(f"kind must be 'official' or 'all', got {kind!r}")
    year = resolve_year(year)
    name = f"qrel_task2_{_QREL_YEAR_TAG[year]}_{kind}.tsv"
    if year == "arqmath1":
        return PROCESSED_DIR / "qrels/task2/arqmath1" / name
    return RAW_DIR / "qrels/task2" / year / name
