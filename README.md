# Formula Search on ARQMath

Final project for **COS 587 – Introduction to Information Retrieval**, University of Southern Maine.

The project studies **mathematical formula retrieval**: given a formula taken from a Math Stack Exchange
question, find formulas in the Math Stack Exchange archive (2010–2018) that a reader would consider
relevant. This is Task 2 of the [ARQMath](https://www.cs.rit.edu/~dprl/ARQMath/) lab, whose collection,
queries and relevance judgments are used throughout.

The retrieval approach under investigation encodes each formula's two tree representations, the
**Symbol Layout Tree** (SLT, how the formula looks) and the **Operator Tree** (OPT, what it computes),
with graph neural networks, and compares the resulting dense retriever against sparse (BM25) and
structural (Tangent-CFT–style) baselines.

## Status

The project restarts an earlier prototype whose evaluation could not be trusted. Work is proceeding
evaluation-first:

| Stage | State |
|---|---|
| Data download, formula index, qrels in one ID space | done |
| Topic (query) loading and run-file / metric harness | in progress |
| BM25 baseline, reproduction of Tangent-CFT numbers | next |
| Graph representation of formulas (symbols, edge types) | planned |
| Training data (pairs, judged negatives) and GNN encoder | planned |

Code under `src/task3/` is from the earlier prototype and is kept only for reference while it is replaced.
It targets the old index layout and should not be run.

## Data

All data comes from ARQMath (collection v1.3, formula index v3). File-level details (formats, columns,
ID spaces, known issues) are in [`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md).

| | |
|---|---|
| Formula instances | 28,320,920 |
| with SLT and OPT (LaTeXML 0.8.5) | 28,267,310 |
| flagged `d` / `dv` (must not be returned) | 4,075,801 |

Queries and judgments come from the three ARQMath Task 2 years. Topics are drawn from posts made after
2018, so none of the queries appear in the collection.

| Year | Role | Judged topics (official / all) |
|---|---|---|
| ARQMath-1 (2020) | training | 45 / 74 |
| ARQMath-2 (2021) | validation | 58 / 70 |
| ARQMath-3 (2022) | test, used only for final numbers | 76 / 86 |

Things that are easy to get wrong with this data (each was verified against the collection):

- **Visual IDs.** Task 2 is judged at the level of *visual IDs* (groups of formulas that look identical).
  ARQMath-2 and ARQMath-3 qrels use the index's `visual_id` column. The ARQMath-1 visual-ID qrels use an
  older numbering that does not exist in the v3 index, so ARQMath-1 is re-keyed from its formula-ID qrels
  (`scripts/convert_arqmath1_qrels.py`).
- **Issue flags.** Formulas flagged `d` are missing from the post XML and must never be returned;
  `v` only marks a corrected visual ID and the formula is usable. Both are stored in the index.
- **Topic formulas.** Queries use ARQMath's official SLT/OPT for each topic's `Formula_Id`. The ARQMath-3
  topic SLT contains unescaped `<` characters that break strict XML parsing for several judged topics and
  must be sanitised before use.

## Evaluation

Runs follow the official ARQMath Task 2 protocol:

- results are deduplicated by visual ID and `d`-flagged formulas are excluded;
- unjudged visual IDs are removed before scoring ("prime" metrics);
- reported metrics are **nDCG′** (graded relevance), **MAP′** and **P′@10** (grades 2–3 count as relevant),
  computed over every topic in the official qrels.

The official scripts (`de_duplicate_2022.py`, `task2_get_results.py`) are downloaded by `setup.sh` and
require the `trec_eval` binary.

## Setup

Data and training live on a GPU server; the steps below are run from the repository root there.

```bash
# 1. Python environment (Python 3.8; versions pinned from the server environment)
pip install -r requirements.txt

# 2. Download ARQMath: collection, formula indexes, topics, topic formulas, qrels, eval scripts
bash scripts/setup.sh

# 3. Build the Parquet formula index (~15 min) → data/processed/formula_index_v2/
python -m src.data.index

# 4. Re-key the ARQMath-1 qrels to v3 visual IDs → data/processed/qrels/task2/arqmath1/
python scripts/convert_arqmath1_qrels.py
```

Optional diagnostics, whose findings are summarised above:

```bash
python scripts/check_qrel_ids.py          # which index column each qrel file is keyed on
python scripts/inspect_flagged_xml.py     # whether 'v'-flagged formulas carry the right SLT
```

## Repository layout

```
scripts/
  setup.sh                   download ARQMath into data/raw/arqmath/
  convert_arqmath1_qrels.py  ARQMath-1 qrels: formula IDs → v3 visual IDs
  check_qrel_ids.py          diagnostic: qrel ID space per file
  inspect_flagged_xml.py     diagnostic: SLT/LaTeX consistency of flagged formulas
src/
  data/
    index.py                 build the Parquet formula index from the ARQMath TSVs
    formula_graph.py         MathML → PyTorch Geometric graphs (to be redesigned)
  task3/                     earlier prototype, reference only
configs/                     experiment configs
docs/
  DATA_ACQUISITION.md        data sources, formats and known issues
data/
  raw/arqmath/               downloaded data (not tracked)
  processed/                 generated index, qrels and reports (not tracked)
  samples/                   small fixtures for local testing
```

## References

- B. Mansouri, R. Zanibbi, D. W. Oard, A. Agarwal. *Overview of ARQMath-3 (2022): Third CLEF Lab on
  Answer Retrieval for Questions on Math.* CLEF 2022 Working Notes.
- B. Mansouri, S. Rohatgi, D. W. Oard, J. Wu, C. L. Giles, R. Zanibbi. *Tangent-CFT: An Embedding Model for
  Mathematical Formulas.* ICTIR 2019.
