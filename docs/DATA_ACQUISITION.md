# Data Acquisition Guide

This project uses the formula retrieval data of the **ARQMath** lab (collection v1.3, formula index v3):
formulas from Math Stack Exchange posts written 2010–2018, query formulas from later posts, and human
relevance judgments for ARQMath Task 2 (formula retrieval) from its three years (2020–2022).

Everything is downloaded by one script and turned into analysis-ready files by two more:

```bash
bash scripts/setup.sh                      # download → data/raw/arqmath/
python -m src.data.index                   # formula index → data/processed/formula_index_v2/
python scripts/convert_arqmath1_qrels.py   # ARQMath-1 qrels → data/processed/qrels/task2/arqmath1/
```

`setup.sh` is idempotent: existing files are not re-downloaded and extracted archives are not
re-extracted. `--no-unzip` downloads without extracting; `--with-collection` also fetches the post
collection (see [Optional: post collection](#optional-post-collection)).

---

## 1. Formula index

`data/raw/arqmath/formulas/` (≈1.8 GB compressed, ≈21 GB unpacked)

| File | Size | Contents |
|------|------|----------|
| `latex_representation_v3.zip` | 406 MB | LaTeX of every formula: the official index |
| `opt_representation_v3.zip` | 587 MB | Operator Trees (Content MathML): what the formula computes |
| `slt_representation_v3.zip` | 835 MB | Symbol Layout Trees (Presentation MathML): how it looks |
| `README_formulas_V3.0.md` | — | Official documentation |

Each archive holds 101 TSV files with the same rows in the same order and these columns:

```
id  post_id  thread_id  type  comment_id  old_visual_id  visual_id  issue  formula
```

- **`id`**: unique formula instance id (28,320,920 instances). 28,267,310 have SLT and OPT
  (converted with LaTeXML 0.8.5); the remaining 53,610 have LaTeX only.
- **`type`**: `question`, `answer`, `comment` or `title`. For comments, `post_id` is the parent post.
- **`visual_id`**: groups instances that look identical (assigned by comparing SLT strings with
  Tangent-S). Task 2 is judged per visual ID, and runs are deduplicated by it.
- **`old_visual_id`**: the ARQMath-2 grouping, kept for comparison with 2021 runs. It differs from
  `visual_id` in only 65,681 rows. Do not use it.
- **`issue`**:
  - `d`: formula missing from the post XML. **Must not be returned** in Task 2 results
    (4,065,932 instances, plus 9,869 flagged `dv`).
  - `v`: the old visual ID was wrong and has been corrected (55,812 instances). The formula itself is
    usable: its SLT was checked against its LaTeX and mismatches were LaTeXML formatting, not wrong
    formulas.

### Built index

`python -m src.data.index` joins the three TSV families into Parquet shards in
`data/processed/formula_index_v2/` (~15 minutes), with columns:

| Column | Notes |
|--------|-------|
| `id`, `post_id`, `thread_id`, `type`, `comment_id` | as in the TSVs |
| `old_visual_id`, `visual_id` | strings |
| `issue` | `d`, `v`, `dv` or null |
| `retrievable` | `false` for `d`/`dv`; filter on this before building any retrieval index |
| `latex`, `opt`, `slt` | `opt`/`slt` are null only where LaTeXML could not convert the formula |

The build prints flag counts and a consistency check between each formula's LaTeX and the LaTeX
embedded in its SLT.

---

## 2. Topics (queries)

`data/raw/arqmath/topics/task2/arqmath{1,2,3}/`

| File | Contents |
|------|----------|
| `Topics*.xml` | One topic per query: `Formula_Id` (e.g. `q_6`), the query formula's `Latex`, and the question post it came from |
| `formulas/*.tsv` | Official LaTeX / OPT / SLT for **every** formula in the topic posts, keyed by `q_…` id. Columns: `id, topic_id, thread_id, type` and the formula (named `formula`, or `opt`/`slt` in ARQMath-1) |
| `README_task2.md` | Official topic documentation for that year |

A topic's query formula is the row in `formulas/` whose id equals the topic's `Formula_Id`. Every topic
in all three years has official SLT and OPT for its query formula; use these instead of converting the
LaTeX locally.

Known formatting issues:

- **ARQMath-3 SLT** contains unescaped `<` (in the `alttext` attribute and as `<mo><</mo>`), which breaks
  strict XML parsing for six query formulas, five of them judged. Drop the `alttext` attribute and escape
  any `<` that cannot start a tag before parsing.
- **ARQMath-1** XML predates the v3 corpus conversion: it is pretty-printed, has an XML declaration, and
  lacks the `<semantics>` wrapper the corpus SLT uses. Normalise it before comparing with corpus
  formulas.

---

## 3. Relevance judgments (qrels)

`data/raw/arqmath/qrels/task2/`, TREC format: `topic_id  0  doc_id  grade`, grades 0–3.
For the binary metrics, grades 2 and 3 count as relevant.

| Year | Files | Keyed by | Judged topics (official / all) |
|------|-------|----------|--------------------------------|
| ARQMath-1 | `qrel_task2_2020{,_all}_formula_id.tsv` | formula `id` | 45 / 74 |
| ARQMath-2 | `qrel_task2_2021_{official,all}.tsv` | `visual_id` | 58 / 70 |
| ARQMath-3 | `qrel_task2_2022_{official,all}.tsv` | `visual_id` | 76 / 86 |

`official` files contain the topics used in each year's official evaluation. `all` files add topics
and judgments collected afterwards.

**ARQMath-1** also published qrels keyed by its own visual IDs, but that numbering does not exist in the
v3 index, so a join against any index column silently pairs topics with unrelated formulas. Only the
formula-ID versions are downloaded. `scripts/convert_arqmath1_qrels.py` maps them to v3 visual IDs and
writes `data/processed/qrels/task2/arqmath1/qrel_task2_2020_{official,all}.tsv`. When several judged
formulas fall into one v3 visual ID with different grades (28 cases in the official file), the visual ID
receives the maximum grade; every such case is logged in a `*_conflicts.tsv` file next to the output.

`scripts/check_qrel_ids.py` re-derives the ID space of each qrel file from the data: for each candidate
column it measures coverage and how much more similar high-grade formulas are to the query than
grade-0 ones.

---

## 4. Evaluation scripts

`data/raw/arqmath/eval_scripts/arqmath3/`

| Script | Purpose |
|--------|---------|
| `de_duplicate_2022.py` | Maps a run's formula ids to visual ids, keeps the first hit per visual id, drops unjudged visual ids ("prime" run) |
| `task2_get_results.py` | Scores prime runs with `trec_eval`: nDCG′, MAP′ and P′@10 (`-l2`: grades 2–3 relevant) |

Runs use the ARQMath Task 2 format `topic_id  formula_id  post_id  rank  score  run_id`. The scripts call
the `trec_eval` binary, which must be installed separately. Note that `trec_eval` averages only over
topics present in the run, so every topic must appear in it.

---

## 5. Split

The topics are temporally disjoint from the collection (queries come from posts written after 2018), so
no query formula's own post is in the index.

| Data | Role |
|------|------|
| ARQMath-1 topics + qrels | training |
| ARQMath-2 topics + qrels | validation (model selection, hyperparameters) |
| ARQMath-3 topics + official qrels | test: final numbers only, never used for decisions |

---

## Optional: post collection

`bash scripts/setup.sh --with-collection` also downloads `data/raw/arqmath/collection/`:

| File | Size | Contents |
|------|------|----------|
| `Posts.V1.3.zip` | 895 MB (~4.1 GB unzipped) | All questions and answers; formulas appear as `<span class="math-container" id="FID">` |
| `PostLinks.V1.3.xml` | 29 MB | Related / duplicate question links |
| `Tags.V1.3.xml` | 169 KB | Tag vocabulary |
| `README_DATA.md` | — | Official documentation |

The formula pipeline does not need these; they are only useful for inspecting a formula's context.

---

## Directory structure

```
data/
├── raw/arqmath/                        (downloaded by scripts/setup.sh, not tracked)
│   ├── formulas/
│   │   ├── README_formulas_V3.0.md
│   │   ├── {latex,opt,slt}_representation_v3.zip
│   │   └── {latex,opt,slt}_representation_v3/     1.tsv … 101.tsv
│   ├── topics/task2/arqmath{1,2,3}/
│   │   ├── Topics*.xml
│   │   ├── README_task2.md
│   │   └── formulas/                              topic formula LaTeX / OPT / SLT
│   ├── qrels/task2/arqmath{1,2,3}/
│   ├── eval_scripts/arqmath3/
│   └── collection/                                (only with --with-collection)
├── processed/                          (generated, not tracked)
│   ├── formula_index_v2/                          shard_001.parquet … shard_101.parquet
│   ├── qrels/task2/arqmath1/                      ARQMath-1 qrels keyed by v3 visual id
│   └── reports/
└── samples/                            (tracked)
    └── formula_sample.jsonl                       50 index rows for local testing
```

---

## Citation

> B. Mansouri, R. Zanibbi, D. W. Oard, A. Agarwal. *Overview of ARQMath-3 (2022): Third CLEF Lab on
> Answer Retrieval for Questions on Math.* CLEF 2022 Working Notes.
