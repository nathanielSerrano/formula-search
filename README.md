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
| Topic (query) loading, visual-ID corpus, run files and metrics | done |
| Validation: matches official scripts; reproduces all 20 published ARQMath-3 runs | done |
| BM25 baseline (dev: nDCG′ 0.514, MAP′ 0.314, P′@10 0.452) | done |
| Graph representation of formulas (symbols, edge types), corpus vocabularies | done |
| Pretraining data: graph store and augmented pairs | done (tested locally) |
| GNN encoder and contrastive pretraining (dev, no labels: nDCG′ 0.511, MAP′ 0.322, P′@10 0.498) | done |
| BM25 + GNN reciprocal rank fusion (dev: nDCG′ 0.589, MAP′ 0.378, P′@10 0.522; all p < 0.01 vs BM25) | done |
| Pretraining variants (batch 2,048; `p_rename` 0.4): GNN alone +0.015–0.017 nDCG′ (p < 0.05), no gain fused | done |
| Supervised fine-tuning (pairs, lr 5e-5; dev fused: nDCG′ 0.599, MAP′ 0.388, P′@10 0.522) | done |
| Structural reranker on RRF (dev, 5-fold CV: nDCG′ 0.610, MAP′ 0.406, P′@10 0.547; nDCG′ p = 0.033, MAP′ p = 0.018 vs RRF) | done |
| Reranker: prototype-inspired feature group, depth up to 500 (ablation vs base features) | implemented (tested locally); server run next |

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

Systems retrieve **visual IDs** from a corpus with one representative formula per retrievable visual ID
(`src/data/visual_index.py`) and write TREC-format run files. `src/eval/metrics.py` computes the metrics
above with `pytrec_eval`, matching the organizers' settings. It also reports **judged@10**, the share of
the unfiltered top 10 that is judged: prime metrics only rank judged formulas (a random ordering of all
judged formulas already scores nDCG′ ≈ 0.64 on ARQMath-2), so they need to be read alongside it.

The official scripts (`de_duplicate_2022.py`, `task2_get_results.py`) are downloaded by `setup.sh`.
`--official` runs them on the same run for comparison; they require the `trec_eval` binary. Every
judged topic counts, and a topic for which a system returns no judged formula scores 0. `--official`
passes `-c` to `trec_eval` for the same behaviour; without it, current `trec_eval` stops with
"result qid … not found in qrels" on such a topic.

## Setup and full pipeline

Data and training live on a GPU server. Every command below runs from the repository root there, in
order; each step depends on the ones before it.

**1. Python environment** (once; versions are pinned from the server's Python 3.8 environment)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Download ARQMath** (≈1.8 GB compressed, ≈21 GB unpacked; formula indexes, topics, topic formulas,
qrels, official eval scripts)

```bash
bash scripts/setup.sh
```

**3. Build the Parquet formula index** (~15 min) → `data/processed/formula_index_v2/`

```bash
python -m src.data.index
```

**4. Re-key the ARQMath-1 qrels to v3 visual IDs** (seconds) → `data/processed/qrels/task2/arqmath1/`

```bash
python scripts/convert_arqmath1_qrels.py
```

**5. Build the visual-ID retrieval corpus** → `data/processed/visual_index/`. Also reports how many
relevant visual IDs are not retrievable (the recall ceiling) for each year.

```bash
python -m src.data.visual_index
```

**6. BM25 baseline**: build the index (→ `data/processed/bm25/`), then retrieve for the dev topics
(→ `runs/bm25_dev.tsv`)

```bash
python -m src.baselines.bm25 build
python -m src.baselines.bm25 search --split dev
```

**7. Evaluate**

```bash
python -m src.eval.evaluate runs/bm25_dev.tsv --split dev
```

**8. Cross-check against the official ARQMath scripts** (once; builds `trec_eval`, no sudo needed)

```bash
git clone https://github.com/usnistgov/trec_eval.git ~/trec_eval
make -C ~/trec_eval
python -m src.eval.evaluate runs/bm25_dev.tsv --split dev --official --trec-eval ~/trec_eval/trec_eval
```

**9. Score the published ARQMath-3 runs** (once): downloads the 20 runs submitted to ARQMath-3 Task 2,
converts them to visual-ID runs in `runs/published/`, and scores them. The results should match Table 4 of
the ARQMath-3 overview, e.g. DPRL's Tangent-CFT run (`DPRL-Task2-CFT-auto-math-A`): nDCG′ 0.641,
MAP′ 0.419, P′@10 0.534. Scoring other systems on the test topics validates the evaluation and does not
inform any of our own choices.

```bash
bash scripts/setup.sh --with-runs
python -m src.eval.import_run data/raw/arqmath/runs/arqmath3/task2/*.tsv
for f in runs/published/*.tsv; do python -m src.eval.evaluate "$f" --split test; done
```

**10. Graph vocabularies and statistics**: converts every visual ID to its SLT and OPT graphs
(`src/data/formula_graph.py`), writes the symbol/tag vocabularies to `data/processed/graph_vocab.json`
and graph-size statistics to `data/processed/reports/graph_stats.json`.

```bash
python -m src.data.graph_stats
```

**11. Pretraining graph store**: converts both graphs of every visual ID once into memory-mapped integer
arrays in `data/processed/pretrain_graphs/` (≈256-node cap), so contrastive pretraining does not parse
MathML every epoch.

```bash
python -m src.pretrain.graph_store
```

**12. Contrastive pretraining** (GPU; settings in `configs/pretrain.yaml`). Run the short smoke test first
to check throughput, then the full run (in `tmux` or with `nohup`, since it takes hours). Checkpoints and
`log.jsonl` go to `checkpoints/pretrain/`; `--resume checkpoints/pretrain/latest.pt` continues a run.

```bash
python -m src.pretrain.train --max-steps 200 --eval-every 100 --out-dir checkpoints/pretrain_smoke
python -m src.pretrain.train
```

**13. Full-corpus retrieval with a checkpoint**: embeds all 8.4M formulas (cached next to the checkpoint,
~4.3 GB), retrieves the top 1,000 per topic and writes `runs/gnn_<checkpoint>_<split>.tsv`; score it with
step 7. Inference fits on a 12 GB GPU, so it can run on a second GPU while training continues.

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.model.retrieve --checkpoint checkpoints/pretrain/best.pt --split dev
python -m src.eval.evaluate runs/gnn_best_dev.tsv --split dev
```

**14. Compare and fuse runs.** `compare` reports per-measure differences against a baseline run with a
paired randomization test, bootstrap 95% intervals and per-topic wins/losses. `fuse` combines runs
(min-max linear or reciprocal rank fusion); with `--tune` it picks the weight on the split's topics and
`--cv` gives a cross-validated estimate. Tune fusion weights on `dev` and reuse them unchanged on `test`.

```bash
python -m src.eval.compare runs/bm25_dev.tsv runs/gnn_best_dev.tsv --split dev --topics 5
python -m src.eval.fuse runs/bm25_dev.tsv runs/gnn_best_dev.tsv --split dev --tune --cv 5 --out runs/fused_dev.tsv
```

**15. Supervised fine-tuning** (GPU; `configs/finetune.yaml`): starts from a pretrained checkpoint and
trains on ARQMath-1 judgments: each batch has 64 distinct topics, each with its official query formula,
one relevant formula (grade 2–3) and one judged non-relevant formula (grade 0–1) as a hard negative.
Checkpoints are selected with the quick dev check, which also runs before the first step for reference.

```bash
python -m src.finetune.train
python -m src.model.retrieve --checkpoint checkpoints/finetune/best.pt --split dev
```

**16. Structural reranker.** Reorders the first-stage top-k with a linear pairwise model over
structural features (Tangent-style symbol-pair and path overlap at exact / unified-variable / structure
level, tree edit distance, size) plus BM25 and *pretrained*-GNN scores; the fine-tuned GNN's scores
are not used as features because they are inflated on its own ARQMath-1 training topics. Trained on
ARQMath-1 judged pairs; regularisation, depth and interpolation with the first stage are chosen on dev.

A second feature group ("proto") ports the ideas of the earlier prototype's hand-written reranker as
learnable features: variables renamed by first appearance (keeps co-reference: x·x ≠ x·y), three-edge
paths, IDF-weighted overlap (document frequencies from a 100k-formula corpus sample, cached in
`data/processed/rerank/idf_100000_0.npz`) and tree edit distance with commutative operands sorted.
`--features base|all` trains without or with it, for the ablation.

```bash
python -m src.rerank.build --split train --judged --qrels all --name train_judged
python -m src.rerank.build --split dev --run runs/rrf_ft_pa05_dev.tsv --depth 500 --name dev_rrf_ft
python -m src.rerank.train --train data/processed/rerank/train_judged.npz --dev data/processed/rerank/dev_rrf_ft.npz \
    --dev-run runs/rrf_ft_pa05_dev.tsv --split dev --features base --model-out data/processed/rerank/model_base.json \
    --out runs/reranked_base_dev.tsv --cv 5 --cv-out runs/reranked_base_cv_dev.tsv
python -m src.rerank.train --train data/processed/rerank/train_judged.npz --dev data/processed/rerank/dev_rrf_ft.npz \
    --dev-run runs/rrf_ft_pa05_dev.tsv --split dev --features all --model-out data/processed/rerank/model_all.json \
    --out runs/reranked_all_dev.tsv --cv 5 --cv-out runs/reranked_all_cv_dev.tsv
python -m src.eval.compare runs/rrf_ft_pa05_dev.tsv runs/reranked_base_cv_dev.tsv runs/reranked_all_cv_dev.tsv --split dev
```

The selected setting's dev score is optimistic (chosen on the same topics); `--cv 5` reranks each fold of
dev topics with the setting chosen on the other folds, an honest estimate to report and compare.

Tune on `dev` only. Run `search --split test` and evaluate on `test` only for final numbers.

**Optional diagnostics** (after step 3; their findings are summarised under [Data](#data)):

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
    paths.py                 data locations; train/dev/test = ARQMath-1/2/3
    index.py                 build the Parquet formula index from the ARQMath TSVs
    visual_index.py          build the visual-ID retrieval corpus
    mathml.py                parse and canonicalise ARQMath MathML
    topics.py                topics with their official query SLT/OPT
    qrels.py                 relevance judgments keyed by visual ID
    formula_graph.py         SLT/OPT graphs: symbol nodes, typed edges; vocabularies; PyG input
    graph_stats.py           build graph vocabularies and size statistics from the corpus
  eval/
    metrics.py               nDCG′ / MAP′ / P′@10 (prime), judged@10
    runs.py                  run files and official submission files
    official.py              scoring with the organizers' scripts
    evaluate.py              command line: score a run file
    import_run.py            convert official ARQMath run files to visual-ID runs
    compare.py               paired significance tests and per-topic comparison of runs
    fuse.py                  linear / reciprocal rank fusion of runs, weight tuning with cross-validation
  pretrain/
    graph_store.py           graphs of every visual ID as memory-mapped arrays
    augment.py               training views: variable renaming, number changes, crops, dropped graph
    dataset.py               positive pairs of views and PyG batching
    train.py                 contrastive pretraining loop
  finetune/
    data.py                  query / positive / hard-negative triples from judgments, topic-distinct batches
    train.py                 supervised fine-tuning loop
  model/
    encoder.py               dual-branch GATv2 encoder (SLT + OPT → one 256-d vector)
    loss.py                  symmetric InfoNCE with learnable temperature
    quick_dev.py             fast dev check during training (judged + random distractors)
    retrieve.py              full-corpus retrieval with a checkpoint → run file
  rerank/
    tree_edit.py             Zhang–Shasha tree edit distance
    features.py              structural query–candidate similarity features (base + proto groups, IDF table)
    build.py                 feature files for judged pairs or first-stage candidates
    train.py                 pairwise linear reranker: training, selection on dev, applying
  baselines/
    bm25.py                  BM25 over SLT symbols
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
