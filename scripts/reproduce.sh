#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# reproduce.sh
#
# Re-runs the whole experiment, from the ARQMath download to the final test run, following the
# protocol in the README: train on ARQMath-1, make every choice on ARQMath-2, retrain the final
# models on ARQMath-1 + 2 with those choices, evaluate ARQMath-3 once.
#
# Stages (in order; each writes logs/reproduce/<stage>.log and, when it succeeds, a .done marker):
#
#   download   ARQMath formulas, topics, qrels, evaluation scripts (scripts/setup.sh)
#   index      Parquet formula index                       (src.data.index)
#   qrels      ARQMath-1 qrels re-keyed to v3 visual IDs   (scripts/convert_arqmath1_qrels.py)
#   corpus     visual-ID retrieval corpus                  (src.data.visual_index)
#   bm25       BM25 index + dev run
#   official   official ARQMath scripts + trec_eval cross-check on the BM25 dev run
#   graphs     graph vocabularies, then the pretraining graph store
#   pretrain   contrastive pretraining (configs/pretrain_selected.yaml); best.pt by quick dev
#   finetune   fine-tuning on ARQMath-1 (configs/finetune.yaml); best.pt by quick dev MAP′
#   dev        dev runs: GNN, RRF, reranker features, reranker selected on ARQMath-2 (+ 5-fold CV)
#   final      final models on ARQMath-1 + 2: fine-tuning stopped at the dev-selected step,
#              reranker trained with the dev-selected C / depth / α
#   test       ARQMath-3, once: BM25, GNN, RRF, rerank, comparison, official cross-check  (--test)
#   published  score the 20 published ARQMath-3 runs (validation of the evaluation)    (--test)
#
# Settings carried from dev to the final models are read from the dev outputs, not hard-coded:
# the fine-tuning step from checkpoints/finetune_pa05/best_ft_pa05.pt and the reranker setting from
# data/processed/rerank/model_base.json. (Our run: step 250; C = 0.1, depth 300, α = 1.0.)
#
# GPU training is not bit-for-bit deterministic even with fixed seeds, so a rerun lands close to,
# not exactly on, the reported numbers. Expect several hours of GPU time and ~60 GB of disk.
#
# Usage:
#   bash scripts/reproduce.sh                 # every stage not yet done, up to (not including) test
#   bash scripts/reproduce.sh --test          # ... and then the one-time test run
#   bash scripts/reproduce.sh --from dev      # redo `dev` and every later stage
#   bash scripts/reproduce.sh --only bm25     # redo one stage
#   bash scripts/reproduce.sh --dry-run --test  # print the commands without running anything
# Options: --gpu N (CUDA device, default 0), PYTHON=/path/to/python, TREC_EVAL=/path/to/trec_eval
# -----------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
TREC_EVAL="${TREC_EVAL:-$HOME/trec_eval/trec_eval}"
LOG_DIR="logs/reproduce"
STAGES=(download index qrels corpus bm25 official graphs pretrain finetune dev final test published)
TEST_STAGES=" test published "

FROM="" ONLY="" WITH_TEST=0 DRY=0 GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --test) WITH_TEST=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
for s in "$FROM" "$ONLY"; do
  if [[ -n "$s" && " ${STAGES[*]} " != *" $s "* ]]; then
    echo "unknown stage: $s (stages: ${STAGES[*]})" >&2; exit 2
  fi
done
export CUDA_VISIBLE_DEVICES="$GPU"

# run CMD...: print the command, then run it (only print with --dry-run)
run() {
  echo "+ $*"
  if [[ "$DRY" -eq 0 ]]; then "$@"; fi
}

# value_or PLACEHOLDER CMD...: the command's output, or the placeholder in a dry run
value_or() {
  local placeholder="$1"; shift
  if [[ "$DRY" -eq 1 ]]; then echo "$placeholder"; else "$@"; fi
}

# ---------------------------------------------------------------------------- stages

stage_download() {
  run bash scripts/setup.sh
}

stage_index() {
  run "$PY" -m src.data.index
}

stage_qrels() {
  run "$PY" scripts/convert_arqmath1_qrels.py
}

stage_corpus() {
  run "$PY" -m src.data.visual_index
}

stage_bm25() {
  run "$PY" -m src.baselines.bm25 build
  run "$PY" -m src.baselines.bm25 search --split dev
  run "$PY" -m src.eval.evaluate runs/bm25_dev.tsv --split dev
}

stage_official() {
  if [[ ! -x "$TREC_EVAL" ]]; then
    run git clone https://github.com/usnistgov/trec_eval.git "$(dirname "$TREC_EVAL")"
    run make -C "$(dirname "$TREC_EVAL")"
  fi
  run "$PY" -m src.eval.evaluate runs/bm25_dev.tsv --split dev --official --trec-eval "$TREC_EVAL"
}

stage_graphs() {
  run "$PY" -m src.data.graph_stats
  run "$PY" -m src.pretrain.graph_store
}

stage_pretrain() {
  run "$PY" -m src.pretrain.train --config configs/pretrain_selected.yaml --out-dir checkpoints/pretrain_rename04
  run cp checkpoints/pretrain_rename04/best.pt checkpoints/pretrain_rename04/best_rename04.pt
}

stage_finetune() {
  # configs/finetune.yaml initialises from checkpoints/pretrain_rename04/best_rename04.pt
  run "$PY" -m src.finetune.train --config configs/finetune.yaml --out-dir checkpoints/finetune_pa05
  run cp checkpoints/finetune_pa05/best.pt checkpoints/finetune_pa05/best_ft_pa05.pt
}

stage_dev() {
  run "$PY" -m src.model.retrieve --checkpoint checkpoints/finetune_pa05/best_ft_pa05.pt --split dev
  run "$PY" -m src.eval.fuse runs/bm25_dev.tsv runs/gnn_best_ft_pa05_dev.tsv --method rrf --run-id rrf_ft_pa05 \
      --out runs/rrf_ft_pa05_dev.tsv
  run "$PY" -m src.rerank.build --split train --judged --qrels all --name train_judged
  run "$PY" -m src.rerank.build --split dev --run runs/rrf_ft_pa05_dev.tsv --depth 500 --name dev_rrf_ft
  run "$PY" -m src.rerank.train --train data/processed/rerank/train_judged.npz \
      --dev data/processed/rerank/dev_rrf_ft.npz --dev-run runs/rrf_ft_pa05_dev.tsv --split dev --features base \
      --model-out data/processed/rerank/model_base.json --out runs/reranked_base_dev.tsv \
      --cv 5 --cv-out runs/reranked_base_cv_dev.tsv
  run "$PY" -m src.eval.compare runs/bm25_dev.tsv runs/gnn_best_ft_pa05_dev.tsv runs/rrf_ft_pa05_dev.tsv \
      runs/reranked_base_cv_dev.tsv --split dev
}

stage_final() {
  local step settings
  step=$(value_or "<STEP>" "$PY" -c "import torch; print(torch.load('checkpoints/finetune_pa05/best_ft_pa05.pt', map_location='cpu')['step'])" 2>/dev/null)
  if [[ "$DRY" -eq 0 && "$step" -le 0 ]]; then
    echo "dev selected fine-tuning step $step (the pretrained model itself); nothing to retrain" >&2; exit 1
  fi
  settings=$(value_or "<C> <DEPTH> <ALPHA>" "$PY" -c "import json; s = json.load(open('data/processed/rerank/model_base.json'))['settings']; print(s['C'], s['depth'], s['alpha'])")
  echo "dev-selected settings: fine-tuning step $step; reranker C, depth, alpha = $settings"
  run "$PY" -m src.finetune.train --config configs/finetune.yaml --split train+dev --no-quick-dev \
      --stop-at "$step" --out-dir checkpoints/finetune_final
  run "$PY" -m src.rerank.build --split train+dev --judged --qrels all --name train_dev_judged
  # shellcheck disable=SC2086  # settings is three words on purpose
  run "$PY" -m src.rerank.train --train data/processed/rerank/train_dev_judged.npz --features base \
      --fixed $settings --model-out data/processed/rerank/model_final.json
}

stage_test() {
  local depth
  depth=$(value_or "<DEPTH>" "$PY" -c "import json; print(json.load(open('data/processed/rerank/model_final.json'))['settings']['depth'])")
  run "$PY" -m src.baselines.bm25 search --split test
  run "$PY" -m src.model.retrieve --checkpoint checkpoints/finetune_final/final.pt --split test --run-id gnn_ft_final
  run "$PY" -m src.eval.fuse runs/bm25_test.tsv runs/gnn_ft_final_test.tsv --method rrf --run-id rrf_final \
      --out runs/rrf_final_test.tsv
  run "$PY" -m src.rerank.build --split test --run runs/rrf_final_test.tsv --depth "$depth" --name test_rrf_final
  run "$PY" -m src.rerank.train --apply data/processed/rerank/model_final.json \
      --dev data/processed/rerank/test_rrf_final.npz --dev-run runs/rrf_final_test.tsv \
      --out runs/reranked_final_test.tsv
  run "$PY" -m src.eval.compare runs/bm25_test.tsv runs/gnn_ft_final_test.tsv runs/rrf_final_test.tsv \
      runs/reranked_final_test.tsv --split test
  run "$PY" -m src.eval.extra_metrics runs/bm25_test.tsv runs/rrf_final_test.tsv runs/reranked_final_test.tsv --split test
  run "$PY" -m src.eval.evaluate runs/reranked_final_test.tsv --split test --official --trec-eval "$TREC_EVAL"
}

stage_published() {
  run bash scripts/setup.sh --with-runs
  run "$PY" -m src.eval.import_run data/raw/arqmath/runs/arqmath3/task2/*.tsv
  local f
  for f in runs/published/*.tsv; do
    run "$PY" -m src.eval.evaluate "$f" --split test
  done
}

# ---------------------------------------------------------------------------- driver

if [[ "$DRY" -eq 0 ]]; then mkdir -p "$LOG_DIR" runs; fi
started=0
for stage in "${STAGES[@]}"; do
  [[ "$stage" == "$FROM" ]] && started=1
  if [[ -n "$ONLY" && "$stage" != "$ONLY" ]]; then continue; fi
  if [[ "$TEST_STAGES" == *" $stage "* && "$WITH_TEST" -eq 0 ]]; then
    echo "== $stage: skipped (touches the ARQMath-3 test topics; pass --test)"; continue
  fi
  if [[ -z "$ONLY" && "$started" -eq 0 && -f "$LOG_DIR/$stage.done" ]]; then
    echo "== $stage: done earlier ($LOG_DIR/$stage.done); --from $stage reruns it"; continue
  fi
  echo "== $stage: $(date '+%F %T')"
  if [[ "$DRY" -eq 1 ]]; then
    "stage_$stage"
  else
    "stage_$stage" 2>&1 | tee "$LOG_DIR/$stage.log"
    date '+%F %T' > "$LOG_DIR/$stage.done"
  fi
done
echo "== finished: $(date '+%F %T')"
