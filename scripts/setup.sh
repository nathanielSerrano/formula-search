#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup.sh
#
# Downloads the ARQMath data needed for formula retrieval (ARQMath Task 2) from
# the official RIT backup server: https://www.cs.rit.edu/~dprl/ARQMath-backup/
#
# What gets downloaded (≈1.8 GB compressed, ≈21 GB unpacked):
#
#   data/raw/arqmath/
#     formulas/
#       README_formulas_V3.0.md
#       latex_representation_v3.zip  (406 MB) + unzipped TSVs — official formula index
#       opt_representation_v3.zip    (587 MB) + unzipped TSVs — Operator Trees (Content MathML)
#       slt_representation_v3.zip    (835 MB) + unzipped TSVs — Symbol Layout Trees (Presentation MathML)
#     topics/task2/arqmath{1,2,3}/
#       Topics*.xml                  — one query formula per topic (Formula_Id + LaTeX)
#       README_task2.md
#       formulas/*.tsv               — official LaTeX / OPT / SLT of every formula in the topic posts
#     qrels/task2/
#       arqmath1/qrel_task2_2020{,_all}_formula_id.tsv   (keyed by formula id; see below)
#       arqmath2/qrel_task2_2021_{official,all}.tsv      (keyed by visual id)
#       arqmath3/qrel_task2_2022_{official,all}.tsv      (keyed by visual id)
#     eval_scripts/arqmath3/
#       de_duplicate_2022.py, task2_get_results.py       — official Task 2 evaluation
#
# ARQMath-1 qrels: ARQMath-1 also published qrels keyed by its own visual ids, but
# those ids do not exist in the v3 formula index, so only the formula-id versions
# are downloaded. scripts/convert_arqmath1_qrels.py re-keys them to v3 visual ids.
#
# Optional (--with-collection): the Math Stack Exchange posts themselves
# (Posts.V1.3, ~4.1 GB unzipped), plus PostLinks and Tags. The formula pipeline
# does not need them; they are only useful for looking at a formula's context.
#
# Optional (--with-runs): the 20 runs submitted to ARQMath-3 Task 2 (≈100 MB),
# in data/raw/arqmath/runs/arqmath3/task2/. Scoring them with our pipeline
# (src.eval.import_run, src.eval.evaluate) and comparing with the published
# results validates the evaluation and gives per-topic reference systems.
#
# Usage:
#   bash scripts/setup.sh [--with-collection] [--with-runs] [--no-unzip]
#
#   --with-collection  Also download the post collection
#   --with-runs        Also download the published ARQMath-3 Task 2 runs
#   --no-unzip         Download only; skip extraction of zip files
#
# The script is idempotent: files that already exist are not re-downloaded,
# and archives that have already been extracted are not re-extracted.
# -----------------------------------------------------------------------------

set -euo pipefail

# ─── Parse flags ─────────────────────────────────────────────────────────────
WITH_COLLECTION=false
WITH_RUNS=false
DO_UNZIP=true

for arg in "$@"; do
  case "$arg" in
    --with-collection) WITH_COLLECTION=true ;;
    --with-runs)       WITH_RUNS=true ;;
    --no-unzip)        DO_UNZIP=false ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--with-collection] [--with-runs] [--no-unzip]"
      exit 1
      ;;
  esac
done

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data/raw/arqmath"

COLLECTION_DIR="$DATA_DIR/collection"
FORMULAS_DIR="$DATA_DIR/formulas"
TOPICS_DIR="$DATA_DIR/topics/task2"
QRELS_DIR="$DATA_DIR/qrels/task2"
EVAL_DIR="$DATA_DIR/eval_scripts"
RUNS_DIR="$DATA_DIR/runs/arqmath3/task2"

# ─── Base URLs ───────────────────────────────────────────────────────────────
BASE="https://www.cs.rit.edu/~dprl/ARQMath-backup"
TOPICS2_BASE="$BASE/Topics/Task2%20Topics"
EVAL_BASE="$BASE/Evaluation/%20Scripts%20%26%20Qrels"

# ─── Helpers ─────────────────────────────────────────────────────────────────
download() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  [skip] $(basename "$dest") already exists"
  else
    echo "  [download] $(basename "$dest")"
    curl --fail --silent --show-error --location \
         --retry 3 --retry-delay 5 \
         --output "$dest" \
         "$url"
  fi
}

# Uses a sentinel file (<zip>.unzipped) to skip re-extraction on subsequent runs.
unzip_archive() {
  local zip="$1"
  local dest_dir="$2"
  local marker="${zip%.zip}.unzipped"
  if [[ -f "$marker" ]]; then
    echo "  [skip] $(basename "$zip") already extracted"
  elif [[ ! -f "$zip" ]]; then
    echo "  [skip] $(basename "$zip") not found, skipping extraction"
  else
    echo "  [unzip] $(basename "$zip")"
    unzip -q "$zip" -d "$dest_dir"
    touch "$marker"
  fi
}

banner() {
  echo ""
  echo "════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════"
}

# ─── Create directories ───────────────────────────────────────────────────────
mkdir -p \
  "$FORMULAS_DIR" \
  "$TOPICS_DIR/arqmath1/formulas" \
  "$TOPICS_DIR/arqmath2/formulas" \
  "$TOPICS_DIR/arqmath3/formulas" \
  "$QRELS_DIR/arqmath1" \
  "$QRELS_DIR/arqmath2" \
  "$QRELS_DIR/arqmath3" \
  "$EVAL_DIR/arqmath3"

# ─── Formula index ────────────────────────────────────────────────────────────
banner "Formula index (LaTeX / OPT / SLT)"
download \
  "$BASE/Formulas/README_formulas_V3.0.md" \
  "$FORMULAS_DIR/README_formulas_V3.0.md"

for rep in latex opt slt; do
  zip="$FORMULAS_DIR/${rep}_representation_v3.zip"
  download "$BASE/Formulas/${rep}_representation_v3.zip" "$zip"
  if [[ "$DO_UNZIP" == "true" ]]; then
    unzip_archive "$zip" "$FORMULAS_DIR"
  fi
done

# ─── Topics ───────────────────────────────────────────────────────────────────
banner "Task 2 topics (query formulas)"

download \
  "$TOPICS2_BASE/ARQMath-1/Topics/Topics_V1.1.xml" \
  "$TOPICS_DIR/arqmath1/Topics_V1.1.xml"

download \
  "$TOPICS2_BASE/ARQMath-2/Topics/Topics_Task2_2021_V1.1.xml" \
  "$TOPICS_DIR/arqmath2/Topics_Task2_2021_V1.1.xml"

download \
  "$TOPICS2_BASE/ARQMath-3/Topics/Topics_Task2_2022_V0.1.xml" \
  "$TOPICS_DIR/arqmath3/Topics_Task2_2022_V0.1.xml"

download \
  "$TOPICS2_BASE/ARQMath-1/00_README_task2.md" \
  "$TOPICS_DIR/arqmath1/README_task2.md"

download \
  "$TOPICS2_BASE/ARQMath-2/01_README_task2.md" \
  "$TOPICS_DIR/arqmath2/README_task2.md"

download \
  "$TOPICS2_BASE/ARQMath-3/README_Task2_V0.1.md" \
  "$TOPICS_DIR/arqmath3/README_task2.md"

# Official LaTeX/OPT/SLT for every formula in the topic posts (keyed by the "q_…"
# Formula_Id in the topic XML), so queries use the organizers' conversions rather
# than a locally installed LaTeXML.
banner "Task 2 topic formulas (official LaTeX / OPT / SLT)"

for rep in latex opt slt; do
  download \
    "$TOPICS2_BASE/ARQMath-1/Formulas/Formula_topics_${rep}_V2.0.tsv" \
    "$TOPICS_DIR/arqmath1/formulas/Formula_topics_${rep}_V2.0.tsv"
done

for rep in Latex OPT SLT; do
  download \
    "$TOPICS2_BASE/ARQMath-2/Formulas/Topics_2021_Formulas_${rep}_V1.1.tsv" \
    "$TOPICS_DIR/arqmath2/formulas/Topics_2021_Formulas_${rep}_V1.1.tsv"
  download \
    "$TOPICS2_BASE/ARQMath-3/Formulas/Topics_Formulas_${rep}.V0.1.tsv" \
    "$TOPICS_DIR/arqmath3/formulas/Topics_Formulas_${rep}.V0.1.tsv"
done

# ─── Qrels ────────────────────────────────────────────────────────────────────
banner "Task 2 qrels (relevance judgments)"

download \
  "$EVAL_BASE/ARQMath-1/Task2/qrel_with_formula_ids/qrel_task2.tsv" \
  "$QRELS_DIR/arqmath1/qrel_task2_2020_formula_id.tsv"

download \
  "$EVAL_BASE/ARQMath-1/Task2/qrel_with_formula_ids/qrel_task2_all.tsv" \
  "$QRELS_DIR/arqmath1/qrel_task2_2020_all_formula_id.tsv"

download \
  "$EVAL_BASE/ARQMath-2/Task2/Qrel%20Files/qrel_task2_2021_test_official_evaluation.tsv" \
  "$QRELS_DIR/arqmath2/qrel_task2_2021_official.tsv"

download \
  "$EVAL_BASE/ARQMath-2/Task2/Qrel%20Files/qrel_task2_2021_all.tsv" \
  "$QRELS_DIR/arqmath2/qrel_task2_2021_all.tsv"

download \
  "$EVAL_BASE/ARQMath-3/Task2/Qrel%20Files/qrel_task2_2022_official.tsv" \
  "$QRELS_DIR/arqmath3/qrel_task2_2022_official.tsv"

download \
  "$EVAL_BASE/ARQMath-3/Task2/Qrel%20Files/qrel_task2_2022_all.tsv" \
  "$QRELS_DIR/arqmath3/qrel_task2_2022_all.tsv"

# ─── Eval scripts ─────────────────────────────────────────────────────────────
banner "Official Task 2 evaluation scripts"

download \
  "$EVAL_BASE/ARQMath-3/Task2/task2_get_results.py" \
  "$EVAL_DIR/arqmath3/task2_get_results.py"

download \
  "$EVAL_BASE/ARQMath-3/Task2/de_duplicate_2022.py" \
  "$EVAL_DIR/arqmath3/de_duplicate_2022.py"

# ─── Collection (optional) ────────────────────────────────────────────────────
if [[ "$WITH_COLLECTION" == "true" ]]; then
  banner "Post collection (optional)"
  mkdir -p "$COLLECTION_DIR"

  download \
    "$BASE/Collection/README_DATA.md" \
    "$COLLECTION_DIR/README_DATA.md"

  download \
    "$BASE/Collection/Tags.V1.3.xml" \
    "$COLLECTION_DIR/Tags.V1.3.xml"

  download \
    "$BASE/Collection/PostLinks.V1.3.xml" \
    "$COLLECTION_DIR/PostLinks.V1.3.xml"

  echo ""
  echo "  [note] Posts.V1.3.zip is 895 MB and unpacks to ~4.1 GB."
  download \
    "$BASE/Collection/Posts.V1.3.zip" \
    "$COLLECTION_DIR/Posts.V1.3.zip"

  if [[ "$DO_UNZIP" == "true" ]]; then
    unzip_archive "$COLLECTION_DIR/Posts.V1.3.zip" "$COLLECTION_DIR"
  fi
fi

# ─── Published runs (optional) ────────────────────────────────────────────────
if [[ "$WITH_RUNS" == "true" ]]; then
  banner "Published ARQMath-3 Task 2 runs (optional)"
  mkdir -p "$RUNS_DIR"
  for run in \
    Baseline-task2-TangentS-auto-math-p \
    DPRL-Task2-CFT-auto-math-A \
    DPRL-Task2-CFTED-auto-math-P \
    DPRL-Task2-LTR-auto-math-A \
    DPRL-Task2-MathAMR-auto-both-A \
    DPRL-Task2-RRAMRCFTED-auto-both-A \
    JU_NITS-task2-formulaL-auto-formula-P \
    JU_NITS-task2-formulaO-auto-formula-A \
    JU_NITS-task2-formulaS-auto-formula-A \
    MathDowsers-task2-L8-auto-math-P \
    MathDowsers-task2-latex_L8_a035-auto-math-A \
    MathDowsers-task2-latex_L8_a040-auto-math-A \
    XYPhoc-task2-xy5-auto-math-a-2022 \
    XYPhoc-task2-xy5IDF-auto-math-a-2022 \
    XYPhoc-task2-xy7o4-auto-math-a-2022 \
    approach0-task2-a0-manual-math-A \
    approach0-task2-fusion02_ctx-auto-both-A \
    approach0-task2-fusion_alpha02-manual-both-A \
    approach0-task2-fusion_alpha03-manual-both-A \
    approach0-task2-fusion_alpha05-manual-both-P; do
    download "$BASE/Runs/ARQMath-3/Task%202/${run}.tsv" "$RUNS_DIR/${run}.tsv"
  done
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
banner "Done"
echo ""
echo "Downloaded to: $DATA_DIR"
echo ""
echo "Next steps (see README.md):"
echo "  python -m src.data.index"
echo "  python scripts/convert_arqmath1_qrels.py"
echo ""
