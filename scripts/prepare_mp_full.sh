#!/usr/bin/env bash
# Prepare the full Materials Project charge-density benchmark.
#
# Usage:
#   export MP_API_KEY="your_materials_project_key"
#   bash scripts/prepare_mp_full.sh
#
# Optional debug overrides:
#   MP_LIMIT_SAMPLES=100 PROBES=5000 bash scripts/prepare_mp_full.sh
#   MP_ID_FILE=mp_ids.txt bash scripts/prepare_mp_full.sh

set -euo pipefail

CONDA_PATH="${CONDA_PATH:-/home/wuhao/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-density-eqgnn}"
REPO_DIR="${REPO_DIR:-/home/wuhao/DensityEQGNN}"

RAW_ROOT="${RAW_ROOT:-data/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed}"
PROBES="${PROBES:-5000}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
MP_LIMIT_SAMPLES="${MP_LIMIT_SAMPLES:-}"
MP_IDS="${MP_IDS:-}"
MP_ID_FILE="${MP_ID_FILE:-}"

if [[ -z "${MP_API_KEY:-}" ]]; then
  echo "ERROR: MP_API_KEY is not set."
  echo "Run: export MP_API_KEY='your_materials_project_key'"
  exit 1
fi

if [[ -n "$MP_IDS" && -n "$MP_ID_FILE" ]]; then
  echo "ERROR: set only one of MP_IDS or MP_ID_FILE."
  exit 1
fi

source "$CONDA_PATH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

mkdir -p job_logs "$RAW_ROOT/materials_project" "$PROCESSED_ROOT/materials_project"

FORCE_ARG=()
if [[ "$FORCE" == "1" ]]; then
  FORCE_ARG=(--force)
fi

LIMIT_ARG=()
if [[ -n "$MP_LIMIT_SAMPLES" ]]; then
  LIMIT_ARG=(--limit-samples "$MP_LIMIT_SAMPLES")
fi

MP_ID_ARGS=()
if [[ -n "$MP_IDS" ]]; then
  # Space-separated list, e.g. MP_IDS="mp-1524357 mp-149"
  MP_ID_ARGS=(--mp-ids $MP_IDS)
elif [[ -n "$MP_ID_FILE" ]]; then
  MP_ID_ARGS=(--mp-id-file "$MP_ID_FILE")
fi

python scripts/prepare_all_benchmarks.py \
  --datasets mp \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --probes "$PROBES" \
  --seed "$SEED" \
  "${LIMIT_ARG[@]}" \
  "${MP_ID_ARGS[@]}" \
  "${FORCE_ARG[@]}"

echo "Full MP preparation done."
echo "Processed data: $REPO_DIR/$PROCESSED_ROOT/materials_project"
echo "Splits: train.txt, val.txt, test.txt"
