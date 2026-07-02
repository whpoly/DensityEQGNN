#!/usr/bin/env bash
# Prepare a small Materials Project density benchmark subset.
#
# Usage:
#   export MP_API_KEY="your_materials_project_key"
#   bash scripts/prepare_mp_test.sh
#
# Override parameters at launch, for example:
#   MP_LIMIT_SAMPLES=5 PROBES=2000 bash scripts/prepare_mp_test.sh

set -euo pipefail

CONDA_PATH="${CONDA_PATH:-/home/wuhao/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-density-eqgnn}"
REPO_DIR="${REPO_DIR:-/home/wuhao/DensityEQGNN}"

RAW_ROOT="${RAW_ROOT:-data/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed}"
MP_LIMIT_SAMPLES="${MP_LIMIT_SAMPLES:-2}"
PROBES="${PROBES:-5000}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
MP_IDS="${MP_IDS:-mp-1524357}"

if [[ -z "${MP_API_KEY:-}" ]]; then
  echo "ERROR: MP_API_KEY is not set."
  echo "Run: export MP_API_KEY='your_materials_project_key'"
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

MP_ID_ARGS=()
if [[ -n "$MP_IDS" ]]; then
  # Space-separated list, e.g. MP_IDS="mp-1524357 mp-149"
  MP_ID_ARGS=(--mp-ids $MP_IDS)
fi

python scripts/prepare_all_benchmarks.py \
  --datasets mp \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --limit-samples "$MP_LIMIT_SAMPLES" \
  --probes "$PROBES" \
  --seed "$SEED" \
  "${MP_ID_ARGS[@]}" \
  "${FORCE_ARG[@]}"

echo "MP preparation done."
echo "Processed data: $REPO_DIR/$PROCESSED_ROOT/materials_project"
echo "Splits: train.txt, val.txt, test.txt"
