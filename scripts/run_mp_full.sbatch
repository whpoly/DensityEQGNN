#!/bin/bash
#SBATCH --job-name=density-mp
#SBATCH --output=./job_logs/output_%j.txt
#SBATCH --error=./job_logs/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00

set -eo pipefail

CONDA_PATH="${CONDA_PATH:-/home/wuhao/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-density-eqgnn}"
REPO_DIR="${REPO_DIR:-/home/wuhao/DensityEQGNN}"
CONFIG_PATH="${CONFIG_PATH:-configs/materials_project.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/materials_project/best.pt}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/materials_project}"

if [[ -z "${MP_API_KEY:-}" ]]; then
  echo "ERROR: MP_API_KEY is not set."
  echo "Submit with: sbatch --export=ALL,MP_API_KEY='your_key' scripts/run_mp_full.sbatch"
  exit 1
fi

cd "$REPO_DIR"
mkdir -p job_logs

# Some conda activation hooks read optional variables before defining them.
set +u
source "$CONDA_PATH"
conda activate "$ENV_NAME"
set -u

export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

bash scripts/prepare_mp_full.sh

if [[ ! -s "$PROCESSED_DIR/train.txt" ]]; then
  echo "ERROR: $PROCESSED_DIR/train.txt was not created."
  echo "Data preparation failed or produced zero usable Materials Project samples."
  exit 1
fi

python -m density_eqgnn.training.train --config "$CONFIG_PATH"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "ERROR: $CHECKPOINT_PATH was not created."
  echo "Training failed before saving a best checkpoint."
  exit 1
fi

python -m density_eqgnn.training.evaluate \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --split test
