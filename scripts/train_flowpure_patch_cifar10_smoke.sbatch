#!/bin/bash
#SBATCH --job-name=flowpure-patch-cifar10-smoke
#SBATCH --partition=dev_accelerated-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowpure_patch_cifar10_smoke_%j.log
#SBATCH --error=logs/flowpure_patch_cifar10_smoke_%j.err

set -euo pipefail

# Resolve project root from submit directory (works from repo root or scripts/).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/pyproject.toml" && -d "${SUBMIT_DIR}/scripts" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../pyproject.toml" && -d "${SUBMIT_DIR}/../scripts" ]]; then
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "[ERROR] Could not resolve project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}"
  echo "[ERROR] Submit from repo root or scripts directory."
  exit 1
fi
cd "${PROJECT_ROOT}"

# Optional runtime overrides (all have sensible defaults for the smoke job):
#   sbatch --export=ALL,MAX_STEPS=2000,BATCH_IMAGES=32,PATCHES_PER_IMAGE=8 \
#          scripts/train_flowpure_patch_cifar10_smoke.sbatch
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_IMAGES="${BATCH_IMAGES:-16}"
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-8}"
PATCH_SIZE="${PATCH_SIZE:-8}"
LR="${LR:-2e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SUBSET_SIZE="${SUBSET_SIZE:-2048}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/flow_matching}"

module purge
module load jupyter/ai/2025-05-23

# Ensure venv exists and dependencies are installed
if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
  echo "[INFO] Creating venv..."
  python -m venv "${PROJECT_ROOT}/venv"
fi

source "${PROJECT_ROOT}/venv/bin/activate"

echo "[INFO] Installing/verifying dependencies..."
pip install -q -e . || {
  echo "[ERROR] Failed to install FlowGuard++. Check pip output above."
  exit 1
}

if [[ "${DATA_PATH}" != /* ]]; then
  DATA_PATH="${PROJECT_ROOT}/${DATA_PATH}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"
RUN_DIR="${OUTPUT_ROOT}/cifar10_flowpure_patch_smoke_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] Torch/CUDA check:"
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY

python scripts/train_flowpure_patch.py \
  --dataset cifar10 \
  --data-path "${DATA_PATH}" \
  --output-dir "${RUN_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --batch-images "${BATCH_IMAGES}" \
  --patches-per-image "${PATCHES_PER_IMAGE}" \
  --patch-size "${PATCH_SIZE}" \
  --lr "${LR}" \
  --num-workers "${NUM_WORKERS}" \
  --subset-size "${SUBSET_SIZE}" \
  --checkpoint-every 500 \
  --log-every 50 \
  --device cuda \
  --download

echo "[INFO] Smoke test finished. Artifacts: ${RUN_DIR}"
