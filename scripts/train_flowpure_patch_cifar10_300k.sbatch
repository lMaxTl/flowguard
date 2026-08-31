#!/bin/bash
#SBATCH --job-name=flowpure-patch-cifar10-300k
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/flowpure_patch_cifar10_300k_%j.log
#SBATCH --error=logs/flowpure_patch_cifar10_300k_%j.err

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

# Optional runtime overrides. This is a pure Gaussian-source patch CNF: no
# victim classifier is required.
#   sbatch --export=ALL,MAX_STEPS=300000,BATCH_IMAGES=64,PATCHES_PER_IMAGE=16 \
#          scripts/train_flowpure_patch_cifar10_300k.sbatch
MAX_STEPS="${MAX_STEPS:-300000}"
BATCH_IMAGES="${BATCH_IMAGES:-32}"
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-8}"
PATCH_SIZE="${PATCH_SIZE:-8}"
LR="${LR:-2e-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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
RUN_DIR="${OUTPUT_ROOT}/cifar10_flowpure_patch_p${PATCH_SIZE}_steps${MAX_STEPS}_bi${BATCH_IMAGES}x${PATCHES_PER_IMAGE}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] CUDA visible:"
nvidia-smi || true

python scripts/train_flowpure_patch.py \
  --dataset cifar10 \
  --data-path "${DATA_PATH}" \
  --output-dir "${RUN_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --batch-images "${BATCH_IMAGES}" \
  --patches-per-image "${PATCHES_PER_IMAGE}" \
  --patch-size "${PATCH_SIZE}" \
  --lr "${LR}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda \
  --download

echo "[INFO] Training finished. Artifacts: ${RUN_DIR}"
