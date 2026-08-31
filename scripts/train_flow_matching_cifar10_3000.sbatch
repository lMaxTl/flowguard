#!/bin/bash
#SBATCH --job-name=fm-cifar10-3000
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fm_cifar10_3000_%j.log
#SBATCH --error=logs/fm_cifar10_3000_%j.err

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

# Optional runtime overrides:
#   sbatch -p accelerated-h100 --export=ALL,EPOCHS=3000,BATCH_SIZE=32,LR=1e-4 scripts/train_flow_matching_cifar10_3000.sbatch
EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/flow_matching}"

# Best practice on HoreKa: keep module state clean.
module purge
module load jupyter/ai/2025-05-23

# Ensure venv exists and dependencies are installed
if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
  echo "[INFO] Creating venv..."
  python -m venv "${PROJECT_ROOT}/venv"
fi

source "${PROJECT_ROOT}/venv/bin/activate"

# Verify and install dependencies
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
RUN_DIR="${OUTPUT_ROOT}/cifar10_ep${EPOCHS}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] CUDA visible:"
nvidia-smi || true

python scripts/train_flow_matching.py \
  --dataset cifar10 \
  --data-path "${DATA_PATH}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda \
  --download

echo "[INFO] Training finished. Artifacts: ${RUN_DIR}"
