#!/bin/bash
#SBATCH --job-name=victim-cifar10-vgg16_bn
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/victim_cifar10_vgg16_bn_%j.log
#SBATCH --error=logs/victim_cifar10_vgg16_bn_%j.err

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

# Optional runtime overrides. Example:
#   sbatch --export=ALL,EPOCHS=150,BATCH_SIZE=256,LR=0.05 scripts/train_victim_cifar10.sbatch
DATASET="${DATASET:-CIFAR10}"
ARCHITECTURE="${ARCHITECTURE:-vgg16_bn}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.01}"
LR_STEP="${LR_STEP:-25}"
LR_GAMMA="${LR_GAMMA:-0.5}"
MOMENTUM="${MOMENTUM:-0.9}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/victim}"

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

DATASET_LOWER="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
RUN_DIR="${OUTPUT_ROOT}/${DATASET_LOWER}_${ARCHITECTURE}_ep${EPOCHS}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] Output dir: ${RUN_DIR}"
echo "[INFO] CUDA visible:"
nvidia-smi || true

python scripts/train_victim.py \
  --dataset "${DATASET}" \
  --architecture "${ARCHITECTURE}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lr-step "${LR_STEP}" \
  --lr-gamma "${LR_GAMMA}" \
  --momentum "${MOMENTUM}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device cuda \
  --download

echo "[INFO] Victim training finished. Artifacts: ${RUN_DIR}"
echo "[INFO] Checkpoint:  ${RUN_DIR}/checkpoint.pth.tar"
echo "[INFO] Params JSON: ${RUN_DIR}/params.json"
echo
echo "[HINT] Point downstream FlowPure training at this checkpoint via:"
echo "       VICTIM_CHECKPOINT_DIR=${RUN_DIR} sbatch scripts/train_flowpure_boundary_cifar10_300k.sbatch"
