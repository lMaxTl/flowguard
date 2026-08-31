#!/bin/bash
#SBATCH --job-name=flowpure-boundary-cifar10-300k
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/flowpure_boundary_cifar10_300k_%j.log
#SBATCH --error=logs/flowpure_boundary_cifar10_300k_%j.err

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

# Optional runtime overrides. Training is attack-agnostic: x_0 is generated
# by a BAM-style ES on the victim's softmax confidence.
#   sbatch -p accelerated-h100 --export=ALL,MAX_STEPS=300000,BATCH_SIZE=64,ES_POP=8 scripts/train_flowpure_boundary_cifar10_300k.sbatch
MAX_STEPS="${MAX_STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-4}"
ES_POP="${ES_POP:-8}"
ES_TOP_K="${ES_TOP_K:-2}"
ES_ITERS="${ES_ITERS:-4}"
ES_GAMMA="${ES_GAMMA:-0.1}"
ES_INIT_EPS="${ES_INIT_EPS:-0.05}"
ES_EVAL_BATCH="${ES_EVAL_BATCH:-256}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/flow_matching}"
VICTIM_CHECKPOINT_DIR="${VICTIM_CHECKPOINT_DIR:-runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model}"

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
if [[ "${VICTIM_CHECKPOINT_DIR}" != /* ]]; then
  VICTIM_CHECKPOINT_DIR="${PROJECT_ROOT}/${VICTIM_CHECKPOINT_DIR}"
fi

if [[ ! -d "${VICTIM_CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Victim checkpoint dir not found: ${VICTIM_CHECKPOINT_DIR}"
  echo "[ERROR] Train the victim first or pass --export=ALL,VICTIM_CHECKPOINT_DIR=..."
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"
RUN_DIR="${OUTPUT_ROOT}/cifar10_flowpure_boundary_steps${MAX_STEPS}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] Victim checkpoint: ${VICTIM_CHECKPOINT_DIR}"
echo "[INFO] CUDA visible:"
nvidia-smi || true

python scripts/train_flowpure_boundary.py \
  --dataset cifar10 \
  --data-path "${DATA_PATH}" \
  --output-dir "${RUN_DIR}" \
  --victim-checkpoint-dir "${VICTIM_CHECKPOINT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --es-pop "${ES_POP}" \
  --es-top-k "${ES_TOP_K}" \
  --es-iters "${ES_ITERS}" \
  --es-gamma "${ES_GAMMA}" \
  --es-init-eps "${ES_INIT_EPS}" \
  --es-eval-batch "${ES_EVAL_BATCH}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda \
  --download

echo "[INFO] Training finished. Artifacts: ${RUN_DIR}"
