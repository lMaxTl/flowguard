#!/bin/bash
#SBATCH --job-name=eval-detection-suite
#SBATCH --partition=dev_accelerated-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/eval_detection_suite_%j.log
#SBATCH --error=logs/eval_detection_suite_%j.err

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
# sbatch --export=ALL,NAME=my-eval,QUERY_BUDGET=200,MAX_RUNS=3 scripts/evaluate_detection_suite.sbatch
NAME="${NAME:-detection-suite-h100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/detection_suite}"
DATASET="${DATASET:-CIFAR10}"
TARGET_ARCHITECTURE="${TARGET_ARCHITECTURE:-resnet18}"
TARGET_CHECKPOINT_DIR="${TARGET_CHECKPOINT_DIR:-}"
QUERY_DATASET="${QUERY_DATASET:-TinyImageNet200}"
QUERY_BUDGET="${QUERY_BUDGET:-120}"
QUERY_TRANSFER_SET_SIZE="${QUERY_TRANSFER_SET_SIZE:-160}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-8}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-1}"
BENIGN_QUERIES="${BENIGN_QUERIES:-160}"
BENIGN_BATCH_SIZE="${BENIGN_BATCH_SIZE:-16}"
DEFENSES="${DEFENSES:-fdinet,prada,flow_matching}"
ATTACKS="${ATTACKS:-naive,maze,disguide}"
MAX_RUNS="${MAX_RUNS:-0}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-runs/flow_matching/cifar10_notebook/checkpoint_latest.pt}"
DATASET_DOWNLOAD="${DATASET_DOWNLOAD:-0}"
DISTRIBUTED="${DISTRIBUTED:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TARGET_DEVICE="${TARGET_DEVICE:-cuda}"
SUBSTITUTE_DEVICE="${SUBSTITUTE_DEVICE:-cuda}"
DEFENSE_OVERRIDES_JSON="${DEFENSE_OVERRIDES_JSON:-}"

module purge
module load jupyter/ai/2025-05-23

# Ensure local .venv exists and dependencies are installed.
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "[INFO] Creating .venv..."
  python -m venv "${PROJECT_ROOT}/.venv"
fi

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "[INFO] Installing/verifying dependencies..."
pip install -q -e . || {
  echo "[ERROR] Failed to install FlowGuard++. Check pip output above."
  exit 1
}

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi
if [[ "${FLOW_CHECKPOINT}" != /* ]]; then
  FLOW_CHECKPOINT="${PROJECT_ROOT}/${FLOW_CHECKPOINT}"
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] CUDA visible:"
nvidia-smi || true
echo "[INFO] Flow checkpoint: ${FLOW_CHECKPOINT}"

CMD=(
  python scripts/evaluate_detection_suite.py
  --name "${NAME}"
  --output-root "${OUTPUT_ROOT}"
  --dataset "${DATASET}"
  --target-architecture "${TARGET_ARCHITECTURE}"
  --query-dataset "${QUERY_DATASET}"
  --target-device "${TARGET_DEVICE}"
  --substitute-device "${SUBSTITUTE_DEVICE}"
  --query-budget "${QUERY_BUDGET}"
  --query-transfer-set-size "${QUERY_TRANSFER_SET_SIZE}"
  --attack-batch-size "${ATTACK_BATCH_SIZE}"
  --training-batch-size "${TRAINING_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --benign-queries "${BENIGN_QUERIES}"
  --benign-batch-size "${BENIGN_BATCH_SIZE}"
  --defenses "${DEFENSES}"
  --attacks "${ATTACKS}"
  --max-runs "${MAX_RUNS}"
  --flow-checkpoint "${FLOW_CHECKPOINT}"
  --num-clients "${NUM_CLIENTS}"
  --num-workers "${NUM_WORKERS}"
)

if [[ -n "${TARGET_CHECKPOINT_DIR}" ]]; then
  CMD+=(--target-checkpoint-dir "${TARGET_CHECKPOINT_DIR}")
fi

if [[ "${DATASET_DOWNLOAD}" == "1" ]]; then
  CMD+=(--dataset-download)
fi

if [[ "${DISTRIBUTED}" == "1" ]]; then
  CMD+=(--distributed)
fi

if [[ -n "${DEFENSE_OVERRIDES_JSON}" ]]; then
  CMD+=(--defense-overrides-json "${DEFENSE_OVERRIDES_JSON}")
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Detection suite finished."
echo "[INFO] Summary JSON: ${OUTPUT_ROOT}/${NAME}/summary.json"
echo "[INFO] Summary CSV:  ${OUTPUT_ROOT}/${NAME}/summary.csv"
