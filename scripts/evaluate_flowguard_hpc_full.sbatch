#!/bin/bash
#SBATCH --job-name=flowguard-hpc-full
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-14%3
#SBATCH --output=logs/flowguard_hpc_full_%A_%a.log
#SBATCH --error=logs/flowguard_hpc_full_%A_%a.err

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

module purge
module load jupyter/ai/2025-05-23

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "[INFO] Creating local .venv..."
  python -m venv "${PROJECT_ROOT}/.venv"
fi
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

echo "[INFO] Installing/verifying package in .venv..."
"${PYTHON}" -m pip install -q -e . || {
  echo "[ERROR] Failed to install FlowGuard++. Check pip output above."
  exit 1
}

mkdir -p "${PROJECT_ROOT}/logs"

# Runtime overrides. Example:
# sbatch --export=ALL,MAZE_QUERY_BUDGET=20000000,MAZE_REPLAY_BUFFER_SIZE=65536 scripts/evaluate_flowguard_hpc_full.sbatch
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/flowguard_hpc/full}"
DEFENSES="${DEFENSES:-flowpure,flowguard_integral,flowguard_userlevel}"
ATTACKS="${ATTACKS:-a1_clean_transfer,a2_projected_maze,a3_flowblind,adaptive_adaptive,maze_baseline}"
DATASET="${DATASET:-CIFAR10}"
QUERY_DATASET="${QUERY_DATASET:-CIFAR10}"
CLEAN_TRANSFER_DATASET="${CLEAN_TRANSFER_DATASET:-CIFAR100}"
TARGET_CHECKPOINT_DIR="${TARGET_CHECKPOINT_DIR:-runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt}"
A3_SURROGATE_CHECKPOINT="${A3_SURROGATE_CHECKPOINT:-${FLOW_CHECKPOINT}}"
DEVICE="${DEVICE:-cuda}"

MAZE_QUERY_BUDGET="${MAZE_QUERY_BUDGET:-20000000}"
CLEAN_TRANSFER_QUERY_BUDGET="${CLEAN_TRANSFER_QUERY_BUDGET:-1000000}"
ATTACK_QUERY_BUDGET="${ATTACK_QUERY_BUDGET:-1000000}"
BENIGN_QUERY_BUDGET="${BENIGN_QUERY_BUDGET:-200000}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-1}"
EXTRACTION_ATTACK_BATCH_SIZE="${EXTRACTION_ATTACK_BATCH_SIZE:-32}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
SEED_SAMPLES="${SEED_SAMPLES:-64}"
TRANSFER_ARTIFACT_SAMPLE_SIZE="${TRANSFER_ARTIFACT_SAMPLE_SIZE:-50000}"

# Keep MAZE long runs bounded. Increase only if the node has enough RAM/time.
MAZE_REPLAY_BUFFER_SIZE="${MAZE_REPLAY_BUFFER_SIZE:-32768}"
MAZE_ARTIFACT_SAMPLE_SIZE="${MAZE_ARTIFACT_SAMPLE_SIZE:-4096}"
MAZE_ITER_EXP="${MAZE_ITER_EXP:-1}"
MAZE_LOG_ITER="${MAZE_LOG_ITER:-5000}"
FORCE="${FORCE:-0}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi
if [[ "${TARGET_CHECKPOINT_DIR}" != /* ]]; then
  TARGET_CHECKPOINT_DIR="${PROJECT_ROOT}/${TARGET_CHECKPOINT_DIR}"
fi
if [[ "${FLOW_CHECKPOINT}" != /* ]]; then
  FLOW_CHECKPOINT="${PROJECT_ROOT}/${FLOW_CHECKPOINT}"
fi
if [[ "${A3_SURROGATE_CHECKPOINT}" != /* ]]; then
  A3_SURROGATE_CHECKPOINT="${PROJECT_ROOT}/${A3_SURROGATE_CHECKPOINT}"
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -d "${TARGET_CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Missing target checkpoint dir: ${TARGET_CHECKPOINT_DIR}"
  exit 1
fi
if [[ ! -f "${FLOW_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing FlowPure/FlowGuard checkpoint: ${FLOW_CHECKPOINT}"
  exit 1
fi

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Array task: ${SLURM_ARRAY_TASK_ID:-none}/${SLURM_ARRAY_TASK_COUNT:-15}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Python: $("${PYTHON}" --version 2>&1)"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] CUDA visible:"
nvidia-smi || true

CMD=(
  "${PYTHON}" scripts/evaluate_flowguard_hpc_suite.py
  --output-root "${OUTPUT_ROOT}"
  --defenses "${DEFENSES}"
  --attacks "${ATTACKS}"
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --num-tasks "${SLURM_ARRAY_TASK_COUNT:-15}"
  --dataset "${DATASET}"
  --query-dataset "${QUERY_DATASET}"
  --clean-transfer-dataset "${CLEAN_TRANSFER_DATASET}"
  --target-checkpoint-dir "${TARGET_CHECKPOINT_DIR}"
  --flow-checkpoint "${FLOW_CHECKPOINT}"
  --a3-surrogate-checkpoint "${A3_SURROGATE_CHECKPOINT}"
  --device "${DEVICE}"
  --maze-query-budget "${MAZE_QUERY_BUDGET}"
  --clean-transfer-query-budget "${CLEAN_TRANSFER_QUERY_BUDGET}"
  --attack-query-budget "${ATTACK_QUERY_BUDGET}"
  --benign-query-budget "${BENIGN_QUERY_BUDGET}"
  --attack-batch-size "${ATTACK_BATCH_SIZE}"
  --extraction-attack-batch-size "${EXTRACTION_ATTACK_BATCH_SIZE}"
  --training-batch-size "${TRAINING_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --seed-samples "${SEED_SAMPLES}"
  --transfer-artifact-sample-size "${TRANSFER_ARTIFACT_SAMPLE_SIZE}"
  --maze-replay-buffer-size "${MAZE_REPLAY_BUFFER_SIZE}"
  --maze-artifact-sample-size "${MAZE_ARTIFACT_SAMPLE_SIZE}"
  --maze-iter-exp "${MAZE_ITER_EXP}"
  --maze-log-iter "${MAZE_LOG_ITER}"
)

if [[ "${FORCE}" == "1" ]]; then
  CMD+=(--force)
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Array task finished."
echo "[INFO] After all array tasks complete, aggregate with:"
echo "       ${PYTHON} scripts/evaluate_flowguard_hpc_suite.py --output-root ${OUTPUT_ROOT} --defenses ${DEFENSES} --attacks ${ATTACKS} --aggregate-only"
