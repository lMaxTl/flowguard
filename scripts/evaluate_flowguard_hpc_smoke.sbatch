#!/bin/bash
#SBATCH --job-name=flowguard-hpc-smoke
#SBATCH --partition=dev_accelerated-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/flowguard_hpc_smoke_%j.log
#SBATCH --error=logs/flowguard_hpc_smoke_%j.err

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

# Smoke defaults are intentionally tiny and sequential so the job also writes
# the aggregate report at the end.
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/flowguard_hpc/smoke}"
DEFENSES="${DEFENSES:-flowpure,flowguard_integral,flowguard_userlevel}"
ATTACKS="${ATTACKS:-a1_clean_transfer,a2_projected_maze,a3_flowblind,adaptive_adaptive,maze_baseline}"
TARGET_CHECKPOINT_DIR="${TARGET_CHECKPOINT_DIR:-runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt}"
A3_SURROGATE_CHECKPOINT="${A3_SURROGATE_CHECKPOINT:-${FLOW_CHECKPOINT}}"
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

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
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
  --target-checkpoint-dir "${TARGET_CHECKPOINT_DIR}"
  --flow-checkpoint "${FLOW_CHECKPOINT}"
  --a3-surrogate-checkpoint "${A3_SURROGATE_CHECKPOINT}"
  --device cuda
  --maze-query-budget 256
  --clean-transfer-query-budget 256
  --attack-query-budget 256
  --benign-query-budget 256
  --attack-batch-size 1
  --extraction-attack-batch-size 16
  --training-batch-size 16
  --epochs 1
  --seed-samples 16
  --transfer-artifact-sample-size 512
  --maze-replay-buffer-size 512
  --maze-artifact-sample-size 64
  --maze-log-iter 50
)

if [[ "${FORCE}" == "1" ]]; then
  CMD+=(--force)
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Smoke evaluation finished."
echo "[INFO] Summary: ${OUTPUT_ROOT}/flowguard_hpc_summary.json"
echo "[INFO] Report:  ${OUTPUT_ROOT}/flowguard_hpc_report.md"
