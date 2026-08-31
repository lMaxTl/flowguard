#!/bin/bash
#SBATCH --job-name=fm-cifar10-smoke
#SBATCH --partition=dev_accelerated-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/fm_cifar10_smoke_%j.log
#SBATCH --error=logs/fm_cifar10_smoke_%j.err

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

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/runs/flow_matching"
RUN_DIR="${PROJECT_ROOT}/runs/flow_matching/cifar10_smoke_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
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

python scripts/train_flow_matching.py \
  --dataset cifar10 \
  --data-path "${PROJECT_ROOT}/data" \
  --output-dir "${RUN_DIR}" \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 8 \
  --device cuda \
  --download

echo "[INFO] Smoke test finished. Artifacts: ${RUN_DIR}"
