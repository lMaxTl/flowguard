#!/usr/bin/env bash
# One-time setup of FlowGuard++ on HoreKa. Run on a LOGIN node (needs internet):
#
#   cp scripts/horeka/site.env.example scripts/horeka/site.env   # optional, edit
#   bash scripts/horeka/setup_env.sh
#
# Steps (each is skipped when already done, so re-running is safe):
#   1. load the Python module, create the virtualenv, install PyTorch + FlowGuard++
#   2. download all datasets and build the 32x32 caches (scripts/prepare_datasets.py)
#   3. pre-fetch google/ddpm-cifar10-32 into the HF cache (only needed to reproduce
#      the published CIFAR-10/VGG16 attack; the new runs train their own prior)
#   4. run scripts/horeka/check_env.py (imports, models, dataset caches)
#
# Environment knobs:
#   DATASETS=CIFAR10,CIFAR100      prepare only these (default: all)
#   SKIP_DATA=1                    skip step 2
#   FETCH_GOOGLE_DDPM=0            skip step 3
#   REMOVE_RAW=1                   delete raw downloads after caching (saves ~25 GB)
#   REINSTALL=1                    force pip install even if flowguard imports

set -euo pipefail

FLOWGUARD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FLOWGUARD_PROJECT_ROOT
cd "${FLOWGUARD_PROJECT_ROOT}"

# Load modules and resolve paths, but do not activate a venv that may not exist yet.
export FLOWGUARD_SKIP_ACTIVATE=1
export HF_HUB_OFFLINE=0
# shellcheck source=/dev/null
source "${FLOWGUARD_PROJECT_ROOT}/scripts/horeka/env.sh"

echo "[setup] project root: ${FLOWGUARD_PROJECT_ROOT}"
echo "[setup] venv:         ${FLOWGUARD_VENV}"
echo "[setup] data root:    ${FLOWGUARD_DATA_ROOT}"
echo "[setup] python:       $(command -v python) ($(python --version 2>&1))"
mkdir -p "${FLOWGUARD_DATA_ROOT}" "${FLOWGUARD_PROJECT_ROOT}/logs"

# --- 1. virtualenv ---------------------------------------------------------------------------
if [[ ! -x "${FLOWGUARD_VENV}/bin/python" ]]; then
  echo "[setup] creating virtualenv"
  python -m venv "${FLOWGUARD_VENV}"
fi
# shellcheck source=/dev/null
source "${FLOWGUARD_VENV}/bin/activate"
PYTHON="${FLOWGUARD_VENV}/bin/python"

if [[ "${REINSTALL:-0}" == "1" ]] || ! "${PYTHON}" -c "import flowguard, torch, diffusers, pyarrow" 2>/dev/null; then
  "${PYTHON}" -m pip install --upgrade pip wheel setuptools
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "${PYTHON}" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
  else
    "${PYTHON}" -m pip install torch torchvision
  fi
  "${PYTHON}" -m pip install -e "${FLOWGUARD_PROJECT_ROOT}[viz,serve,diffusion,data]" || {
    # Known broken-metadata case (see docs/INSTALL.md).
    "${PYTHON}" -m pip install --force-reinstall --no-cache-dir typing-inspection
    "${PYTHON}" -m pip install -e "${FLOWGUARD_PROJECT_ROOT}[viz,serve,diffusion,data]"
  }
fi
"${PYTHON}" -c "import torch; print('[setup] torch', torch.__version__, 'CUDA build', torch.version.cuda)"

# --- 2. datasets ---------------------------------------------------------------------------------
if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  PREP_ARGS=(--datasets "${DATASETS:-all}")
  [[ "${REMOVE_RAW:-0}" == "1" ]] && PREP_ARGS+=(--remove-raw)
  "${PYTHON}" scripts/prepare_datasets.py "${PREP_ARGS[@]}"
fi

# --- 3. optional published prior ------------------------------------------------------------------
if [[ "${FETCH_GOOGLE_DDPM:-1}" == "1" ]]; then
  "${PYTHON}" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("google/ddpm-cifar10-32")
print(f"[setup] google/ddpm-cifar10-32 cached at {path}")
PY
fi

# --- 4. self-check --------------------------------------------------------------------------------
HF_HUB_OFFLINE=1 "${PYTHON}" scripts/horeka/check_env.py

echo
echo "[setup] done. Next:"
echo "  python scripts/schedule_experiments.py --dry-run     # show what would be submitted"
echo "  python scripts/schedule_experiments.py               # submit all missing jobs"
