#!/usr/bin/env bash
# One-time setup of FlowGuard++ on HoreKa. Run on a LOGIN node (needs internet):
#
#   cp scripts/horeka/site.env.example scripts/horeka/site.env   # optional, edit
#   bash scripts/horeka/setup_env.sh
#
# Steps (each is skipped when already done, so re-running is safe):
#   1. load the Python module, create the virtualenv, install FlowGuard++
#   2. download all datasets and build the 32x32 caches (scripts/prepare_datasets.py)
#   3. pre-fetch google/ddpm-cifar10-32 into the HF cache (only needed to reproduce
#      the published CIFAR-10/VGG16 attack; the new runs train their own prior)
#   4. run scripts/horeka/check_env.py (imports, models, dataset caches)
#
# Environment knobs (all prefixed, because HoreKa itself exports e.g. $DATASETS):
#   FLOWGUARD_DATASETS=CIFAR10,CIFAR100   prepare only these (default: all)
#   FLOWGUARD_SKIP_DATA=1                 skip step 2
#   FLOWGUARD_FETCH_GOOGLE_DDPM=0         skip step 3
#   FLOWGUARD_REMOVE_RAW=1                delete raw downloads after caching (saves ~25 GB)
#   FLOWGUARD_REINSTALL=1                 force pip install even if everything imports
#   FLOWGUARD_VENV_SYSTEM_SITE=0          fully isolated venv (installs its own torch)
#   FLOWGUARD_TORCH_SPEC / FLOWGUARD_TORCH_INDEX_URL   torch version / wheel index

set -euo pipefail

FLOWGUARD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FLOWGUARD_PROJECT_ROOT
cd "${FLOWGUARD_PROJECT_ROOT}"

# Load modules and resolve paths, but do not activate a venv that may not exist yet.
export FLOWGUARD_SKIP_ACTIVATE=1
export HF_HUB_OFFLINE=0
# shellcheck source=/dev/null
source "${FLOWGUARD_PROJECT_ROOT}/scripts/horeka/env.sh"

# The venv reuses the module's packages (torch 2.7.0+cu126, numpy, scipy, ...)
# through --system-site-packages; its own installs take precedence over them.
: "${FLOWGUARD_VENV_SYSTEM_SITE:=1}"
# Same torch build as the jupyter/ai/2025-05-23 module, which is known to work
# with the HoreKa GPU drivers. With system site packages this is already
# satisfied and nothing is downloaded.
: "${FLOWGUARD_TORCH_SPEC:=torch==2.7.0 torchvision==0.22.0}"
: "${FLOWGUARD_TORCH_INDEX_URL:=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}}"

echo "[setup] project root: ${FLOWGUARD_PROJECT_ROOT}"
echo "[setup] venv:         ${FLOWGUARD_VENV} (system site packages: ${FLOWGUARD_VENV_SYSTEM_SITE})"
echo "[setup] data root:    ${FLOWGUARD_DATA_ROOT}"
echo "[setup] python:       $(command -v python) ($(python --version 2>&1))"
if [[ -n "${FLOWGUARD_MODULE_PYTHONPATH:-}" ]]; then
  echo "[setup] removed module PYTHONPATH (it would shadow venv packages): ${FLOWGUARD_MODULE_PYTHONPATH}"
fi
mkdir -p "${FLOWGUARD_DATA_ROOT}" "${FLOWGUARD_PROJECT_ROOT}/logs"

# --- 1. virtualenv ---------------------------------------------------------------------------
VENV_ARGS=()
[[ "${FLOWGUARD_VENV_SYSTEM_SITE}" == "1" ]] && VENV_ARGS+=(--system-site-packages)
if [[ ! -x "${FLOWGUARD_VENV}/bin/python" ]]; then
  echo "[setup] creating virtualenv"
  python -m venv ${VENV_ARGS[@]+"${VENV_ARGS[@]}"} "${FLOWGUARD_VENV}"
fi
# A venv created by an earlier version of this script lacks system site
# packages; switch it over instead of rebuilding it (the flag is plain config).
PYVENV_CFG="${FLOWGUARD_VENV}/pyvenv.cfg"
if [[ "${FLOWGUARD_VENV_SYSTEM_SITE}" == "1" ]] && grep -q "^include-system-site-packages = false" "${PYVENV_CFG}"; then
  echo "[setup] enabling system site packages in ${PYVENV_CFG}"
  sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' "${PYVENV_CFG}"
fi
# shellcheck source=/dev/null
source "${FLOWGUARD_VENV}/bin/activate"
PYTHON="${FLOWGUARD_VENV}/bin/python"

if [[ "${FLOWGUARD_REINSTALL:-0}" == "1" ]] || ! "${PYTHON}" -c "import flowguard, torch, diffusers, pyarrow, skopt" 2>/dev/null; then
  "${PYTHON}" -m pip install --upgrade pip wheel setuptools
  # shellcheck disable=SC2086  # word splitting of the spec is intended
  "${PYTHON}" -m pip install ${FLOWGUARD_TORCH_SPEC} --extra-index-url "${FLOWGUARD_TORCH_INDEX_URL}"
  "${PYTHON}" -m pip install -e "${FLOWGUARD_PROJECT_ROOT}[viz,serve,diffusion,data]" || {
    # Known broken-metadata case (see docs/INSTALL.md).
    "${PYTHON}" -m pip install --force-reinstall --no-cache-dir typing-inspection
    "${PYTHON}" -m pip install -e "${FLOWGUARD_PROJECT_ROOT}[viz,serve,diffusion,data]"
  }
fi
# Show which copy of the packages that exist in both places is actually imported.
"${PYTHON}" - <<'PY'
import importlib, torch
print(f"[setup] torch {torch.__version__} (CUDA build {torch.version.cuda}) from {torch.__file__}")
for name in ("typing_extensions", "click", "numpy", "diffusers", "flowguard"):
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        print(f"[setup]   {name:17s} NOT IMPORTABLE: {error}")
        continue
    version = getattr(module, "__version__", "")
    print(f"[setup]   {name:17s} {version:10s} {module.__file__}")
PY

# --- 2. datasets ---------------------------------------------------------------------------------
if [[ "${FLOWGUARD_SKIP_DATA:-0}" != "1" ]]; then
  PREP_ARGS=(--datasets "${FLOWGUARD_DATASETS:-all}")
  [[ "${FLOWGUARD_REMOVE_RAW:-0}" == "1" ]] && PREP_ARGS+=(--remove-raw)
  "${PYTHON}" scripts/prepare_datasets.py "${PREP_ARGS[@]}"
fi

# --- 3. optional published prior ------------------------------------------------------------------
if [[ "${FLOWGUARD_FETCH_GOOGLE_DDPM:-1}" == "1" ]]; then
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
