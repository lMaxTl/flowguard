#!/usr/bin/env bash
# Shared environment for every FlowGuard++ job on HoreKa (sourced, not executed).
#
# Sourced by scripts/horeka/setup_env.sh (login node) and by run_stage.sbatch
# (compute nodes). Site-specific values go into scripts/horeka/site.env, which
# is sourced first if it exists (copy site.env.example).
#
# Compute jobs never install anything and never download anything: the venv,
# the datasets and the HuggingFace cache are all prepared on the login node by
# setup_env.sh, and HF_HUB_OFFLINE=1 makes a missing file fail loudly instead
# of hanging on a network call.

if [[ -z "${FLOWGUARD_PROJECT_ROOT:-}" ]]; then
  FLOWGUARD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export FLOWGUARD_PROJECT_ROOT

if [[ -f "${FLOWGUARD_PROJECT_ROOT}/scripts/horeka/site.env" ]]; then
  # shellcheck source=/dev/null
  source "${FLOWGUARD_PROJECT_ROOT}/scripts/horeka/site.env"
fi

# Module that provides Python (the one the existing .sbatch scripts use).
: "${FLOWGUARD_MODULES:=jupyter/ai/2025-05-23}"
: "${FLOWGUARD_VENV:=${FLOWGUARD_PROJECT_ROOT}/.venv}"
: "${FLOWGUARD_DATA_ROOT:=${FLOWGUARD_PROJECT_ROOT}/data}"
export FLOWGUARD_MODULES FLOWGUARD_VENV FLOWGUARD_DATA_ROOT

if ! command -v module >/dev/null 2>&1; then
  # Non-login batch shells do not always define the Lmod function.
  for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
    if [[ -f "${init}" ]]; then
      # shellcheck source=/dev/null
      source "${init}"
      break
    fi
  done
fi
if command -v module >/dev/null 2>&1; then
  # Lmod reads variables that may be unset; do not trip callers using `set -u`.
  _flowguard_restore_u=0
  [[ $- == *u* ]] && _flowguard_restore_u=1 && set +u
  module purge
  for mod in ${FLOWGUARD_MODULES}; do
    module load "${mod}"
  done
  [[ ${_flowguard_restore_u} == 1 ]] && set -u
  unset _flowguard_restore_u
else
  echo "[WARN] 'module' command not found; using the system Python." >&2
fi

# The jupyter/ai module puts its own site-packages on PYTHONPATH, and Python
# searches PYTHONPATH *before* the virtualenv. Module packages then shadow the
# newer versions pip installed into the venv (observed on HoreKa: the module's
# typing_extensions 4.13 and click 8.2 hiding the venv's 4.16 and 8.5). The venv
# is created with --system-site-packages instead (setup_env.sh), which still
# reuses the module's torch/numpy/scipy but lets venv packages take precedence.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export FLOWGUARD_MODULE_PYTHONPATH="${PYTHONPATH}"
  unset PYTHONPATH
fi
# Keep ~/.local packages out, for the same reason.
export PYTHONNOUSERSITE=1

if [[ "${FLOWGUARD_SKIP_ACTIVATE:-0}" != "1" ]]; then
  if [[ ! -x "${FLOWGUARD_VENV}/bin/python" ]]; then
    echo "[ERROR] No virtualenv at ${FLOWGUARD_VENV}. Run scripts/horeka/setup_env.sh on the login node first." >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck source=/dev/null
  source "${FLOWGUARD_VENV}/bin/activate"
  PYTHON="${FLOWGUARD_VENV}/bin/python"
  export PYTHON
fi

# Set explicitly (not "if unset"): a module may point these at a shared or
# read-only cache, and jobs must read exactly what setup_env.sh downloaded.
export HF_HOME="${FLOWGUARD_HF_HOME:-${FLOWGUARD_DATA_ROOT}/hf_home}"
export TORCH_HOME="${FLOWGUARD_TORCH_HOME:-${FLOWGUARD_DATA_ROOT}/torch_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
