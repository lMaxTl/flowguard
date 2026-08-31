#!/usr/bin/env bash
# Shared HPC virtualenv bootstrap for Slurm jobs.
#
# Usage (from project root, after module load):
#   source scripts/hpc_venv_bootstrap.sh
#   hpc_bootstrap_venv "${PROJECT_ROOT}"
#
# Environment overrides:
#   FORCE_VENV_INSTALL=1     Always run pip install -e . even if import works.
#   RECREATE_VENV_ON_FAILURE=1  Recreate .venv when pip still fails (default: 1).
#   SKIP_VENV_INSTALL=1      Never run pip; fail if flowguard cannot be imported.

hpc_bootstrap_venv() {
  local project_root="${1:?project_root required}"
  local python_bin="${project_root}/.venv/bin/python"

  if [[ "${SKIP_VENV_INSTALL:-0}" == "1" ]]; then
    if [[ -x "${python_bin}" ]] && "${python_bin}" -c "import flowguard" 2>/dev/null; then
      echo "[INFO] SKIP_VENV_INSTALL=1 and flowguard is importable."
      PYTHON="${python_bin}"
      export PYTHON
      return 0
    fi
    echo "[ERROR] SKIP_VENV_INSTALL=1 but flowguard is not importable in ${python_bin}"
    return 1
  fi

  if [[ ! -d "${project_root}/.venv" ]]; then
    echo "[INFO] Creating local .venv..."
    python -m venv "${project_root}/.venv"
  fi

  if [[ ! -x "${python_bin}" ]]; then
    echo "[ERROR] Missing interpreter: ${python_bin}"
    return 1
  fi

  PYTHON="${python_bin}"
  export PYTHON

  if [[ "${FORCE_VENV_INSTALL:-0}" != "1" ]]; then
    if "${PYTHON}" -c "import flowguard" 2>/dev/null; then
      echo "[INFO] flowguard already importable; skipping pip install."
      echo "[INFO] Set FORCE_VENV_INSTALL=1 to reinstall editable package."
      return 0
    fi
  fi

  echo "[INFO] Installing/verifying package in .venv..."
  if "${PYTHON}" -m pip install -q -e "${project_root}[viz,serve]"; then
    return 0
  fi

  echo "[WARN] pip install -e . failed; repairing typing-inspection (common broken-metadata case)..."
  "${PYTHON}" -m pip install --force-reinstall --no-cache-dir typing-inspection 2>/dev/null || true
  if "${PYTHON}" -m pip install -q -e "${project_root}[viz,serve]"; then
    echo "[INFO] Editable install succeeded after typing-inspection repair."
    return 0
  fi

  if [[ "${RECREATE_VENV_ON_FAILURE:-1}" != "1" ]]; then
    echo "[ERROR] Failed to install FlowGuard++. Set RECREATE_VENV_ON_FAILURE=1 to recreate .venv."
    return 1
  fi

  echo "[WARN] Recreating .venv from scratch..."
  rm -rf "${project_root}/.venv"
  python -m venv "${project_root}/.venv"
  PYTHON="${project_root}/.venv/bin/python"
  export PYTHON
  "${PYTHON}" -m pip install -q --upgrade pip wheel setuptools
  if ! "${PYTHON}" -m pip install -q -e "${project_root}[viz,serve]"; then
    echo "[ERROR] Failed to install FlowGuard++ after recreating .venv."
    return 1
  fi
  echo "[INFO] Editable install succeeded in fresh .venv."
  return 0
}
