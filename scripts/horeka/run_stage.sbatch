#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=accelerated
#
# Generic FlowGuard++ stage runner used by scripts/schedule_experiments.py.
#
#   sbatch [resources] --chdir=<project root> run_stage.sbatch <done-marker> <script.py> [args...]
#
# If <done-marker> already exists the job exits immediately. That is what makes
# chained continuation jobs cheap: the scheduler submits a long stage as N jobs
# linked with --dependency=afterany, every job resumes from the stage's own
# checkpoint, and the ones after completion return at once.

set -euo pipefail

DONE_MARKER="${1:?done marker path required}"
shift

export FLOWGUARD_PROJECT_ROOT="${FLOWGUARD_PROJECT_ROOT:-$(pwd)}"
cd "${FLOWGUARD_PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${FLOWGUARD_PROJECT_ROOT}/scripts/horeka/env.sh"

echo "[stage] job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} at $(date -Is)"
echo "[stage] done marker: ${DONE_MARKER}"
if [[ -f "${DONE_MARKER}" ]]; then
  echo "[stage] already complete, nothing to do."
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[stage] ${PYTHON} $*"
"${PYTHON}" "$@"
if [[ ! -f "${DONE_MARKER}" ]]; then
  echo "[stage] WARNING: command exited 0 but did not write ${DONE_MARKER}." >&2
  exit 3
fi
echo "[stage] finished at $(date -Is)"
