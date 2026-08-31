#!/bin/bash
#SBATCH --job-name=fgpp-runtime
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fgpp_runtime_%j.log
#SBATCH --error=logs/fgpp_runtime_%j.err

# Serving cost of the FlowGuard++ components on the A100 the paper reports.
#
# Why this must run on HoreKa rather than a workstation: the paper states its
# hardware, and a latency table measured elsewhere answers a different question
# than the one a reviewer asks ("can this run at API throughput on the stated
# deployment?"). A local RTX 4060 additionally cannot hold the C3 solve at
# batch 64 in 8 GB -- it falls back to host memory and the timing then measures
# paging, not the detector.
#
# Usage:
#   sbatch scripts/benchmark_flowguard_runtime.sbatch
#   sbatch --export=ALL,PRESET=mnist scripts/benchmark_flowguard_runtime.sbatch
#   BATCH_SIZES=1,8,64,256 sbatch --export=ALL scripts/benchmark_flowguard_runtime.sbatch
#
# Comma-separated values must be set in the submitting environment as above,
# never inside --export: Slurm splits --export on commas and would keep only
# BATCH_SIZES=1. See scripts/hpc_list_guard.sh.
#
# Output:
#   runs/runtime/flowguard_runtime_<preset>_a100.json
#   paper/tables/runtime_overhead.tex   (rendered at LATEX_BATCH_SIZE)

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/pyproject.toml" && -d "${SUBMIT_DIR}/scripts" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../pyproject.toml" && -d "${SUBMIT_DIR}/../scripts" ]]; then
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "[ERROR] Could not resolve project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}"
  exit 1
fi
cd "${PROJECT_ROOT}"

module purge
module load jupyter/ai/2025-05-23
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/hpc_venv_bootstrap.sh"
hpc_bootstrap_venv "${PROJECT_ROOT}" || exit 1
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/hpc_list_guard.sh"
hpc_warn_export_truncation
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/runs/runtime"

PRESET="${PRESET:-cifar10}"
case "${PRESET}" in
  cifar10)
    : "${DATASET:=CIFAR10}"
    : "${NUM_CLASSES:=10}"
    : "${IMAGE_SIZE:=32}"
    : "${TARGET_CHECKPOINT_DIR:=runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model}"
    # C1/C2 need the FlowPure-family checkpoint, C3 the Gaussian-source CNF;
    # see evaluate_adaptive_generators.sbatch for why they are not
    # interchangeable.
    : "${FLOW_CHECKPOINT:=runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt}"
    : "${LIKELIHOOD_FLOW_CHECKPOINT:=runs/flow_matching/cifar10_notebook/checkpoint_latest.pt}"
    : "${DATA_ROOT:=data/cifar10}"
    ;;
  mnist)
    : "${DATASET:=MNIST}"
    : "${NUM_CLASSES:=10}"
    : "${IMAGE_SIZE:=32}"
    : "${TARGET_CHECKPOINT_DIR:=runs/victim/mnist_lenet}"
    : "${FLOW_CHECKPOINT:=runs/flow_matching/mnist_cnf/checkpoint_latest.pt}"
    : "${LIKELIHOOD_FLOW_CHECKPOINT:=runs/flow_matching/mnist_cnf/checkpoint_latest.pt}"
    : "${DATA_ROOT:=data}"
    ;;
  *) echo "[ERROR] Unknown PRESET='${PRESET}'"; exit 1 ;;
esac

# Batch sizes span the serving regimes that matter: 1 is the latency-bound
# single-request case an API actually sees, 64 is what the evaluation used, 256
# shows how far throughput can be recovered by batching.
: "${BATCH_SIZES:=1,8,64,256}"
: "${SELECTIVE_FRACTIONS:=0.05,0.1,0.25}"
: "${EXTRA_LIKELIHOOD_STEPS:=0.1,0.2}"
: "${NUM_STEPS:=8}"
: "${LIKELIHOOD_STEP_SIZE:=0.05}"
: "${LIKELIHOOD_METHOD:=midpoint}"
: "${WARMUP:=3}"
: "${REPEATS:=10}"
: "${LATEX_BATCH_SIZE:=64}"
: "${OUTPUT:=runs/runtime/flowguard_runtime_${PRESET}_a100.json}"
: "${LATEX_OUT:=paper/tables/runtime_overhead.tex}"

echo "[INFO] preset=${PRESET} repeats=${REPEATS}"
hpc_echo_list BATCH_SIZES SELECTIVE_FRACTIONS EXTRA_LIKELIHOOD_STEPS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PYTHON}" scripts/benchmark_flowguard_runtime.py \
  --dataset "${DATASET}" \
  --num-classes "${NUM_CLASSES}" \
  --image-size "${IMAGE_SIZE}" \
  --data-root "${DATA_ROOT}" \
  --target-checkpoint-dir "${TARGET_CHECKPOINT_DIR}" \
  --flow-checkpoint "${FLOW_CHECKPOINT}" \
  --likelihood-flow-checkpoint "${LIKELIHOOD_FLOW_CHECKPOINT}" \
  --device cuda \
  --batch-sizes "${BATCH_SIZES}" \
  --selective-fractions "${SELECTIVE_FRACTIONS}" \
  --extra-likelihood-steps "${EXTRA_LIKELIHOOD_STEPS}" \
  --num-steps "${NUM_STEPS}" \
  --likelihood-step-size "${LIKELIHOOD_STEP_SIZE}" \
  --likelihood-method "${LIKELIHOOD_METHOD}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --output "${OUTPUT}" \
  --latex-out "${LATEX_OUT}" \
  --latex-batch-size "${LATEX_BATCH_SIZE}"

echo "[INFO] wrote ${OUTPUT} and ${LATEX_OUT}"
