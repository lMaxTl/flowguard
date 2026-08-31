#!/bin/bash
#SBATCH --job-name=fgpp-active
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/fgpp_active_%A_%a.log
#SBATCH --error=logs/fgpp_active_%A_%a.err

# End-to-end extraction against an ENFORCING FlowGuard++ deployment.
#
# Every FlowGuard++ number published so far is an audit measurement, and the
# high-budget filtered experiment enforces FlowPure, not FlowGuard++. This job
# closes that gap: the composite runs with a fused risk threshold calibrated on
# benign traffic and suppresses flagged queries per request, so the substitute
# only learns from released answers.
#
# What it reports per attack: accepted-query rate, benign rejection rate
# measured on a held-out benign stream through the same enforcing detector,
# substitute accuracy and fidelity, and detection AUROC/F1 on the same stream.
#
# Usage:
#   # The three attacks the reviewer asked for, one array task each:
#   sbatch scripts/evaluate_active_flowguard_extraction.sbatch
#
#   # One million queries is enough evidence; the 100M budget is not needed here:
#   sbatch --export=ALL,QUERY_BUDGET=1000000 scripts/evaluate_active_flowguard_extraction.sbatch
#
#   # Smoke first (minutes):
#   sbatch --export=ALL,MODE=smoke --time=01:00:00 scripts/evaluate_active_flowguard_extraction.sbatch
#
#   # Distributed variant, to pair with the Sybil tables:
#   sbatch --export=ALL,SYBIL_IDENTITIES=1250 scripts/evaluate_active_flowguard_extraction.sbatch
#
# NOTE: exactly one enforcing detector runs here. Two enforcing detectors would
# each change the stream the other sees, so their results would describe
# different experiments.

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
mkdir -p "${PROJECT_ROOT}/logs"

PRESET="${PRESET:-cifar10}"
case "${PRESET}" in
  cifar10)
    : "${DATASET:=CIFAR10}"
    : "${TARGET_ARCHITECTURE:=vgg16_bn}"
    : "${SUBSTITUTE_ARCHITECTURE:=vgg16_bn}"
    : "${TARGET_CHECKPOINT_DIR:=runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model}"
    # Two CNFs, because the components have mutually exclusive requirements:
    #   FLOW_CHECKPOINT (velocity, C1/C2) must be FlowPure-family
    #     (x_0 = adversarial, x_1 = clean) -- that flow's velocity IS the
    #     paper's detection signal. A Gaussian CNF's velocity does not separate
    #     adversarial from benign.
    #   LIKELIHOOD_FLOW_CHECKPOINT (C3 typicality, flow_matching) must be
    #     Gaussian-source, since log p(x) is computed against a standard-Gaussian
    #     p_0. Under a purification flow it does not measure benign density.
    # Using one checkpoint for both silently invalidates whichever component it
    # does not suit, and that is not visible in the results.
    : "${FLOW_CHECKPOINT:=runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt}"
    : "${LIKELIHOOD_FLOW_CHECKPOINT:=runs/flow_matching/cifar10_notebook/checkpoint_latest.pt}"
    : "${DIFFUSION_MODEL_ID:=google/ddpm-cifar10-32}"
    : "${ADAPTIVE_ATTACKER_POOL:=CIFAR100}"
    : "${CLEAN_TRANSFER_DATASET:=CIFAR100}"
    ;;
  mnist)
    : "${DATASET:=MNIST}"
    : "${TARGET_ARCHITECTURE:=lenet}"
    : "${SUBSTITUTE_ARCHITECTURE:=lenet}"
    : "${TARGET_CHECKPOINT_DIR:=runs/victim/mnist_lenet}"
    # MNIST has only the Gaussian-source CNF: no FlowPure^PGD checkpoint was
    # trained for it, so C1 there is a velocity score under a generative flow,
    # not the paper's purification-flow signal. Noted rather than hidden.
    : "${FLOW_CHECKPOINT:=runs/flow_matching/mnist_cnf/checkpoint_latest.pt}"
    : "${LIKELIHOOD_FLOW_CHECKPOINT:=runs/flow_matching/mnist_cnf/checkpoint_latest.pt}"
    : "${DIFFUSION_MODEL_ID:=1aurent/ddpm-mnist}"
    : "${ADAPTIVE_ATTACKER_POOL:=EMNISTLetters}"
    : "${CLEAN_TRANSFER_DATASET:=FashionMNIST}"
    ;;
  *) echo "[ERROR] Unknown PRESET='${PRESET}'"; exit 1 ;;
esac

MODE="${MODE:-full}"
case "${MODE}" in
  full)
    : "${QUERY_BUDGET:=1000000}"
    : "${BENIGN_QUERY_BUDGET:=50000}"
    # Benign calibration batch size. The benign stream is a transfer-set recipe,
    # so it uses ATTACK_BATCH_SIZE -- which defaults to 1. With a likelihood-ODE
    # detector in the mix that means one ODE solve per single image: 50k queries
    # took ~18h and consumed the whole allocation before the attack started.
    : "${ATTACK_BATCH_SIZE:=64}"
    : "${EXTRACTION_ATTACK_BATCH_SIZE:=64}"
    : "${TRAINING_BATCH_SIZE:=128}"
    : "${EPOCHS:=20}"
    : "${ATTACK_ARTIFACT_SAMPLE_SIZE:=50000}"
    : "${LABEL_HISTOGRAM_SAMPLES:=10000}"
    : "${ADAPTIVE_CALIBRATION_SAMPLES:=8192}"
    : "${ADAPTIVE_DISTILL_EPOCHS:=150}"
    : "${DISGUIDE_BYPASS_EPOCH_ITRS:=100}"
    : "${QUERY_SNAPSHOT_MILESTONES:=1,1000,10000,100000,1000000,2000000}"
    : "${PLATEAU_MIN_QUERIES:=400000}"
    : "${PLATEAU_PATIENCE_QUERIES:=300000}"
    : "${PLATEAU_SMOOTHING_WINDOW:=5}"
    : "${ADAPTIVE_DISTILL_MIN_CORRELATION:=0.90}"
    # Snapshot attack state so a task killed at the wall clock resumes instead
    # of restarting. Measured throughput is ~11-12 queries/s, so a 2M-query cell
    # needs ~46-50h and cannot finish in one 24h allocation: without this, seven
    # of twelve tasks were killed and lost all their work.
    : "${ATTACK_CHECKPOINT_EVERY_QUERIES:=100000}"
    ;;
  smoke)
    : "${QUERY_BUDGET:=12800}"
    : "${BENIGN_QUERY_BUDGET:=512}"
    : "${ATTACK_BATCH_SIZE:=32}"
    : "${EXTRACTION_ATTACK_BATCH_SIZE:=32}"
    : "${TRAINING_BATCH_SIZE:=64}"
    : "${EPOCHS:=1}"
    : "${ATTACK_ARTIFACT_SAMPLE_SIZE:=2000}"
    : "${LABEL_HISTOGRAM_SAMPLES:=512}"
    : "${ADAPTIVE_CALIBRATION_SAMPLES:=256}"
    : "${ADAPTIVE_DISTILL_EPOCHS:=10}"
    : "${DISGUIDE_BYPASS_EPOCH_ITRS:=25}"
    : "${QUERY_SNAPSHOT_MILESTONES:=1,1000,10000}"
    : "${PLATEAU_MIN_QUERIES:=0}"
    : "${PLATEAU_PATIENCE_QUERIES:=0}"
    : "${PLATEAU_SMOOTHING_WINDOW:=1}"
    : "${ADAPTIVE_DISTILL_MIN_CORRELATION:=0}"
    : "${ATTACK_CHECKPOINT_EVERY_QUERIES:=0}"
    ;;
  *) echo "[ERROR] Unknown MODE='${MODE}'"; exit 1 ;;
esac

SYBIL_IDENTITIES="${SYBIL_IDENTITIES:-1}"
GENERATOR="${GENERATOR:-latent}"
case "${GENERATOR}" in procedural|latent|conv) ;; *)
  echo "[ERROR] GENERATOR must be procedural, latent or conv (got '${GENERATOR}')."; exit 1 ;;
esac

ATTACK_LIST="${ATTACK_LIST:-flowguard_d1_latent_manifold,flowguard_d5_adaptive_composite,flowguard_d6_adaptive_stateful}"
IFS=',' read -r -a ATTACK_ARRAY <<< "${ATTACK_LIST}"
hpc_guard_array_range ATTACK_LIST "${#ATTACK_ARRAY[@]}" || exit 1
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID >= ${#ATTACK_ARRAY[@]} )); then
    echo "[ERROR] Array index ${SLURM_ARRAY_TASK_ID} exceeds ATTACK_LIST size ${#ATTACK_ARRAY[@]}."
    hpc_export_hint ATTACK_LIST
    exit 1
  fi
  ATTACKS="${ATTACK_ARRAY[${SLURM_ARRAY_TASK_ID}]}"
  TASK_SUFFIX="${ATTACKS}"
else
  ATTACKS="${ATTACK_LIST}"
  TASK_SUFFIX="all"
fi

# The enforcing driver deploys the composite only; the defense list exists to
# keep the shared recipe machinery happy.
DEFENSES="${DEFENSES:-flowguard_composite}"
RUN_LABEL="${RUN_LABEL:-active-fgpp-${GENERATOR}}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/${RUN_LABEL}}"
OUTPUT_ROOT="${OUTPUT_BASE}/${TASK_SUFFIX}"
DEVICE="${DEVICE:-cuda}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-8}"
ADAPTIVE_LIKELIHOOD_STEPS="${ADAPTIVE_LIKELIHOOD_STEPS:-4}"
ADAPTIVE_DISTILL_SURROGATE="${ADAPTIVE_DISTILL_SURROGATE:-1}"
QUERY_SNAPSHOT_IMAGES="${QUERY_SNAPSHOT_IMAGES:-9}"
PLATEAU_MIN_DELTA="${PLATEAU_MIN_DELTA:-1.0}"
TARGET_FPR="${TARGET_FPR:-0.05}"
A3_SURROGATE_CHECKPOINT="${A3_SURROGATE_CHECKPOINT:-${FLOW_CHECKPOINT}}"
# Attacker's C3 surrogate mirrors the defender's likelihood CNF, so D4-D6
# optimize the same statistic the detector computes.
A3_LIKELIHOOD_SURROGATE_CHECKPOINT="${A3_LIKELIHOOD_SURROGATE_CHECKPOINT:-${LIKELIHOOD_FLOW_CHECKPOINT}}"
CALIBRATION_CACHE="${CALIBRATION_CACHE:-}"

for var in OUTPUT_ROOT TARGET_CHECKPOINT_DIR FLOW_CHECKPOINT LIKELIHOOD_FLOW_CHECKPOINT A3_SURROGATE_CHECKPOINT A3_LIKELIHOOD_SURROGATE_CHECKPOINT; do
  value="${!var}"
  [[ "${value}" != /* ]] && printf -v "${var}" '%s' "${PROJECT_ROOT}/${value}"
done
mkdir -p "${OUTPUT_ROOT}"

[[ -d "${TARGET_CHECKPOINT_DIR}" ]] || { echo "[ERROR] Missing victim: ${TARGET_CHECKPOINT_DIR}"; exit 1; }
[[ -f "${FLOW_CHECKPOINT}" ]] || { echo "[ERROR] Missing velocity CNF: ${FLOW_CHECKPOINT}"; exit 1; }
[[ -f "${LIKELIHOOD_FLOW_CHECKPOINT}" ]] || { echo "[ERROR] Missing likelihood CNF: ${LIKELIHOOD_FLOW_CHECKPOINT}"; exit 1; }

if [[ -n "${CALIBRATION_CACHE}" ]]; then
  [[ "${CALIBRATION_CACHE}" != /* ]] && CALIBRATION_CACHE="${PROJECT_ROOT}/${CALIBRATION_CACHE}"
  if [[ -f "${CALIBRATION_CACHE}" ]]; then
    echo "[INFO] Reusing attacker calibration from ${CALIBRATION_CACHE}"
    cp "${CALIBRATION_CACHE}" "${OUTPUT_ROOT}/attacker_surrogate_calibration.json"
  fi
fi

echo "[INFO] Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "[INFO] preset=${PRESET} mode=${MODE} generator=${GENERATOR} attack=${ATTACKS}"
echo "[INFO] enforcing detector: ${DEFENSES} (per-query suppression)"
echo "[INFO] sybil identities: ${SYBIL_IDENTITIES}"
hpc_echo_list ATTACK_LIST DEFENSES
echo "[INFO] velocity CNF (C1/C2): ${FLOW_CHECKPOINT}"
echo "[INFO] likelihood CNF (C3):  ${LIKELIHOOD_FLOW_CHECKPOINT}"
echo "[INFO] budget=${QUERY_BUDGET} output=${OUTPUT_ROOT}"
nvidia-smi || true

CMD=(
  "${PYTHON}" scripts/evaluate_active_flowguard_extraction.py
  --run-label "${RUN_LABEL}"
  --output-root "${OUTPUT_ROOT}"
  --attacks "${ATTACKS}"
  --defenses "${DEFENSES}"
  --sybil-identities "${SYBIL_IDENTITIES}"
  --dataset "${DATASET}"
  --query-dataset "${DATASET}"
  --clean-transfer-dataset "${CLEAN_TRANSFER_DATASET}"
  --target-architecture "${TARGET_ARCHITECTURE}"
  --substitute-architecture "${SUBSTITUTE_ARCHITECTURE}"
  --target-checkpoint-dir "${TARGET_CHECKPOINT_DIR}"
  --flow-checkpoint "${FLOW_CHECKPOINT}"
  --likelihood-flow-checkpoint "${LIKELIHOOD_FLOW_CHECKPOINT}"
  --a3-surrogate-checkpoint "${A3_SURROGATE_CHECKPOINT}"
  --a3-likelihood-surrogate-checkpoint "${A3_LIKELIHOOD_SURROGATE_CHECKPOINT}"
  --device "${DEVICE}"
  --query-budget "${QUERY_BUDGET}"
  --benign-query-budget "${BENIGN_QUERY_BUDGET}"
  --target-fpr "${TARGET_FPR}"
  --attack-batch-size "${ATTACK_BATCH_SIZE}"
  --extraction-attack-batch-size "${EXTRACTION_ATTACK_BATCH_SIZE}"
  --training-batch-size "${TRAINING_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --attack-artifact-sample-size "${ATTACK_ARTIFACT_SAMPLE_SIZE}"
  --label-histogram-samples "${LABEL_HISTOGRAM_SAMPLES}"
  --disguide-bypass-epoch-itrs "${DISGUIDE_BYPASS_EPOCH_ITRS}"
  --diffusion-model-id "${DIFFUSION_MODEL_ID}"
  --diffusion-steps "${DIFFUSION_STEPS}"
  --adaptive-generator "${GENERATOR}"
  --adaptive-attacker-pool "${ADAPTIVE_ATTACKER_POOL}"
  --adaptive-calibration-samples "${ADAPTIVE_CALIBRATION_SAMPLES}"
  --adaptive-likelihood-steps "${ADAPTIVE_LIKELIHOOD_STEPS}"
  --adaptive-distill-epochs "${ADAPTIVE_DISTILL_EPOCHS}"
  --adaptive-distill-min-correlation "${ADAPTIVE_DISTILL_MIN_CORRELATION}"
  --query-snapshot-milestones "${QUERY_SNAPSHOT_MILESTONES}"
  --query-snapshot-images "${QUERY_SNAPSHOT_IMAGES}"
  --attack-plateau-min-queries "${PLATEAU_MIN_QUERIES}"
  --attack-plateau-patience-queries "${PLATEAU_PATIENCE_QUERIES}"
  --attack-plateau-min-delta "${PLATEAU_MIN_DELTA}"
  --attack-plateau-smoothing-window "${PLATEAU_SMOOTHING_WINDOW}"
  --attack-checkpoint-every-queries "${ATTACK_CHECKPOINT_EVERY_QUERIES}"
)
[[ "${ADAPTIVE_DISTILL_SURROGATE}" == "1" ]] && CMD+=(--adaptive-distill-surrogate)
[[ "${DOWNLOAD:-0}" == "1" ]] && CMD+=(--download)
[[ "${VERBOSE:-0}" == "1" ]] && CMD+=(--verbose)

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

if [[ -n "${CALIBRATION_CACHE}" && ! -f "${CALIBRATION_CACHE}" ]]; then
  if [[ -f "${OUTPUT_ROOT}/attacker_surrogate_calibration.json" ]]; then
    mkdir -p "$(dirname "${CALIBRATION_CACHE}")"
    cp "${OUTPUT_ROOT}/attacker_surrogate_calibration.json" "${CALIBRATION_CACHE}"
  fi
fi

echo "[INFO] Done: ${OUTPUT_ROOT}/active_flowguard_${RUN_LABEL}_summary.json"
