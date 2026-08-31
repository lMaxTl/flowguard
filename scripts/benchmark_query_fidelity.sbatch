#!/bin/bash
#SBATCH --job-name=qfid
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/qfid_%A_%a.log
#SBATCH --error=logs/qfid_%A_%a.err

# Queries-to-fidelity benchmark for the FlowGuard++-adaptive attackers (D1-D6).
#
# Answers "how many victim queries buy how much clone fidelity, and is the
# attack detected while doing it" in a single pass per attack: the DisGUIDE loop
# evaluates the substitute every epoch, so one run yields the whole curve and a
# budget sweep is unnecessary.
#
# Submitted as a job ARRAY, one attack per task, because the matrix harness runs
# cells sequentially and a single high-budget CIFAR-10 cell can approach the
# wall clock on its own. Tasks are independent; the per-attack summaries and
# snapshot directories are merged afterwards (see MERGE below).
#
# Usage:
#   # CIFAR-10, default 2M queries per attack, all six attacks in parallel:
#   sbatch scripts/benchmark_query_fidelity.sbatch
#
#   # Smoke test the wiring first (minutes, not hours):
#   sbatch --array=0-5 --time=01:00:00 --partition=dev_accelerated-h100 \
#          --export=ALL,MODE=smoke scripts/benchmark_query_fidelity.sbatch
#
#   # Reproduce the MNIST benchmark:
#   sbatch --export=ALL,PRESET=mnist scripts/benchmark_query_fidelity.sbatch
#
#   # Put D3-D6 on D1's steered-diffusion generator instead of D2's procedural
#   # one. Use where the procedural parameterization cannot reach useful
#   # fidelity in the domain and would otherwise cap those rows for reasons
#   # unrelated to the adaptive penalties:
#   sbatch --export=ALL,ADAPTIVE_GENERATOR=latent,RUN_LABEL=qfid-latent \
#          scripts/benchmark_query_fidelity.sbatch
#
#   # A single attack (array index into ATTACK_LIST):
#   sbatch --array=0 scripts/benchmark_query_fidelity.sbatch
#
# MERGE (after the array finishes; runs on a login node in seconds):
#   python scripts/report_query_fidelity_benchmark.py \
#       $(for d in runs/qfid/*/; do echo --summary $d/attack_defense_matrix_*_summary.json; done) \
#       --targets 50,70,90,95 --output runs/qfid/queries_to_fidelity.md
#   python scripts/plot_query_evolution.py \
#       $(for d in runs/qfid/*/; do echo --snapshot-dir $d/query_snapshots; done) \
#       --output runs/qfid/query_evolution --formats png,pdf --per-attack
#
# ATTACKER CALIBRATION: D3-D6 each need benign (mu, sigma) per component from
# the attacker's own pool, which costs a likelihood pass over that pool. Array
# tasks do not share an output root, so each would recompute it. Point
# CALIBRATION_CACHE at a shared file to compute it once and reuse:
#   sbatch --array=2 --export=ALL,CALIBRATION_CACHE=runs/qfid/calibration.json ...   # seeds it
#   sbatch --array=3-5 --export=ALL,CALIBRATION_CACHE=runs/qfid/calibration.json ... # reuses it

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

mkdir -p "${PROJECT_ROOT}/logs"

# ---------------------------------------------------------------------------
# Dataset preset. Selects victim, CNF, generator prior and attacker pool
# together, since mixing them across datasets silently produces nonsense
# (e.g. a 3-channel CIFAR DDPM against a 1-channel MNIST victim).
# ---------------------------------------------------------------------------
PRESET="${PRESET:-cifar10}"
case "${PRESET}" in
  cifar10)
    : "${DATASET:=CIFAR10}"
    : "${TARGET_ARCHITECTURE:=vgg16_bn}"
    : "${SUBSTITUTE_ARCHITECTURE:=vgg16_bn}"
    : "${TARGET_CHECKPOINT_DIR:=runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model}"
    : "${FLOW_CHECKPOINT:=runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt}"
    : "${DIFFUSION_MODEL_ID:=google/ddpm-cifar10-32}"
    # Related-domain pool the attacker owns. Measured 0.09 sigma from the
    # defender's benign likelihood band, i.e. an almost ideal proxy.
    : "${ADAPTIVE_ATTACKER_POOL:=CIFAR100}"
    : "${CLEAN_TRANSFER_DATASET:=CIFAR100}"
    ;;
  mnist)
    : "${DATASET:=MNIST}"
    : "${TARGET_ARCHITECTURE:=lenet}"
    : "${SUBSTITUTE_ARCHITECTURE:=lenet}"
    : "${TARGET_CHECKPOINT_DIR:=runs/victim/mnist_lenet}"
    : "${FLOW_CHECKPOINT:=runs/flow_matching/mnist_cnf/checkpoint_latest.pt}"
    : "${DIFFUSION_MODEL_ID:=1aurent/ddpm-mnist}"
    # Closest MNIST-family pool (same acquisition process, disjoint labels).
    # Still ~5 sigma from the defender's band -- far worse than CIFAR-100 is for
    # CIFAR-10 -- so the C3-targeting rows (D4-D6) aim at a displaced target.
    : "${ADAPTIVE_ATTACKER_POOL:=EMNISTLetters}"
    : "${CLEAN_TRANSFER_DATASET:=FashionMNIST}"
    ;;
  *)
    echo "[ERROR] Unknown PRESET='${PRESET}' (expected 'cifar10' or 'mnist')."
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Size profile.
# ---------------------------------------------------------------------------
MODE="${MODE:-full}"
case "${MODE}" in
  full)
    # 2M is a compromise: the published CIFAR-10 data-free numbers use 20M, but
    # the curve's knee and the detection verdict both land far earlier, and
    # early stopping ends a run once fidelity stops moving.
    : "${QUERY_BUDGET:=2000000}"
    : "${BENIGN_QUERY_BUDGET:=50000}"
    : "${EXTRACTION_ATTACK_BATCH_SIZE:=64}"
    : "${TRAINING_BATCH_SIZE:=128}"
    : "${EPOCHS:=20}"
    : "${ATTACK_ARTIFACT_SAMPLE_SIZE:=50000}"
    : "${LABEL_HISTOGRAM_SAMPLES:=10000}"
    # CIFAR-10 needs far more than MNIST to fit the score surrogate: at 256
    # samples / 10 epochs the components correlated r=0.25-0.40 with the exact
    # CNF score (MNIST reached 0.98-1.00 at 2048/30). Verified via
    # ADAPTIVE_DISTILL_MIN_CORRELATION below, which aborts on a bad fit.
    : "${ADAPTIVE_CALIBRATION_SAMPLES:=8192}"
    : "${ADAPTIVE_DISTILL_EPOCHS:=150}"
    : "${DISGUIDE_BYPASS_EPOCH_ITRS:=100}"
    : "${QUERY_SNAPSHOT_MILESTONES:=1,1000,10000,100000,1000000,2000000}"
    # Never stop before 20% of budget; stop after 15% without a 1pp gain in the
    # SMOOTHED curve. The window matters: CIFAR-10 fidelity swings +-2-3pp
    # epoch to epoch while gaining ~1pp per 100k queries, and the unsmoothed
    # rule truncated four of six attacks mid-progress in the first run.
    : "${PLATEAU_MIN_QUERIES:=400000}"
    : "${PLATEAU_PATIENCE_QUERIES:=300000}"
    : "${PLATEAU_SMOOTHING_WINDOW:=5}"
    : "${ATTACK_CHECKPOINT_EVERY_QUERIES:=250000}"
    : "${ADAPTIVE_DISTILL_MIN_CORRELATION:=0.90}"
    ;;
  smoke)
    : "${QUERY_BUDGET:=12800}"
    : "${BENIGN_QUERY_BUDGET:=512}"
    : "${EXTRACTION_ATTACK_BATCH_SIZE:=32}"
    : "${TRAINING_BATCH_SIZE:=64}"
    : "${EPOCHS:=1}"
    : "${ATTACK_ARTIFACT_SAMPLE_SIZE:=2000}"
    : "${LABEL_HISTOGRAM_SAMPLES:=512}"
    : "${ADAPTIVE_CALIBRATION_SAMPLES:=256}"
    : "${DISGUIDE_BYPASS_EPOCH_ITRS:=25}"
    : "${QUERY_SNAPSHOT_MILESTONES:=1,1000,10000}"
    : "${PLATEAU_MIN_QUERIES:=0}"
    : "${PLATEAU_PATIENCE_QUERIES:=0}"
    : "${PLATEAU_SMOOTHING_WINDOW:=1}"
    : "${ATTACK_CHECKPOINT_EVERY_QUERIES:=0}"
    # Smoke mode deliberately under-fits the surrogate; gating here would abort
    # a run whose only job is to prove the wiring works.
    : "${ADAPTIVE_DISTILL_MIN_CORRELATION:=0}"
    ;;
  *)
    echo "[ERROR] Unknown MODE='${MODE}' (expected 'full' or 'smoke')."
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# One attack per array task.
# ---------------------------------------------------------------------------
ATTACK_LIST="${ATTACK_LIST:-flowguard_d1_latent_manifold,flowguard_d2_procedural_natural,flowguard_d3_adaptive_c1c2,flowguard_d4_adaptive_c1c3,flowguard_d5_adaptive_composite,flowguard_d6_adaptive_stateful}"
IFS=',' read -r -a ATTACK_ARRAY <<< "${ATTACK_LIST}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID >= ${#ATTACK_ARRAY[@]} )); then
    echo "[ERROR] Array index ${SLURM_ARRAY_TASK_ID} exceeds ATTACK_LIST size ${#ATTACK_ARRAY[@]}."
    exit 1
  fi
  ATTACKS="${ATTACK_ARRAY[${SLURM_ARRAY_TASK_ID}]}"
  TASK_SUFFIX="${ATTACKS}"
else
  # Not an array job: run every attack sequentially in this one task.
  ATTACKS="${ATTACK_LIST}"
  TASK_SUFFIX="all"
fi

RUN_LABEL="${RUN_LABEL:-qfid-${PRESET}}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/${RUN_LABEL}}"
OUTPUT_ROOT="${OUTPUT_BASE}/${TASK_SUFFIX}"

DEFENSES="${DEFENSES:-flowpure}"
QUERY_DATASET="${QUERY_DATASET:-${DATASET}}"
DEVICE="${DEVICE:-cuda}"
# Audit-only measures the extraction ceiling: defenses score every query but
# never block, so substitute quality reflects the generator constraint alone.
# Set 0 to make the defense actually reject and measure the enforced setting.
AUDIT_ONLY="${AUDIT_ONLY:-1}"
ADAPTIVE_GENERATOR="${ADAPTIVE_GENERATOR:-procedural}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-8}"
ADAPTIVE_LIKELIHOOD_STEPS="${ADAPTIVE_LIKELIHOOD_STEPS:-4}"
# The exact likelihood gradient costs a double-backward per ODE step, which is
# not affordable at these budgets. Distillation gives composite gradients at one
# forward/backward; check attacker_distilled_score.report.json afterwards --
# component correlations below ~0.9 mean the attack optimized noise.
ADAPTIVE_DISTILL_SURROGATE="${ADAPTIVE_DISTILL_SURROGATE:-1}"
ADAPTIVE_DISTILL_EPOCHS="${ADAPTIVE_DISTILL_EPOCHS:-30}"
# Abort before spending the budget if the distilled surrogate does not track the
# exact CNF score. Only affects D3-D6 (D1/D2 never build one). Set 0 to disable.
ADAPTIVE_DISTILL_MIN_CORRELATION="${ADAPTIVE_DISTILL_MIN_CORRELATION}"
QUERY_SNAPSHOT_IMAGES="${QUERY_SNAPSHOT_IMAGES:-9}"
PLATEAU_MIN_DELTA="${PLATEAU_MIN_DELTA:-1.0}"
SYBIL_IDENTITIES="${SYBIL_IDENTITIES:-1}"
TARGET_FPR="${TARGET_FPR:-0.05}"
A3_SURROGATE_CHECKPOINT="${A3_SURROGATE_CHECKPOINT:-${FLOW_CHECKPOINT}}"
CALIBRATION_CACHE="${CALIBRATION_CACHE:-}"

for var in OUTPUT_ROOT TARGET_CHECKPOINT_DIR FLOW_CHECKPOINT A3_SURROGATE_CHECKPOINT; do
  value="${!var}"
  if [[ "${value}" != /* ]]; then
    printf -v "${var}" '%s' "${PROJECT_ROOT}/${value}"
  fi
done

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -d "${TARGET_CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Missing victim checkpoint dir: ${TARGET_CHECKPOINT_DIR}"
  exit 1
fi
if [[ ! -f "${FLOW_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing CNF checkpoint: ${FLOW_CHECKPOINT}"
  exit 1
fi

# Reuse a shared attacker calibration when one is provided, so parallel array
# tasks do not each repeat the likelihood pass over the attacker's pool.
if [[ -n "${CALIBRATION_CACHE}" ]]; then
  if [[ "${CALIBRATION_CACHE}" != /* ]]; then
    CALIBRATION_CACHE="${PROJECT_ROOT}/${CALIBRATION_CACHE}"
  fi
  if [[ -f "${CALIBRATION_CACHE}" ]]; then
    echo "[INFO] Reusing attacker calibration from ${CALIBRATION_CACHE}"
    cp "${CALIBRATION_CACHE}" "${OUTPUT_ROOT}/attacker_surrogate_calibration.json"
  else
    echo "[INFO] No calibration cache yet; this task will compute and publish one."
  fi
fi

echo "[INFO] Job: ${SLURM_JOB_ID:-local} array-task ${SLURM_ARRAY_TASK_ID:-none}"
echo "[INFO] Preset: ${PRESET} | mode: ${MODE} | attack(s): ${ATTACKS}"
echo "[INFO] Generator for D3-D6: ${ADAPTIVE_GENERATOR}"
echo "[INFO] Budget: ${QUERY_BUDGET} | plateau: stop after ${PLATEAU_PATIENCE_QUERIES} queries without ${PLATEAU_MIN_DELTA}pp (min ${PLATEAU_MIN_QUERIES})"
echo "[INFO] Output: ${OUTPUT_ROOT}"
nvidia-smi || true

CMD=(
  "${PYTHON}" scripts/evaluate_attack_defense_matrix_smoke.py
  --run-label "${RUN_LABEL}"
  --output-root "${OUTPUT_ROOT}"
  --attacks "${ATTACKS}"
  --defenses "${DEFENSES}"
  --dataset "${DATASET}"
  --query-dataset "${QUERY_DATASET}"
  --clean-transfer-dataset "${CLEAN_TRANSFER_DATASET}"
  --target-architecture "${TARGET_ARCHITECTURE}"
  --substitute-architecture "${SUBSTITUTE_ARCHITECTURE}"
  --target-checkpoint-dir "${TARGET_CHECKPOINT_DIR}"
  --flow-checkpoint "${FLOW_CHECKPOINT}"
  --a3-surrogate-checkpoint "${A3_SURROGATE_CHECKPOINT}"
  --device "${DEVICE}"
  --query-budget "${QUERY_BUDGET}"
  --benign-query-budget "${BENIGN_QUERY_BUDGET}"
  --target-fpr "${TARGET_FPR}"
  --extraction-attack-batch-size "${EXTRACTION_ATTACK_BATCH_SIZE}"
  --training-batch-size "${TRAINING_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --sybil-identities "${SYBIL_IDENTITIES}"
  --attack-artifact-sample-size "${ATTACK_ARTIFACT_SAMPLE_SIZE}"
  --label-histogram-samples "${LABEL_HISTOGRAM_SAMPLES}"
  --disguide-bypass-epoch-itrs "${DISGUIDE_BYPASS_EPOCH_ITRS}"
  --diffusion-model-id "${DIFFUSION_MODEL_ID}"
  --diffusion-steps "${DIFFUSION_STEPS}"
  --adaptive-generator "${ADAPTIVE_GENERATOR}"
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

if [[ "${ADAPTIVE_DISTILL_SURROGATE}" == "1" ]]; then
  CMD+=(--adaptive-distill-surrogate)
fi
if [[ "${AUDIT_ONLY}" == "0" ]]; then
  CMD+=(--no-audit-only)
fi
if [[ "${DOWNLOAD:-0}" == "1" ]]; then
  CMD+=(--download)
fi
if [[ "${VERBOSE:-0}" == "1" ]]; then
  CMD+=(--verbose)
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

# Publish the calibration for sibling array tasks that have not started yet.
if [[ -n "${CALIBRATION_CACHE}" && ! -f "${CALIBRATION_CACHE}" ]]; then
  if [[ -f "${OUTPUT_ROOT}/attacker_surrogate_calibration.json" ]]; then
    mkdir -p "$(dirname "${CALIBRATION_CACHE}")"
    cp "${OUTPUT_ROOT}/attacker_surrogate_calibration.json" "${CALIBRATION_CACHE}"
    echo "[INFO] Published attacker calibration to ${CALIBRATION_CACHE}"
  fi
fi

echo "[INFO] Done. Summary: ${OUTPUT_ROOT}/attack_defense_matrix_${RUN_LABEL}_summary.json"
echo "[INFO] Merge all tasks with the commands in this script's header."
