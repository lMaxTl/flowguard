#!/bin/bash
#SBATCH --job-name=ad-combined
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/attack_defense_combined_%j.log
#SBATCH --error=logs/attack_defense_combined_%j.err

# Merged evaluation harness.
#
# This single script replaces the four previous sbatch files:
#   - evaluate_attack_defense_matrix_full.sbatch   (extraction + adaptive A* attacks)
#   - evaluate_attack_defense_matrix_smoke.sbatch
#   - evaluate_flowpure_bypass_research_full.sbatch  (data-free D1-D4 bypass attacks)
#   - evaluate_flowpure_bypass_research_smoke.sbatch
#
# It runs the union of both attack families against the union of all defenses in
# one matrix, and is profiled by MODE.
#
# Usage:
#   # Full run (heavy, default profile + sbatch resources above):
#   sbatch scripts/evaluate_attack_defense_combined.sbatch
#
#   # Smoke run (small budgets; also drop to the dev partition / short wall-clock,
#   # since #SBATCH directives cannot be set from inside the body):
#   sbatch --partition=dev_accelerated-h100 --mem=64G --time=02:00:00 \
#          --job-name=ad-combined-smoke \
#          --export=ALL,MODE=smoke scripts/evaluate_attack_defense_combined.sbatch
#
#   # Any default below can be overridden, e.g. the Sybil sweep:
#   sbatch --export=ALL,SYBIL_IDENTITIES=32 scripts/evaluate_attack_defense_combined.sbatch
#
#   # Resume after a crash (summary is flushed after every matrix cell):
#   #   echo '{"start_defense":"fdinet","start_attack":"maze"}' > runs/resume.json
#   sbatch --export=ALL,RESUME_FILE=runs/resume.json \
#          scripts/evaluate_attack_defense_combined.sbatch
#
# If pip fails with a broken typing-inspection in .venv, either let the job
# recreate the venv (default) or on the login node once:
#   rm -rf .venv && python -m venv .venv && .venv/bin/pip install -e .

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

# Bootstrap .venv (skip pip when import works; repair/recreate on corruption).
# See scripts/hpc_venv_bootstrap.sh for FORCE_VENV_INSTALL, SKIP_VENV_INSTALL, etc.
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/hpc_venv_bootstrap.sh"
hpc_bootstrap_venv "${PROJECT_ROOT}" || exit 1
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/hpc_list_guard.sh"
hpc_warn_export_truncation

mkdir -p "${PROJECT_ROOT}/logs"

# ---------------------------------------------------------------------------
# Profile: full | smoke. Drives only the size-dependent defaults below; every
# individual variable can still be overridden via --export=ALL,VAR=value.
# ---------------------------------------------------------------------------
MODE="${MODE:-full}"
case "${MODE}" in
  full)
    : "${QUERY_BUDGET:=50000}"
    : "${BENIGN_QUERY_BUDGET:=50000}"
    : "${EXTRACTION_ATTACK_BATCH_SIZE:=32}"
    : "${TRAINING_BATCH_SIZE:=64}"
    : "${EVALUATION_BATCH_SIZE:=128}"
    # Substitute training epochs. Use a convergent value for real fidelity; this
    # multiplies per-cell wall-clock, so size --time accordingly.
    : "${EPOCHS:=50}"
    : "${SEED_SAMPLES:=64}"
    : "${ATTACK_ARTIFACT_SAMPLE_SIZE:=${QUERY_BUDGET}}"
    : "${LABEL_HISTOGRAM_SAMPLES:=10000}"
    : "${CEM_POPULATION_SIZE:=128}"
    : "${PROJECTION_STEPS:=10}"
    ;;
  smoke)
    : "${QUERY_BUDGET:=256}"
    : "${BENIGN_QUERY_BUDGET:=256}"
    : "${EXTRACTION_ATTACK_BATCH_SIZE:=4}"
    : "${TRAINING_BATCH_SIZE:=8}"
    : "${EVALUATION_BATCH_SIZE:=64}"
    : "${EPOCHS:=1}"
    : "${SEED_SAMPLES:=4}"
    : "${ATTACK_ARTIFACT_SAMPLE_SIZE:=32}"
    : "${LABEL_HISTOGRAM_SAMPLES:=512}"
    : "${CEM_POPULATION_SIZE:=16}"
    : "${PROJECTION_STEPS:=5}"
    ;;
  *)
    echo "[ERROR] Unknown MODE='${MODE}' (expected 'full' or 'smoke')."
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Mode-independent defaults (override any via --export=ALL,VAR=value).
# ---------------------------------------------------------------------------
RUN_LABEL="${RUN_LABEL:-combined-${MODE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/attack_defense_combined_${MODE}}"
# Optional JSON resume spec; see header comments. Relative paths resolve from PROJECT_ROOT.
RESUME_FILE="${RESUME_FILE:-}"

# Combined attack set: extraction/adaptive (A*) + data-free FlowPure bypass (D*).
ATTACKS="${ATTACKS:-prada,maze,disguide,transfer_naive,transfer_top1,transfer_s4l,transfer_smoothing,flowguard_a1_clean_transfer,flowguard_a2_projected_maze,flowguard_a3_flowblind,flowguard_adaptive_adaptive,flowguard_d1_latent_manifold,flowguard_d2_procedural_natural,flowguard_d3_projected_latent,flowguard_d4_rejection_oracle}"
# Union of all defenses (composite only matters for the D* attacks but is harmless elsewhere).
DEFENSES="${DEFENSES:-prada,flowpure,fdinet,flowguard_integral,flowguard_userlevel,flowguard_labelhist,flowguard_composite}"

DATASET="${DATASET:-CIFAR10}"
QUERY_DATASET="${QUERY_DATASET:-CIFAR10}"
CLEAN_TRANSFER_DATASET="${CLEAN_TRANSFER_DATASET:-CIFAR100}"
TARGET_CHECKPOINT_DIR="${TARGET_CHECKPOINT_DIR:-runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt}"
A3_SURROGATE_CHECKPOINT="${A3_SURROGATE_CHECKPOINT:-${FLOW_CHECKPOINT}}"
DIFFUSION_MODEL_ID="${DIFFUSION_MODEL_ID:-google/ddpm-cifar10-32}"
DIFFUSION_SCHEDULER="${DIFFUSION_SCHEDULER:-ddim}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
DIFFUSION_TRAIN_STEPS="${DIFFUSION_TRAIN_STEPS:-1}"
DIFFUSION_STEERING_SCALE="${DIFFUSION_STEERING_SCALE:-1.0}"
DIFFUSION_GUIDANCE_SCALE="${DIFFUSION_GUIDANCE_SCALE:-0.0}"
DIFFUSION_GUIDANCE_START_FRAC="${DIFFUSION_GUIDANCE_START_FRAC:-0.5}"
DISGUIDE_BYPASS_ENSEMBLE_SIZE="${DISGUIDE_BYPASS_ENSEMBLE_SIZE:-2}"
DISGUIDE_BYPASS_G_ITER="${DISGUIDE_BYPASS_G_ITER:-1}"
DISGUIDE_BYPASS_D_ITER="${DISGUIDE_BYPASS_D_ITER:-1}"
DISGUIDE_BYPASS_REP_ITER="${DISGUIDE_BYPASS_REP_ITER:-1}"
DISGUIDE_BYPASS_LAMBDA_DIV="${DISGUIDE_BYPASS_LAMBDA_DIV:-0.2}"
DISGUIDE_BYPASS_EPOCH_ITRS="${DISGUIDE_BYPASS_EPOCH_ITRS:-4}"
DISGUIDE_BYPASS_LR_STUDENT="${DISGUIDE_BYPASS_LR_STUDENT:-0.03}"
DISGUIDE_BYPASS_LR_GENERATOR="${DISGUIDE_BYPASS_LR_GENERATOR:-1e-4}"
DEVICE="${DEVICE:-cuda}"

TARGET_FPR="${TARGET_FPR:-0.05}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-1}"
# Sybil sweep: spread each attacker's queries across this many identities so
# stateful per-user defenses (KS window, label histogram, composite) lose
# evidence per identity. 1 = single-identity baseline.
SYBIL_IDENTITIES="${SYBIL_IDENTITIES:-1}"
# 1 = audit-only (score but do not block). 0 = enforcing mode (defenses reject at runtime).
AUDIT_ONLY="${AUDIT_ONLY:-1}"

# Data-free bypass knobs.
# D1/D2 use the DisGUIDE-style loop (ensemble + generator + surrogate flow penalty).
# D3/D4 remain CEM-based (projected latent / rejection oracle).
CEM_ELITE_FRACTION="${CEM_ELITE_FRACTION:-0.20}"
CEM_LATENT_DIM="${CEM_LATENT_DIM:-128}"
CEM_SELECTOR="${CEM_SELECTOR:-entropy}"
CEM_LAMBDA_DETECTOR="${CEM_LAMBDA_DETECTOR:-1.0}"
PROJECTION_NAME="${PROJECTION_NAME:-gaussian_blur}"
PROJECTION_STRENGTH="${PROJECTION_STRENGTH:-0.1}"
REJECTION_ORACLE_PROPOSAL="${REJECTION_ORACLE_PROPOSAL:-latent}"
PROCEDURAL_GRID_SIZE="${PROCEDURAL_GRID_SIZE:-4}"
PROCEDURAL_FOURIER_MODES="${PROCEDURAL_FOURIER_MODES:-3}"
PROCEDURAL_BLOBS="${PROCEDURAL_BLOBS:-4}"
PROCEDURAL_LAMBDA_TV="${PROCEDURAL_LAMBDA_TV:-0.01}"
PROCEDURAL_LAMBDA_FREQ="${PROCEDURAL_LAMBDA_FREQ:-0.01}"
PROCEDURAL_LAMBDA_COLOR="${PROCEDURAL_LAMBDA_COLOR:-0.01}"

# Defense detector knobs.
INTEGRAL_NUM_STEPS="${INTEGRAL_NUM_STEPS:-8}"
KS_THRESHOLD="${KS_THRESHOLD:-0.25}"
KS_MIN_WINDOW="${KS_MIN_WINDOW:-64}"
KS_MAX_WINDOW="${KS_MAX_WINDOW:-2048}"
LABELHIST_THRESHOLD="${LABELHIST_THRESHOLD:-0.05}"
LABELHIST_MIN_WINDOW="${LABELHIST_MIN_WINDOW:-128}"
LABELHIST_MAX_WINDOW="${LABELHIST_MAX_WINDOW:-4096}"
FLOWGUARDPP_DECISION_MODE="${FLOWGUARDPP_DECISION_MODE:-hybrid}"
FLOWGUARDPP_RESPONSE_POLICY="${FLOWGUARDPP_RESPONSE_POLICY:-reject}"
# 1 = also report a 'flowguard_composite_learned' column whose per-component
# weights are learned via cross-fitted logistic regression (oracle/upper bound
# vs the equal-weight composite default). Requires flowguard_composite in DEFENSES.
COMPOSITE_EVAL_LEARNED="${COMPOSITE_EVAL_LEARNED:-0}"
COMPOSITE_LEARNED_FOLDS="${COMPOSITE_LEARNED_FOLDS:-5}"
# 1 = also dump per-query composite scores (benign + attack) to
# <OUTPUT_ROOT>/composite_scores_<RUN_LABEL>/<attack>.npz for each
# flowguard_composite cell. Feeds scripts/plot_composite_scores.py.
DUMP_COMPOSITE_SCORES="${DUMP_COMPOSITE_SCORES:-0}"

# Extraction-attack knobs.
MAZE_REPLAY_BUFFER_SIZE="${MAZE_REPLAY_BUFFER_SIZE:-32768}"
MAZE_LOG_ITER="${MAZE_LOG_ITER:-5000}"
DISGUIDE_REPLAY_SIZE="${DISGUIDE_REPLAY_SIZE:-4096}"

# FlowGuard++-adaptive attackers (D3-D6). These extend the D2 generator loss
# with penalties on the composite's own components. They are NOT in the default
# ATTACKS list because each one costs a likelihood pass over the attacker's pool
# plus a heavier generator step; enable them explicitly, e.g.
#   ATTACKS=flowguard_d3_adaptive_c1c2,flowguard_d4_adaptive_c1c3,flowguard_d5_adaptive_composite,flowguard_d6_adaptive_stateful \
#     sbatch --export=ALL,SYBIL_IDENTITIES=1250 \
#          scripts/evaluate_attack_defense_combined.sbatch
ADAPTIVE_ATTACKER_POOL="${ADAPTIVE_ATTACKER_POOL:-CIFAR100}"
ADAPTIVE_CALIBRATION_SAMPLES="${ADAPTIVE_CALIBRATION_SAMPLES:-2048}"
ADAPTIVE_CALIBRATION_BATCH_SIZE="${ADAPTIVE_CALIBRATION_BATCH_SIZE:-64}"
ADAPTIVE_BAND_QUANTILE="${ADAPTIVE_BAND_QUANTILE:-0.95}"
ADAPTIVE_INTEGRAL_STEPS="${ADAPTIVE_INTEGRAL_STEPS:-8}"
# Each likelihood step costs a double-backward when optimized directly. 4 steps
# fits an A100-40GB at the generator batch sizes used here; raise it (or set
# ADAPTIVE_DISTILL_SURROGATE=1) if memory allows.
ADAPTIVE_LIKELIHOOD_STEPS="${ADAPTIVE_LIKELIHOOD_STEPS:-4}"
ADAPTIVE_HUTCHINSON_SAMPLES="${ADAPTIVE_HUTCHINSON_SAMPLES:-1}"
ADAPTIVE_LAMBDA_INTEGRAL="${ADAPTIVE_LAMBDA_INTEGRAL:-1.0}"
ADAPTIVE_LAMBDA_LIKELIHOOD="${ADAPTIVE_LAMBDA_LIKELIHOOD:-1.0}"
ADAPTIVE_LAMBDA_COMPOSITE="${ADAPTIVE_LAMBDA_COMPOSITE:-1.0}"
ADAPTIVE_LAMBDA_DISTRIBUTION="${ADAPTIVE_LAMBDA_DISTRIBUTION:-1.0}"
ADAPTIVE_LAMBDA_LABEL="${ADAPTIVE_LAMBDA_LABEL:-1.0}"
ADAPTIVE_REFERENCE_SAMPLE_SIZE="${ADAPTIVE_REFERENCE_SAMPLE_SIZE:-512}"
# 1 = distill the CNF component scores into a fast regressor so the generator
# gets composite gradients at one forward/backward per step.
ADAPTIVE_DISTILL_SURROGATE="${ADAPTIVE_DISTILL_SURROGATE:-0}"
ADAPTIVE_DISTILL_EPOCHS="${ADAPTIVE_DISTILL_EPOCHS:-30}"
# Snapshot generator/ensemble/optimizer/replay state every N queries so a cell
# that outlives the wall clock resumes instead of restarting. Mandatory for
# high-budget runs, where one cell cannot finish inside a single allocation.
# 0 disables. Only the DisGUIDE-loop attacks (D1-D6) support this.
ATTACK_CHECKPOINT_EVERY_QUERIES="${ATTACK_CHECKPOINT_EVERY_QUERIES:-0}"
# 1 = ignore an existing attack_checkpoint.pt and retrain from scratch.
ATTACK_NO_RESUME="${ATTACK_NO_RESUME:-0}"

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

if [[ ! -d "${TARGET_CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Missing target checkpoint dir: ${TARGET_CHECKPOINT_DIR}"
  exit 1
fi
if [[ ! -f "${FLOW_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing FlowPure/FlowGuard checkpoint: ${FLOW_CHECKPOINT}"
  exit 1
fi
if [[ ! -f "${A3_SURROGATE_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing A3 surrogate checkpoint: ${A3_SURROGATE_CHECKPOINT}"
  exit 1
fi

echo "[INFO] Job ID: ${SLURM_JOB_ID:-local}"
echo "[INFO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[INFO] Mode: ${MODE} | run-label: ${RUN_LABEL}"
echo "[INFO] Python: $("${PYTHON}" --version 2>&1)"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Query budget: ${QUERY_BUDGET} | epochs: ${EPOCHS} | sybil identities: ${SYBIL_IDENTITIES} | audit_only: ${AUDIT_ONLY}"
hpc_echo_list ATTACKS DEFENSES
echo "[INFO] CUDA visible:"
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
  --target-checkpoint-dir "${TARGET_CHECKPOINT_DIR}"
  --flow-checkpoint "${FLOW_CHECKPOINT}"
  --a3-surrogate-checkpoint "${A3_SURROGATE_CHECKPOINT}"
  --device "${DEVICE}"
  --query-budget "${QUERY_BUDGET}"
  --benign-query-budget "${BENIGN_QUERY_BUDGET}"
  --target-fpr "${TARGET_FPR}"
  --attack-batch-size "${ATTACK_BATCH_SIZE}"
  --extraction-attack-batch-size "${EXTRACTION_ATTACK_BATCH_SIZE}"
  --training-batch-size "${TRAINING_BATCH_SIZE}"
  --evaluation-batch-size "${EVALUATION_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --sybil-identities "${SYBIL_IDENTITIES}"
  --seed-samples "${SEED_SAMPLES}"
  --attack-artifact-sample-size "${ATTACK_ARTIFACT_SAMPLE_SIZE}"
  --label-histogram-samples "${LABEL_HISTOGRAM_SAMPLES}"
  --cem-population-size "${CEM_POPULATION_SIZE}"
  --cem-elite-fraction "${CEM_ELITE_FRACTION}"
  --cem-latent-dim "${CEM_LATENT_DIM}"
  --cem-selector "${CEM_SELECTOR}"
  --cem-lambda-detector "${CEM_LAMBDA_DETECTOR}"
  --projection-name "${PROJECTION_NAME}"
  --projection-strength "${PROJECTION_STRENGTH}"
  --projection-steps "${PROJECTION_STEPS}"
  --procedural-grid-size "${PROCEDURAL_GRID_SIZE}"
  --procedural-fourier-modes "${PROCEDURAL_FOURIER_MODES}"
  --procedural-blobs "${PROCEDURAL_BLOBS}"
  --procedural-lambda-tv "${PROCEDURAL_LAMBDA_TV}"
  --procedural-lambda-freq "${PROCEDURAL_LAMBDA_FREQ}"
  --procedural-lambda-color "${PROCEDURAL_LAMBDA_COLOR}"
  --rejection-oracle-proposal "${REJECTION_ORACLE_PROPOSAL}"
  --integral-num-steps "${INTEGRAL_NUM_STEPS}"
  --ks-threshold "${KS_THRESHOLD}"
  --ks-min-window "${KS_MIN_WINDOW}"
  --ks-max-window "${KS_MAX_WINDOW}"
  --labelhist-threshold "${LABELHIST_THRESHOLD}"
  --labelhist-min-window "${LABELHIST_MIN_WINDOW}"
  --labelhist-max-window "${LABELHIST_MAX_WINDOW}"
  --flowguardpp-decision-mode "${FLOWGUARDPP_DECISION_MODE}"
  --flowguardpp-response-policy "${FLOWGUARDPP_RESPONSE_POLICY}"
  --maze-replay-buffer-size "${MAZE_REPLAY_BUFFER_SIZE}"
  --maze-log-iter "${MAZE_LOG_ITER}"
  --disguide-replay-size "${DISGUIDE_REPLAY_SIZE}"
  --diffusion-model-id "${DIFFUSION_MODEL_ID}"
  --diffusion-scheduler "${DIFFUSION_SCHEDULER}"
  --diffusion-steps "${DIFFUSION_STEPS}"
  --diffusion-train-steps "${DIFFUSION_TRAIN_STEPS}"
  --diffusion-steering-scale "${DIFFUSION_STEERING_SCALE}"
  --diffusion-guidance-scale "${DIFFUSION_GUIDANCE_SCALE}"
  --diffusion-guidance-start-frac "${DIFFUSION_GUIDANCE_START_FRAC}"
  --disguide-bypass-ensemble-size "${DISGUIDE_BYPASS_ENSEMBLE_SIZE}"
  --disguide-bypass-g-iter "${DISGUIDE_BYPASS_G_ITER}"
  --disguide-bypass-d-iter "${DISGUIDE_BYPASS_D_ITER}"
  --disguide-bypass-rep-iter "${DISGUIDE_BYPASS_REP_ITER}"
  --disguide-bypass-lambda-div "${DISGUIDE_BYPASS_LAMBDA_DIV}"
  --disguide-bypass-epoch-itrs "${DISGUIDE_BYPASS_EPOCH_ITRS}"
  --disguide-bypass-lr-student "${DISGUIDE_BYPASS_LR_STUDENT}"
  --disguide-bypass-lr-generator "${DISGUIDE_BYPASS_LR_GENERATOR}"
  --adaptive-attacker-pool "${ADAPTIVE_ATTACKER_POOL}"
  --adaptive-calibration-samples "${ADAPTIVE_CALIBRATION_SAMPLES}"
  --adaptive-calibration-batch-size "${ADAPTIVE_CALIBRATION_BATCH_SIZE}"
  --adaptive-band-quantile "${ADAPTIVE_BAND_QUANTILE}"
  --adaptive-integral-steps "${ADAPTIVE_INTEGRAL_STEPS}"
  --adaptive-likelihood-steps "${ADAPTIVE_LIKELIHOOD_STEPS}"
  --adaptive-hutchinson-samples "${ADAPTIVE_HUTCHINSON_SAMPLES}"
  --adaptive-lambda-integral "${ADAPTIVE_LAMBDA_INTEGRAL}"
  --adaptive-lambda-likelihood "${ADAPTIVE_LAMBDA_LIKELIHOOD}"
  --adaptive-lambda-composite "${ADAPTIVE_LAMBDA_COMPOSITE}"
  --adaptive-lambda-distribution "${ADAPTIVE_LAMBDA_DISTRIBUTION}"
  --adaptive-lambda-label "${ADAPTIVE_LAMBDA_LABEL}"
  --adaptive-reference-sample-size "${ADAPTIVE_REFERENCE_SAMPLE_SIZE}"
  --adaptive-distill-epochs "${ADAPTIVE_DISTILL_EPOCHS}"
  --attack-checkpoint-every-queries "${ATTACK_CHECKPOINT_EVERY_QUERIES}"
)

if [[ "${ADAPTIVE_DISTILL_SURROGATE}" == "1" ]]; then
  CMD+=(--adaptive-distill-surrogate)
fi
if [[ "${ATTACK_NO_RESUME}" == "1" ]]; then
  CMD+=(--attack-no-resume)
fi
if [[ "${VERBOSE:-0}" == "1" ]]; then
  CMD+=(--verbose)
fi
if [[ "${DOWNLOAD:-0}" == "1" ]]; then
  CMD+=(--download)
fi
if [[ "${AUDIT_ONLY}" == "0" ]]; then
  CMD+=(--no-audit-only)
fi
if [[ "${COMPOSITE_EVAL_LEARNED}" == "1" ]]; then
  CMD+=(--composite-eval-learned --composite-learned-folds "${COMPOSITE_LEARNED_FOLDS}")
fi
if [[ "${DUMP_COMPOSITE_SCORES}" == "1" ]]; then
  CMD+=(--dump-composite-scores)
fi
if [[ -n "${RESUME_FILE}" ]]; then
  if [[ "${RESUME_FILE}" != /* ]]; then
    RESUME_FILE="${PROJECT_ROOT}/${RESUME_FILE}"
  fi
  if [[ ! -f "${RESUME_FILE}" ]]; then
    echo "[ERROR] RESUME_FILE not found: ${RESUME_FILE}"
    exit 1
  fi
  CMD+=(--resume-file "${RESUME_FILE}")
  echo "[INFO] Resume file: ${RESUME_FILE}"
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Combined evaluation finished (mode=${MODE})."
echo "[INFO] Summary: ${OUTPUT_ROOT}/attack_defense_matrix_${RUN_LABEL}_summary.json"
echo "[INFO] Report:  ${OUTPUT_ROOT}/attack_defense_matrix_${RUN_LABEL}_report.md"
