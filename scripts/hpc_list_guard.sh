#!/usr/bin/env bash
# Guards against silently truncated list-valued Slurm variables.
#
# `sbatch --export` takes a COMMA-separated list of assignments, so
#
#   sbatch --export=ALL,DEFENSES=flowpure,flowguard_composite job.sbatch
#
# does not do what it reads like. Slurm splits on every comma, so the job sees
# DEFENSES=flowpure plus a bare token `flowguard_composite`, which it reads as
# "re-export the variable named flowguard_composite from the submitting
# environment" and drops when no such variable exists. The job then runs a
# quietly smaller experiment and exits 0. Every element after the first is lost
# this way; later KEY=VALUE pairs in the same --export are unaffected.
#
# Pass list-valued variables in the submitting environment instead, where the
# shell -- not Slurm -- parses them, and let ALL carry them into the job:
#
#   DEFENSES=flowpure,flowguard_composite sbatch --export=ALL job.sbatch
#
# Usage (from project root, after resolving PROJECT_ROOT):
#   source scripts/hpc_list_guard.sh
#   hpc_warn_export_truncation
#   hpc_guard_array_range ATTACK_LIST "${#ATTACK_ARRAY[@]}" || exit 1

# Print the fix for one truncated variable.
hpc_export_hint() {
  local name="${1:-VAR}"
  echo "[ERROR] Likely cause: a comma-separated value inside --export. Slurm splits"
  echo "[ERROR] --export on commas, so only the first element of the value survives."
  echo "[ERROR] Pass list variables in the submitting environment instead:"
  echo "[ERROR]   ${name}=a,b,c sbatch --export=ALL <script>.sbatch"
}

# hpc_guard_array_range <list_var_name> <parsed_element_count>
#
# Fails EVERY task -- task 0 included -- when --array asks for more indices than
# the list holds. Erroring only in the out-of-range tasks is worse than useless:
# task 0 runs the truncated experiment to completion and exits 0, so a 5x9
# matrix quietly becomes a 1x1 cell after four GPU-hours.
hpc_guard_array_range() {
  local list_name="${1:?list variable name required}"
  local count="${2:?element count required}"
  local max="${SLURM_ARRAY_TASK_MAX:-}"

  [[ -z "${max}" ]] && return 0
  (( count > max )) && return 0

  echo "[ERROR] --array requests indices up to ${max}, but ${list_name} holds only"
  echo "[ERROR] ${count} element(s): ${list_name}=${!list_name-<unset>}"
  hpc_export_hint "${list_name}"
  echo "[ERROR] Failing every task on purpose: letting task 0 proceed would run a"
  echo "[ERROR] silently truncated experiment to completion and report success."
  return 1
}

_hpc_report_truncation() {
  local name="${1}" dropped="${2}"
  local kept="${!name-}"
  echo "[WARN] --export looks like it split ${name}. The job received:"
  echo "[WARN]   ${name}=${kept}"
  echo "[WARN] and Slurm discarded: ${dropped}"
  echo "[WARN] Re-submit with the list in the submitting environment:"
  echo "[WARN]   ${name}=${kept}${dropped:+,${dropped}} sbatch --export=ALL <script>.sbatch"
}

# Best-effort warning for the non-array case, which hpc_guard_array_range
# cannot see. Slurm records the raw --export string in SLURM_EXPORT_ENV; a bare
# token following an assignment there is an element that was split off that
# assignment's value.
#
# Advisory only -- it never fails a job. SLURM_EXPORT_ENV is not guaranteed to
# be populated on every Slurm version, and a deliberately bare export ("forward
# PATH") is indistinguishable from a dropped element, so treat a warning as a
# prompt to check the echoed command rather than as proof.
hpc_warn_export_truncation() {
  local raw="${SLURM_EXPORT_ENV:-}"
  [[ -z "${raw}" ]] && return 0

  local -a tokens=()
  IFS=',' read -r -a tokens <<< "${raw}"

  local current="" dropped="" token
  for token in "${tokens[@]}"; do
    if [[ "${token}" == *=* ]]; then
      [[ -n "${dropped}" ]] && _hpc_report_truncation "${current}" "${dropped}"
      current="${token%%=*}"
      dropped=""
      continue
    fi
    # A bare token before any assignment (the leading ALL) is a real export
    # directive, not a symptom.
    [[ -n "${current}" ]] && dropped+="${dropped:+,}${token}"
  done
  [[ -n "${dropped}" ]] && _hpc_report_truncation "${current}" "${dropped}"
  return 0
}

# hpc_echo_list <name> [<name> ...]
# Print each list variable with its element count, so a truncated list is
# visible in the first screen of the log instead of only in the final summary.
hpc_echo_list() {
  local name value count
  for name in "$@"; do
    value="${!name-}"
    if [[ -z "${value}" ]]; then
      echo "[INFO] ${name}: (empty)"
      continue
    fi
    count="$(awk -F',' '{print NF}' <<< "${value}")"
    echo "[INFO] ${name}: ${count} element(s) -- ${value}"
  done
}
