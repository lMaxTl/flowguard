"""End-to-end extraction against an *enforcing* FlowGuard++ deployment.

Why this exists
---------------
Every FlowGuard++ number in the paper is an audit measurement: the detector
scores the stream and the stream continues unchanged. The high-budget filtered
experiment enforces FlowPure, not FlowGuard++. So the manuscript shows a strong
offline detector, not a working defense -- the gap a TDSC reviewer will name
first.

Two things were missing to close it, both now in place:

1. ``FlowGuardCompositeDefense`` gained ``fused_threshold``. Until then the
   deployed object could only enforce individual component thresholds, which
   the evaluation sets to +-inf; the fused rule existed only inside the offline
   metric computation.
2. A ``per_query_suppress`` response policy. A batch-level policy cannot
   express a serving decision, and blanking a whole batch would misalign the
   attacker's inputs from its labels, which corrupts the extraction measurement
   rather than defending against it.

Protocol
--------
Phase A  Calibrate on benign traffic in audit mode: collect the per-component
         benign references, then set the fused threshold at the ``--target-fpr``
         quantile of the benign fused-risk distribution.
Phase B  Replay a *held-out* benign stream through the enforcing detector and
         record how much of it is suppressed. This is the achieved benign
         rejection rate at the deployed operating point -- the number the paper
         currently asserts from calibration rather than measures.
Phase C  Run the attack against the enforcing detector. Flagged queries receive
         an uninformative uniform response, so the substitute only learns from
         released answers. Records the accepted-query rate over the run.

Reported per attack: accepted-query rate, benign rejection rate, substitute
accuracy and fidelity, detection AUROC/F1 on the same stream, and the queries
actually consumed.

Usage::

    python scripts/evaluate_active_flowguard_extraction.py \\
        --run-label active-fgpp --output-root runs/active_flowguard \\
        --attacks flowguard_d1_latent_manifold,flowguard_d5_adaptive_composite \\
        --query-budget 1000000 --benign-query-budget 50000 \\
        --target-checkpoint-dir runs/notebook/.../target_model \\
        --flow-checkpoint runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt \\
        --likelihood-flow-checkpoint runs/flow_matching/cifar10_notebook/checkpoint_latest.pt
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_attack_defense_matrix_smoke as matrix  # noqa: E402
import numpy as np  # noqa: E402
from evaluate_attack_defense_matrix_smoke import (  # noqa: E402
    CalibrationData,
    DefenseRecipe,
    _attack_recipes,
    _attacker_surrogate_calibration,
    _build_label_histogram,
    _compute_metrics,
    _defense_recipes,
    _extract_samples,
    _force_threshold,
    _run_one,
    _score_threshold,
    _split_csv,
)

DEFENSE_KEY = "flowguard_composite"


def _series(records: list[Any], field: str) -> list[float]:
    """Concatenate one per-query series the composite recorded.

    The parallel driver's equivalent reads only the ``by_defense`` namespace,
    which exists only when several detectors share one stream. Here the
    composite runs alone and writes its series at the top level, so a
    namespace-only reader silently returns an empty list -- and the composite
    then falls back to an unstandardized max score, making the calibrated
    threshold describe a different rule than the one enforced.
    """
    values: list[float] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        scoped = (metadata.get("by_defense") or {}).get(DEFENSE_KEY)
        source = scoped if isinstance(scoped, dict) else metadata
        series = source.get(field)
        if isinstance(series, (list, tuple)):
            values.extend(
                float(item)
                for item in series
                if isinstance(item, (int, float)) and np.isfinite(float(item))
            )
    return values


def _window_scalars(records: list[Any], field: str, *, min_window: int) -> list[float]:
    """Per-batch values of a stateful signal, from batches whose window was full.

    A batch that has not yet filled its window reports 0.0 for the KS and MMD
    statistics. Including those zeros would drag the benign quantile down and
    produce a threshold that fires on ordinary traffic, which is the opposite of
    the intended calibration.
    """
    values: list[float] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        scoped = (metadata.get("by_defense") or {}).get(DEFENSE_KEY)
        source = scoped if isinstance(scoped, dict) else metadata
        if int(source.get("score_window_size", 0) or 0) < min_window:
            continue
        value = source.get(field)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _stateful_threshold(
    values: list[float], *, target_fpr: float, default: float
) -> float:
    """Quantile of a benign stateful statistic, at the same operating point.

    The composite ships fixed defaults (KS 0.25, MMD 0.05) that no benign
    calibration ever touches. Deployed unchanged they reject most benign traffic
    -- which is why previous runs recorded an achieved FPR of 1.0 while the
    published F1 was reconstructed at a nominal 5%. Calibrating them here makes
    the *whole* decision rule, not only its per-query half, correspond to the
    stated operating point.
    """
    if not values:
        return default
    threshold = float(np.quantile(np.asarray(values, dtype=np.float64), 1.0 - target_fpr))
    # A degenerate benign distribution (all identical) would give a threshold the
    # statistic reaches exactly, flagging benign traffic; keep it strictly above.
    return max(threshold, float(np.max(values))) if threshold <= 0.0 else threshold


def _fused_scores(records: list[Any]) -> list[float]:
    """Per-query fused risk scores recorded by the composite."""
    return _series(records, "composite_anomaly_score")


def _release_counts(records: list[Any]) -> dict[str, int]:
    """Break the suppression decision down by which half of the rule caused it.

    The two halves fail differently and need to be read apart. The per-query
    fused rule suppresses individual queries at its calibrated rate; a
    window-level flag condemns the whole batch, so at batch size B one
    false-positive batch costs B queries. Reporting only the total would hide
    which of the two drives a given benign rejection rate.
    """
    counts = {"released": 0, "total": 0, "per_query": 0, "stateful": 0}
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        released_mask = metadata.get("flowguard++_released")
        if not (isinstance(released_mask, (list, tuple)) and released_mask):
            # No mask: the pass was not enforcing, or predates the policy. Count
            # the batch as released so the rate is never flattered by missing
            # instrumentation.
            size = int(getattr(record, "batch_size", 0) or 0)
            counts["released"] += size
            counts["total"] += size
            continue
        query_flags = metadata.get("flowguard++_query_flags") or []
        for index, released in enumerate(released_mask):
            counts["total"] += 1
            if released:
                counts["released"] += 1
            elif index < len(query_flags) and bool(query_flags[index]):
                counts["per_query"] += 1
            else:
                counts["stateful"] += 1
    return counts


def _enforcing_recipe(
    audit_recipe: DefenseRecipe,
    *,
    fused_threshold: float,
    stateful_thresholds: dict[str, float] | None = None,
) -> DefenseRecipe:
    parameters = dict(audit_recipe.parameters)
    parameters["audit_only"] = False
    parameters["response_policy"] = "per_query_suppress"
    parameters["fused_threshold"] = float(fused_threshold)
    parameters.update(stateful_thresholds or {})
    return replace(
        audit_recipe,
        key=DEFENSE_KEY,
        display_name="FlowGuard++ (enforcing, per-query suppression)",
        parameters=parameters,
        notes=(
            "Fused risk threshold calibrated on benign traffic; flagged queries "
            "receive an uninformative response."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = matrix._build_parser()
    parser.add_argument(
        "--skip-benign-holdout",
        action="store_true",
        help="Skip phase B (measuring the benign rejection rate under enforcement).",
    )
    args = parser.parse_args(argv)
    args.target_checkpoint_dir = Path(args.target_checkpoint_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root = output_root / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    attack_order = _split_csv(args.attacks)
    num_classes, label_histogram, entropy_histogram = _build_label_histogram(
        args, device=args.device
    )
    calibration = CalibrationData(
        num_classes=num_classes,
        flowpure_scores=[],
        label_histogram=label_histogram,
        likelihood_scores=[],
        entropy_histogram=entropy_histogram,
    )

    defenses = _defense_recipes(args, calibration)
    benign_recipe = matrix.AttackRecipe(
        key="benign_reference",
        display_name="Benign reference",
        kind=matrix.AttackKind.TRANSFER_SET,
        mode="naive",
        query_dataset=args.dataset,
        extra={"transfer_artifact_sample_size": 0},
        notes="Benign calibration stream from the defended dataset.",
    )

    # --- Phase A0: component references ---------------------------------
    # The composite only emits its fused score once it has benign per-component
    # means; without them it falls back to a scale-dominated max of the raw
    # components. Collect those references first, so the threshold in phase A
    # is computed from the same score the deployment will enforce.
    print("[phase A0] benign component references (audit mode)")
    _, _, error, reference_records = _run_one(
        recipe=benign_recipe,
        defense=defenses[DEFENSE_KEY],
        args=args,
        scratch_root=scratch_root,
        query_budget=int(args.benign_query_budget),
        gt=0,
        skip_substitute_training=True,
    )
    if error:
        raise RuntimeError(f"Benign reference pass failed: {error}")
    calibration = CalibrationData(
        num_classes=calibration.num_classes,
        flowpure_scores=_series(reference_records, "flowpure_t0_score"),
        label_histogram=calibration.label_histogram,
        likelihood_scores=_series(reference_records, "likelihood_score"),
        entropy_histogram=calibration.entropy_histogram,
    )
    defenses = _defense_recipes(args, calibration)
    audit_recipe = defenses[DEFENSE_KEY]
    print(
        f"[phase A0] references: {len(calibration.flowpure_scores)} t0, "
        f"{len(calibration.likelihood_scores)} likelihood"
    )

    # --- Phase A: calibrate the fused threshold on benign traffic -----------
    print("[phase A] benign calibration (audit mode)")
    _, _, error, benign_records = _run_one(
        recipe=benign_recipe,
        defense=audit_recipe,
        args=args,
        scratch_root=scratch_root,
        query_budget=int(args.benign_query_budget),
        gt=0,
        skip_substitute_training=True,
    )
    if error:
        raise RuntimeError(f"Benign calibration failed: {error}")

    benign_samples = _extract_samples(benign_records, defense_key=DEFENSE_KEY, gt=0)
    fused_threshold = _score_threshold(benign_samples, target_fpr=float(args.target_fpr))
    benign_fused = _fused_scores(benign_records)
    print(
        f"[phase A] fused threshold={fused_threshold:.6g} from {len(benign_samples)} "
        f"benign queries (target FPR {float(args.target_fpr):.2%})"
    )

    # Calibrate the stateful half of the rule on the same benign stream. Without
    # this the composite enforces a calibrated per-query threshold alongside
    # hard-coded window thresholds, and the window terms decide almost every
    # batch: a smoke run rejected 76.6% of benign traffic while the per-query
    # rule was firing at its intended 5.5%.
    min_window = int(audit_recipe.parameters.get("min_window", 64))
    target_fpr = float(args.target_fpr)
    stateful_thresholds = {
        "score_ks_threshold": _stateful_threshold(
            _window_scalars(benign_records, "score_window_ks", min_window=min_window),
            target_fpr=target_fpr,
            default=float(audit_recipe.parameters.get("score_ks_threshold", 0.25)),
        ),
        "label_mmd_threshold": _stateful_threshold(
            _window_scalars(benign_records, "label_histogram_mmd", min_window=min_window),
            target_fpr=target_fpr,
            default=float(audit_recipe.parameters.get("label_mmd_threshold", 0.05)),
        ),
        "entropy_mmd_threshold": _stateful_threshold(
            _window_scalars(benign_records, "entropy_histogram_mmd", min_window=min_window),
            target_fpr=target_fpr,
            default=float(audit_recipe.parameters.get("entropy_mmd_threshold", 0.05)),
        ),
    }
    for name, value in stateful_thresholds.items():
        print(f"[phase A] {name}={value:.6g}")

    enforcing_recipe = _enforcing_recipe(
        audit_recipe,
        fused_threshold=fused_threshold,
        stateful_thresholds=stateful_thresholds,
    )

    # --- Phase B: benign rejection rate under enforcement -------------------
    benign_rejection_rate: float | None = None
    benign_counts: dict[str, int] = {}
    if not args.skip_benign_holdout:
        print("[phase B] held-out benign stream through the enforcing detector")
        _, _, error, holdout_records = _run_one(
            recipe=benign_recipe,
            defense=enforcing_recipe,
            args=args,
            scratch_root=scratch_root,
            query_budget=int(args.benign_query_budget),
            gt=0,
            skip_substitute_training=True,
        )
        if error:
            raise RuntimeError(f"Benign holdout failed: {error}")
        benign_counts = _release_counts(holdout_records)
        total = benign_counts["total"]
        benign_rejection_rate = (
            1.0 - (benign_counts["released"] / total) if total else None
        )
        print(
            f"[phase B] benign rejection rate={benign_rejection_rate:.4f} "
            f"({total - benign_counts['released']}/{total} suppressed: "
            f"{benign_counts['per_query']} by the per-query rule, "
            f"{benign_counts['stateful']} by a window-level flag)"
        )

    # --- Phase C: extraction against the enforcing detector -----------------
    adaptive_calibration = None
    if set(attack_order) & (
        set(matrix.FULLY_ADAPTIVE_ATTACKS) | set(matrix.VELOCITY_ADAPTIVE_ATTACKS)
    ):
        adaptive_calibration = _attacker_surrogate_calibration(
            args, cache_path=output_root / "attacker_surrogate_calibration.json"
        )
    attacks = _attack_recipes(
        args,
        adaptive_calibration=adaptive_calibration,
        benign_label_histogram=calibration.label_histogram,
        benign_entropy_moments=matrix._entropy_moments_from_histogram(
            calibration.entropy_histogram, num_classes=calibration.num_classes
        ),
    )
    unknown = [key for key in attack_order if key not in attacks]
    if unknown:
        raise ValueError(f"Unknown attacks: {unknown}")

    rows: list[dict[str, Any]] = []
    for attack_key in attack_order:
        print(f"[phase C] {attack_key} against the enforcing detector")
        started = time.perf_counter()
        _, attack_meta, attack_error, attack_records = _run_one(
            recipe=attacks[attack_key],
            defense=enforcing_recipe,
            args=args,
            scratch_root=scratch_root,
            query_budget=int(args.query_budget),
            gt=1,
            skip_substitute_training=False,
        )
        elapsed = time.perf_counter() - started
        if attack_error:
            print(f"[FAILED] {attack_key} after {elapsed / 60:.1f} min: {attack_error}")
            rows.append({"attack": attack_key, "error": attack_error, "elapsed_seconds": elapsed})
            continue

        attack_counts = _release_counts(attack_records)
        released, total = attack_counts["released"], attack_counts["total"]
        attack_samples = _extract_samples(attack_records, defense_key=DEFENSE_KEY, gt=1)
        metrics = _compute_metrics(
            _force_threshold(benign_samples, fused_threshold)
            + _force_threshold(attack_samples, fused_threshold)
        )
        row = {
            "attack": attack_key,
            "defense": "flowguard_composite_enforcing",
            "error": None,
            **metrics,
            "fused_threshold": float(fused_threshold),
            "accepted_queries": released,
            "submitted_queries": total,
            "accepted_query_rate": (released / total) if total else None,
            "suppressed_by_per_query_rule": attack_counts["per_query"],
            "suppressed_by_window_flag": attack_counts["stateful"],
            "benign_rejection_rate": benign_rejection_rate,
            "benign_rejection_breakdown": (
                benign_counts if not args.skip_benign_holdout else None
            ),
            "accuracy": attack_meta.get("accuracy"),
            "fidelity": attack_meta.get("fidelity"),
            "total_queries": attack_meta.get("total_queries"),
            "query_curve": matrix._query_curve(attack_meta),
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(
            f"[done] {attack_key} | accepted={row['accepted_query_rate']:.4f} "
            f"| fidelity={row['fidelity']} | AUROC={matrix._fmt(row.get('auroc'))}"
        )

        payload = {
            "config": {
                "run_label": args.run_label,
                "dataset": args.dataset,
                "query_budget": int(args.query_budget),
                "benign_query_budget": int(args.benign_query_budget),
                "target_fpr": float(args.target_fpr),
                "sybil_identities": int(getattr(args, "sybil_identities", 1)),
                "attacks": attack_order,
                "defenses": ["flowguard_composite_enforcing"],
                "enforcing": True,
                "response_policy": "per_query_suppress",
                "fused_threshold": float(fused_threshold),
                "stateful_thresholds": stateful_thresholds,
                "benign_fused_mean": float(np.mean(benign_fused)) if benign_fused else None,
                "adaptive_generator": str(getattr(args, "adaptive_generator", "")),
            },
            "results": rows,
        }
        matrix._atomic_write_text(
            output_root / f"active_flowguard_{args.run_label}_summary.json",
            json.dumps(payload, indent=2),
        )
    print(f"[write] {output_root / f'active_flowguard_{args.run_label}_summary.json'}")


if __name__ == "__main__":
    main()
