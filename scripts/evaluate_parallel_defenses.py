"""Score one attack run against several detectors at once.

``evaluate_attack_defense_matrix_smoke.py`` runs one attack per (defense,
attack) cell, so comparing N detectors retrains the attack N times. In
audit-only mode that is wasted work: an auditing detector records a score and
returns the batch untouched, so the attack trajectory is independent of which
detector is watching. This driver runs each attack **once** and fans the query
stream out to every detector via :class:`MultiAuditQueryDefense`.

Two benefits:

- Cost drops from ``N_defenses x N_attacks`` attack runs to ``N_attacks``.
- The comparison becomes *paired*. Every detector scores the identical queries
  from the identical generator state, instead of N independently seeded runs
  whose attacks diverge -- so a difference between detectors is a difference
  between detectors, not between runs.

Enforcement is not supported here: two detectors that both reject would each
change the stream the other sees. Use the matrix driver, one defense at a time,
for enforcing evaluations.

Outputs match the matrix driver's schema (``..._summary.json`` with one row per
(defense, attack) plus ``query_curve``), so the existing reporting works:

    python scripts/report_query_fidelity_benchmark.py --summary <summary.json>
    python scripts/plot_query_evolution.py --snapshot-dir <root>/query_snapshots

Usage::

    python scripts/evaluate_parallel_defenses.py \\
        --run-label par --output-root runs/par \\
        --dataset CIFAR10 --target-architecture vgg16_bn \\
        --substitute-architecture vgg16_bn \\
        --target-checkpoint-dir runs/notebook/.../target_model \\
        --attacks flowguard_d3_adaptive_c1c2,flowguard_d4_adaptive_c1c3 \\
        --defenses flowpure,flowguard_composite \\
        --adaptive-generator latent
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Reuse the matrix driver's recipes, runner and metrics rather than
# reimplementing them: this script differs only in how many detectors observe a
# single run, and any divergence in scoring would make the two incomparable.
import evaluate_attack_defense_matrix_smoke as matrix
import numpy as np
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
    _matrix_artifact_paths,
    _query_curve,
    _run_one,
    _score_threshold,
    _split_csv,
)


def _scoped_series(records: list[Any], defense_key: str, field: str) -> list[float]:
    """Concatenate a per-query metadata list emitted by one detector.

    Reads from the ``by_defense`` namespace that MultiAuditQueryDefense writes,
    so a shared run can be mined for one detector's raw component scores.
    """
    values: list[float] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        scoped = (metadata.get("by_defense") or {}).get(defense_key)
        series = (scoped or {}).get(field)
        if isinstance(series, (list, tuple)):
            values.extend(
                float(item)
                for item in series
                if isinstance(item, (int, float)) and np.isfinite(float(item))
            )
    return values


def _multi_defense_recipe(
    defenses: dict[str, DefenseRecipe],
    keys: list[str],
) -> DefenseRecipe:
    """Compose one audit-only recipe that fans out to every requested detector."""
    children: list[dict[str, Any]] = []
    for key in keys:
        recipe = defenses[key]
        parameters = dict(recipe.parameters)
        # Force auditing: MultiAuditQueryDefense rejects enforcing children, and
        # a blocking detector would invalidate its siblings' scores anyway.
        parameters["audit_only"] = True
        children.append(
            {"key": key, "name": recipe.query_defense, "parameters": parameters}
        )
    return DefenseRecipe(
        key="multi_audit",
        display_name="Parallel audit (" + ", ".join(keys) + ")",
        query_defense="multi_audit",
        parameters={"defenses": children},
        notes="One query stream scored by every detector simultaneously.",
    )


def _build_parser():
    parser = matrix._build_parser()
    parser.description = __doc__
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.device = matrix._resolve_device(args.device)
    args.target_checkpoint_dir = Path(args.target_checkpoint_dir)
    if not args.target_checkpoint_dir.exists():
        raise FileNotFoundError(f"--target-checkpoint-dir not found: {args.target_checkpoint_dir}")
    args.flow_checkpoint = matrix._resolve_flow_checkpoint(args.flow_checkpoint)
    args.a3_surrogate_checkpoint = (
        Path(args.a3_surrogate_checkpoint)
        if args.a3_surrogate_checkpoint is not None
        else args.flow_checkpoint
    )
    # The attacker mirrors the detector it is adapting to, so its likelihood
    # surrogate defaults to the defender's likelihood CNF rather than to the
    # velocity checkpoint -- which would leave the C3 penalty undefined.
    args.a3_likelihood_surrogate_checkpoint = (
        Path(args.a3_likelihood_surrogate_checkpoint)
        if args.a3_likelihood_surrogate_checkpoint is not None
        else (
            Path(args.likelihood_flow_checkpoint)
            if args.likelihood_flow_checkpoint is not None
            else args.a3_surrogate_checkpoint
        )
    )
    if not args.a3_likelihood_surrogate_checkpoint.exists():
        raise FileNotFoundError(
            f"--a3-likelihood-surrogate-checkpoint not found: "
            f"{args.a3_likelihood_surrogate_checkpoint}"
        )
    if not bool(args.audit_only):
        raise SystemExit(
            "Parallel defense scoring requires audit-only mode. Two enforcing "
            "detectors would each alter the query stream the other observes. "
            "Drop --no-audit-only, or use the matrix driver for enforcement."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root = output_root / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    artifacts = _matrix_artifact_paths(output_root, str(args.run_label))

    attack_order = _split_csv(args.attacks)
    defense_order = _split_csv(args.defenses)

    print("[calibration] building benign label and entropy histograms")
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
    unknown = [key for key in defense_order if key not in defenses]
    if unknown:
        raise ValueError(f"Unknown defenses: {unknown}. Available: {sorted(defenses)}")

    benign_recipe = matrix.AttackRecipe(
        key="benign_reference",
        display_name="Benign reference",
        kind=matrix.AttackKind.TRANSFER_SET,
        mode="naive",
        query_dataset=args.dataset,
        extra={"transfer_artifact_sample_size": 0},
        notes="Benign calibration stream from the defended dataset.",
    )

    # --- Pass 0: benign references the composite standardizes against --------
    #
    # FlowGuardCompositeDefense only emits its fused anomaly score once it has
    # benign per-component means; without them it silently falls back to
    # max(t0, integral, -loglik), which is dominated by t0. Those references
    # come from the benign stream itself, so they have to be collected before
    # the pass whose scores set the operating point -- otherwise the threshold
    # is computed from the fallback score while the attack is scored with the
    # calibrated one, and the two are not comparable.
    if "flowguard_composite" in defense_order:
        print(f"[benign] pass 0/2: component references -> {len(defense_order)} detectors")
        reference_multi = _multi_defense_recipe(defenses, defense_order)
        _, _, reference_error, reference_records = _run_one(
            recipe=benign_recipe,
            defense=reference_multi,
            args=args,
            scratch_root=scratch_root,
            query_budget=int(args.benign_query_budget),
            gt=0,
            skip_substitute_training=True,
        )
        if reference_error:
            raise RuntimeError(f"Benign reference pass failed: {reference_error}")
        # Take both references from the composite's own per-component output.
        # It already records the raw t0 and log-likelihood of every query
        # whether or not benign statistics exist, so there is no need to
        # conscript the standalone flow_matching detector -- which is a second
        # full likelihood ODE per query for a detector the caller did not ask
        # for.
        calibration.flowpure_scores = _scoped_series(
            reference_records, "flowguard_composite", "flowpure_t0_score"
        )
        calibration.likelihood_scores = _scoped_series(
            reference_records, "flowguard_composite", "likelihood_score"
        )
        defenses = _defense_recipes(args, calibration)
        print(
            f"[benign] references: {len(calibration.flowpure_scores)} t0, "
            f"{len(calibration.likelihood_scores)} likelihood"
        )

    # --- Pass 1: benign stream, scored by every detector at once -------------
    print(f"[benign] pass 1/2: one stream -> {len(defense_order)} detectors")
    multi = _multi_defense_recipe(defenses, defense_order)
    _, _, benign_error, benign_records = _run_one(
        recipe=benign_recipe,
        defense=multi,
        args=args,
        scratch_root=scratch_root,
        query_budget=int(args.benign_query_budget),
        gt=0,
        skip_substitute_training=True,
    )
    if benign_error:
        raise RuntimeError(f"Benign calibration failed: {benign_error}")

    benign_by_defense = {
        key: _extract_samples(benign_records, defense_key=key, gt=0)
        for key in defense_order
    }

    thresholds = {
        key: _score_threshold(samples, target_fpr=float(args.target_fpr))
        for key, samples in benign_by_defense.items()
        if samples
    }
    for key, value in thresholds.items():
        print(f"[benign] {key}: threshold={value:.6g} n={len(benign_by_defense[key])}")

    # --- Attacker-side calibration (D1-D6 need their own benign stats) -------
    adaptive_calibration: dict[str, Any] | None = None
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
    unknown_attacks = [key for key in attack_order if key not in attacks]
    if unknown_attacks:
        raise ValueError(f"Unknown attacks: {unknown_attacks}")

    # --- Pass 2: one run per attack, every detector scores it ----------------
    rows: list[dict[str, Any]] = []
    for attack_key in attack_order:
        print(f"[run] {attack_key} -> {len(defense_order)} detectors (single run)")
        started = time.perf_counter()
        _, attack_meta, attack_error, attack_records = _run_one(
            recipe=attacks[attack_key],
            defense=multi,
            args=args,
            scratch_root=scratch_root,
            query_budget=int(args.query_budget),
            gt=1,
            skip_substitute_training=False,
        )
        elapsed = time.perf_counter() - started
        curve = _query_curve(attack_meta)

        if attack_error:
            # Loudly, once per attack. Previously a failed attack produced no
            # [done] line, no traceback and exit code 0, so the job looked
            # successful and the failure was only discoverable by opening the
            # summary JSON.
            print(
                f"[FAILED] {attack_key} after {elapsed / 60:.1f} min: {attack_error}",
                flush=True,
            )
        for defense_key in defense_order:
            if attack_error:
                rows.append(
                    {
                        "defense": defense_key,
                        "attack": attack_key,
                        "error": attack_error,
                        "elapsed_seconds": elapsed,
                    }
                )
                continue
            attack_samples = _extract_samples(
                attack_records, defense_key=defense_key, gt=1
            )
            threshold = thresholds.get(defense_key)
            # Apply the calibrated threshold to BOTH sides. Thresholding only
            # the attack side mixes decision rules within one metric: recall
            # would come from the calibrated threshold while precision counted
            # benign queries flagged by the detector's own runtime rule. For the
            # composite, whose stateful terms flag nearly all benign traffic,
            # that produced FPR=1.0 and an F1 that is not an operating-point F1
            # at all.
            scored_benign = benign_by_defense[defense_key]
            scored_attack = attack_samples
            if threshold is not None:
                scored_benign = _force_threshold(scored_benign, threshold)
                scored_attack = _force_threshold(scored_attack, threshold)
            metrics = _compute_metrics(scored_benign + scored_attack)
            row = {
                "defense": defense_key,
                "attack": attack_key,
                "error": None,
                **metrics,
                "accuracy": attack_meta.get("accuracy"),
                "fidelity": attack_meta.get("fidelity"),
                "total_queries": attack_meta.get("total_queries"),
                "elapsed_seconds": elapsed,
                "query_curve": curve,
                # Every row for this attack came from one shared run.
                "shared_attack_run": True,
            }
            rows.append(row)
            print(
                f"[done] {defense_key} x {attack_key} | "
                f"AUROC={matrix._fmt(row.get('auroc'))} "
                f"TPR={matrix._fmt(row.get('tpr'))} "
                f"FPR={matrix._fmt(row.get('fpr'))}"
            )

        payload = {
            "config": {
                "run_label": args.run_label,
                "dataset": args.dataset,
                "query_budget": int(args.query_budget),
                "benign_query_budget": int(args.benign_query_budget),
                "target_fpr": float(args.target_fpr),
                "sybil_identities": int(getattr(args, "sybil_identities", 1)),
                "diffusion_steps": int(getattr(args, "diffusion_steps", 0)),
                "diffusion_guidance_scale": float(
                    getattr(args, "diffusion_guidance_scale", 0.0)
                ),
                "attack_plateau": {
                    "min_queries": int(getattr(args, "attack_plateau_min_queries", 0)),
                    "patience_queries": int(
                        getattr(args, "attack_plateau_patience_queries", 0)
                    ),
                    "min_delta": float(getattr(args, "attack_plateau_min_delta", 0.0)),
                    "smoothing_window": int(
                        getattr(args, "attack_plateau_smoothing_window", 1)
                    ),
                },
                "attacks": attack_order,
                "defenses": defense_order,
                "adaptive_generator": str(args.adaptive_generator),
                "parallel_defense_scoring": True,
            },
            "thresholds": thresholds,
            "results": rows,
        }
        matrix._atomic_write_text(
            artifacts["summary"], json.dumps(payload, indent=2, default=str)
        )
        print(f"[report] wrote {artifacts['summary']}")

    print(
        f"[summary] {len(attack_order)} attack run(s) scored by "
        f"{len(defense_order)} detector(s) "
        f"= {len(rows)} cells from {len(attack_order)} runs "
        f"(matrix driver would need {len(attack_order) * len(defense_order)})"
    )
    # Exit non-zero when any cell failed, so Slurm reports the job as FAILED
    # rather than COMPLETED. A silent exit 0 on a broken run is worse than a
    # crash: it looks like a result.
    failed = sorted({str(row["attack"]) for row in rows if row.get("error")})
    if failed:
        raise SystemExit(
            f"[FAILED] {len(failed)} attack(s) produced no result: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
