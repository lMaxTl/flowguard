"""Evaluation runner for the attack/defense detection matrix.

The script runs the requested attack x query-defense matrix and writes only
textual outputs:

- ``attack_defense_matrix_<run-label>_summary.json`` (updated after every cell)
- ``attack_defense_matrix_<run-label>_report.md`` (updated after every cell)

Resume a partial run with ``--resume-file`` pointing at JSON containing
``start_defense`` and ``start_attack`` (optional ``state_path`` to the summary).

All per-run training/checkpoint/transfer artifacts are written to a scratch
directory and removed immediately after the metrics have been collected.
Smoke and full evaluations use the same script shape; the full Slurm script
only increases ``--query-budget`` / ``--benign-query-budget``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defenses import datasets as legacy_datasets
from flowguard.experiments.factory import build_experiment_spec
from flowguard.experiments.spec import AttackKind
from flowguard.orchestration import run_experiment
from flowguard.serving.model_loader import load_legacy_model

DEFAULT_ATTACKS: tuple[str, ...] = (
    "prada",
    "maze",
    "disguide",
    "transfer_naive",
    "transfer_top1",
    "transfer_s4l",
    "transfer_smoothing",
    "flowguard_a1_clean_transfer",
    "flowguard_a2_projected_maze",
    "flowguard_a3_flowblind",
    "flowguard_adaptive_adaptive",
)

FLOWPURE_BYPASS_ATTACKS: tuple[str, ...] = (
    "flowguard_d1_latent_manifold",
    "flowguard_d2_procedural_natural",
    "flowguard_d3_projected_latent",
    "flowguard_d4_rejection_oracle",
)

# Attackers adaptive to FlowGuard++ itself rather than to the single-point
# velocity score it extends. These require the attacker-side surrogate
# calibration pass (see ``_attacker_surrogate_calibration``).
# Adaptive to the single-point velocity score only. They still need the
# attacker's benign statistics: without them the C1 term degenerates to the raw
# mean ||v(0,x)||^2, whose scale is dataset-dependent (~2.5e-3 on the CIFAR-10
# FlowPure CNF vs ~3e2 on MNIST), so a fixed lambda makes the penalty either
# inert or dominant. Listed separately from FULLY_ADAPTIVE_ATTACKS because they
# need the calibration but not the composite-component terms.
VELOCITY_ADAPTIVE_ATTACKS: tuple[str, ...] = (
    "flowguard_d1_latent_manifold",
    "flowguard_d2_procedural_natural",
)

FULLY_ADAPTIVE_ATTACKS: tuple[str, ...] = (
    "flowguard_d3_adaptive_c1c2",
    "flowguard_d4_adaptive_c1c3",
    "flowguard_d5_adaptive_composite",
    "flowguard_d6_adaptive_stateful",
)

DEFAULT_DEFENSES: tuple[str, ...] = (
    "prada",
    "flowpure",
    "fdinet",
    "flowguard_integral",
    "flowguard_userlevel",
    "flowguard_labelhist",
    "flowguard_composite",
)

# Defenses whose operating threshold is (re)calibrated on the realized benign
# stream at the target FPR, then applied to attack scores. FDINet is included
# because its internal threshold is calibrated on cherry-picked high-confidence
# training samples and does not transfer to the benign query stream (observed
# ~44% FPR vs a 5% target). Recalibrating here yields a fair TPR@FPR baseline.
CALIBRATED_SCORE_DEFENSES: frozenset[str] = frozenset(
    {"flowpure", "flowguard_integral", "fdinet", "flow_matching"}
)

LIKELIHOOD_DEFENSES: frozenset[str] = frozenset({"flow_matching", "flowguard_composite"})

# Report-only defense column derived from the equal-weight composite run by
# learning per-component weights via cross-fitted logistic regression.
LEARNED_COMPOSITE_KEY: str = "flowguard_composite_learned"


@dataclass(frozen=True, slots=True)
class AttackRecipe:
    key: str
    display_name: str
    kind: AttackKind
    mode: str
    query_dataset: str
    extra: dict[str, Any]
    notes: str


@dataclass(frozen=True, slots=True)
class DefenseRecipe:
    key: str
    display_name: str
    query_defense: str
    parameters: dict[str, Any]
    notes: str


@dataclass(slots=True)
class CalibrationData:
    num_classes: int
    flowpure_scores: list[float]
    label_histogram: list[float]
    likelihood_scores: list[float]
    entropy_histogram: list[float]


@dataclass(frozen=True, slots=True)
class ResumeFrom:
    """Matrix cell to start (or restart) from; earlier cells are skipped."""

    defense: str
    attack: str


@dataclass(frozen=True, slots=True)
class MatrixProgress:
    status: Literal["running", "complete"]
    last_completed: dict[str, Any] | None = None
    updated_at: str | None = None


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _matrix_artifact_paths(output_root: Path, run_label: str) -> dict[str, Path]:
    safe_label = _safe_name(str(run_label).lower())
    return {
        "summary": output_root / f"attack_defense_matrix_{safe_label}_summary.json",
        "report": output_root / f"attack_defense_matrix_{safe_label}_report.md",
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _matrix_progress_now(
    *,
    status: Literal["running", "complete"],
    last_completed: dict[str, Any] | None,
) -> MatrixProgress:
    return MatrixProgress(
        status=status,
        last_completed=last_completed,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _parse_resume_file(resume_path: Path) -> tuple[ResumeFrom, Path | None]:
    """Load resume instructions from a small JSON file.

    Expected fields:
    - ``start_defense`` / ``defense``: first defense to execute (inclusive).
    - ``start_attack`` / ``attack``: first attack to execute for that defense (inclusive).
    - ``state_path`` (optional): existing summary JSON to merge with.
    """
    payload = json.loads(resume_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Resume file must be a JSON object: {resume_path}")
    defense = payload.get("start_defense", payload.get("defense"))
    attack = payload.get("start_attack", payload.get("attack"))
    if not defense or not attack:
        raise ValueError(
            f"Resume file must define start_defense/start_attack (or defense/attack): {resume_path}"
        )
    state_raw = payload.get("state_path")
    state_path = Path(state_raw) if state_raw else None
    return ResumeFrom(defense=str(defense), attack=str(attack)), state_path


def _defense_before_resume(defense_key: str, resume_from: ResumeFrom, defense_order: list[str]) -> bool:
    return defense_order.index(defense_key) < defense_order.index(resume_from.defense)


def _attack_before_resume(attack_key: str, resume_from: ResumeFrom, attack_order: list[str]) -> bool:
    return attack_order.index(attack_key) < attack_order.index(resume_from.attack)


def _upsert_result_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    key = (str(row["defense"]), str(row["attack"]))
    merged = [
        existing
        for existing in rows
        if (str(existing["defense"]), str(existing["attack"])) != key
    ]
    merged.append(row)
    return merged


def _load_matrix_state(
    summary_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any] | None]:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot resume: summary state not found at {summary_path}. "
            "Run without --resume-file first or set state_path in the resume file."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(payload.get("results", []))
    calibrations = dict(payload.get("calibrations", {}))
    global_calibration = payload.get("global_calibration")
    if global_calibration is not None and not isinstance(global_calibration, dict):
        raise ValueError(f"global_calibration must be an object in {summary_path}")
    return rows, calibrations, global_calibration


def _global_calibration_payload(calibration: CalibrationData) -> dict[str, Any]:
    return {
        "num_classes": int(calibration.num_classes),
        "label_histogram": list(calibration.label_histogram),
        "entropy_histogram": list(calibration.entropy_histogram),
        "flowpure_scores": list(calibration.flowpure_scores),
        "likelihood_scores": list(calibration.likelihood_scores),
    }


def _calibration_from_global_payload(payload: dict[str, Any]) -> CalibrationData:
    return CalibrationData(
        num_classes=int(payload["num_classes"]),
        flowpure_scores=[float(value) for value in payload.get("flowpure_scores", [])],
        label_histogram=[float(value) for value in payload.get("label_histogram", [])],
        likelihood_scores=[float(value) for value in payload.get("likelihood_scores", [])],
        entropy_histogram=[float(value) for value in payload.get("entropy_histogram", [])],
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _resolve_flow_checkpoint(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(f"--flow-checkpoint not found: {requested}")
        return requested
    candidates = [
        PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd" / "checkpoint_latest.pt",
        PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd_smoke" / "checkpoint_latest.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "--flow-checkpoint is required. Checked:\n"
        + "\n".join(f"- {candidate}" for candidate in candidates)
    )


def _build_label_histogram(args: argparse.Namespace, *, device: str) -> tuple[int, list[float], list[float]]:
    loaded = load_legacy_model(args.target_checkpoint_dir, device=device)
    dataset_name = str(args.dataset)
    modelfamily = legacy_datasets.dataset_to_modelfamily[dataset_name]
    transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    dataset = legacy_datasets.__dict__[dataset_name](
        train=False,
        transform=transform,
        download=bool(args.download),
    )
    sample_count = min(int(args.label_histogram_samples), len(dataset))
    subset = Subset(dataset, list(range(sample_count)))
    loader = DataLoader(
        subset,
        batch_size=int(args.evaluation_batch_size),
        shuffle=False,
        num_workers=0,
    )
    num_classes = int(loaded.num_classes)
    histogram = np.zeros(num_classes, dtype=np.float64)
    entropy_values: list[float] = []
    loaded.model.eval()
    with torch.inference_mode():
        for inputs, _targets in loader:
            logits = loaded.model(inputs.to(loaded.device))
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=1).detach().cpu().numpy()
            histogram += np.bincount(predictions, minlength=num_classes)
            batch_entropy = (
                -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log())
                .sum(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            entropy_values.extend(float(value) for value in batch_entropy)
    total = float(histogram.sum())
    if total <= 0.0:
        raise RuntimeError("Could not build benign label histogram: no labels collected.")
    if not entropy_values:
        raise RuntimeError("Could not build benign entropy histogram: no entropies collected.")
    entropy_hist, _ = np.histogram(
        np.asarray(entropy_values, dtype=np.float64),
        bins=num_classes,
        range=(0.0, math.log(max(num_classes, 2))),
    )
    entropy_total = float(entropy_hist.sum())
    if entropy_total <= 0.0:
        raise RuntimeError("Could not build benign entropy histogram: empty histogram.")
    return num_classes, (histogram / total).tolist(), (entropy_hist / entropy_total).tolist()


def _attacker_surrogate_calibration(
    args: argparse.Namespace,
    *,
    cache_path: Path,
) -> dict[str, Any]:
    """Calibrate the attacker's own benign statistics for the adaptive objective.

    A FlowGuard++-adaptive attacker needs ``(mu, sigma)`` and a tolerated band
    per component to standardize its penalty. It estimates them the same way the
    defender does, but on *its own* pool of benign-looking images
    (``--adaptive-attacker-pool``, by default the related-domain CIFAR-100 pool
    the transfer attacker already has) and with *its own* surrogate CNF. The
    defender's calibration set, thresholds, and operating point are never read.

    Results are cached because the likelihood pass over the pool is the second
    most expensive step in the whole matrix after attack training itself.
    """
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[attacker-calibration] restored from {cache_path}")
        return payload

    from flowguard.attacks.adaptive import (
        CompositeWeights,
        SurrogateVelocityRegularizer,
        calibrate_surrogate_stats,
    )

    pool_name = str(args.adaptive_attacker_pool)
    modelfamily = legacy_datasets.dataset_to_modelfamily[pool_name]
    transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    dataset = legacy_datasets.__dict__[pool_name](
        train=True,
        transform=transform,
        download=bool(args.download),
    )
    sample_count = min(int(args.adaptive_calibration_samples), len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(sample_count))),
        batch_size=int(args.adaptive_calibration_batch_size),
        shuffle=False,
        num_workers=0,
    )

    mean, std = legacy_datasets.modelfamily_to_mean_std[
        "imagenet" if modelfamily == "tinyimagenet" else modelfamily
    ].values()
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)

    batches: list[torch.Tensor] = []
    for inputs, _targets in loader:
        batches.append(torch.clamp(inputs * std_t + mean_t, 0.0, 1.0))

    regularizer = SurrogateVelocityRegularizer(
        str(args.a3_surrogate_checkpoint), device=str(args.device)
    )
    # The benign likelihood statistics must be estimated with the same flow the
    # attack will optimize against, or the attacker standardizes its penalty
    # against a distribution it never sees again.
    likelihood_path = getattr(args, "a3_likelihood_surrogate_checkpoint", None)
    likelihood_regularizer = (
        SurrogateVelocityRegularizer(str(likelihood_path), device=str(args.device))
        if likelihood_path and str(likelihood_path) != str(args.a3_surrogate_checkpoint)
        else None
    )
    print(
        f"[attacker-calibration] scoring {sample_count} images from {pool_name} "
        f"with the attacker's surrogate CNF"
    )
    stats, columns = calibrate_surrogate_stats(
        regularizer,
        batches,
        integral_num_steps=int(args.adaptive_integral_steps),
        likelihood_num_steps=int(args.adaptive_likelihood_steps),
        hutchinson_samples=int(args.adaptive_hutchinson_samples),
        band_quantile=float(args.adaptive_band_quantile),
        composite_weights=CompositeWeights(),
        likelihood_regularizer=likelihood_regularizer,
    )
    payload: dict[str, Any] = {
        "pool": pool_name,
        "num_samples": sample_count,
        "benign_stats": {
            field: getattr(stats, field)
            for field in stats.__dataclass_fields__
            if getattr(stats, field) is not None
        },
        # Reference sample for the attacker's distributional (C4) term. The
        # composite column is what the defender's window test would buffer.
        "benign_reference_scores": columns["composite"],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[attacker-calibration] wrote {cache_path}")

    if bool(args.adaptive_distill_surrogate):
        payload["distilled_surrogate_path"] = _train_distilled_surrogate(
            args,
            regularizer=regularizer,
            batches=batches,
            stats=stats,
            output_path=cache_path.with_name("attacker_distilled_score.pt"),
            likelihood_regularizer=likelihood_regularizer,
        )
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _train_distilled_surrogate(
    args: argparse.Namespace,
    *,
    regularizer,
    batches: list[torch.Tensor],
    stats,
    output_path: Path,
    likelihood_regularizer=None,
) -> str:
    """Distill the attacker's CNF component scores into a fast regressor.

    Backpropagating the exact likelihood ODE costs one double-backward per
    solver step. Distillation replaces that with a single forward/backward,
    which is what makes the composite-adaptive attack affordable at the query
    budgets used for the extraction study.
    """
    from flowguard.attacks.score_distillation import (
        build_distillation_targets,
        train_score_surrogate,
    )

    print("[attacker-calibration] building distillation targets")
    images, targets = build_distillation_targets(
        regularizer,
        batches,
        stats=stats,
        integral_num_steps=int(args.adaptive_integral_steps),
        likelihood_num_steps=int(args.adaptive_likelihood_steps),
        hutchinson_samples=int(args.adaptive_hutchinson_samples),
        likelihood_regularizer=likelihood_regularizer,
    )
    surrogate, report = train_score_surrogate(
        images,
        targets,
        device=str(args.device),
        epochs=int(args.adaptive_distill_epochs),
        batch_size=int(args.adaptive_calibration_batch_size),
    )
    surrogate.save(output_path, stats=stats)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(
        f"[attacker-calibration] distilled surrogate -> {output_path} "
        f"(component r: {report.component_correlation})"
    )
    # A poorly fitted surrogate does not fail loudly: the generator still gets a
    # gradient, just of the wrong function, and the run completes with numbers
    # that look like an evaluation of the adaptive attack but are not. Refuse to
    # continue rather than spend the budget optimizing noise.
    minimum = float(args.adaptive_distill_min_correlation)
    correlations = {
        name: float(value)
        for name, value in (report.component_correlation or {}).items()
    }
    if minimum > 0.0 and correlations:
        weakest_name = min(correlations, key=lambda key: correlations[key])
        weakest = correlations[weakest_name]
        if weakest < minimum:
            raise RuntimeError(
                "Distilled score surrogate is too inaccurate to attack with: "
                f"component '{weakest_name}' correlates r={weakest:.3f} with the "
                f"exact CNF score (minimum {minimum:.3f}). The adaptive attacks "
                "would optimize noise. Raise --adaptive-calibration-samples / "
                "--adaptive-distill-epochs, drop --adaptive-distill-surrogate to "
                "use the exact (slower) gradient, or lower "
                "--adaptive-distill-min-correlation to accept this fit. "
                f"All components: {correlations}"
            )
    return str(output_path)


def _adaptive_common_extra(
    args: argparse.Namespace,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Attacker-side knobs shared by every FlowGuard++-adaptive attack."""
    extra: dict[str, Any] = {
        "benign_stats": calibration["benign_stats"],
        "integral_num_steps": int(args.adaptive_integral_steps),
        "likelihood_num_steps": int(args.adaptive_likelihood_steps),
        "likelihood_hutchinson_samples": int(args.adaptive_hutchinson_samples),
    }
    if calibration.get("distilled_surrogate_path"):
        extra["distilled_surrogate_path"] = calibration["distilled_surrogate_path"]
    if getattr(args, "a3_likelihood_surrogate_checkpoint", None):
        extra["likelihood_surrogate_checkpoint_path"] = str(
            args.a3_likelihood_surrogate_checkpoint
        )
    return extra


def _disguide_bypass_common_extra(
    args: argparse.Namespace,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared DisGUIDE-style knobs for D1/D2 bypass attacks.

    ``benign_stats`` is what lets the generator's C1 penalty use the
    benign-normalized hinge instead of the raw ``scores.mean()`` fallback. The
    raw form is not comparable to the extraction loss: benign ``||v(0, x)||^2``
    on the CIFAR-10 FlowPure CNF is ~2.5e-3, so at ``lambda=1`` the penalty is
    ~3 orders of magnitude below the ensemble-disagreement term and the
    generator is effectively unconstrained -- i.e. D1/D2 stop being adaptive to
    FlowPure at all. D3-D6 already received these stats via
    :func:`_adaptive_common_extra`; passing them here too puts every D-family
    attack on the same footing.
    """
    surrogate_checkpoint = args.a3_surrogate_checkpoint or args.flow_checkpoint
    benign_stats = dict((calibration or {}).get("benign_stats") or {})
    return {
        "benign_stats": benign_stats,
        "ensemble_size": int(args.disguide_bypass_ensemble_size),
        "latent_dim": int(args.cem_latent_dim),
        "g_iter": int(args.disguide_bypass_g_iter),
        "d_iter": int(args.disguide_bypass_d_iter),
        "rep_iter": int(args.disguide_bypass_rep_iter),
        "replay_size": int(args.disguide_replay_size),
        "lambda_div": float(args.disguide_bypass_lambda_div),
        "loss": "l1",
        "epoch_itrs": int(args.disguide_bypass_epoch_itrs),
        "scheduler": "none",
        "lr_student": float(args.disguide_bypass_lr_student),
        "lr_generator": float(args.disguide_bypass_lr_generator),
        "velocity_regularizer_weight": float(args.cem_lambda_detector),
        "surrogate_checkpoint_path": str(surrogate_checkpoint),
        "velocity_regularizer_device": str(args.device),
        "disable_pbar": not bool(args.verbose),
        "verbose": bool(args.verbose),
        # Bound the in-memory transfer records and snapshot training state
        # periodically. Both matter only for high-budget runs, where an
        # uncapped record store exhausts host memory and a single cell can
        # outlive its Slurm allocation.
        "artifact_sample_size": int(args.attack_artifact_sample_size),
        "checkpoint_every_queries": int(args.attack_checkpoint_every_queries),
        "resume_from_checkpoint": not bool(args.attack_no_resume),
        # Query-evolution snapshots for the qualitative figure. The scratch
        # directory is deleted per cell, so _run_one copies the file out.
        "query_snapshot_milestones": [
            int(value) for value in _split_csv(str(args.query_snapshot_milestones))
        ],
        "query_snapshot_images": int(args.query_snapshot_images),
        "plateau_patience_queries": int(args.attack_plateau_patience_queries),
        "plateau_min_delta": float(args.attack_plateau_min_delta),
        "plateau_min_queries": int(args.attack_plateau_min_queries),
        "plateau_smoothing_window": int(args.attack_plateau_smoothing_window),
    }


def _uses_disguide_bypass_loop(recipe: AttackRecipe) -> bool:
    if recipe.kind in {AttackKind.MAZE, AttackKind.DISGUIDE}:
        return True
    if recipe.kind in {AttackKind.LATENT_MANIFOLD, AttackKind.PROCEDURAL_NATURAL}:
        return str(recipe.extra.get("engine", "disguide")).lower() != "cem"
    return False


def _entropy_moments_from_histogram(
    entropy_histogram: list[float] | None,
    *,
    num_classes: int,
) -> tuple[float | None, float | None]:
    """Recover approximate benign entropy mean/std from the calibrated histogram.

    The histogram is built over ``num_classes`` equal-width bins spanning
    ``[0, log(num_classes)]``, so bin centers give a moment estimate accurate to
    within half a bin width. That is well inside the tolerance of a soft
    matching penalty, and it avoids threading raw entropy values through the
    calibration checkpoint format.
    """
    if not entropy_histogram:
        return None, None
    weights = np.asarray(entropy_histogram, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return None, None
    weights = weights / total
    edges = np.linspace(0.0, math.log(max(num_classes, 2)), len(weights) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = float((weights * centers).sum())
    variance = float((weights * (centers - mean) ** 2).sum())
    return mean, float(math.sqrt(max(variance, 0.0)))


def _attack_recipes(
    args: argparse.Namespace,
    *,
    adaptive_calibration: dict[str, Any] | None = None,
    benign_label_histogram: list[float] | None = None,
    benign_entropy_moments: tuple[float | None, float | None] = (None, None),
) -> dict[str, AttackRecipe]:
    adaptive_calibration = dict(adaptive_calibration or {"benign_stats": {}})
    entropy_mean, entropy_std = benign_entropy_moments
    if entropy_mean is not None:
        stats = dict(adaptive_calibration.get("benign_stats") or {})
        stats["entropy_mean"] = entropy_mean
        stats["entropy_std"] = entropy_std
        adaptive_calibration["benign_stats"] = stats
    maze_common = {
        "latent_dim": int(args.maze_latent_dim),
        "iter_gen": int(args.maze_iter_gen),
        "iter_clone": int(args.maze_iter_clone),
        "iter_exp": int(args.maze_iter_exp),
        "ndirs": int(args.maze_ndirs),
        "log_iter": int(args.maze_log_iter),
        "disable_pbar": True,
        "verbose": bool(args.verbose),
        "replay_buffer_size": int(args.maze_replay_buffer_size),
        "artifact_sample_size": int(args.attack_artifact_sample_size),
    }
    latent_generator_extra = (
        {"generator": str(args.latent_generator_checkpoint)}
        if args.latent_generator_checkpoint is not None
        else {}
    )
    diffusion_generator_extra = (
        {
            "generator_hf_model_id": str(args.diffusion_model_id),
            "diffusion_scheduler": str(args.diffusion_scheduler),
            "diffusion_steps": int(args.diffusion_steps),
            "diffusion_use_safetensors": True,
        }
        if args.diffusion_model_id
        else {}
    )
    cem_common = {
        "population_size": int(args.cem_population_size),
        "elite_fraction": float(args.cem_elite_fraction),
        "latent_dim": int(args.cem_latent_dim),
        "selector": str(args.cem_selector),
        "lambda_detector": float(args.cem_lambda_detector),
        "artifact_sample_size": int(args.attack_artifact_sample_size),
    }
    projection_extra = {
        "projection": {
            "name": str(args.projection_name),
            "strength": float(args.projection_strength),
            "steps": int(args.projection_steps),
            "differentiable": False,
        }
    }

    # Generator backing the FlowGuard++-adaptive attackers (D3-D6). They were
    # written on top of D2's procedural parameterization, which models natural
    # *color* image statistics. That is a poor fit for domains it was not
    # designed for (on MNIST it neither reaches useful fidelity nor evades the
    # CNF), and it then caps D3-D6 for reasons unrelated to their penalties.
    # Selecting D1's steered-diffusion generator instead measures what the
    # adaptive penalties cost on a generator that works in the domain.
    adaptive_generator = str(args.adaptive_generator).lower()
    if adaptive_generator == "conv":
        # Learned convolutional generator trained from scratch inside the attack
        # loop (the DFME/DisGUIDE generator). Unlike 'latent' it carries no
        # pretrained prior, so the attacker is assumed to know nothing about the
        # victim's data distribution -- data-free in the strict sense.
        adaptive_kind = AttackKind.LATENT_MANIFOLD
        adaptive_mode = "latent_manifold"
        adaptive_generator_extra: dict[str, Any] = {}
    elif adaptive_generator == "latent":
        adaptive_kind = AttackKind.LATENT_MANIFOLD
        adaptive_mode = "latent_manifold"
        adaptive_generator_extra: dict[str, Any] = {
            **latent_generator_extra,
            **diffusion_generator_extra,
            "diffusion_train_steps": int(args.diffusion_train_steps),
            "diffusion_steering_scale": float(args.diffusion_steering_scale),
            "diffusion_guidance_scale": float(args.diffusion_guidance_scale),
            "diffusion_guidance_start_frac": float(args.diffusion_guidance_start_frac),
        }
    else:
        adaptive_kind = AttackKind.PROCEDURAL_NATURAL
        adaptive_mode = "procedural_natural"
        adaptive_generator_extra = {
            "procedural_grid_size": int(args.procedural_grid_size),
            "procedural_fourier_modes": int(args.procedural_fourier_modes),
            "procedural_blobs": int(args.procedural_blobs),
        }
    return {
        "prada": AttackRecipe(
            key="prada",
            display_name="PRADA",
            kind=AttackKind.PRADA,
            mode="prada",
            query_dataset=args.query_dataset,
            extra={
                "initial_seed_size": int(args.seed_samples),
                "duplication_rounds": 2,
                "expansion_factor": 1,
                "max_iter": 50,
                "epsilon": 0.05,
                "hyperparameter_search_budget": 0,
                "verbose": bool(args.verbose),
            },
            notes="Iterative PRADA adversarial extraction baseline.",
        ),
        "maze": AttackRecipe(
            key="maze",
            display_name="MAZE",
            kind=AttackKind.MAZE,
            mode="maze",
            query_dataset=args.query_dataset,
            extra=dict(maze_common),
            notes="Unmodified MAZE data-free extraction baseline.",
        ),
        "disguide": AttackRecipe(
            key="disguide",
            display_name="DisGuide",
            kind=AttackKind.DISGUIDE,
            mode="disguide",
            query_dataset=args.query_dataset,
            extra={
                "ensemble_size": 2,
                "latent_dim": 16,
                "g_iter": 1,
                "d_iter": 1,
                "rep_iter": 1,
                "replay_size": int(args.disguide_replay_size),
                "epoch_itrs": 3,
                "disable_pbar": True,
                "verbose": bool(args.verbose),
                "artifact_sample_size": int(args.attack_artifact_sample_size),
        "plateau_patience_queries": int(args.attack_plateau_patience_queries),
        "plateau_min_delta": float(args.attack_plateau_min_delta),
        "plateau_min_queries": int(args.attack_plateau_min_queries),
            },
            notes="Unmodified DisGuide ensemble disagreement extraction baseline.",
        ),
        "transfer_naive": AttackRecipe(
            key="transfer_naive",
            display_name="TransferSet naive",
            kind=AttackKind.TRANSFER_SET,
            mode="naive",
            query_dataset=args.query_dataset,
            extra={"transfer_artifact_sample_size": int(args.attack_artifact_sample_size)},
            notes="Soft-label transfer-set extraction on the default query dataset.",
        ),
        "transfer_top1": AttackRecipe(
            key="transfer_top1",
            display_name="TransferSet top-1",
            kind=AttackKind.TRANSFER_SET,
            mode="top1",
            query_dataset=args.query_dataset,
            extra={"transfer_artifact_sample_size": int(args.attack_artifact_sample_size)},
            notes="Hard-label transfer-set extraction.",
        ),
        "transfer_s4l": AttackRecipe(
            key="transfer_s4l",
            display_name="TransferSet S4L",
            kind=AttackKind.TRANSFER_SET,
            mode="s4l",
            query_dataset=args.query_dataset,
            extra={"transfer_artifact_sample_size": int(args.attack_artifact_sample_size)},
            notes="Semi-supervised transfer-set extraction.",
        ),
        "transfer_smoothing": AttackRecipe(
            key="transfer_smoothing",
            display_name="TransferSet smoothing",
            kind=AttackKind.TRANSFER_SET,
            mode="smoothing",
            query_dataset=args.query_dataset,
            extra={"transfer_artifact_sample_size": int(args.attack_artifact_sample_size)},
            notes="Transfer-set extraction with repeated smoothed queries per image.",
        ),
        "flowguard_a1_clean_transfer": AttackRecipe(
            key="flowguard_a1_clean_transfer",
            display_name="FlowGuard A1 Clean-Transfer",
            kind=AttackKind.TRANSFER_SET,
            mode="naive",
            query_dataset=args.clean_transfer_dataset,
            extra={"transfer_artifact_sample_size": int(args.attack_artifact_sample_size)},
            notes="Clean external query pool intended to look benign to FlowPure.",
        ),
        "flowguard_a2_projected_maze": AttackRecipe(
            key="flowguard_a2_projected_maze",
            display_name="FlowGuard A2 ProjectedMAZE",
            kind=AttackKind.MAZE,
            mode="maze",
            query_dataset=args.query_dataset,
            extra={
                **maze_common,
                "denoise": {
                    "kind": args.a2_denoise_kind,
                    "sigma": float(args.a2_denoise_sigma),
                    "kernel_size": int(args.a2_denoise_kernel),
                    "mix": float(args.a2_denoise_mix),
                },
            },
            notes="MAZE with attacker-side query denoising before submission.",
        ),
        "flowguard_a3_flowblind": AttackRecipe(
            key="flowguard_a3_flowblind",
            display_name="FlowGuard A3 FlowBlind",
            kind=AttackKind.MAZE,
            mode="maze",
            query_dataset=args.query_dataset,
            extra={
                **maze_common,
                "surrogate_checkpoint_path": str(args.a3_surrogate_checkpoint),
                "velocity_regularizer_weight": float(args.a3_regularizer_weight),
                "velocity_regularizer_device": args.device,
                "velocity_regularizer_is_pixel_space": True,
                "record_surrogate_scores": True,
            },
            notes="MAZE regularized against a surrogate CNF velocity score.",
        ),
        "flowguard_adaptive_adaptive": AttackRecipe(
            key="flowguard_adaptive_adaptive",
            display_name="FlowGuard Adaptive-Adaptive",
            kind=AttackKind.MAZE,
            mode="maze",
            query_dataset=args.query_dataset,
            extra={
                **maze_common,
                "surrogate_checkpoint_path": str(args.a3_surrogate_checkpoint),
                "velocity_regularizer_weight": float(args.a3_regularizer_weight) * 5.0,
                "velocity_regularizer_device": args.device,
                "velocity_regularizer_is_pixel_space": True,
                "record_surrogate_scores": True,
                "denoise": {
                    "kind": args.a2_denoise_kind,
                    "sigma": float(args.a2_denoise_sigma),
                    "kernel_size": int(args.a2_denoise_kernel),
                    "mix": float(args.a2_denoise_mix),
                },
            },
            notes="Combined A2 denoising and A3 CNF regularization.",
        ),
        "flowguard_d1_latent_manifold": AttackRecipe(
            key="flowguard_d1_latent_manifold",
            display_name="FlowGuard D1 Latent-Manifold",
            kind=AttackKind.LATENT_MANIFOLD,
            mode="latent_manifold",
            query_dataset=args.query_dataset,
            extra={
                **_disguide_bypass_common_extra(args, adaptive_calibration),
                **diffusion_generator_extra,
                "diffusion_train_steps": int(args.diffusion_train_steps),
                "diffusion_steering_scale": float(args.diffusion_steering_scale),
                "diffusion_guidance_scale": float(args.diffusion_guidance_scale),
                "diffusion_guidance_start_frac": float(args.diffusion_guidance_start_frac),
            },
            notes=(
                "DisGUIDE loop with frozen pretrained DDPM, trainable latent steering, "
                "ensemble disagreement, and surrogate flow-penalty on the generator."
            ),
        ),
        "flowguard_d2_procedural_natural": AttackRecipe(
            key="flowguard_d2_procedural_natural",
            display_name="FlowGuard D2 Procedural-Natural",
            kind=AttackKind.PROCEDURAL_NATURAL,
            mode="procedural_natural",
            query_dataset=args.query_dataset,
            extra={
                **_disguide_bypass_common_extra(args, adaptive_calibration),
                "procedural_grid_size": int(args.procedural_grid_size),
                "procedural_fourier_modes": int(args.procedural_fourier_modes),
                "procedural_blobs": int(args.procedural_blobs),
            },
            notes=(
                "DisGUIDE loop with procedural-natural generator, ensemble disagreement, "
                "and surrogate flow-penalty on the generator."
            ),
        ),
        "flowguard_d3_projected_latent": AttackRecipe(
            key="flowguard_d3_projected_latent",
            display_name="FlowGuard D3 Projected Latent",
            kind=AttackKind.LATENT_MANIFOLD,
            mode="latent_manifold",
            query_dataset=args.query_dataset,
            extra={
                "engine": "cem",
                **cem_common,
                **latent_generator_extra,
                **diffusion_generator_extra,
                **projection_extra,
            },
            notes="D1 latent CEM with a projection stage applied before victim and defense evaluation.",
        ),
        "flowguard_d4_rejection_oracle": AttackRecipe(
            key="flowguard_d4_rejection_oracle",
            display_name="FlowGuard D4 Rejection Oracle",
            kind=AttackKind.REJECTION_ORACLE,
            mode="rejection_oracle",
            query_dataset=args.query_dataset,
            extra={
                **cem_common,
                **latent_generator_extra,
                **diffusion_generator_extra,
                "proposal": str(args.rejection_oracle_proposal),
                "accepted_only": True,
            },
            notes="Accepted/blocked-only CEM adaptation; labels are retained only for accepted queries.",
        ),
        # -------------------------------------------------------------------
        # FlowGuard++-adaptive attackers.
        #
        # D1/D2 above optimize only the t=0 velocity, i.e. they are adaptive to
        # single-point flow scoring. The four recipes below extend the same
        # generator objective to the components FlowGuard++ actually decides on,
        # so the defense is evaluated against attackers adapted to itself rather
        # than to the detector it replaces. All four share one generator,
        # selected by --adaptive-generator: D2's procedural parameterization by
        # default (the stronger of the two bypasses on CIFAR-10 in Table 2), or
        # D1's steered diffusion generator where the procedural one is a poor
        # fit for the domain and would cap these rows on its own.
        # -------------------------------------------------------------------
        "flowguard_d3_adaptive_c1c2": AttackRecipe(
            key="flowguard_d3_adaptive_c1c2",
            display_name="D3 adaptive C1+C2",
            kind=adaptive_kind,
            mode=adaptive_mode,
            query_dataset=args.query_dataset,
            extra={
                **_disguide_bypass_common_extra(args, adaptive_calibration),
                **_adaptive_common_extra(args, adaptive_calibration),
                **adaptive_generator_extra,
                "integral_regularizer_weight": float(args.adaptive_lambda_integral),
            },
            notes=(
                "Generator penalized on the surrogate t=0 velocity and the "
                "surrogate trajectory-integral energy (C1+C2)."
            ),
        ),
        "flowguard_d4_adaptive_c1c3": AttackRecipe(
            key="flowguard_d4_adaptive_c1c3",
            display_name="D4 adaptive C1+C3",
            kind=adaptive_kind,
            mode=adaptive_mode,
            query_dataset=args.query_dataset,
            extra={
                **_disguide_bypass_common_extra(args, adaptive_calibration),
                **_adaptive_common_extra(args, adaptive_calibration),
                **adaptive_generator_extra,
                "likelihood_regularizer_weight": float(args.adaptive_lambda_likelihood),
            },
            notes=(
                "Generator penalized on the surrogate t=0 velocity and the "
                "two-sided likelihood-typicality score (C1+C3), the component "
                "that carries detection under Sybil fan-out."
            ),
        ),
        "flowguard_d5_adaptive_composite": AttackRecipe(
            key="flowguard_d5_adaptive_composite",
            display_name="D5 adaptive composite",
            kind=adaptive_kind,
            mode=adaptive_mode,
            query_dataset=args.query_dataset,
            extra={
                **_disguide_bypass_common_extra(args, adaptive_calibration),
                **_adaptive_common_extra(args, adaptive_calibration),
                **adaptive_generator_extra,
                "composite_regularizer_weight": float(args.adaptive_lambda_composite),
                # The deployed detector blocks on an OR of per-component
                # thresholds and ranks on the fused risk, so this attacker
                # suppresses both rather than the fused risk alone.
                "integral_regularizer_weight": float(args.adaptive_lambda_integral),
                "likelihood_regularizer_weight": float(args.adaptive_lambda_likelihood),
            },
            notes=(
                "Generator penalized on the attacker's replica of the fused "
                "per-query risk R(x) = z0 + z_int + s_typ, plus the individual "
                "component hinges the detector blocks on."
            ),
        ),
        "flowguard_d6_adaptive_stateful": AttackRecipe(
            key="flowguard_d6_adaptive_stateful",
            display_name="D6 adaptive composite+stateful",
            kind=adaptive_kind,
            mode=adaptive_mode,
            query_dataset=args.query_dataset,
            extra={
                **_disguide_bypass_common_extra(args, adaptive_calibration),
                **_adaptive_common_extra(args, adaptive_calibration),
                **adaptive_generator_extra,
                "composite_regularizer_weight": float(args.adaptive_lambda_composite),
                "integral_regularizer_weight": float(args.adaptive_lambda_integral),
                "likelihood_regularizer_weight": float(args.adaptive_lambda_likelihood),
                "distribution_regularizer_weight": float(args.adaptive_lambda_distribution),
                "label_regularizer_weight": float(args.adaptive_lambda_label),
                "benign_reference_scores": adaptive_calibration.get(
                    "benign_reference_scores", []
                )[: int(args.adaptive_reference_sample_size)],
                "benign_label_histogram": list(benign_label_histogram or []),
            },
            notes=(
                "D5 plus batch-level objectives: match the benign score "
                "distribution (C4) and the benign label/entropy profile (C5)."
            ),
        ),
    }


def _defense_for_attack_run(
    defense: DefenseRecipe,
    *,
    defense_key: str,
    threshold: float | None,
    audit_only: bool,
) -> DefenseRecipe:
    """Clone a defense recipe for an attack cell in enforcing (non-audit) mode.

    Benign calibration always stays in audit-only mode so the full benign
    stream can be scored. Attack runs can block/reject at runtime when
    ``audit_only`` is false and a calibrated threshold is available.
    """
    if audit_only:
        return defense
    params = dict(defense.parameters)
    params["audit_only"] = False
    composite_thresholds = params.pop("composite_thresholds", None)
    if threshold is not None:
        if defense_key == "flowpure":
            params["velocity_threshold"] = float(threshold)
        elif defense_key == "flowguard_integral":
            params["integral_threshold"] = float(threshold)
        elif defense_key == "fdinet":
            params["detection_threshold"] = float(threshold)
        elif defense_key == "flow_matching":
            # The calibrated ``threshold`` is a quantile of the anomaly score the
            # defense emits. With a benign typicality reference that score is the
            # two-sided typicality statistic, so the operating point is a direct
            # upper threshold on it; otherwise fall back to the legacy lower-tail
            # log-likelihood threshold.
            if params.get("scoring") == "typicality" and params.get("benign_likelihood_mean") is not None:
                params["typicality_threshold"] = float(threshold)
            else:
                params["likelihood_threshold"] = _negated_likelihood_threshold(float(threshold))
        elif defense_key == "flowguard_composite":
            params["flowpure_threshold"] = float(threshold)
    if defense_key == "flowguard_composite" and isinstance(composite_thresholds, dict):
        params.update(composite_thresholds)
    return replace(defense, parameters=params)


def _defense_recipes(args: argparse.Namespace, calibration: CalibrationData) -> dict[str, DefenseRecipe]:
    # Recipe defaults stay audit-only so benign calibration can score the full stream.
    flow_base = {
        "fm_checkpoint_path": str(args.flow_checkpoint),
        "dataset_name": args.dataset,
        "device": args.device,
        "inputs_normalized": True,
        "audit_only": True,
    }
    composite_likelihood_threshold = -float("inf")
    benign_likelihood_mean: float | None = None
    benign_likelihood_std: float | None = None
    if calibration.likelihood_scores:
        likelihood_array = np.asarray(calibration.likelihood_scores, dtype=np.float64)
        composite_likelihood_threshold = float(
            np.quantile(likelihood_array, float(args.target_fpr))
        )
        # Two-sided typicality reference for likelihood-based detection: attacks
        # can be atypical with either lower OR higher likelihood than benign.
        benign_likelihood_mean = float(np.mean(likelihood_array))
        benign_likelihood_std = float(np.std(likelihood_array))
    # Benign reference for the FlowPure ||v(t=0)||^2 score (== composite t0 score),
    # used to standardize the composite's per-component anomaly scores.
    benign_t0_mean: float | None = None
    benign_t0_std: float | None = None
    if calibration.flowpure_scores:
        t0_array = np.asarray(calibration.flowpure_scores, dtype=np.float64)
        benign_t0_mean = float(np.mean(t0_array))
        benign_t0_std = float(np.std(t0_array))
    likelihood_checkpoint = str(
        args.likelihood_flow_checkpoint or args.flow_checkpoint
    )
    composite_parameters: dict[str, Any] = {
        **flow_base,
        "likelihood_fm_checkpoint_path": likelihood_checkpoint,
        "flowpure_threshold": float("inf"),
        "integral_threshold": float("inf"),
        "likelihood_threshold": composite_likelihood_threshold,
        "num_steps": int(args.integral_num_steps),
        "benign_reference_scores": calibration.flowpure_scores,
        "benign_label_histogram": calibration.label_histogram,
        "benign_entropy_histogram": calibration.entropy_histogram,
        "benign_likelihood_mean": benign_likelihood_mean,
        "benign_likelihood_std": benign_likelihood_std,
        "benign_t0_mean": benign_t0_mean,
        "benign_t0_std": benign_t0_std,
        "num_classes": int(calibration.num_classes),
        "score_ks_threshold": float(args.ks_threshold),
        "label_mmd_threshold": float(args.labelhist_threshold),
        "entropy_mmd_threshold": float(args.labelhist_threshold),
        "min_window": int(args.ks_min_window),
        "max_window": int(args.ks_max_window),
        "decision_mode": str(args.flowguardpp_decision_mode),
        "response_policy": str(args.flowguardpp_response_policy),
    }
    return {
        "prada": DefenseRecipe(
            key="prada",
            display_name="PRADA",
            query_defense="prada",
            parameters={"audit_only": True},
            notes="Growing-distance PRADA query detector.",
        ),
        "flowpure": DefenseRecipe(
            key="flowpure",
            display_name="FlowPure",
            query_defense="flowpure",
            parameters={**flow_base, "velocity_threshold": float("inf")},
            notes="Paper-style FlowPure score ||v(t=0, x)||^2.",
        ),
        "fdinet": DefenseRecipe(
            key="fdinet",
            display_name="FDINet",
            query_defense="fdinet",
            parameters={"audit_only": True, "target_fpr": float(args.target_fpr)},
            notes="Feature Distortion Index detector (operating point recalibrated on benign stream at target FPR).",
        ),
        "flowguard_integral": DefenseRecipe(
            key="flowguard_integral",
            display_name="FlowGuard++ C1 Integral",
            query_defense="flowguard_integral",
            parameters={
                **flow_base,
                "integral_threshold": float("inf"),
                "num_steps": int(args.integral_num_steps),
            },
            notes="Trajectory-integral CNF energy score.",
        ),
        "flowguard_userlevel": DefenseRecipe(
            key="flowguard_userlevel",
            display_name="FlowGuard++ C3 User-Level",
            query_defense="flowguard_userlevel",
            parameters={
                **flow_base,
                "benign_reference_scores": calibration.flowpure_scores,
                "ks_threshold": float(args.ks_threshold),
                "min_window": int(args.ks_min_window),
                "max_window": int(args.ks_max_window),
            },
            notes="KS test over a sliding user window of FlowPure scores.",
        ),
        "flowguard_labelhist": DefenseRecipe(
            key="flowguard_labelhist",
            display_name="FlowGuard++ C4 Label-Hist",
            query_defense="flowguard_labelhist",
            parameters={
                "num_classes": int(calibration.num_classes),
                "benign_label_histogram": calibration.label_histogram,
                "mmd_threshold": float(args.labelhist_threshold),
                "min_window": int(args.labelhist_min_window),
                "max_window": int(args.labelhist_max_window),
                "audit_only": True,
            },
            notes="MMD over the victim top-1 label histogram.",
        ),
        "flow_matching": DefenseRecipe(
            key="flow_matching",
            display_name="Flow Matching Likelihood",
            query_defense="flow_matching",
            parameters={
                **flow_base,
                # Density estimation needs the Gaussian-source CNF, not the
                # purification flow that drives the velocity components.
                "fm_checkpoint_path": likelihood_checkpoint,
                "likelihood_threshold": -float("inf"),
                "scoring": "typicality",
                "benign_likelihood_mean": benign_likelihood_mean,
                "benign_likelihood_std": benign_likelihood_std,
                "typicality_threshold": float("inf"),
            },
            notes=(
                "Reverse-flow ODE log-likelihood, scored as a two-sided "
                "typicality statistic |log p(x) - mu| / sigma against the benign set."
            ),
        ),
        "flowguard_composite": DefenseRecipe(
            key="flowguard_composite",
            display_name="FlowGuard++ Composite",
            query_defense="flowguard++",
            parameters=dict(composite_parameters),
            notes=(
                "Composite FlowGuard++ monitor (equal-weight standardized "
                "combination of FlowPure t0, trajectory integral, ODE "
                "log-likelihood, KS, label-TV, and entropy-TV signals)."
            ),
        ),
        # Same monitor; per-query component weights are learned offline via
        # cross-fitted logistic regression at metric time (see --composite-eval-learned).
        "flowguard_composite_learned": DefenseRecipe(
            key="flowguard_composite_learned",
            display_name="FlowGuard++ Composite (learned weights)",
            query_defense="flowguard++",
            parameters=dict(composite_parameters),
            notes=(
                "FlowGuard++ Composite with per-component weights learned by "
                "cross-fitted logistic regression (oracle/upper-bound vs the "
                "equal-weight default)."
            ),
        ),
    }


def _build_spec(
    *,
    name: str,
    recipe: AttackRecipe,
    defense: DefenseRecipe,
    args: argparse.Namespace,
    query_budget: int,
    skip_substitute_training: bool = False,
) -> Any:
    extra = dict(recipe.extra)
    if skip_substitute_training:
        extra["skip_substitute_training"] = True
        extra["transfer_artifact_sample_size"] = 0
    attack_batch_size = int(args.attack_batch_size)
    if _uses_disguide_bypass_loop(recipe):
        attack_batch_size = max(2, int(args.extraction_attack_batch_size))
    return build_experiment_spec(
        name=name,
        dataset=args.dataset,
        target_architecture=args.target_architecture,
        substitute_architecture=args.substitute_architecture,
        attack_kind=recipe.kind,
        attack_mode=recipe.mode,
        query_dataset=recipe.query_dataset,
        query_budget=query_budget,
        query_transfer_set_size=None,
        target_checkpoint_dir=str(args.target_checkpoint_dir),
        target_device=args.device,
        substitute_device=args.device,
        distributed=False,
        num_workers=1,
        num_clients=max(1, int(getattr(args, "sybil_identities", 1))),
        attack_batch_size=attack_batch_size,
        attack_extra=extra,
        training_batch_size=int(args.training_batch_size),
        epochs=int(args.epochs),
        query_download=bool(args.download),
        dataset_download=bool(args.download),
        query_defense=defense.query_defense,
        query_defense_parameters=defense.parameters,
        verbose=bool(args.verbose),
    )


def _finite_score(value: Any, *, default: float = 0.0) -> float:
    """Coerce defense metadata scores to finite floats for sklearn metrics."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    return score if np.isfinite(score) else float(default)


def _metadata_list(metadata: dict[str, Any], key: str, batch_size: int, default: float) -> list[float]:
    raw = metadata.get(key)
    if isinstance(raw, list):
        return [_finite_score(value, default=default) for value in raw]
    if raw is None:
        return [float(default) for _ in range(batch_size)]
    return [_finite_score(raw, default=default) for _ in range(batch_size)]


def _metadata_flags(metadata: dict[str, Any], key: str, batch_size: int) -> list[bool]:
    raw = metadata.get(key)
    if isinstance(raw, list):
        return [bool(value) for value in raw]
    if raw is None:
        return [False for _ in range(batch_size)]
    return [bool(raw) for _ in range(batch_size)]


def _metadata_flags(metadata: dict[str, Any], key: str, batch_size: int) -> list[bool]:
    raw = metadata.get(key)
    if isinstance(raw, list):
        return [bool(value) for value in raw]
    if raw is None:
        return [False for _ in range(batch_size)]
    return [bool(raw) for _ in range(batch_size)]


def _metadata_score_list(records: list[Any], metadata_key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        raw = metadata.get(metadata_key)
        if isinstance(raw, list):
            values.extend(_finite_score(value) for value in raw)
        elif raw is not None:
            batch_size = int(getattr(record, "batch_size", metadata.get("batch_size", 1)))
            values.extend(_finite_score(raw) for _ in range(max(batch_size, 1)))
    return values


def _calibrate_composite_thresholds(
    records: list[Any],
    *,
    target_fpr: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    t0_scores = _metadata_score_list(records, "flowpure_t0_score")
    integral_scores = _metadata_score_list(records, "trajectory_integral_score")
    likelihood_scores = _metadata_score_list(records, "likelihood_score")
    if t0_scores:
        thresholds["flowpure_threshold"] = float(
            np.quantile(np.asarray(t0_scores, dtype=np.float64), 1.0 - float(target_fpr))
        )
    if integral_scores:
        thresholds["integral_threshold"] = float(
            np.quantile(np.asarray(integral_scores, dtype=np.float64), 1.0 - float(target_fpr))
        )
    if likelihood_scores:
        thresholds["likelihood_threshold"] = float(
            np.quantile(np.asarray(likelihood_scores, dtype=np.float64), float(target_fpr))
        )
    return thresholds


def _negated_likelihood_threshold(calibrated_negated_threshold: float) -> float:
    return float(-calibrated_negated_threshold)


# Which quantity the C4 (user-level KS) column is ranked by. See the branch in
# _extract_samples for why the default changed.
_USERLEVEL_SCORE_SOURCE = "ks"


def set_userlevel_score_source(source: str) -> None:
    global _USERLEVEL_SCORE_SOURCE
    if source not in {"ks", "flowpure"}:
        raise ValueError(f"userlevel score source must be 'ks' or 'flowpure', got {source!r}")
    _USERLEVEL_SCORE_SOURCE = source


def _extract_samples(records: list[Any], *, defense_key: str, gt: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        batch_size = int(getattr(record, "batch_size", metadata.get("batch_size", 0)))
        if batch_size <= 0:
            continue

        # When several detectors audited one query stream (see
        # MultiAuditQueryDefense) their outputs are namespaced, because every
        # flow detector mirrors its score into the same flowpure_* keys and they
        # would otherwise overwrite each other. Overlay the requested
        # detector's own metadata so the per-defense branches below are
        # unchanged whether the run was shared or dedicated.
        scoped = metadata.get("by_defense")
        if isinstance(scoped, dict) and defense_key in scoped:
            metadata = {**metadata, **scoped[defense_key]}

        if defense_key == "prada":
            blocked = metadata.get("prada_blocked_indices")
            blocked_set = set(blocked) if isinstance(blocked, list) else set()
            flags = [index in blocked_set for index in range(batch_size)]
            scores = [1.0 if flag else 0.0 for flag in flags]
        elif defense_key == "fdinet":
            flags = _metadata_flags(metadata, "fdinet_flags", batch_size)
            scores = _metadata_list(metadata, "fdinet_scores", batch_size, 0.0)
        elif defense_key == "flowguard_integral":
            flags = _metadata_flags(metadata, "flowpure_blocked", batch_size)
            scores = _metadata_list(metadata, "flowguard_integral_scores", batch_size, 0.0)
        elif defense_key == "flowguard_userlevel":
            # Rank by the detector's own statistic. The previous behaviour ranked
            # by the per-query FlowPure score that merely *populates* the window,
            # which made C4's AUROC an exact duplicate of C1's in every published
            # table -- including the rows where the window never filled and no KS
            # test was evaluated at all. Only the F1 differed, because it used the
            # KS flag. Ranking by the window statistic makes the column report the
            # component it is labelled with; set ``userlevel_score_source`` to
            # "flowpure" to reproduce the earlier numbers.
            flags = _metadata_flags(metadata, "flowpure_blocked", batch_size)
            if _USERLEVEL_SCORE_SOURCE == "flowpure":
                scores = _metadata_list(metadata, "flowpure_scores", batch_size, 0.0)
            else:
                window_ks = _finite_score(metadata.get("flowguard_userlevel_ks", 0.0))
                scores = [window_ks for _ in range(batch_size)]
        elif defense_key == "flowguard_labelhist":
            flag = bool(metadata.get("flowguard_labelhist_flag", False))
            score = _finite_score(metadata.get("flowguard_labelhist_mmd", 0.0))
            flags = [flag for _ in range(batch_size)]
            scores = [score for _ in range(batch_size)]
        elif defense_key == "flowguard_composite":
            flags = _metadata_flags(metadata, "flowpure_blocked", batch_size)
            raw_scores = metadata.get("composite_anomaly_score")
            if isinstance(raw_scores, list):
                scores = [_finite_score(value) for value in raw_scores]
            else:
                raw_scores = metadata.get("flowpure_t0_score")
                if isinstance(raw_scores, list):
                    scores = [_finite_score(value) for value in raw_scores]
                else:
                    scores = _metadata_list(metadata, "flowpure_scores", batch_size, 0.0)
        elif defense_key == "flow_matching":
            flags = _metadata_flags(metadata, "flow_matching_blocked", batch_size)
            # Prefer the two-sided typicality anomaly score emitted by the defense;
            # fall back to the legacy lower-tail score (-log p(x)) if unavailable.
            anomaly_scores = metadata.get("flow_matching_scores")
            if isinstance(anomaly_scores, list):
                scores = [_finite_score(value) for value in anomaly_scores]
            else:
                raw_scores = metadata.get("queries_blocked")
                if isinstance(raw_scores, list):
                    scores = [_finite_score(-value) for value in raw_scores]
                else:
                    scores = [1.0 if flag else 0.0 for flag in flags]
        else:
            flags = _metadata_flags(metadata, "flowpure_blocked", batch_size)
            scores = _metadata_list(metadata, "flowpure_scores", batch_size, 0.0)

        length = min(batch_size, len(flags), len(scores))
        for index in range(length):
            samples.append(
                {
                    "gt": int(gt),
                    "pred": int(bool(flags[index])),
                    "score": _finite_score(scores[index]),
                }
            )
    return samples


def _force_threshold(samples: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "gt": int(sample["gt"]),
            "pred": int(float(sample["score"]) > threshold),
            "score": float(sample["score"]),
        }
        for sample in samples
    ]


def _score_threshold(samples: list[dict[str, Any]], *, target_fpr: float) -> float | None:
    if not samples:
        return None
    scores = np.asarray([float(sample["score"]) for sample in samples], dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return None
    return float(np.quantile(scores, 1.0 - float(target_fpr)))


def _compute_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "num_samples": 0,
            "num_benign": 0,
            "num_attack": 0,
            "auroc": None,
            "tpr": 0.0,
            "fpr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "score_mean_benign": None,
            "score_mean_attack": None,
        }
    y_true = np.asarray([int(sample["gt"]) for sample in samples], dtype=np.int64)
    y_pred = np.asarray([int(sample["pred"]) for sample in samples], dtype=np.int64)
    y_score = np.asarray([_finite_score(sample["score"]) for sample in samples], dtype=np.float64)
    finite_mask = np.isfinite(y_score)
    nonfinite_count = int(np.sum(~finite_mask))
    if nonfinite_count:
        print(
            f"[warn] _compute_metrics: {nonfinite_count} non-finite score(s); "
            "AUROC uses finite values only"
        )
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    auroc: float | None = None
    y_true_finite = y_true[finite_mask]
    y_score_finite = y_score[finite_mask]
    if (
        len(np.unique(y_true_finite)) > 1
        and y_score_finite.size > 0
        and len(np.unique(y_score_finite)) > 1
    ):
        auroc = float(roc_auc_score(y_true_finite, y_score_finite))
    benign_scores = y_score[(y_true == 0) & finite_mask]
    attack_scores = y_score[(y_true == 1) & finite_mask]
    return {
        "num_samples": int(len(samples)),
        "num_benign": int(np.sum(y_true == 0)),
        "num_attack": int(np.sum(y_true == 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "auroc": auroc,
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "score_mean_benign": float(benign_scores.mean()) if benign_scores.size else None,
        "score_mean_attack": float(attack_scores.mean()) if attack_scores.size else None,
    }


def _composite_component_matrix(records: list[Any]) -> tuple[np.ndarray, list[str]]:
    """Stack the per-query standardized component anomalies emitted by the composite.

    Returns an ``(n_queries, n_components)`` matrix and the component name order.
    """
    columns: dict[str, list[float]] = {}
    names: list[str] | None = None
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        component_scores = metadata.get("composite_component_scores")
        if not isinstance(component_scores, dict) or not component_scores:
            continue
        record_names = sorted(component_scores.keys())
        if names is None:
            names = record_names
        for name in record_names:
            columns.setdefault(name, []).extend(
                _finite_score(value) for value in component_scores[name]
            )
    if names is None:
        return np.empty((0, 0), dtype=np.float64), []
    lengths = {len(columns[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent composite component lengths: {lengths}")
    matrix = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in names]
    )
    return matrix, names


def _safe_folds(n_benign: int, n_attack: int, requested: int) -> int:
    return max(2, min(int(requested), min(n_benign, n_attack)))


def _cross_fitted_anomaly(
    features: np.ndarray, labels: np.ndarray, *, folds: int
) -> np.ndarray:
    """Out-of-fold logistic-regression decision scores = learned-weight anomaly.

    Cross-fitting (out-of-fold prediction) keeps the comparison honest: every
    query is scored by a model that never saw it in training, so the learned
    weights cannot trivially memorize the eval attacks. It is still an
    oracle/upper bound because the *same attack family* appears in train and
    test folds; use it to see how much headroom supervised weighting buys over
    the equal-weight default.
    """
    out_of_fold = np.zeros(labels.shape[0], dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for train_index, test_index in splitter.split(features, labels):
        model = LogisticRegression(max_iter=1000)
        model.fit(features[train_index], labels[train_index])
        out_of_fold[test_index] = model.decision_function(features[test_index])
    return out_of_fold


def _learned_component_weights(
    features: np.ndarray, labels: np.ndarray, names: list[str]
) -> dict[str, float]:
    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)
    coefficients = np.asarray(model.coef_, dtype=np.float64).ravel()
    return {name: float(weight) for name, weight in zip(names, coefficients, strict=True)}


def _compute_learned_composite_row(
    benign_records: list[Any],
    attack_records: list[Any],
    *,
    attack_meta: dict[str, Any],
    target_fpr: float,
    folds: int,
) -> dict[str, Any]:
    """Learned-weight composite metrics from cross-fitted per-component scores."""
    benign_matrix, benign_names = _composite_component_matrix(benign_records)
    attack_matrix, attack_names = _composite_component_matrix(attack_records)
    if benign_matrix.size == 0 or attack_matrix.size == 0:
        return {"error": "no composite component scores for learned weighting"}
    if benign_names != attack_names:
        return {"error": f"component mismatch benign={benign_names} attack={attack_names}"}
    n_benign = int(benign_matrix.shape[0])
    n_attack = int(attack_matrix.shape[0])
    if min(n_benign, n_attack) < 2:
        return {"error": "too few samples to learn composite weights"}
    features = np.vstack([benign_matrix, attack_matrix])
    labels = np.concatenate(
        [np.zeros(n_benign, dtype=np.int64), np.ones(n_attack, dtype=np.int64)]
    )
    folds_effective = _safe_folds(n_benign, n_attack, folds)
    out_of_fold = _cross_fitted_anomaly(features, labels, folds=folds_effective)
    benign_scores = out_of_fold[:n_benign]
    attack_scores = out_of_fold[n_benign:]
    threshold = float(np.quantile(benign_scores, 1.0 - float(target_fpr)))
    predictions = (out_of_fold > threshold).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {
        "error": None,
        "auroc": float(roc_auc_score(labels, out_of_fold)),
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "num_benign": n_benign,
        "num_attack": n_attack,
        "score_mean_benign": float(benign_scores.mean()) if benign_scores.size else None,
        "score_mean_attack": float(attack_scores.mean()) if attack_scores.size else None,
        "learned_component_weights": _learned_component_weights(features, labels, benign_names),
        "learned_folds": int(folds_effective),
        "accuracy": attack_meta.get("accuracy"),
        "fidelity": attack_meta.get("fidelity"),
        "elapsed_seconds": attack_meta.get("elapsed_seconds"),
    }


def _rescue_query_snapshots(
    attack_metadata: Any,
    *,
    destination: Path,
) -> None:
    """Copy an attack's query snapshots out of the scratch dir before deletion."""
    if not isinstance(attack_metadata, dict):
        return
    source = attack_metadata.get("query_snapshots_path")
    if not source:
        return
    source_path = Path(source)
    if not source_path.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)


def _query_curve(attack_meta: dict[str, Any]) -> list[dict[str, float]] | None:
    """Return the attack's per-epoch (queries, accuracy, fidelity) history.

    Only the DisGUIDE-loop attacks (D1-D6) record one; everything else returns
    ``None`` rather than an empty list so a missing curve is distinguishable
    from an attack that ran zero epochs.
    """
    metadata = attack_meta.get("attack_metadata")
    if not isinstance(metadata, dict):
        return None
    curve = metadata.get("disguide_metrics")
    if not isinstance(curve, list) or not curve:
        return None
    return [
        {
            key: float(value)
            for key, value in point.items()
            if isinstance(value, (int, float))
        }
        for point in curve
        if isinstance(point, dict)
    ]


def _collect_metadata_series(records: list[Any], key: str) -> np.ndarray:
    """Concatenate a per-query metadata list field across all batch records."""
    values: list[float] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        series = metadata.get(key)
        if isinstance(series, (list, tuple)):
            values.extend(_finite_score(item) for item in series)
    return np.asarray(values, dtype=np.float64)


def _dump_composite_component_scores(
    *,
    output_root: Path,
    run_label: str,
    attack_key: str,
    benign_records: list[Any],
    attack_records: list[Any],
) -> Path | None:
    """Persist per-query composite scores (benign + attack) to an ``.npz``.

    The dump pairs the standardized component matrix used for fusion with the
    raw t0 / integral / log-likelihood signals, so post-hoc plots (per-component
    distributions, the log-likelihood typicality-failure histogram, the
    component correlation matrix, and leave-one-out fusion ablations) can be
    regenerated offline without re-running the attack/defense matrix.
    """
    benign_matrix, benign_names = _composite_component_matrix(benign_records)
    attack_matrix, attack_names = _composite_component_matrix(attack_records)
    if benign_matrix.size == 0 or attack_matrix.size == 0:
        return None
    if benign_names != attack_names:
        return None
    scores_dir = output_root / f"composite_scores_{_safe_name(run_label.lower())}"
    scores_dir.mkdir(parents=True, exist_ok=True)
    out_path = scores_dir / f"{_safe_name(attack_key)}.npz"
    payload: dict[str, np.ndarray] = {
        "component_names": np.asarray(benign_names),
        "benign_components": benign_matrix,
        "attack_components": attack_matrix,
    }
    # alias -> defense metadata key carrying the raw (un-standardized) per-query signal
    raw_keys = {
        "t0": "flowpure_t0_score",
        "integral": "trajectory_integral_score",
        "loglik": "likelihood_score",
        "composite": "composite_anomaly_score",
    }
    for alias, metadata_key in raw_keys.items():
        payload[f"benign_raw_{alias}"] = _collect_metadata_series(benign_records, metadata_key)
        payload[f"attack_raw_{alias}"] = _collect_metadata_series(attack_records, metadata_key)
    np.savez_compressed(out_path, **payload)
    return out_path


def _run_one(
    *,
    recipe: AttackRecipe,
    defense: DefenseRecipe,
    args: argparse.Namespace,
    scratch_root: Path,
    query_budget: int,
    gt: int,
    skip_substitute_training: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None, list[Any]]:
    run_name = _safe_name(f"{defense.key}__{recipe.key}")
    if gt == 0:
        run_name = _safe_name(f"{defense.key}__benign_reference")
    run_dir = scratch_root / run_name
    started = time.perf_counter()
    succeeded = False
    try:
        spec = _build_spec(
            name=run_name,
            recipe=recipe,
            defense=defense,
            args=args,
            query_budget=query_budget,
            skip_substitute_training=skip_substitute_training,
        )
        result = run_experiment(spec, output_dir=run_dir)
        samples = _extract_samples(
            result.query_summary.history.records,
            defense_key=defense.key,
            gt=gt,
        )
        metrics = {
            "total_queries": int(result.query_summary.total_queries),
            "total_batches": int(result.query_summary.total_batches),
            "elapsed_seconds": float(time.perf_counter() - started),
            "accuracy": result.evaluation_summary.metrics.get("accuracy"),
            "fidelity": result.evaluation_summary.metrics.get("fidelity"),
            "joint_accuracy": result.evaluation_summary.metrics.get("joint_accuracy"),
            "attack_output_dir": result.attack_summary.output_dir,
            "attack_metadata": result.attack_summary.metadata,
        }
        _rescue_query_snapshots(
            metrics.get("attack_metadata"),
            destination=scratch_root.parent / "query_snapshots" / f"{run_name}.pt",
        )
        succeeded = True
        return samples, metrics, None, list(result.query_summary.history.records)
    except Exception as exc:  # noqa: BLE001 - report every failed cell.
        traceback.print_exc()
        print(f"[error] {run_name}: {exc!r}", flush=True)
        return [], {"elapsed_seconds": float(time.perf_counter() - started)}, repr(exc), []
    finally:
        # Only discard the scratch directory when the cell actually succeeded.
        # It holds attack_checkpoint.pt, so deleting it on the failure path
        # destroys the very state a retry would resume from -- which turned four
        # crashed resumes into four runs that had to start again from zero.
        if succeeded and run_dir.exists():
            shutil.rmtree(run_dir)
        elif run_dir.exists():
            print(
                f"[error] keeping {run_dir} for resume (contains attack state)",
                flush=True,
            )


def _write_report(
    *,
    output_root: Path,
    args: argparse.Namespace,
    attack_order: list[str],
    defense_order: list[str],
    attacks: dict[str, AttackRecipe],
    defenses: dict[str, DefenseRecipe],
    rows: list[dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
    progress: MatrixProgress | None = None,
    global_calibration: dict[str, Any] | None = None,
) -> None:
    run_label = _safe_name(str(args.run_label).lower())
    artifact_paths = _matrix_artifact_paths(output_root, run_label)
    report_path = artifact_paths["report"]
    summary_path = artifact_paths["summary"]
    lookup = {(row["defense"], row["attack"]): row for row in rows}
    title = "Smoke" if run_label == "smoke" else "Full"
    description = (
        [
            "This is a wiring smoke test with the full requested matrix and small query budgets.",
            "Scientific conclusions require the later full-budget run.",
        ]
        if run_label == "smoke"
        else [
            "This is the full-budget run of the requested attack/defense matrix.",
            "Per-run binary artifacts are deleted after textual metrics are collected.",
        ]
    )
    lines: list[str] = [
        f"# Attack/Defense Matrix {title} Report",
        "",
        *description,
        "",
    ]
    if progress is not None:
        last = progress.last_completed or {}
        lines.extend(
            [
                "## Progress",
                "",
                f"- Status: `{progress.status}`",
                f"- Updated at (UTC): `{progress.updated_at or 'n/a'}`",
                f"- Last completed: `{last.get('phase', 'n/a')}` "
                f"defense=`{last.get('defense', 'n/a')}` attack=`{last.get('attack', 'n/a')}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Configuration",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Query dataset: `{args.query_dataset}`",
        f"- Clean-transfer dataset: `{args.clean_transfer_dataset}`",
        f"- Query budget per attack: `{args.query_budget}`",
        f"- Benign calibration budget per defense: `{args.benign_query_budget}`",
        f"- Target FPR for calibrated score defenses: `{args.target_fpr}`",
        f"- Device: `{args.device}`",
        f"- Target checkpoint: `{args.target_checkpoint_dir}`",
        f"- Flow checkpoint: `{args.flow_checkpoint}`",
        "",
        "## Calibration",
        "",
        "| defense | threshold/statistic | benign samples | notes |",
        "|---|---:|---:|---|",
        ]
    )
    for defense_key in defense_order:
        calibration = calibrations.get(defense_key, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    defense_key,
                    _fmt(calibration.get("threshold")),
                    str(calibration.get("num_benign", "n/a")),
                    str(calibration.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## AUROC Matrix", "", "| attack \\ defense | " + " | ".join(defense_order) + " |"])
    lines.append("|" + "---|" * (len(defense_order) + 1))
    for attack_key in attack_order:
        cells = []
        for defense_key in defense_order:
            row = lookup.get((defense_key, attack_key), {})
            cells.append(_fmt(row.get("auroc")))
        lines.append(f"| {attack_key} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Detection Rate Matrix", "", "| attack \\ defense | " + " | ".join(defense_order) + " |"])
    lines.append("|" + "---|" * (len(defense_order) + 1))
    for attack_key in attack_order:
        cells = []
        for defense_key in defense_order:
            row = lookup.get((defense_key, attack_key), {})
            cells.append(_fmt(row.get("tpr")))
        lines.append(f"| {attack_key} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Full Results",
            "",
            "| defense | attack | status | AUROC | TPR | FPR | F1 | benign n | attack n | score mean benign | score mean attack | accuracy | fidelity | seconds |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        status = "ok" if not row.get("error") else f"error: `{row['error']}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["defense"],
                    row["attack"],
                    status,
                    _fmt(row.get("auroc")),
                    _fmt(row.get("tpr")),
                    _fmt(row.get("fpr")),
                    _fmt(row.get("f1")),
                    str(row.get("num_benign", 0)),
                    str(row.get("num_attack", 0)),
                    _fmt(row.get("score_mean_benign")),
                    _fmt(row.get("score_mean_attack")),
                    _fmt(row.get("accuracy"), digits=2),
                    _fmt(row.get("fidelity"), digits=2),
                    _fmt(row.get("elapsed_seconds"), digits=1),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Attack Notes", ""])
    for attack_key in attack_order:
        lines.append(f"- `{attack_key}`: {attacks[attack_key].notes}")
    lines.extend(["", "## Defense Notes", ""])
    for defense_key in defense_order:
        lines.append(f"- `{defense_key}`: {defenses[defense_key].notes}")

    _atomic_write_text(report_path, "\n".join(lines) + "\n")
    payload: dict[str, Any] = {
        "config": {
            "run_label": run_label,
            "dataset": args.dataset,
            "query_dataset": args.query_dataset,
            "clean_transfer_dataset": args.clean_transfer_dataset,
            "query_budget": args.query_budget,
            "benign_query_budget": args.benign_query_budget,
            "target_fpr": args.target_fpr,
            "device": args.device,
            "target_checkpoint_dir": str(args.target_checkpoint_dir),
            "flow_checkpoint": str(args.flow_checkpoint),
            "likelihood_flow_checkpoint": str(
                getattr(args, "likelihood_flow_checkpoint", "") or ""
            ),
            # Without this a distributed run is indistinguishable from a
            # single-client one once the job has finished, so no table cell can
            # be attributed to an identity count after the fact.
            "sybil_identities": int(getattr(args, "sybil_identities", 1)),
            "userlevel_score_source": str(getattr(args, "userlevel_score_source", "ks")),
            # D1's decode knobs. A sweep over the guidance scale is otherwise
            # indistinguishable from eight repeats of the same run once the
            # jobs have finished -- which is exactly what a wrapper that does
            # not forward --diffusion-guidance-scale produces.
            "diffusion_steps": int(getattr(args, "diffusion_steps", 0)),
            "diffusion_steering_scale": float(
                getattr(args, "diffusion_steering_scale", 0.0)
            ),
            "diffusion_guidance_scale": float(
                getattr(args, "diffusion_guidance_scale", 0.0)
            ),
            "diffusion_guidance_start_frac": float(
                getattr(args, "diffusion_guidance_start_frac", 0.5)
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
        },
        "calibrations": calibrations,
        "results": rows,
    }
    if global_calibration is not None:
        payload["global_calibration"] = global_calibration
    if progress is not None:
        payload["progress"] = {
            "status": progress.status,
            "last_completed": progress.last_completed,
            "updated_at": progress.updated_at,
        }
    _atomic_write_text(summary_path, json.dumps(payload, indent=2))
    print(f"[report] wrote {summary_path}")
    print(f"[report] wrote {report_path}")


def _persist_matrix_state(
    *,
    output_root: Path,
    args: argparse.Namespace,
    attack_order: list[str],
    defense_order: list[str],
    attacks: dict[str, AttackRecipe],
    defenses: dict[str, DefenseRecipe],
    rows: list[dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
    progress: MatrixProgress,
    global_calibration: dict[str, Any] | None = None,
) -> None:
    _write_report(
        output_root=output_root,
        args=args,
        attack_order=attack_order,
        defense_order=defense_order,
        attacks=attacks,
        defenses=defenses,
        rows=rows,
        calibrations=calibrations,
        progress=progress,
        global_calibration=global_calibration,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", default="smoke",
                        help="Label used in report filenames, e.g. smoke or full.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "attack_defense_matrix_smoke")
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--defenses", default=",".join(DEFAULT_DEFENSES))
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--query-dataset", default="CIFAR10")
    parser.add_argument("--clean-transfer-dataset", default="CIFAR100")
    parser.add_argument("--target-architecture", default="vgg16_bn")
    parser.add_argument("--substitute-architecture", default="vgg16_bn")
    parser.add_argument(
        "--target-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "notebook" / "training-victim-cifar10-vgg16_bn-nodefense" / "target_model",
    )
    parser.add_argument("--flow-checkpoint", type=Path, default=None,
                        help="CNF for the velocity-based components (C1 t=0 velocity, "
                             "C2 trajectory integral). Must be a FlowPure-family "
                             "checkpoint (x_0 = adversarial, x_1 = clean); that flow's "
                             "velocity is the detection signal.")
    parser.add_argument("--likelihood-flow-checkpoint", type=Path, default=None,
                        help="CNF for the likelihood-based components (C3 typicality, "
                             "and the flow_matching defense). Must be Gaussian-source, "
                             "since log p(x) is computed against a standard-Gaussian "
                             "p_0. Defaults to --flow-checkpoint, which is correct only "
                             "if that checkpoint is itself Gaussian-source -- a "
                             "FlowPure-family checkpoint cannot serve both roles.")
    parser.add_argument("--a3-surrogate-checkpoint", type=Path, default=None,
                        help="Attacker's surrogate CNF for the velocity components. "
                             "Defaults to --flow-checkpoint (strongest white-box "
                             "threat model: the attacker holds the deployed flow).")
    parser.add_argument("--a3-likelihood-surrogate-checkpoint", type=Path, default=None,
                        help="Attacker's surrogate CNF for the C3 typicality term. "
                             "Must be Gaussian-source for the same reason as "
                             "--likelihood-flow-checkpoint. Defaults to that flag, "
                             "so the attacker mirrors the detector it is adapting to.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--query-budget", type=int, default=8)
    parser.add_argument("--benign-query-budget", type=int, default=8)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument(
        "--audit-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Audit-only mode (default): defenses score queries but do not block. "
            "Pass --no-audit-only for enforcing mode where defenses reject "
            "suspicious queries at runtime (benign calibration stays audit-only)."
        ),
    )
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--extraction-attack-batch-size", type=int, default=4)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--sybil-identities",
        type=int,
        default=1,
        help=(
            "Number of attacker identities to spread queries across (Sybil "
            "sweep). 1 = single identity (baseline). Larger values fragment the "
            "per-user evidence seen by stateful defenses (KS window, label "
            "histogram, composite)."
        ),
    )
    parser.add_argument("--seed-samples", type=int, default=4)
    parser.add_argument("--attack-artifact-sample-size", type=int, default=8)

    parser.add_argument("--maze-latent-dim", type=int, default=16)
    parser.add_argument("--maze-iter-gen", type=int, default=1)
    parser.add_argument("--maze-iter-clone", type=int, default=2)
    parser.add_argument("--maze-iter-exp", type=int, default=1)
    parser.add_argument("--maze-ndirs", type=int, default=1)
    parser.add_argument("--maze-log-iter", type=int, default=50)
    parser.add_argument("--maze-replay-buffer-size", type=int, default=512)
    parser.add_argument("--disguide-replay-size", type=int, default=64)
    parser.add_argument("--disguide-bypass-ensemble-size", type=int, default=2)
    parser.add_argument("--disguide-bypass-g-iter", type=int, default=1)
    parser.add_argument("--disguide-bypass-d-iter", type=int, default=1)
    parser.add_argument("--disguide-bypass-rep-iter", type=int, default=1)
    parser.add_argument("--disguide-bypass-lambda-div", type=float, default=0.2)
    parser.add_argument("--disguide-bypass-epoch-itrs", type=int, default=4)
    parser.add_argument("--disguide-bypass-lr-student", type=float, default=0.03)
    parser.add_argument("--disguide-bypass-lr-generator", type=float, default=1e-4)

    parser.add_argument("--a2-denoise-kind", default="gaussian_blur", choices=["none", "gaussian_blur", "box_blur", "pixel_soft"])
    parser.add_argument("--a2-denoise-sigma", type=float, default=0.8)
    parser.add_argument("--a2-denoise-kernel", type=int, default=5)
    parser.add_argument("--a2-denoise-mix", type=float, default=1.0)
    parser.add_argument("--a3-regularizer-weight", type=float, default=0.01)

    parser.add_argument("--latent-generator-checkpoint", type=Path, default=None)
    parser.add_argument("--diffusion-model-id", default="google/ddpm-cifar10-32")
    parser.add_argument("--diffusion-scheduler", default="ddim",
                        choices=["ddpm", "ddim", "pndm"])
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument(
        "--diffusion-train-steps",
        type=int,
        default=1,
        help="Generator training steps per DisGUIDE inner loop for D1 (steering updates).",
    )
    parser.add_argument(
        "--diffusion-steering-scale",
        type=float,
        default=1.0,
        help="Residual scale for the trainable latent steering map in D1.",
    )
    parser.add_argument(
        "--diffusion-guidance-scale",
        type=float,
        default=0.0,
        help=(
            "Per-denoising-step detector-guidance scale for D1's frozen DDPM "
            "decode. 0.0 (default) disables guidance and reproduces the "
            "steering-only behavior."
        ),
    )
    parser.add_argument(
        "--diffusion-guidance-start-frac",
        type=float,
        default=0.5,
        help=(
            "Fraction of the D1 decode trajectory (from t=1 toward t=0) after "
            "which per-step detector guidance is applied."
        ),
    )
    parser.add_argument("--cem-population-size", type=int, default=16)
    parser.add_argument("--cem-elite-fraction", type=float, default=0.25)
    parser.add_argument("--cem-latent-dim", type=int, default=128)
    parser.add_argument("--cem-selector", default="entropy",
                        choices=["entropy", "class_rarity", "low_detector"])
    parser.add_argument("--cem-lambda-detector", type=float, default=1.0)
    parser.add_argument("--projection-name", default="gaussian_blur",
                        choices=["none", "gaussian_blur", "pixel_soft", "diffusion_denoise", "autoencoder", "cnf_purify"])
    parser.add_argument("--projection-strength", type=float, default=0.1)
    parser.add_argument("--projection-steps", type=int, default=5)
    parser.add_argument("--procedural-grid-size", type=int, default=4)
    parser.add_argument("--procedural-fourier-modes", type=int, default=3)
    parser.add_argument("--procedural-blobs", type=int, default=4)
    parser.add_argument("--procedural-lambda-tv", type=float, default=0.01)
    parser.add_argument("--procedural-lambda-freq", type=float, default=0.01)
    parser.add_argument("--procedural-lambda-color", type=float, default=0.01)
    parser.add_argument("--rejection-oracle-proposal", default="latent", choices=["latent", "procedural"])

    # --- FlowGuard++-adaptive attackers (D3-D6) ----------------------------
    adaptive = parser.add_argument_group(
        "FlowGuard++-adaptive attackers",
        "Attackers that optimize against the composite's own components rather "
        "than only against the t=0 velocity score.",
    )
    adaptive.add_argument(
        "--adaptive-attacker-pool", default="CIFAR100",
        help="Benign-looking image pool the attacker owns and calibrates on. "
             "Must not be the defender's calibration set.",
    )
    adaptive.add_argument(
        "--adaptive-generator", default="procedural",
        choices=["procedural", "latent", "conv"],
        help="Generator backing D3-D6. 'procedural' keeps D2's parameterization "
             "(the published configuration, but it cannot reach natural images "
             "in any domain tested); 'latent' uses D1's steered diffusion "
             "generator, which reaches far higher fidelity but assumes a "
             "pretrained prior for the victim's domain; 'conv' uses the learned "
             "DFME/DisGUIDE generator, which carries no prior and so is "
             "data-free in the strict sense.",
    )
    adaptive.add_argument("--adaptive-calibration-samples", type=int, default=2048)
    adaptive.add_argument("--adaptive-calibration-batch-size", type=int, default=64)
    adaptive.add_argument("--adaptive-band-quantile", type=float, default=0.95,
                          help="Benign quantile the attacker treats as the tolerated band.")
    adaptive.add_argument("--adaptive-integral-steps", type=int, default=8,
                          help="Euler steps for the attacker's trajectory-integral surrogate.")
    adaptive.add_argument("--adaptive-likelihood-steps", type=int, default=4,
                          help="Euler steps for the attacker's likelihood surrogate. Each step "
                               "costs a double-backward when optimized directly.")
    adaptive.add_argument("--adaptive-hutchinson-samples", type=int, default=1)
    adaptive.add_argument("--adaptive-lambda-integral", type=float, default=1.0)
    adaptive.add_argument("--adaptive-lambda-likelihood", type=float, default=1.0)
    adaptive.add_argument("--adaptive-lambda-composite", type=float, default=1.0)
    adaptive.add_argument("--adaptive-lambda-distribution", type=float, default=1.0)
    adaptive.add_argument("--adaptive-lambda-label", type=float, default=1.0)
    adaptive.add_argument("--adaptive-reference-sample-size", type=int, default=512,
                          help="Benign score sample the attacker matches its batch "
                               "distribution against (C4 term).")
    adaptive.add_argument("--adaptive-distill-surrogate", action="store_true",
                          help="Distill the CNF component scores into a fast regressor so the "
                               "generator gets composite gradients at one forward/backward "
                               "per step instead of one double-backward per ODE step.")
    adaptive.add_argument("--adaptive-distill-epochs", type=int, default=30)
    adaptive.add_argument(
        "--adaptive-distill-min-correlation", type=float, default=0.0,
        help="Abort if any distilled component correlates below this with the "
             "exact CNF score. The distilled surrogate replaces the true "
             "gradient, so a bad fit silently turns the adaptive attacks into "
             "noise optimization. 0 (default) disables the check.",
    )
    adaptive.add_argument(
        "--attack-checkpoint-every-queries", type=int, default=0,
        help="Snapshot generator/ensemble/optimizer/replay state every N queries so a "
             "cell that outlives its wall-clock limit resumes instead of restarting. "
             "0 disables. Only the DisGUIDE-loop attacks (D1-D6) support this.",
    )
    adaptive.add_argument(
        "--attack-no-resume", action="store_true",
        help="Ignore an existing attack_checkpoint.pt and train from scratch.",
    )

    parser.add_argument(
        "--attack-plateau-patience-queries", type=int, default=0,
        help="Stop a data-free attack early once its substitute fidelity has "
             "gone this many queries without improving by --attack-plateau-min-delta. "
             "0 (default) always spends the full budget.",
    )
    parser.add_argument("--attack-plateau-min-delta", type=float, default=1.0,
                        help="Fidelity gain (percentage points) that counts as progress.")
    parser.add_argument("--attack-plateau-min-queries", type=int, default=0,
                        help="Never stop early before this many queries.")
    parser.add_argument(
        "--attack-plateau-smoothing-window", type=int, default=1,
        help="Evaluations averaged before testing for a plateau. Raise it when "
             "epoch-to-epoch fidelity noise exceeds --attack-plateau-min-delta "
             "(CIFAR-10 swings +-2-3pp), or one lucky epoch sets a best the "
             "honest trend cannot beat and the run is killed mid-progress.",
    )
    parser.add_argument(
        "--query-snapshot-milestones", default="",
        help="Comma-separated query counts at which to save the queries in "
             "flight (e.g. '1,1000,10000,100000') for the query-evolution "
             "figure. Empty disables snapshotting.",
    )
    parser.add_argument("--query-snapshot-images", type=int, default=8,
                        help="Images kept per snapshot milestone.")
    parser.add_argument(
        "--userlevel-score-source",
        choices=("ks", "flowpure"),
        default="ks",
        help=(
            "What the C4 column is ranked by: its own window KS statistic "
            "(default) or the per-query FlowPure score that populates the "
            "window (the earlier convention, which duplicated C1)."
        ),
    )
    parser.add_argument("--integral-num-steps", type=int, default=8)
    parser.add_argument("--ks-threshold", type=float, default=0.25)
    parser.add_argument("--ks-min-window", type=int, default=64)
    parser.add_argument("--ks-max-window", type=int, default=2048)
    parser.add_argument("--labelhist-threshold", type=float, default=0.05)
    parser.add_argument("--labelhist-min-window", type=int, default=128)
    parser.add_argument("--labelhist-max-window", type=int, default=4096)
    parser.add_argument("--label-histogram-samples", type=int, default=512)
    parser.add_argument("--flowguardpp-decision-mode", default="hybrid",
                        choices=["per_query_only", "stateful_user", "stateful_global", "hybrid"])
    parser.add_argument("--flowguardpp-response-policy", default="reject",
                        choices=["reject", "rate_limit", "randomized_review", "label_coarsening", "silent_low_information_response"])
    parser.add_argument(
        "--composite-eval-learned",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also report a 'flowguard_composite_learned' column whose per-component "
            "weights are learned by cross-fitted logistic regression on the same "
            "composite run (oracle/upper bound vs the equal-weight default). "
            "Requires 'flowguard_composite' in --defenses."
        ),
    )
    parser.add_argument(
        "--composite-learned-folds",
        type=int,
        default=5,
        help="Cross-fitting folds for composite learned-weight evaluation.",
    )
    parser.add_argument(
        "--dump-composite-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Persist per-query composite scores (benign + attack) to "
            "'<output_root>/composite_scores_<run-label>/<attack>.npz' for every "
            "flowguard_composite cell. Consumed by scripts/plot_composite_scores.py "
            "to produce the post-hoc analysis figures."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help=(
            "JSON file with start_defense/start_attack (or defense/attack) and optional "
            "state_path to an existing summary. Skips earlier matrix cells and merges "
            "new results into the summary/report after each cell."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    set_userlevel_score_source(str(getattr(args, "userlevel_score_source", "ks")))
    args.device = _resolve_device(args.device)
    args.target_checkpoint_dir = Path(args.target_checkpoint_dir)
    if not args.target_checkpoint_dir.exists():
        raise FileNotFoundError(f"--target-checkpoint-dir not found: {args.target_checkpoint_dir}")
    args.flow_checkpoint = _resolve_flow_checkpoint(args.flow_checkpoint)
    args.a3_surrogate_checkpoint = (
        Path(args.a3_surrogate_checkpoint)
        if args.a3_surrogate_checkpoint is not None
        else args.flow_checkpoint
    )
    if not args.a3_surrogate_checkpoint.exists():
        raise FileNotFoundError(f"--a3-surrogate-checkpoint not found: {args.a3_surrogate_checkpoint}")
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
    if args.latent_generator_checkpoint is not None:
        args.latent_generator_checkpoint = Path(args.latent_generator_checkpoint)
        if not args.latent_generator_checkpoint.exists():
            raise FileNotFoundError(
                f"--latent-generator-checkpoint not found: {args.latent_generator_checkpoint}"
            )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = _matrix_artifact_paths(output_root, str(args.run_label))
    scratch_root = output_root / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    resume_from: ResumeFrom | None = None
    state_path = artifact_paths["summary"]
    if args.resume_file is not None:
        resume_path = Path(args.resume_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-file not found: {resume_path}")
        resume_from, resume_state_path = _parse_resume_file(resume_path)
        if resume_state_path is not None:
            state_path = resume_state_path
        print(
            f"[resume] continuing from defense={resume_from.defense} attack={resume_from.attack} "
            f"using state {state_path}"
        )

    attack_order = _split_csv(args.attacks)
    defense_order = _split_csv(args.defenses)

    # The learned-weight composite is a byproduct of the equal-weight composite
    # run (cross-fitted at metric time), never a standalone run. Listing it in
    # --defenses or passing --composite-eval-learned both enable it; it is
    # appended at the end so it shows as its own report column.
    eval_learned = bool(args.composite_eval_learned) or (
        LEARNED_COMPOSITE_KEY in defense_order
    )
    defense_order = [key for key in defense_order if key != LEARNED_COMPOSITE_KEY]
    if eval_learned and "flowguard_composite" not in defense_order:
        raise ValueError(
            "Composite learned-weight evaluation requires 'flowguard_composite' in --defenses."
        )
    if eval_learned:
        defense_order.append(LEARNED_COMPOSITE_KEY)

    dump_scores = bool(args.dump_composite_scores)

    # Provisional recipes for key validation. The FlowGuard++-adaptive rows are
    # rebuilt below once the attacker-side surrogate calibration exists.
    attacks = _attack_recipes(args)
    unknown_attacks = [key for key in attack_order if key not in attacks]
    if unknown_attacks:
        raise ValueError(f"Unknown attacks: {unknown_attacks}. Available: {sorted(attacks)}")
    if resume_from is not None:
        if resume_from.defense not in defense_order:
            raise ValueError(
                f"Resume defense '{resume_from.defense}' not in --defenses: {defense_order}"
            )
        if resume_from.attack not in attack_order:
            raise ValueError(
                f"Resume attack '{resume_from.attack}' not in --attacks: {attack_order}"
            )

    rows: list[dict[str, Any]] = []
    calibrations: dict[str, dict[str, Any]] = {}
    global_calibration_payload: dict[str, Any] | None = None
    if resume_from is not None:
        rows, calibrations, global_calibration_payload = _load_matrix_state(state_path)

    progress = _matrix_progress_now(status="running", last_completed=None)

    if global_calibration_payload is not None:
        print("[calibration] restoring global calibration from checkpoint")
        calibration = _calibration_from_global_payload(global_calibration_payload)
    else:
        print("[calibration] building benign label and entropy histograms")
        num_classes, label_histogram, entropy_histogram = _build_label_histogram(
            args,
            device=args.device,
        )
        calibration = CalibrationData(
            num_classes=num_classes,
            flowpure_scores=[],
            label_histogram=label_histogram,
            likelihood_scores=[],
            entropy_histogram=entropy_histogram,
        )
    defenses = _defense_recipes(args, calibration)
    unknown_defenses = [key for key in defense_order if key not in defenses]
    if unknown_defenses:
        raise ValueError(f"Unknown defenses: {unknown_defenses}. Available: {sorted(defenses)}")

    # Rebuild the attack recipes now that the benign label/entropy reference
    # exists, running the attacker-side surrogate calibration only if a
    # FlowGuard++-adaptive row was actually requested (it costs a likelihood
    # pass over the attacker's image pool).
    if set(attack_order) & (set(FULLY_ADAPTIVE_ATTACKS) | set(VELOCITY_ADAPTIVE_ATTACKS)):
        adaptive_calibration = _attacker_surrogate_calibration(
            args,
            cache_path=output_root / "attacker_surrogate_calibration.json",
        )
        attacks = _attack_recipes(
            args,
            adaptive_calibration=adaptive_calibration,
            benign_label_histogram=calibration.label_histogram,
            benign_entropy_moments=_entropy_moments_from_histogram(
                calibration.entropy_histogram,
                num_classes=calibration.num_classes,
            ),
        )

    flowpure_benign_recipe = AttackRecipe(
        key="benign_reference",
        display_name="Benign reference",
        kind=AttackKind.TRANSFER_SET,
        mode="naive",
        query_dataset=args.dataset,
        extra={"transfer_artifact_sample_size": 0},
        notes="Benign calibration stream from the defended dataset.",
    )

    needs_flowpure_reference = bool(
        {"flowpure", "flowguard_userlevel", "flowguard_composite"} & set(defense_order)
    )
    needs_likelihood_reference = bool(LIKELIHOOD_DEFENSES & set(defense_order))

    skip_flowpure_global = bool(
        resume_from is not None
        and calibration.flowpure_scores
        and needs_flowpure_reference
    )
    if not needs_flowpure_reference:
        skip_flowpure_global = True
    if skip_flowpure_global:
        if needs_flowpure_reference:
            print("[calibration] skipping FlowPure benign collection (restored from checkpoint)")
    else:
        print("[calibration] collecting FlowPure benign scores for KS/composite references")
        flowpure_samples, _flowpure_meta, flowpure_error, _flowpure_records = _run_one(
            recipe=flowpure_benign_recipe,
            defense=defenses["flowpure"],
            args=args,
            scratch_root=scratch_root,
            query_budget=int(args.benign_query_budget),
            gt=0,
            skip_substitute_training=True,
        )
        if flowpure_error:
            raise RuntimeError(f"FlowPure benign calibration failed: {flowpure_error}")
        calibration.flowpure_scores = [float(sample["score"]) for sample in flowpure_samples]
        defenses = _defense_recipes(args, calibration)
        global_calibration_payload = _global_calibration_payload(calibration)
        progress = _matrix_progress_now(
            status="running",
            last_completed={
                "phase": "global_calibration",
                "defense": "flowpure",
                "attack": None,
            },
        )
        _persist_matrix_state(
            output_root=output_root,
            args=args,
            attack_order=attack_order,
            defense_order=defense_order,
            attacks=attacks,
            defenses=defenses,
            rows=rows,
            calibrations=calibrations,
            progress=progress,
            global_calibration=global_calibration_payload,
        )

    skip_likelihood_global = bool(
        resume_from is not None
        and calibration.likelihood_scores
        and needs_likelihood_reference
    )
    if not needs_likelihood_reference:
        skip_likelihood_global = True
    if skip_likelihood_global:
        if needs_likelihood_reference:
            print("[calibration] skipping likelihood benign collection (restored from checkpoint)")
    else:
        print("[calibration] collecting Flow Matching benign log-likelihood scores")
        _likelihood_samples, _likelihood_meta, likelihood_error, likelihood_records = _run_one(
            recipe=flowpure_benign_recipe,
            defense=defenses["flow_matching"],
            args=args,
            scratch_root=scratch_root,
            query_budget=int(args.benign_query_budget),
            gt=0,
            skip_substitute_training=True,
        )
        if likelihood_error:
            raise RuntimeError(f"Likelihood benign calibration failed: {likelihood_error}")
        calibration.likelihood_scores = _metadata_score_list(likelihood_records, "queries_blocked")
        defenses = _defense_recipes(args, calibration)
        global_calibration_payload = _global_calibration_payload(calibration)
        progress = _matrix_progress_now(
            status="running",
            last_completed={
                "phase": "global_calibration",
                "defense": "flow_matching",
                "attack": None,
            },
        )
        _persist_matrix_state(
            output_root=output_root,
            args=args,
            attack_order=attack_order,
            defense_order=defense_order,
            attacks=attacks,
            defenses=defenses,
            rows=rows,
            calibrations=calibrations,
            progress=progress,
            global_calibration=global_calibration_payload,
        )

    for defense_key in defense_order:
        if defense_key == LEARNED_COMPOSITE_KEY:
            # Produced as a byproduct of the flowguard_composite cells below.
            continue
        if resume_from is not None and _defense_before_resume(defense_key, resume_from, defense_order):
            print(f"[resume] skipping defense {defense_key} (before resume point)")
            continue

        defense = defenses[defense_key]
        print(f"[defense] {defense_key}: benign reference")
        benign_samples, benign_meta, benign_error, benign_records = _run_one(
            recipe=flowpure_benign_recipe,
            defense=defense,
            args=args,
            scratch_root=scratch_root,
            query_budget=int(args.benign_query_budget),
            gt=0,
            skip_substitute_training=True,
        )
        threshold = None
        composite_thresholds: dict[str, float] | None = None
        calibration_notes = "uses defense-native flags"
        if defense_key == "flowguard_composite" and benign_error is None:
            composite_thresholds = _calibrate_composite_thresholds(
                benign_records,
                target_fpr=float(args.target_fpr),
            )
            threshold = _score_threshold(benign_samples, target_fpr=float(args.target_fpr))
            if threshold is not None:
                benign_samples = _force_threshold(benign_samples, threshold)
            defense = replace(
                defense,
                parameters={
                    **defense.parameters,
                    "composite_thresholds": composite_thresholds,
                },
            )
            defenses[defense_key] = defense
            calibration_notes = (
                f"composite multi-signal thresholds at target FPR={args.target_fpr}"
            )
        elif defense_key in CALIBRATED_SCORE_DEFENSES and benign_error is None:
            threshold = _score_threshold(benign_samples, target_fpr=float(args.target_fpr))
            if threshold is not None:
                benign_samples = _force_threshold(benign_samples, threshold)
                calibration_notes = f"score quantile at target FPR={args.target_fpr}"
        calibrations[defense_key] = {
            "threshold": threshold,
            "composite_thresholds": composite_thresholds,
            "num_benign": len(benign_samples),
            "error": benign_error,
            "elapsed_seconds": benign_meta.get("elapsed_seconds"),
            "notes": calibration_notes,
        }
        progress = _matrix_progress_now(
            status="running",
            last_completed={
                "phase": "benign_calibration",
                "defense": defense_key,
                "attack": None,
            },
        )
        _persist_matrix_state(
            output_root=output_root,
            args=args,
            attack_order=attack_order,
            defense_order=defense_order,
            attacks=attacks,
            defenses=defenses,
            rows=rows,
            calibrations=calibrations,
            progress=progress,
            global_calibration=global_calibration_payload,
        )

        if benign_error:
            for attack_key in attack_order:
                if resume_from is not None and _attack_before_resume(
                    attack_key, resume_from, attack_order
                ):
                    continue
                rows = _upsert_result_row(
                    rows,
                    {
                        "defense": defense_key,
                        "attack": attack_key,
                        "error": f"benign reference failed: {benign_error}",
                    },
                )
            _persist_matrix_state(
                output_root=output_root,
                args=args,
                attack_order=attack_order,
                defense_order=defense_order,
                attacks=attacks,
                defenses=defenses,
                rows=rows,
                calibrations=calibrations,
                progress=progress,
                global_calibration=global_calibration_payload,
            )
            continue

        for attack_key in attack_order:
            if resume_from is not None and _attack_before_resume(
                attack_key, resume_from, attack_order
            ):
                print(f"[resume] skipping {defense_key} x {attack_key} (before resume point)")
                continue

            attack = attacks[attack_key]
            print(f"[run] {defense_key} x {attack_key}")
            attack_defense = _defense_for_attack_run(
                defense,
                defense_key=defense_key,
                threshold=threshold,
                audit_only=bool(args.audit_only),
            )
            attack_samples, attack_meta, attack_error, attack_records = _run_one(
                recipe=attack,
                defense=attack_defense,
                args=args,
                scratch_root=scratch_root,
                query_budget=int(args.query_budget),
                gt=1,
                skip_substitute_training=False,
            )
            scored_benign = benign_samples
            scored_attack = attack_samples
            if threshold is not None:
                scored_attack = _force_threshold(attack_samples, threshold)
            metrics = _compute_metrics(scored_benign + scored_attack)
            row = {
                "defense": defense_key,
                "attack": attack_key,
                "error": attack_error,
                **metrics,
                "accuracy": attack_meta.get("accuracy"),
                "fidelity": attack_meta.get("fidelity"),
                "joint_accuracy": attack_meta.get("joint_accuracy"),
                "total_queries": attack_meta.get("total_queries"),
                "total_batches": attack_meta.get("total_batches"),
                "elapsed_seconds": attack_meta.get("elapsed_seconds"),
                # Per-epoch (queries, accuracy, fidelity) history from the
                # DisGUIDE-loop attacks. Retained so one run answers "how many
                # queries buy how much fidelity" without a budget sweep; the
                # scratch directory holding it is deleted right after the cell.
                "query_curve": _query_curve(attack_meta),
            }
            rows = _upsert_result_row(rows, row)

            if (
                eval_learned
                and defense_key == "flowguard_composite"
                and attack_error is None
            ):
                learned_row = _compute_learned_composite_row(
                    benign_records,
                    attack_records,
                    attack_meta=attack_meta,
                    target_fpr=float(args.target_fpr),
                    folds=int(args.composite_learned_folds),
                )
                learned_row.update({"defense": LEARNED_COMPOSITE_KEY, "attack": attack_key})
                rows = _upsert_result_row(rows, learned_row)
                weights = learned_row.get("learned_component_weights")
                print(
                    f"[done] {LEARNED_COMPOSITE_KEY} x {attack_key} | "
                    f"AUROC={_fmt(learned_row.get('auroc'))} "
                    f"TPR={_fmt(learned_row.get('tpr'))} "
                    f"weights={weights} error={learned_row.get('error') or 'none'}"
                )

            if (
                dump_scores
                and defense_key == "flowguard_composite"
                and attack_error is None
            ):
                dump_path = _dump_composite_component_scores(
                    output_root=output_root,
                    run_label=str(args.run_label),
                    attack_key=attack_key,
                    benign_records=benign_records,
                    attack_records=attack_records,
                )
                if dump_path is not None:
                    print(f"[dump] composite scores -> {dump_path}")
                else:
                    print(
                        f"[dump] composite scores skipped for {attack_key} "
                        "(no standardized component scores available)"
                    )

            progress = _matrix_progress_now(
                status="running",
                last_completed={
                    "phase": "attack",
                    "defense": defense_key,
                    "attack": attack_key,
                },
            )
            _persist_matrix_state(
                output_root=output_root,
                args=args,
                attack_order=attack_order,
                defense_order=defense_order,
                attacks=attacks,
                defenses=defenses,
                rows=rows,
                calibrations=calibrations,
                progress=progress,
                global_calibration=global_calibration_payload,
            )
            print(
                f"[done] {defense_key} x {attack_key} | "
                f"AUROC={_fmt(row.get('auroc'))} TPR={_fmt(row.get('tpr'))} "
                f"FPR={_fmt(row.get('fpr'))} error={attack_error or 'none'}"
            )

    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    progress = _matrix_progress_now(
        status="complete",
        last_completed=progress.last_completed,
    )
    _persist_matrix_state(
        output_root=output_root,
        args=args,
        attack_order=attack_order,
        defense_order=defense_order,
        attacks=attacks,
        defenses=defenses,
        rows=rows,
        calibrations=calibrations,
        progress=progress,
        global_calibration=global_calibration_payload,
    )


if __name__ == "__main__":
    main()
