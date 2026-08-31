"""Unified FlowPure-weakness attack-suite evaluation.

Runs the attack catalogue from the PhD research plan
(``.cursor/plans/flowpure_weakness_research_*.plan.md``) against the
paper-faithful ``FlowPure^PGD`` detector and optionally against the
composite ``FlowGuard++`` defense. Produces AUROC, detection-rate, and
substitute-accuracy metrics per attack so the hypotheses in plan Teil 2
can be empirically falsified.

Attack catalogue (selectable via ``--attacks``):

- ``a1_clean_transfer``    - Clean-transfer knockoff (TRANSFER_SET, naive) with
                             a user-specified clean query dataset (default
                             CIFAR-100). Threat model: strict black-box. The
                             query pool is already clean, so the detector has
                             no high-frequency signal to latch onto.
- ``a2_projected_maze``    - MAZE with the ``apply_query_denoising`` stage
                             enabled so the submitted query is a Gaussian-blur
                             projection of the generator output. Threat model:
                             black-box that knows FlowPure is deployed.
- ``a3_flowblind``         - MAZE with the surrogate-CNF velocity regularizer
                             enabled so the generator is gradient-trained to
                             minimize ``||v_surrogate(0, G(z))||^2``. Threat
                             model: white-box against the defense (attacker has
                             access to a surrogate CNF).
- ``maze_baseline``        - Unmodified MAZE, serves as the Section-3.3 paper
                             baseline (expected AUROC ~1.0).
- ``prada_baseline``       - Unmodified PRADA for reference.
- ``disguide_baseline``    - Unmodified DisGuide for reference.

Each run records AUROC, detection rate at calibrated FPR, and substitute
accuracy pulled from the attack artifacts when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flowguard.evaluation.detection import compute_detection_metrics, extract_detection_samples
from flowguard.experiments.factory import build_experiment_spec
from flowguard.experiments.spec import AttackKind
from flowguard.orchestration import run_experiment

# ---------------------------------------------------------------------------
# Attack catalogue
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AttackRecipe:
    name: str
    attack_kind: AttackKind
    attack_mode: str
    attack_extra: dict[str, Any]
    query_dataset: str
    drop_prefix: int = 0
    notes: str = ""


def _attack_recipes(args: argparse.Namespace) -> dict[str, AttackRecipe]:
    seed_samples = int(args.seed_samples)
    dataset = str(args.dataset)
    query_default = str(args.query_dataset)
    clean_transfer_dataset = str(args.clean_transfer_dataset)
    maze_common = {
        "latent_dim": int(args.maze_latent_dim),
        "iter_gen": int(args.maze_iter_gen),
        "iter_clone": int(args.maze_iter_clone),
        "iter_exp": int(args.maze_iter_exp),
        "ndirs": int(args.maze_ndirs),
        "log_iter": int(args.maze_log_iter),
        "disable_pbar": True,
        "verbose": True,
        "replay_buffer_size": int(args.maze_replay_buffer_size),
        "artifact_sample_size": int(args.maze_artifact_sample_size),
    }
    return {
        "a1_clean_transfer": AttackRecipe(
            name="a1_clean_transfer",
            attack_kind=AttackKind.TRANSFER_SET,
            attack_mode="naive",
            attack_extra={
                "transfer_artifact_sample_size": int(args.transfer_artifact_sample_size),
            },
            query_dataset=clean_transfer_dataset,
            drop_prefix=0,
            notes=(
                "Naive transfer_set with a clean external query pool. "
                "Expected to bypass FlowPure^PGD because queries are "
                "statistically benign."
            ),
        ),
        "a2_projected_maze": AttackRecipe(
            name="a2_projected_maze",
            attack_kind=AttackKind.MAZE,
            attack_mode="maze",
            attack_extra={
                **maze_common,
                "denoise": {
                    "kind": args.a2_denoise_kind,
                    "sigma": float(args.a2_denoise_sigma),
                    "kernel_size": int(args.a2_denoise_kernel),
                    "mix": float(args.a2_denoise_mix),
                },
            },
            query_dataset=query_default,
            drop_prefix=0,
            notes=(
                "MAZE with an attacker-side denoiser applied to each query "
                "right before submission, suppressing high-frequency "
                "structure that drives ||v(0, x)||^2."
            ),
        ),
        "a3_flowblind": AttackRecipe(
            name="a3_flowblind",
            attack_kind=AttackKind.MAZE,
            attack_mode="maze",
            attack_extra={
                **maze_common,
                "surrogate_checkpoint_path": str(args.a3_surrogate_checkpoint),
                "velocity_regularizer_weight": float(args.a3_regularizer_weight),
                "velocity_regularizer_device": args.device,
                "velocity_regularizer_is_pixel_space": True,
                "record_surrogate_scores": True,
            },
            query_dataset=query_default,
            drop_prefix=0,
            notes=(
                "MAZE with a differentiable surrogate-CNF velocity "
                "regularizer added to the generator loss. Tests whether the "
                "score function is spoofable via transferability."
            ),
        ),
        "maze_baseline": AttackRecipe(
            name="maze_baseline",
            attack_kind=AttackKind.MAZE,
            attack_mode="maze",
            attack_extra=dict(maze_common),
            query_dataset=query_default,
            drop_prefix=0,
            notes="Unmodified MAZE (Section-3.3 paper baseline).",
        ),
        "prada_baseline": AttackRecipe(
            name="prada_baseline",
            attack_kind=AttackKind.PRADA,
            attack_mode="prada",
            attack_extra={
                "initial_seed_size": seed_samples,
                "duplication_rounds": 2,
                "expansion_factor": 1,
                "max_iter": 50,
                "epsilon": 0.05,
                "hyperparameter_search_budget": 0,
                "verbose": True,
            },
            query_dataset=query_default,
            drop_prefix=seed_samples,
            notes="Unmodified PRADA baseline.",
        ),
        "adaptive_adaptive": AttackRecipe(
            name="adaptive_adaptive",
            attack_kind=AttackKind.MAZE,
            attack_mode="maze",
            attack_extra={
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
            query_dataset=query_default,
            drop_prefix=0,
            notes=(
                "Adaptive-Adaptive attack: combines A2 denoising and A3 "
                "surrogate-CNF regularization at increased weight, targeting "
                "FlowGuard++ integral/KS defenses simultaneously."
            ),
        ),
        "disguide_baseline": AttackRecipe(
            name="disguide_baseline",
            attack_kind=AttackKind.DISGUIDE,
            attack_mode="disguide",
            attack_extra={
                "ensemble_size": 2,
                "latent_dim": 16,
                "g_iter": 1,
                "d_iter": 1,
                "rep_iter": 1,
                "replay_size": 64,
                "epoch_itrs": 3,
                "disable_pbar": True,
                "verbose": True,
            },
            query_dataset=query_default,
            drop_prefix=0,
            notes="Unmodified DisGuide baseline.",
        ),
    }


# ---------------------------------------------------------------------------
# Spec construction helpers
# ---------------------------------------------------------------------------


def _build_flowpure_params(args: argparse.Namespace, flow_checkpoint: Path) -> dict[str, Any]:
    return {
        "fm_checkpoint_path": str(flow_checkpoint),
        "dataset_name": args.dataset,
        "device": args.device,
        "velocity_threshold": float(args.velocity_threshold),
        "inputs_normalized": True,
        "audit_only": True,
    }


def _build_defense_spec(args: argparse.Namespace, flow_checkpoint: Path) -> tuple[str, dict[str, Any]]:
    """Return the (query_defense_name, parameters) pair for the requested defense."""
    defense = args.defense.lower()
    base = {
        "fm_checkpoint_path": str(flow_checkpoint),
        "dataset_name": args.dataset,
        "device": args.device,
        "inputs_normalized": True,
        "audit_only": True,
    }
    if defense == "flowpure":
        base["velocity_threshold"] = float(args.velocity_threshold)
        return "flowpure", base
    if defense in {"flowguard_integral", "c1"}:
        base["integral_threshold"] = float(args.velocity_threshold)
        base["num_steps"] = int(args.integral_num_steps)
        return "flowguard_integral", base
    if defense in {"flowguard_userlevel", "c3"}:
        base["ks_threshold"] = float(args.ks_threshold)
        base["min_window"] = int(args.ks_min_window)
        base["max_window"] = int(args.ks_max_window)
        # Without a benign reference the defense raises; we calibrate it on the
        # benign reference run and persist it via defense.set_benign_reference.
        base["benign_reference_scores"] = []
        return "flowguard_userlevel", base
    raise ValueError(
        f"Unsupported defense '{args.defense}'. Choose one of: flowpure, "
        f"flowguard_integral, flowguard_userlevel."
    )


def _build_common_kwargs(
    args: argparse.Namespace,
    *,
    recipe: AttackRecipe,
    query_budget: int,
    attack_batch_size: int | None = None,
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "target_architecture": args.target_architecture,
        "substitute_architecture": args.substitute_architecture,
        "query_dataset": recipe.query_dataset,
        "query_budget": query_budget,
        "query_transfer_set_size": None,
        "target_checkpoint_dir": str(args.target_checkpoint_dir),
        "target_device": args.device,
        "substitute_device": args.device,
        "distributed": False,
        "num_workers": 1,
        "num_clients": 1,
        "attack_batch_size": int(
            attack_batch_size if attack_batch_size is not None else args.attack_batch_size
        ),
        "training_batch_size": args.training_batch_size,
        "epochs": args.epochs,
        "verbose": True,
    }


def _is_generator_style_attack(recipe: AttackRecipe) -> bool:
    return recipe.attack_kind in {AttackKind.MAZE, AttackKind.DISGUIDE}


# ---------------------------------------------------------------------------
# Execution and scoring
# ---------------------------------------------------------------------------


def _force_gt_label(samples: list[dict[str, Any]], *, gt_label: int) -> list[dict[str, Any]]:
    return [
        {
            "gt": int(gt_label),
            "pred": int(row.get("pred", 0)),
            "score": float(row["score"]),
            "source": row.get("source", "unknown"),
        }
        for row in samples
    ]


def _apply_threshold(samples: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "gt": int(row["gt"]),
            "pred": int(float(row["score"]) > threshold),
            "score": float(row["score"]),
            "source": row.get("source", "unknown"),
        }
        for row in samples
    ]


def _calibrate_threshold(benign: list[dict[str, Any]], *, target_fpr: float) -> float:
    if not benign:
        raise RuntimeError("Cannot calibrate threshold: no benign samples.")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1); got {target_fpr}.")
    scores = np.asarray([float(row["score"]) for row in benign], dtype=np.float64)
    return float(np.quantile(scores, 1.0 - target_fpr))


def _run_benign_reference(
    args: argparse.Namespace,
    run_root: Path,
    defense_name: str,
    defense_params: dict[str, Any],
) -> list[dict[str, Any]]:
    benign_recipe = AttackRecipe(
        name="benign_reference",
        attack_kind=AttackKind.TRANSFER_SET,
        attack_mode="naive",
        attack_extra={
            "transfer_artifact_sample_size": int(args.transfer_artifact_sample_size),
            "skip_substitute_training": True,
        },
        query_dataset=args.dataset,
    )
    spec = build_experiment_spec(
        name=f"benign_reference__{defense_name}",
        attack_kind=benign_recipe.attack_kind,
        attack_mode=benign_recipe.attack_mode,
        query_defense=defense_name,
        query_defense_parameters=defense_params,
        **_build_common_kwargs(
            args,
            recipe=benign_recipe,
            query_budget=int(args.benign_query_budget),
        ),
    )
    result = run_experiment(spec, output_dir=run_root / spec.name)
    samples = extract_detection_samples(
        result.query_summary.history.records,
        defense_name="flowpure",
        fallback_gt=0,
    )
    return _force_gt_label(samples, gt_label=0)


def _run_attack(
    attack_key: str,
    recipe: AttackRecipe,
    args: argparse.Namespace,
    run_root: Path,
    defense_name: str,
    defense_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    batch_size = args.attack_batch_size
    if _is_generator_style_attack(recipe):
        batch_size = max(2, int(args.extraction_attack_batch_size))
    spec = build_experiment_spec(
        name=f"attack-{attack_key}__{defense_name}",
        attack_kind=recipe.attack_kind,
        attack_mode=recipe.attack_mode,
        query_defense=defense_name,
        query_defense_parameters=defense_params,
        attack_extra=recipe.attack_extra,
        **_build_common_kwargs(
            args,
            recipe=recipe,
            query_budget=int(args.attack_query_budget),
            attack_batch_size=batch_size,
        ),
    )
    output_dir = run_root / spec.name
    result = run_experiment(spec, output_dir=output_dir)
    samples = extract_detection_samples(
        result.query_summary.history.records,
        defense_name="flowpure",
        fallback_gt=1,
    )
    samples = _force_gt_label(samples, gt_label=1)
    if recipe.drop_prefix > 0:
        samples = samples[recipe.drop_prefix :]
    return samples, output_dir


def _read_substitute_accuracy(output_dir: Path) -> float | None:
    # MAZE/DisGuide write metrics.csv with surrogate_accuracy columns.
    for candidate in output_dir.rglob("metrics.csv"):
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
            if len(lines) < 2:
                continue
            header = lines[0].split(",")
            if "surrogate_accuracy" not in header:
                continue
            column = header.index("surrogate_accuracy")
            last_row = lines[-1].split(",")
            if column >= len(last_row):
                continue
            return float(last_row[column])
        except (OSError, ValueError):
            continue
    # Transfer-set / PRADA substitute is evaluated via spec evaluation summary
    # saved as evaluation.json.
    for candidate in output_dir.rglob("evaluation.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            accuracy = payload.get("accuracy")
            if accuracy is not None:
                return float(accuracy)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _compute_metrics_with_auroc(
    benign_samples: list[dict[str, Any]],
    attack_samples: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    benign_scored = _apply_threshold(benign_samples, threshold=threshold)
    attack_scored = _apply_threshold(attack_samples, threshold=threshold)
    combined = benign_scored + attack_scored
    metrics = compute_detection_metrics(combined)
    if not combined:
        metrics["auroc"] = None
        metrics["f1_benign"] = 0.0
        metrics["f1_attack"] = 0.0
        return metrics
    y_true = np.asarray([row["gt"] for row in combined], dtype=np.int64)
    y_pred = np.asarray([row["pred"] for row in combined], dtype=np.int64)
    y_score = np.asarray([row["score"] for row in combined], dtype=np.float64)
    auroc: float | None
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_score))
    else:
        auroc = None
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    metrics["auroc"] = auroc
    metrics["f1_benign"] = float(per_class[0])
    metrics["f1_attack"] = float(per_class[1])
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_attacks(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_flow_checkpoint(requested: Path | None) -> Path:
    candidates = [
        PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd" / "checkpoint_latest.pt",
        PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd_smoke" / "checkpoint_latest.pt",
    ]
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(f"--flow-checkpoint not found: {requested}")
        return requested
    for candidate in candidates:
        if candidate.exists():
            return candidate
    listing = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"No FlowPure checkpoint found. Provide --flow-checkpoint or create one of:\n{listing}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="flowpure-attack-suite")
    parser.add_argument("--output-root", default="runs/notebook")
    parser.add_argument(
        "--attacks",
        default="a1_clean_transfer,a2_projected_maze,a3_flowblind,maze_baseline",
        help=(
            "Comma-separated attack keys to run. Available: "
            "a1_clean_transfer, a2_projected_maze, a3_flowblind, "
            "maze_baseline, prada_baseline, disguide_baseline."
        ),
    )
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--query-dataset", default="CIFAR10",
                        help="Default query pool for non-A1 attacks.")
    parser.add_argument("--clean-transfer-dataset", default="CIFAR100",
                        help="Query pool for the A1 clean-transfer knockoff.")
    parser.add_argument("--target-architecture", default="vgg16_bn")
    parser.add_argument("--substitute-architecture", default="vgg16_bn")
    parser.add_argument(
        "--target-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "notebook"
        / "training-victim-cifar10-vgg16_bn-nodefense"
        / "target_model",
    )
    parser.add_argument("--flow-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--velocity-threshold", type=float, default=5000.0)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--seed-samples", type=int, default=16)
    parser.add_argument("--attack-query-budget", type=int, default=144)
    parser.add_argument("--benign-query-budget", type=int, default=144)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--extraction-attack-batch-size", type=int, default=16)
    parser.add_argument("--training-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--transfer-artifact-sample-size", type=int, default=0,
                        help=(
                            "Limit saved TRANSFER_SET samples used for artifacts/substitute "
                            "training; 0 preserves all queried samples."
                        ))

    # MAZE / MAZE-variant knobs. Defaults preserve the previous smoke-oriented
    # settings, while HPC scripts can increase budgets and bound replay memory.
    parser.add_argument("--maze-latent-dim", type=int, default=16)
    parser.add_argument("--maze-iter-gen", type=int, default=1)
    parser.add_argument("--maze-iter-clone", type=int, default=2)
    parser.add_argument("--maze-iter-exp", type=int, default=1)
    parser.add_argument("--maze-ndirs", type=int, default=1)
    parser.add_argument("--maze-log-iter", type=int, default=50)
    parser.add_argument("--maze-replay-buffer-size", type=int, default=0,
                        help="Bound in-memory MAZE replay records; 0 keeps all records.")
    parser.add_argument("--maze-artifact-sample-size", type=int, default=0,
                        help="Limit saved MAZE transfer/visualization records; 0 saves all.")

    # A2 knobs.
    parser.add_argument("--a2-denoise-kind", default="gaussian_blur",
                        choices=["none", "gaussian_blur", "box_blur", "pixel_soft"])
    parser.add_argument("--a2-denoise-sigma", type=float, default=0.8)
    parser.add_argument("--a2-denoise-kernel", type=int, default=5)
    parser.add_argument("--a2-denoise-mix", type=float, default=1.0)

    # A3 knobs.
    parser.add_argument(
        "--a3-surrogate-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "flow_matching"
        / "cifar10_flowpure_pgd"
        / "checkpoint_latest.pt",
        help=(
            "Surrogate-CNF checkpoint for FlowBlind. Using the real defense "
            "checkpoint corresponds to the strongest (white-box-vs-defense) "
            "threat model; pass a different CNF for transferability studies."
        ),
    )
    parser.add_argument("--a3-regularizer-weight", type=float, default=0.01)

    # Defense selection.
    parser.add_argument(
        "--defense",
        default="flowpure",
        choices=["flowpure", "flowguard_integral", "c1", "flowguard_userlevel", "c3"],
        help="Query defense to evaluate the attacks against.",
    )
    parser.add_argument("--integral-num-steps", type=int, default=8,
                        help="Reverse-ODE steps for FlowGuardIntegralDefense (C1).")
    parser.add_argument("--ks-threshold", type=float, default=0.25,
                        help="KS statistic threshold for FlowGuardUserLevelDefense (C3).")
    parser.add_argument("--ks-min-window", type=int, default=64,
                        help="Minimum window size before KS test is applied (C3).")
    parser.add_argument("--ks-max-window", type=int, default=2048,
                        help="Maximum sliding window size (C3).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    flow_checkpoint = _resolve_flow_checkpoint(args.flow_checkpoint)
    if not args.target_checkpoint_dir.exists():
        raise FileNotFoundError(f"--target-checkpoint-dir not found: {args.target_checkpoint_dir}")

    attack_keys = _parse_attacks(args.attacks)
    recipes = _attack_recipes(args)
    unknown = [key for key in attack_keys if key not in recipes]
    if unknown:
        raise ValueError(
            f"Unknown attack keys: {unknown}. Available: {sorted(recipes.keys())}"
        )

    run_root = Path(args.output_root) / args.name
    run_root.mkdir(parents=True, exist_ok=True)
    defense_name, defense_params = _build_defense_spec(args, flow_checkpoint)
    flowpure_params = _build_flowpure_params(args, flow_checkpoint)
    print(f"Using FlowPure checkpoint: {flow_checkpoint}")
    print(f"Defense: {defense_name}")
    print(f"Device: {args.device}")

    print("\n=== Benign reference run (for threshold calibration) ===")
    # User-level defenses require a benign reference; we always run the
    # reference via the base flowpure scoring and then reuse the scores
    # both for threshold calibration and for seeding the distribution tests.
    benign_samples = _run_benign_reference(
        args, run_root, defense_name="flowpure", defense_params=flowpure_params,
    )
    if not benign_samples:
        raise RuntimeError("Benign reference produced no samples.")
    threshold = _calibrate_threshold(benign_samples, target_fpr=float(args.target_fpr))
    benign_scores = np.asarray([row["score"] for row in benign_samples], dtype=np.float64)
    print(
        f"Calibrated threshold = {threshold:.4f} | "
        f"benign score mean={benign_scores.mean():.2f}, "
        f"median={np.median(benign_scores):.2f}, "
        f"max={benign_scores.max():.2f}"
    )

    # Propagate threshold / benign reference into the actual defense parameters
    # once we have real benign data.
    if defense_name == "flowguard_userlevel":
        defense_params["benign_reference_scores"] = benign_scores.tolist()
    elif defense_name == "flowguard_integral":
        defense_params["integral_threshold"] = float(threshold)
    elif defense_name == "flowpure":
        defense_params["velocity_threshold"] = float(threshold)

    summary_rows: list[dict[str, Any]] = []
    for attack_key in attack_keys:
        recipe = recipes[attack_key]
        print(f"\n=== Attack: {attack_key} ({recipe.attack_kind.value}, {recipe.attack_mode}) ===")
        print(f"    notes: {recipe.notes}")
        try:
            attack_samples, out_dir = _run_attack(
                attack_key, recipe, args, run_root,
                defense_name=defense_name, defense_params=defense_params,
            )
        except Exception as exc:
            print(f"    ERROR running {attack_key}: {exc!r}")
            summary_rows.append(
                {
                    "attack": attack_key,
                    "error": repr(exc),
                }
            )
            continue
        if not attack_samples:
            print(f"    WARNING: {attack_key} produced no detection samples.")
            summary_rows.append({"attack": attack_key, "error": "no detection samples"})
            continue

        metrics = _compute_metrics_with_auroc(
            benign_samples, attack_samples, threshold=threshold
        )
        substitute_accuracy = _read_substitute_accuracy(out_dir)

        row = {
            "attack": attack_key,
            "kind": recipe.attack_kind.value,
            "mode": recipe.attack_mode,
            "query_dataset": recipe.query_dataset,
            "num_benign": int(metrics["num_benign"]),
            "num_attack": int(metrics["num_malicious"]),
            "detection_rate_attack": float(metrics["detection_rate"]),
            "auroc": metrics["auroc"],
            "fpr": float(metrics["fpr"]),
            "f1_attack": float(metrics["f1_attack"]),
            "f1_benign": float(metrics["f1_benign"]),
            "substitute_accuracy": substitute_accuracy,
        }
        summary_rows.append(row)
        auroc_display = f"{row['auroc']:.4f}" if row["auroc"] is not None else "n/a"
        acc_display = (
            f"{substitute_accuracy:.2f}%" if substitute_accuracy is not None else "n/a"
        )
        print(
            f"    AUROC={auroc_display} | "
            f"detection_rate={row['detection_rate_attack']:.4f} | "
            f"FPR={row['fpr']:.4f} | "
            f"F1_attack={row['f1_attack']:.4f} | "
            f"substitute_acc={acc_display}"
        )

    payload = {
        "config": {
            "dataset": args.dataset,
            "query_dataset": args.query_dataset,
            "clean_transfer_dataset": args.clean_transfer_dataset,
            "target_architecture": args.target_architecture,
            "substitute_architecture": args.substitute_architecture,
            "target_checkpoint_dir": str(args.target_checkpoint_dir),
            "flow_checkpoint": str(flow_checkpoint),
            "defense_name": defense_name,
            "defense_parameters": defense_params,
            "flowpure_parameters": flowpure_params,
            "attack_query_budget": args.attack_query_budget,
            "benign_query_budget": args.benign_query_budget,
            "attack_batch_size": args.attack_batch_size,
            "extraction_attack_batch_size": args.extraction_attack_batch_size,
            "target_fpr": args.target_fpr,
            "calibrated_threshold": threshold,
            "seed_samples": args.seed_samples,
            "a2_denoise_kind": args.a2_denoise_kind,
            "a2_denoise_sigma": args.a2_denoise_sigma,
            "a2_denoise_kernel": args.a2_denoise_kernel,
            "a2_denoise_mix": args.a2_denoise_mix,
            "a3_surrogate_checkpoint": str(args.a3_surrogate_checkpoint),
            "a3_regularizer_weight": args.a3_regularizer_weight,
        },
        "results": summary_rows,
    }
    out_path = run_root / "attack_suite_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved suite summary to: {out_path}")


if __name__ == "__main__":
    main()
