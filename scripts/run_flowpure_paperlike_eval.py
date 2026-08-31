from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

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


def _parse_attacks(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _attack_config(name: str, *, seed_samples: int) -> tuple[AttackKind, str, dict[str, Any], int]:
    """Return attack kind/mode/extras and prefix samples to ignore in evaluation."""
    if name == "prada":
        return (
            AttackKind.PRADA,
            "prada",
            {
                "initial_seed_size": seed_samples,
                "duplication_rounds": 2,
                "expansion_factor": 1,
                "max_iter": 50,
                "epsilon": 0.05,
                "hyperparameter_search_budget": 0,
                "verbose": True,
            },
            seed_samples,
        )
    if name == "maze":
        return (
            AttackKind.MAZE,
            "maze",
            {
                "latent_dim": 16,
                "iter_gen": 1,
                "iter_clone": 2,
                "iter_exp": 1,
                "ndirs": 1,
                "log_iter": 50,
                "disable_pbar": True,
                "verbose": True,
            },
            0,
        )
    if name == "disguide":
        return (
            AttackKind.DISGUIDE,
            "disguide",
            {
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
            0,
        )
    raise ValueError(f"Unsupported attack '{name}'. Supported: prada, maze, disguide")


def _build_flowpure_params(*, flow_checkpoint: Path, dataset: str, device: str, velocity_threshold: float) -> dict[str, Any]:
    return {
        "fm_checkpoint_path": str(flow_checkpoint),
        "dataset_name": dataset,
        "device": device,
        "velocity_threshold": velocity_threshold,
        "inputs_normalized": True,
        "audit_only": True,
    }


def _default_flow_checkpoint_candidates() -> list[Path]:
    """Prefer the paper detector checkpoint, then fall back to Gaussian CNF."""
    return [
        PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd" / "checkpoint_latest.pt",
        PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd_smoke" / "checkpoint_latest.pt",
        PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_notebook" / "checkpoint_latest.pt",
    ]


def _resolve_flow_checkpoint(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(f"Flow checkpoint not found: {requested}")
        return requested

    for candidate in _default_flow_checkpoint_candidates():
        if candidate.exists():
            return candidate

    candidate_lines = "\n".join(f"- {path}" for path in _default_flow_checkpoint_candidates())
    raise FileNotFoundError(
        "No FlowPure checkpoint found. Provide --flow-checkpoint or create one of:\n"
        f"{candidate_lines}"
    )


def _calibrate_threshold_from_samples(samples: list[dict[str, Any]], *, target_fpr: float) -> float:
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1); got {target_fpr}.")
    if not samples:
        raise RuntimeError("Cannot calibrate threshold: no benign samples available.")

    benign_scores = np.asarray([float(row["score"]) for row in samples], dtype=np.float64)
    if benign_scores.size == 0:
        raise RuntimeError("Cannot calibrate threshold: benign score list is empty.")
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def _apply_threshold(samples: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    relabeled: list[dict[str, Any]] = []
    for row in samples:
        score = float(row["score"])
        relabeled.append(
            {
                "gt": int(row["gt"]),
                "pred": int(score > threshold),
                "score": score,
                "source": row.get("source", "unknown"),
            }
        )
    return relabeled


def _force_gt_label(samples: list[dict[str, Any]], *, gt_label: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in samples:
        normalized.append(
            {
                "gt": int(gt_label),
                "pred": int(row.get("pred", 0)),
                "score": float(row["score"]),
                "source": row.get("source", "unknown"),
            }
        )
    return normalized


def _build_common_spec_kwargs(
    args: argparse.Namespace,
    *,
    query_budget: int,
    attack_batch_size: int | None = None,
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "target_architecture": args.target_architecture,
        "substitute_architecture": args.substitute_architecture,
        "query_dataset": args.query_dataset,
        "query_budget": query_budget,
        "query_transfer_set_size": None,
        "target_checkpoint_dir": str(args.target_checkpoint_dir),
        "target_device": args.device,
        "substitute_device": args.device,
        "distributed": False,
        "num_workers": 1,
        "num_clients": 1,
        "attack_batch_size": int(attack_batch_size if attack_batch_size is not None else args.attack_batch_size),
        "training_batch_size": args.training_batch_size,
        "epochs": args.epochs,
        "verbose": True,
    }


def _compute_class_f1(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {"f1_benign": 0.0, "f1_attack": 0.0}

    y_true = np.asarray([int(row["gt"]) for row in samples], dtype=np.int64)
    y_pred = np.asarray([int(row["pred"]) for row in samples], dtype=np.int64)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "f1_benign": float(f1_per_class[0]),
        "f1_attack": float(f1_per_class[1]),
    }


def _run_benign_reference(
    *,
    args: argparse.Namespace,
    run_root: Path,
    flowpure_params: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = build_experiment_spec(
        name="flowpure-paperlike-benign",
        attack_kind=AttackKind.TRANSFER_SET,
        attack_mode="naive",
        query_defense="flowpure",
        query_defense_parameters=flowpure_params,
        **_build_common_spec_kwargs(args, query_budget=args.benign_query_budget),
    )
    result = run_experiment(spec, output_dir=run_root / spec.name)
    samples = extract_detection_samples(
        result.query_summary.history.records,
        defense_name="flowpure",
        fallback_gt=0,
    )
    # Do not trust optional metadata labels here; this run is the benign reference by construction.
    return _force_gt_label(samples, gt_label=0)


def _run_single_attack(
    *,
    attack_name: str,
    args: argparse.Namespace,
    run_root: Path,
    flowpure_params: dict[str, Any],
) -> list[dict[str, Any]]:
    attack_kind, attack_mode, attack_extra, drop_prefix = _attack_config(
        attack_name,
        seed_samples=args.seed_samples,
    )
    # MAZE/DisGuide train generator components with BatchNorm, so batch_size=1
    # can crash during forward passes. Use a dedicated extraction batch size.
    effective_attack_batch_size = args.attack_batch_size
    if attack_name in {"maze", "disguide"}:
        effective_attack_batch_size = max(2, int(args.extraction_attack_batch_size))

    spec = build_experiment_spec(
        name=f"flowpure-paperlike-{attack_name}",
        attack_kind=attack_kind,
        attack_mode=attack_mode,
        query_defense="flowpure",
        query_defense_parameters=flowpure_params,
        attack_extra=attack_extra,
        **_build_common_spec_kwargs(
            args,
            query_budget=args.attack_query_budget,
            attack_batch_size=effective_attack_batch_size,
        ),
    )
    result = run_experiment(spec, output_dir=run_root / spec.name)
    samples = extract_detection_samples(
        result.query_summary.history.records,
        defense_name="flowpure",
        fallback_gt=1,
    )
    # Do not trust optional metadata labels here; this run is an attack run by construction.
    samples = _force_gt_label(samples, gt_label=1)
    if drop_prefix > 0:
        samples = samples[drop_prefix:]
    return samples


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run notebook-like FlowPure paper-style defense evaluation (no visualizations)."
    )
    parser.add_argument("--name", default="flowpure-paperlike-eval")
    parser.add_argument("--output-root", default="runs/notebook")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--query-dataset", default="CIFAR10")
    parser.add_argument("--target-architecture", default="vgg16_bn")
    parser.add_argument("--substitute-architecture", default="vgg16_bn")
    parser.add_argument(
        "--target-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "notebook" / "training-victim-cifar10-vgg16_bn-nodefense" / "target_model",
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        default=None,
        help=(
            "FlowPure checkpoint path. If omitted, auto-selects a checkpoint in this order: "
            "cifar10_flowpure_pgd, cifar10_flowpure_pgd_smoke, cifar10_notebook."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--velocity-threshold", type=float, default=5000.0)
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target false-positive rate used to calibrate the score threshold on benign queries.",
    )
    parser.add_argument("--attacks", default="prada,maze,disguide")
    parser.add_argument("--seed-samples", type=int, default=16)
    parser.add_argument("--attack-query-budget", type=int, default=144)
    parser.add_argument("--benign-query-budget", type=int, default=144)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument(
        "--extraction-attack-batch-size",
        type=int,
        default=16,
        help="Batch size for MAZE/DisGuide extraction attacks (must be >1 for BatchNorm).",
    )
    parser.add_argument("--training-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.device == "cuda":
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    args.flow_checkpoint = _resolve_flow_checkpoint(args.flow_checkpoint)
    if not args.target_checkpoint_dir.exists():
        raise FileNotFoundError(f"Target checkpoint dir not found: {args.target_checkpoint_dir}")

    attack_names = _parse_attacks(args.attacks)
    run_root = Path(args.output_root) / args.name
    run_root.mkdir(parents=True, exist_ok=True)

    flowpure_params = _build_flowpure_params(
        flow_checkpoint=args.flow_checkpoint,
        dataset=args.dataset,
        device=args.device,
        velocity_threshold=args.velocity_threshold,
    )
    print(f"Using FlowPure checkpoint: {args.flow_checkpoint}")

    print("Running benign reference with FlowPure...")
    benign_samples_raw = _run_benign_reference(args=args, run_root=run_root, flowpure_params=flowpure_params)
    if not benign_samples_raw:
        raise RuntimeError("Benign run did not produce detection samples.")

    calibrated_threshold = _calibrate_threshold_from_samples(
        benign_samples_raw,
        target_fpr=float(args.target_fpr),
    )
    benign_samples = _apply_threshold(benign_samples_raw, threshold=calibrated_threshold)

    benign_scores = np.asarray([float(row["score"]) for row in benign_samples_raw], dtype=np.float64)
    benign_flag_rate = float(np.mean([row["pred"] for row in benign_samples])) if benign_samples else 0.0
    print(
        "Calibrated score threshold from benign run: "
        f"{calibrated_threshold:.4f} (target FPR={args.target_fpr:.2%}, observed={benign_flag_rate:.2%})"
    )
    print(
        "Benign score stats: "
        f"mean={benign_scores.mean():.2f}, median={np.median(benign_scores):.2f}, max={benign_scores.max():.2f}"
    )

    summary_rows: list[dict[str, Any]] = []
    for attack_name in attack_names:
        attack_kind_debug, attack_mode_debug, attack_extra_debug, _ = _attack_config(
            attack_name, seed_samples=args.seed_samples,
        )
        print(f"\nRunning attack: {attack_name} (kind={attack_kind_debug}, mode={attack_mode_debug})")
        attack_samples_raw = _run_single_attack(
            attack_name=attack_name,
            args=args,
            run_root=run_root,
            flowpure_params=flowpure_params,
        )
        if not attack_samples_raw:
            raise RuntimeError(f"Attack run '{attack_name}' did not produce detection samples.")

        attack_samples = _apply_threshold(attack_samples_raw, threshold=calibrated_threshold)

        combined_samples = benign_samples + attack_samples
        metrics = compute_detection_metrics(combined_samples)
        class_f1 = _compute_class_f1(combined_samples)

        row = {
            "attack": attack_name,
            "num_benign": len(benign_samples),
            "num_attack": len(attack_samples),
            "detection_rate_attack": metrics["detection_rate"],
            "f1_attack": class_f1["f1_attack"],
            "f1_benign": class_f1["f1_benign"],
            "f1_macro_total": metrics["f1_macro"],
            "f1_binary_attack_total": metrics["f1"],
            "precision_attack_total": metrics["precision"],
            "recall_attack_total": metrics["recall"],
            "false_positive_rate": metrics["fpr"],
        }
        summary_rows.append(row)

        print(
            "  detection_rate(attack): "
            f"{row['detection_rate_attack']:.4f} | "
            f"F1 attack: {row['f1_attack']:.4f} | "
            f"F1 benign: {row['f1_benign']:.4f} | "
            f"F1 macro total: {row['f1_macro_total']:.4f}"
        )

    payload = {
        "config": {
            "dataset": args.dataset,
            "query_dataset": args.query_dataset,
            "target_architecture": args.target_architecture,
            "substitute_architecture": args.substitute_architecture,
            "target_checkpoint_dir": str(args.target_checkpoint_dir),
            "flow_checkpoint": str(args.flow_checkpoint),
            "device": args.device,
            "flowpure_parameters": flowpure_params,
            "seed_samples": args.seed_samples,
            "attack_query_budget": args.attack_query_budget,
            "benign_query_budget": args.benign_query_budget,
            "attack_batch_size": args.attack_batch_size,
            "extraction_attack_batch_size": args.extraction_attack_batch_size,
            "target_fpr": args.target_fpr,
            "calibrated_threshold": calibrated_threshold,
        },
        "results": summary_rows,
    }
    out_path = run_root / "flowpure_paperlike_metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
