from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defenses import datasets as legacy_datasets
from flowguard.defenses.query.fdinet import FDINetQueryDefense
from flowguard.defenses.query.flow_matching import FlowMatchingQueryDefense
from flowguard.defenses.query.flowpure import FlowPureQueryDefense
from flowguard.defenses.query.prada import PradaQueryDefense
from flowguard.evaluation.detection import compute_detection_metrics, extract_detection_samples
from flowguard.experiments.factory import build_experiment_spec
from flowguard.experiments.spec import AttackKind
from flowguard.orchestration.runner import run_experiment
from flowguard.querying.engine import QueryEngine
from flowguard.serving.model_loader import load_legacy_model
from flowguard.serving.target_service import TargetService


def _parse_csv_argument(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _flow_checkpoint_default(repo_root: Path) -> Path:
    return repo_root / "runs" / "flow_matching" / "cifar10_notebook" / "checkpoint_latest.pt"


def _resolve_target_checkpoint_dir(raw: str | None) -> str | None:
    if raw is None:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        if candidate.name.lower().startswith("checkpoint"):
            return str(candidate.parent)
        raise ValueError(
            "--target-checkpoint-dir points to a file that is not a checkpoint. "
            f"Received: {candidate}"
        )
    return str(candidate)


def _defense_parameters(
    *,
    defense_name: str,
    dataset: str,
    target_device: str,
    flow_checkpoint: Path,
) -> dict[str, Any]:
    normalized = defense_name.lower()
    if normalized == "fdinet":
        return {
            "audit_only": True,
            "bootstrap": True,
            "num_anchor_samples_per_class": 10,
            "anchor_candidates_per_class": 48,
            "benign_calibration_samples": 192,
            "malicious_calibration_samples": 192,
            "malicious_noise_std": 0.2,
            "target_fpr": 0.05,
            "calibration_batch_size": 64,
            "dataset_download": False,
        }
    if normalized == "prada":
        return {
            "audit_only": True,
            "check_interval": 5,
            "min_class_distances": 5,
            "min_distribution_samples": 40,
            "shapiro_threshold": 0.95,
        }
    if normalized in {"flow_matching", "flowmatching", "fm"}:
        return {
            "audit_only": True,
            "fm_checkpoint_path": str(flow_checkpoint),
            "dataset_name": dataset,
            "device": target_device,
            "likelihood_threshold": 5000.0,
        }
    if normalized == "flowpure":
        return {
            "audit_only": True,
            "fm_checkpoint_path": str(flow_checkpoint),
            "dataset_name": dataset,
            "device": target_device,
            "velocity_threshold": 5000.0,
        }
    raise ValueError(f"Unsupported defense '{defense_name}'.")


def _parse_defense_overrides(raw: str | None) -> dict[str, dict[str, Any]]:
    if raw is None or not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--defense-overrides-json must decode to an object/dict.")
    normalized: dict[str, dict[str, Any]] = {}
    for defense_name, parameters in payload.items():
        if not isinstance(parameters, dict):
            raise ValueError(
                "Each defense override must be a JSON object. "
                f"Invalid entry for '{defense_name}'."
            )
        normalized[str(defense_name).lower()] = dict(parameters)
    return normalized


def _attack_spec_overrides(attack_name: str) -> tuple[AttackKind, str, dict[str, Any]]:
    normalized = attack_name.lower()
    if normalized in {"naive", "transfer_set", "transfer"}:
        return AttackKind.TRANSFER_SET, "naive", {}
    if normalized == "maze":
        return AttackKind.MAZE, "maze", {
            "latent_dim": 16,
            "iter_gen": 1,
            "iter_clone": 2,
            "iter_exp": 1,
            "ndirs": 1,
            "log_iter": 50,
            "disable_pbar": True,
        }
    if normalized == "disguide":
        return AttackKind.DISGUIDE, "disguide", {
            "ensemble_size": 2,
            "latent_dim": 16,
            "g_iter": 1,
            "d_iter": 1,
            "rep_iter": 1,
            "replay_size": 64,
            "epoch_itrs": 3,
            "disable_pbar": True,
        }
    raise ValueError(f"Unsupported attack '{attack_name}'.")


def _build_query_defense_instance(
    *,
    defense_name: str,
    defense_params: dict[str, Any],
    loaded_model,
):
    normalized = defense_name.lower()
    if normalized == "fdinet":
        return FDINetQueryDefense(loaded_model=loaded_model, **defense_params)
    if normalized == "prada":
        return PradaQueryDefense(**defense_params)
    if normalized in {"flow_matching", "flowmatching", "fm"}:
        return FlowMatchingQueryDefense(**defense_params)
    if normalized == "flowpure":
        return FlowPureQueryDefense(**defense_params)
    raise ValueError(f"Unsupported defense '{defense_name}'.")


def _run_benign_reference(
    *,
    checkpoint_dir: Path,
    target_device: str,
    defense_name: str,
    defense_params: dict[str, Any],
    benign_queries: int,
    benign_batch_size: int,
    dataset_download: bool,
) -> list[dict[str, Any]]:
    loaded_model = load_legacy_model(checkpoint_dir, device=target_device)
    defense = _build_query_defense_instance(
        defense_name=defense_name,
        defense_params=defense_params,
        loaded_model=loaded_model,
    )
    engine = QueryEngine(
        TargetService(loaded_model),
        query_defenses=[defense],
    )

    transform = legacy_datasets.modelfamily_to_transforms[loaded_model.modelfamily]["test"]
    dataset = legacy_datasets.__dict__[loaded_model.dataset_name](
        train=False,
        transform=transform,
        download=dataset_download,
    )

    queried = 0
    for offset in range(0, len(dataset), max(1, benign_batch_size)):
        if queried >= benign_queries:
            break
        samples = []
        for index in range(offset, min(len(dataset), offset + max(1, benign_batch_size))):
            sample, _ = dataset[index]
            samples.append(sample)
        if not samples:
            continue
        remaining = benign_queries - queried
        batch = samples[:remaining]
        import torch

        tensor_batch = torch.stack(batch, dim=0)
        engine.query_batch(
            tensor_batch,
            record=True,
            metadata={"fdinet_gt": 0, "fdinet_source": "benign"},
        )
        queried += len(batch)

    return extract_detection_samples(
        engine.history.records,
        defense_name=defense_name,
        fallback_gt=0,
    )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate query-defense detection metrics across attacks.")
    parser.add_argument("--name", default="detection-suite-quick")
    parser.add_argument("--output-root", default="runs/detection_suite")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--target-architecture", default="resnet18")
    parser.add_argument(
        "--target-checkpoint-dir",
        default="runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model",
        help="Target model directory or direct path to checkpoint.pth.tar.",
    )
    parser.add_argument("--query-dataset", default="TinyImageNet200")
    parser.add_argument("--target-device", default="cuda")
    parser.add_argument("--substitute-device", default="cuda")
    parser.add_argument("--query-budget", type=int, default=120)
    parser.add_argument("--query-transfer-set-size", type=int, default=160)
    parser.add_argument("--attack-batch-size", type=int, default=8)
    parser.add_argument("--training-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--benign-queries", type=int, default=160)
    parser.add_argument("--benign-batch-size", type=int, default=16)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed query execution for attack runs.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
        help="Number of logical clients/workers. Use 100 for a 100-client setup.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Thread-pool workers used by the distributed coordinator.",
    )
    parser.add_argument("--dataset-download", action="store_true")
    parser.add_argument("--defenses", default="fdinet,prada,flow_matching")
    parser.add_argument("--attacks", default="naive,maze,disguide")
    parser.add_argument("--max-runs", type=int, default=0, help="Debug limiter; 0 means run full matrix.")
    parser.add_argument("--flow-checkpoint", default=None)
    parser.add_argument(
        "--defense-overrides-json",
        default="",
        help=(
            "JSON dict of per-defense parameter overrides, e.g. "
            "'{\"prada\": {\"check_interval\": 1}}'"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    flow_checkpoint = Path(args.flow_checkpoint) if args.flow_checkpoint else _flow_checkpoint_default(repo_root)
    if not flow_checkpoint.exists():
        raise FileNotFoundError(
            "Flow Matching checkpoint not found. "
            f"Expected at '{flow_checkpoint}'. Provide --flow-checkpoint."
        )

    defenses = _parse_csv_argument(args.defenses)
    attacks = _parse_csv_argument(args.attacks)
    defense_overrides = _parse_defense_overrides(args.defense_overrides_json)
    distributed_enabled = bool(args.distributed)
    num_clients = max(1, int(args.num_clients))
    num_workers = max(1, int(args.num_workers))

    run_root = Path(args.output_root) / args.name
    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    target_checkpoint_dir = _resolve_target_checkpoint_dir(args.target_checkpoint_dir)
    completed = 0

    for defense_name in defenses:
        defense_params = _defense_parameters(
            defense_name=defense_name,
            dataset=args.dataset,
            target_device=args.target_device,
            flow_checkpoint=flow_checkpoint,
        )
        defense_params.update(defense_overrides.get(defense_name.lower(), {}))

        for attack_name in attacks:
            if args.max_runs > 0 and completed >= args.max_runs:
                break

            attack_kind, attack_mode, attack_extra = _attack_spec_overrides(attack_name)
            run_name = f"{defense_name}-{attack_name}"
            run_dir = run_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            spec = build_experiment_spec(
                name=run_name,
                dataset=args.dataset,
                target_architecture=args.target_architecture,
                target_checkpoint_dir=target_checkpoint_dir,
                attack_kind=attack_kind,
                attack_mode=attack_mode,
                query_dataset=args.query_dataset,
                query_budget=args.query_budget,
                query_transfer_set_size=args.query_transfer_set_size,
                query_defense=defense_name,
                query_defense_parameters=defense_params,
                target_device=args.target_device,
                substitute_device=args.substitute_device,
                attack_batch_size=args.attack_batch_size,
                training_batch_size=args.training_batch_size,
                epochs=args.epochs,
                attack_extra=attack_extra,
                query_download=args.dataset_download,
                dataset_download=args.dataset_download,
                distributed=distributed_enabled,
                num_clients=num_clients,
                num_workers=num_workers,
                verbose=True,
            )

            result = run_experiment(spec, output_dir=run_dir)
            target_checkpoint_dir = str(result.artifacts.get("target_model", target_checkpoint_dir))

            malicious_samples = extract_detection_samples(
                result.query_summary.history.records,
                defense_name=defense_name,
                fallback_gt=1,
            )
            benign_samples = _run_benign_reference(
                checkpoint_dir=Path(target_checkpoint_dir),
                target_device=args.target_device,
                defense_name=defense_name,
                defense_params=defense_params,
                benign_queries=args.benign_queries,
                benign_batch_size=args.benign_batch_size,
                dataset_download=args.dataset_download,
            )

            all_samples = benign_samples + malicious_samples
            metrics = compute_detection_metrics(all_samples)

            metrics_payload = {
                "defense": defense_name,
                "attack": attack_name,
                "attack_kind": attack_kind.value,
                "attack_mode": attack_mode,
                "query_budget": args.query_budget,
                "benign_queries": args.benign_queries,
                "distributed": distributed_enabled,
                "num_clients": num_clients,
                "num_workers": num_workers,
                "defense_parameters": defense_params,
                "attack_extra": attack_extra,
                "metrics": metrics,
            }
            with (run_dir / "detection_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle, indent=2)

            _write_rows_csv(
                run_dir / "detection_samples.csv",
                all_samples,
                fieldnames=["gt", "pred", "score", "source"],
            )

            summary_rows.append(
                {
                    "defense": defense_name,
                    "attack": attack_name,
                    "attack_kind": attack_kind.value,
                    "attack_mode": attack_mode,
                    "num_samples": metrics["num_samples"],
                    "num_benign": metrics["num_benign"],
                    "num_malicious": metrics["num_malicious"],
                    "detection_rate": metrics["detection_rate"],
                    "tpr": metrics["tpr"],
                    "fpr": metrics["fpr"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "f1_macro": metrics["f1_macro"],
                    "roc_auc": metrics["roc_auc"],
                    "distributed": distributed_enabled,
                    "num_clients": num_clients,
                    "num_workers": num_workers,
                    "run_dir": str(run_dir),
                }
            )
            completed += 1

        if args.max_runs > 0 and completed >= args.max_runs:
            break

    with (run_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    _write_rows_csv(
        run_root / "summary.csv",
        summary_rows,
        fieldnames=[
            "defense",
            "attack",
            "attack_kind",
            "attack_mode",
            "num_samples",
            "num_benign",
            "num_malicious",
            "detection_rate",
            "tpr",
            "fpr",
            "precision",
            "recall",
            "f1",
            "f1_macro",
            "roc_auc",
            "distributed",
            "num_clients",
            "num_workers",
            "run_dir",
        ],
    )

    print(f"Completed {completed} defense/attack runs.")
    print(f"Summary JSON: {run_root / 'summary.json'}")
    print(f"Summary CSV: {run_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
