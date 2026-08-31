from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from defenses import datasets as legacy_datasets
from flowguard.defenses.query.fdinet import FDINetQueryDefense
from flowguard.experiments.factory import build_experiment_spec
from flowguard.experiments.spec import AttackKind
from flowguard.orchestration.runner import run_experiment
from flowguard.querying.engine import QueryEngine
from flowguard.serving.model_loader import load_legacy_model
from flowguard.serving.target_service import TargetService


def _parse_attack_kind(raw: str) -> AttackKind:
    normalized = raw.strip().lower()
    for kind in AttackKind:
        if kind.value == normalized:
            return kind
    supported = ", ".join(kind.value for kind in AttackKind)
    raise ValueError(f"Unsupported attack kind '{raw}'. Supported: {supported}")


def _extract_samples_from_history(records: list[Any], fallback_gt: int | None = None) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        flags = metadata.get("fdinet_flags")
        scores = metadata.get("fdinet_scores")
        if not isinstance(flags, list):
            continue

        raw_gt = metadata.get("fdinet_gt", fallback_gt)
        if isinstance(raw_gt, list):
            gts = [int(value) for value in raw_gt]
        else:
            if raw_gt is None:
                gts = [fallback_gt for _ in flags]
            else:
                gts = [int(raw_gt) for _ in flags]

        source = str(metadata.get("fdinet_source", "unknown"))
        for index, flagged in enumerate(flags):
            gt_value = gts[index] if index < len(gts) else gts[-1]
            if gt_value is None:
                continue
            score = float(flagged)
            if isinstance(scores, list) and index < len(scores):
                score = float(scores[index])
            expanded.append(
                {
                    "gt": int(gt_value),
                    "pred": int(bool(flagged)),
                    "score": score,
                    "source": source,
                }
            )
    return expanded


def _run_benign_reference(
    *,
    checkpoint_dir: Path,
    dataset_download: bool,
    query_count: int,
    batch_size: int,
    target_device: str,
    fdinet_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    loaded_model = load_legacy_model(checkpoint_dir, device=target_device)
    defense = FDINetQueryDefense(loaded_model=loaded_model, **fdinet_parameters)
    engine = QueryEngine(
        TargetService(loaded_model),
        query_defenses=[defense],
    )

    dataset_name = loaded_model.dataset_name
    modelfamily = loaded_model.modelfamily
    transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    dataset = legacy_datasets.__dict__[dataset_name](
        train=False,
        transform=transform,
        download=dataset_download,
    )
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, num_workers=0)

    queried = 0
    for inputs, _ in loader:
        if queried >= query_count:
            break
        remaining = query_count - queried
        batch = inputs[:remaining]
        engine.query_batch(
            batch,
            record=True,
            metadata={
                "fdinet_gt": 0,
                "fdinet_source": "benign",
            },
        )
        queried += len(batch)

    return _extract_samples_from_history(engine.history.records, fallback_gt=0)


def _compute_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "num_samples": 0,
            "num_benign": 0,
            "num_malicious": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tpr": 0.0,
            "fpr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
        }

    y_true = np.asarray([row["gt"] for row in samples], dtype=np.int64)
    y_pred = np.asarray([row["pred"] for row in samples], dtype=np.int64)
    y_score = np.asarray([row["score"] for row in samples], dtype=np.float64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))

    return {
        "num_samples": int(len(samples)),
        "num_benign": int(np.sum(y_true == 0)),
        "num_malicious": int(np.sum(y_true == 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FDINet detection evaluation.")
    parser.add_argument("--name", default="fdinet-detection", help="Experiment run name.")
    parser.add_argument("--output-root", default="runs/fdinet_detection", help="Directory for outputs.")
    parser.add_argument("--dataset", default="CIFAR10", help="Target dataset name.")
    parser.add_argument("--target-architecture", default="resnet18", help="Target model architecture.")
    parser.add_argument("--target-checkpoint-dir", default=None, help="Existing target checkpoint dir.")
    parser.add_argument("--query-dataset", default="TinyImageNet200", help="Attacker query dataset.")
    parser.add_argument("--attack-kind", default="prada", help="Attack kind value (default: prada).")
    parser.add_argument("--attack-mode", default="prada", help="Attack mode name (default: prada).")
    parser.add_argument("--query-budget", type=int, default=3_000, help="Maximum attack queries.")
    parser.add_argument(
        "--query-transfer-set-size",
        type=int,
        default=2_000,
        help="Optional subset size from query pool for faster debug runs.",
    )
    parser.add_argument("--attack-batch-size", type=int, default=16, help="Attack query batch size.")
    parser.add_argument("--training-batch-size", type=int, default=64, help="Substitute training batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Substitute training epochs.")
    parser.add_argument("--target-device", default="cpu", help="Target model device.")
    parser.add_argument("--substitute-device", default="cpu", help="Substitute model device.")
    parser.add_argument("--benign-queries", type=int, default=1_000, help="Number of benign reference queries.")
    parser.add_argument("--calibration-batch-size", type=int, default=64, help="FDINet calibration batch size.")
    parser.add_argument(
        "--num-anchor-samples-per-class",
        type=int,
        default=10,
        help="Number of anchor samples per class.",
    )
    parser.add_argument(
        "--anchor-candidates-per-class",
        type=int,
        default=64,
        help="Number of high-confidence anchor candidates per class.",
    )
    parser.add_argument(
        "--benign-calibration-samples",
        type=int,
        default=256,
        help="Number of benign calibration samples for FDINet classifier fitting.",
    )
    parser.add_argument(
        "--malicious-calibration-samples",
        type=int,
        default=256,
        help="Number of malicious calibration samples for FDINet classifier fitting.",
    )
    parser.add_argument(
        "--malicious-noise-std",
        type=float,
        default=0.2,
        help="Noise std used to generate synthetic malicious calibration samples.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Target false-positive rate for threshold calibration.")
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=None,
        help="Optional fixed malicious-score threshold (overrides target_fpr calibration).",
    )
    parser.add_argument("--dataset-download", action="store_true", help="Allow dataset auto-download.")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--audit-only",
        action="store_true",
        dest="audit_only",
        default=True,
        help="Record suspicious queries without blocking (default).",
    )
    mode_group.add_argument(
        "--blocking",
        action="store_false",
        dest="audit_only",
        help="Block suspicious queries instead of only auditing.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    attack_kind = _parse_attack_kind(args.attack_kind)
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    fdinet_parameters = {
        "num_anchor_samples_per_class": args.num_anchor_samples_per_class,
        "anchor_candidates_per_class": args.anchor_candidates_per_class,
        "benign_calibration_samples": args.benign_calibration_samples,
        "malicious_calibration_samples": args.malicious_calibration_samples,
        "malicious_noise_std": args.malicious_noise_std,
        "calibration_batch_size": args.calibration_batch_size,
        "target_fpr": args.target_fpr,
        "detection_threshold": args.detection_threshold,
        "audit_only": args.audit_only,
        "dataset_download": args.dataset_download,
    }

    spec = build_experiment_spec(
        name=args.name,
        dataset=args.dataset,
        target_architecture=args.target_architecture,
        attack_kind=attack_kind,
        attack_mode=args.attack_mode,
        query_dataset=args.query_dataset,
        query_budget=args.query_budget,
        query_transfer_set_size=args.query_transfer_set_size,
        query_defense="fdinet",
        query_defense_parameters=fdinet_parameters,
        target_checkpoint_dir=args.target_checkpoint_dir,
        target_device=args.target_device,
        substitute_device=args.substitute_device,
        attack_batch_size=args.attack_batch_size,
        training_batch_size=args.training_batch_size,
        epochs=args.epochs,
        query_download=args.dataset_download,
        dataset_download=args.dataset_download,
    )

    result = run_experiment(spec, output_dir=run_dir)
    malicious_samples = _extract_samples_from_history(result.query_summary.history.records)

    target_checkpoint_dir = Path(result.artifacts["target_model"])
    benign_samples = _run_benign_reference(
        checkpoint_dir=target_checkpoint_dir,
        dataset_download=args.dataset_download,
        query_count=args.benign_queries,
        batch_size=args.attack_batch_size,
        target_device=args.target_device,
        fdinet_parameters=fdinet_parameters,
    )

    all_samples = benign_samples + malicious_samples
    metrics = _compute_metrics(all_samples)
    metrics_payload = {
        "experiment": {
            "name": args.name,
            "attack_kind": attack_kind.value,
            "attack_mode": args.attack_mode,
            "query_budget": args.query_budget,
            "benign_queries": args.benign_queries,
        },
        "fdinet_parameters": fdinet_parameters,
        "metrics": metrics,
    }

    json_path = run_dir / "fdinet_detection_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    csv_path = run_dir / "fdinet_detection_samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gt", "pred", "score", "source"])
        writer.writeheader()
        writer.writerows(all_samples)

    print("FDINet detection evaluation completed.")
    print(f"Metrics written to: {json_path}")
    print(f"Per-query samples written to: {csv_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
