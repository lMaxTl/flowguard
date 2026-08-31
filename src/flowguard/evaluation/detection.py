from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

_DETECTION_QUANTILES: tuple[tuple[str, float], ...] = (
    ("p50", 0.50),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)


def _resolve_flags_and_scores(
    metadata: dict[str, Any],
    *,
    defense_name: str,
    batch_size: int,
) -> tuple[list[bool], list[float]]:
    normalized = defense_name.lower()

    if normalized == "fdinet":
        raw_flags = metadata.get("fdinet_flags")
        raw_scores = metadata.get("fdinet_scores")
        flags = list(raw_flags) if isinstance(raw_flags, list) else [False for _ in range(batch_size)]
        scores = (
            [float(value) for value in raw_scores]
            if isinstance(raw_scores, list)
            else [1.0 if flag else 0.0 for flag in flags]
        )
        return flags, scores

    if normalized == "prada":
        blocked = metadata.get("prada_blocked_indices")
        blocked_set = set(blocked) if isinstance(blocked, list) else set()
        flags = [index in blocked_set for index in range(batch_size)]
        scores = [1.0 if flag else 0.0 for flag in flags]
        return flags, scores

    if normalized in {"flow_matching", "flowmatching", "fm"}:
        raw_flags = metadata.get("flow_matching_blocked")
        raw_scores = metadata.get("queries_blocked")
        flags = list(raw_flags) if isinstance(raw_flags, list) else [False for _ in range(batch_size)]
        scores = (
            [float(value) for value in raw_scores]
            if isinstance(raw_scores, list)
            else [1.0 if flag else 0.0 for flag in flags]
        )
        return flags, scores

    if normalized == "flowpure":
        raw_flags = metadata.get("flowpure_blocked")
        raw_scores = metadata.get("flowpure_scores")
        flags = list(raw_flags) if isinstance(raw_flags, list) else [False for _ in range(batch_size)]
        scores = (
            [float(value) for value in raw_scores]
            if isinstance(raw_scores, list)
            else [1.0 if flag else 0.0 for flag in flags]
        )
        return flags, scores

    raise ValueError(f"Unsupported defense_name '{defense_name}' for detection extraction.")


def extract_detection_samples(
    records: list[Any],
    *,
    defense_name: str,
    gt_key: str = "fdinet_gt",
    fallback_gt: int | None = None,
) -> list[dict[str, Any]]:
    """Expand query-history records into per-query detection labels and scores."""

    samples: list[dict[str, Any]] = []
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        record_batch_size = int(getattr(record, "batch_size", metadata.get("batch_size", 0)))

        flags, scores = _resolve_flags_and_scores(
            metadata,
            defense_name=defense_name,
            batch_size=record_batch_size,
        )

        length = min(len(flags), len(scores))
        if length <= 0:
            continue

        raw_gt = metadata.get(gt_key, fallback_gt)
        if isinstance(raw_gt, list):
            gts = [int(value) for value in raw_gt]
        elif raw_gt is None:
            gts = [fallback_gt for _ in range(length)]
        else:
            gts = [int(raw_gt) for _ in range(length)]

        source = str(metadata.get("fdinet_source", metadata.get("source", "unknown")))
        for index in range(length):
            gt_value = gts[index] if index < len(gts) else gts[-1]
            if gt_value is None:
                continue
            samples.append(
                {
                    "gt": int(gt_value),
                    "pred": int(bool(flags[index])),
                    "score": float(scores[index]),
                    "source": source,
                }
            )
    return samples


def _score_quantiles(scores: np.ndarray) -> dict[str, float | None]:
    if scores.size == 0:
        return {name: None for name, _ in _DETECTION_QUANTILES}
    return {
        name: float(np.quantile(scores, quantile))
        for name, quantile in _DETECTION_QUANTILES
    }


def _score_mean(scores: np.ndarray) -> float | None:
    if scores.size == 0:
        return None
    return float(np.mean(scores))


def _tpr_at_calibrated_fpr(
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    *,
    target_fpr: float,
) -> tuple[float | None, float | None]:
    if benign_scores.size == 0 or attack_scores.size == 0:
        return None, None
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1); got {target_fpr}.")
    threshold = float(np.quantile(benign_scores, 1.0 - target_fpr))
    tpr = float(np.mean(attack_scores > threshold))
    return tpr, threshold


def compute_detection_metrics(
    samples: list[dict[str, Any]],
    *,
    calibrated_fpr: float = 0.05,
) -> dict[str, Any]:
    """Compute detection metrics from per-query samples."""

    if not samples:
        return {
            "num_samples": 0,
            "num_benign": 0,
            "num_malicious": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "detection_rate": 0.0,
            "detection_rate_overall": 0.0,
            "tpr": 0.0,
            "fpr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_macro": 0.0,
            "roc_auc": None,
            "auroc": None,
            "tpr_at_calibrated_fpr": None,
            "calibrated_threshold": None,
            "score_mean_benign": None,
            "score_mean_attack": None,
            "score_quantiles_benign": _score_quantiles(np.asarray([], dtype=np.float64)),
            "score_quantiles_attack": _score_quantiles(np.asarray([], dtype=np.float64)),
            "accepted_queries": 0,
            "blocked_queries": 0,
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
    detection_rate_overall = float(np.mean(y_pred == 1)) if len(y_pred) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    roc_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
    benign_scores = y_score[y_true == 0]
    attack_scores = y_score[y_true == 1]
    calibrated_tpr, calibrated_threshold = _tpr_at_calibrated_fpr(
        benign_scores,
        attack_scores,
        target_fpr=calibrated_fpr,
    )

    return {
        "num_samples": int(len(samples)),
        "num_benign": int(np.sum(y_true == 0)),
        "num_malicious": int(np.sum(y_true == 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detection_rate": tpr,
        "detection_rate_overall": detection_rate_overall,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "auroc": roc_auc,
        "tpr_at_calibrated_fpr": calibrated_tpr,
        "calibrated_threshold": calibrated_threshold,
        "score_mean_benign": _score_mean(benign_scores),
        "score_mean_attack": _score_mean(attack_scores),
        "score_quantiles_benign": _score_quantiles(benign_scores),
        "score_quantiles_attack": _score_quantiles(attack_scores),
        "accepted_queries": int(np.sum(y_pred == 0)),
        "blocked_queries": int(np.sum(y_pred == 1)),
    }


def build_attack_defense_result_block(
    samples: list[dict[str, Any]],
    *,
    query_budget: int,
    substitute_accuracy: float | None = None,
    substitute_agreement: float | None = None,
    substitute_fidelity: float | None = None,
    calibrated_fpr: float = 0.05,
) -> dict[str, Any]:
    """Return the shared FlowPure-bypass result block required by reports.

    The primary detector operating-point metric is
    ``tpr_at_calibrated_fpr``. AUROC is kept as ``auroc`` for ranking but is
    not the only detector result.
    """

    metrics = compute_detection_metrics(samples, calibrated_fpr=calibrated_fpr)
    return {
        "auroc": metrics["auroc"],
        "tpr_at_calibrated_fpr": metrics["tpr_at_calibrated_fpr"],
        "fpr": metrics["fpr"],
        "score_mean_benign": metrics["score_mean_benign"],
        "score_mean_attack": metrics["score_mean_attack"],
        "score_quantiles_benign": metrics["score_quantiles_benign"],
        "score_quantiles_attack": metrics["score_quantiles_attack"],
        "substitute_accuracy": substitute_accuracy,
        "substitute_agreement": substitute_agreement,
        "substitute_fidelity": substitute_fidelity,
        "query_budget": int(query_budget),
        "accepted_queries": metrics["accepted_queries"],
        "blocked_queries": metrics["blocked_queries"],
    }
