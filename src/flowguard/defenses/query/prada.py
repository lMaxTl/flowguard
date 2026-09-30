from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats
import torch

from flowguard.defenses.query.base import QueryContext, QueryDefense, query_identities


def _l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the flattened L2 distance between two queries."""
    return float(np.linalg.norm(left.reshape(-1) - right.reshape(-1)))


def _default_threshold_update(distances: np.ndarray) -> float:
    """Update the growing-set threshold using mean plus standard deviation."""
    if distances.size == 0:
        return 0.0
    return float(np.mean(distances) + np.std(distances))


class _GrowingDistanceDetector:
    """Stateful PRADA detector operating on one query stream."""

    def __init__(
        self,
        *,
        shapiro_threshold: float,
        distance_metric: Callable[[np.ndarray, np.ndarray], float],
        threshold_update_rule: Callable[[np.ndarray], float],
        check_interval: int,
        min_class_distances: int,
        min_distribution_samples: int,
        outlier_std_factor: float,
    ) -> None:
        self.shapiro_threshold = float(shapiro_threshold)
        self.distance_metric = distance_metric
        self.threshold_update_rule = threshold_update_rule
        self.check_interval = int(check_interval)
        self.min_class_distances = int(min_class_distances)
        self.min_distribution_samples = int(min_distribution_samples)
        self.outlier_std_factor = float(outlier_std_factor)

        self.growing_set: dict[int, list[np.ndarray]] = defaultdict(list)
        self.growing_set_distances: dict[int, list[float]] = defaultdict(list)
        self.thresholds: dict[int, float] = defaultdict(float)
        self.input_distances: dict[int, list[float]] = defaultdict(list)

        self.queries_processed = 0
        self.attacker_detected = False

    def process_query(self, sample: np.ndarray, target_class: int) -> bool:
        """Update the detector state and report whether PRADA fired."""
        class_queries = self.growing_set[target_class]
        if not class_queries:
            class_queries.append(sample)
            self.growing_set_distances[target_class].append(0.0)
        else:
            distances = np.asarray(
                [self.distance_metric(reference, sample) for reference in class_queries],
                dtype=np.float64,
            )
            minimum_distance = float(np.min(distances))
            self.input_distances[target_class].append(minimum_distance)

            if minimum_distance > self.thresholds[target_class]:
                class_queries.append(sample)
                class_growing_distances = self.growing_set_distances[target_class]
                class_growing_distances.append(minimum_distance)
                updated_threshold = self.threshold_update_rule(
                    np.asarray(class_growing_distances, dtype=np.float64)
                )
                self.thresholds[target_class] = max(
                    float(updated_threshold),
                    float(self.thresholds[target_class]),
                )

        self.queries_processed += 1
        if self.queries_processed % self.check_interval == 0:
            self.attacker_detected = self._detect_attack()
        return self.attacker_detected

    def _detect_attack(self) -> bool:
        pooled_distances: list[float] = []
        for class_distances in self.input_distances.values():
            if len(class_distances) >= self.min_class_distances:
                pooled_distances.extend(class_distances)
        if len(pooled_distances) < self.min_distribution_samples:
            return False

        filtered_distances = self._reject_outliers(np.asarray(pooled_distances, dtype=np.float64))
        if len(filtered_distances) < self.min_distribution_samples:
            return False

        # Only the W statistic is used, and scipy documents it as accurate for
        # any N; its N > 5000 warning concerns the p-value alone. Unsilenced it
        # fires on every check of a long stream (the message embeds N, so it
        # is never deduplicated) and buries the Slurm logs.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*computed p-value may not be accurate.*",
                category=UserWarning,
            )
            shapiro_statistic, _ = stats.shapiro(filtered_distances)
        return float(shapiro_statistic) < self.shapiro_threshold

    def _reject_outliers(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        std = float(np.std(values))
        if std == 0.0:
            return values
        mean = float(np.mean(values))
        distance_from_mean = np.abs(values - mean)
        return values[distance_from_mean < self.outlier_std_factor * std]


class PradaQueryDefense(QueryDefense):
    """Detect model extraction attempts using the PRADA growing-distance test."""

    name = "prada"

    def __init__(
        self,
        shapiro_threshold: float = 0.95,
        *,
        check_interval: int = 10,
        min_class_distances: int = 10,
        min_distribution_samples: int = 100,
        outlier_std_factor: float = 3.0,
        audit_only: bool = False,
        **parameters: Any,
    ) -> None:
        """Initialise the PRADA detector state.

        Args:
            shapiro_threshold: Threshold for the Shapiro-Wilk statistic. Lower
                observed values indicate non-natural query distributions.
            check_interval: Run the statistical test every N processed queries.
            min_class_distances: Minimum number of collected distances per class
                before they contribute to the pooled distribution.
            min_distribution_samples: Minimum number of pooled distances needed
                before invoking the Shapiro-Wilk test.
            outlier_std_factor: Standard-deviation multiplier used for simple
                outlier rejection before the normality test.
            audit_only: If True, the defense will not raise blocking exceptions.
                Instead, it will record which queries would have been blocked in
                the query context metadata.
            **parameters: Additional serialisable parameters retained for
                experiment metadata compatibility.
        """
        super().__init__(
            shapiro_threshold=shapiro_threshold,
            check_interval=check_interval,
            min_class_distances=min_class_distances,
            min_distribution_samples=min_distribution_samples,
            outlier_std_factor=outlier_std_factor,
            audit_only=audit_only,
            **parameters,
        )
        # PRADA is a per-client monitor: each identity gets its own growing set
        # and its own Shapiro-Wilk test. A single shared detector (the previous
        # implementation) pooled every identity's queries, so splitting an
        # attack across Sybil identities could not affect it at all.
        self._detector_kwargs = dict(
            shapiro_threshold=shapiro_threshold,
            distance_metric=_l2_distance,
            threshold_update_rule=_default_threshold_update,
            check_interval=check_interval,
            min_class_distances=min_class_distances,
            min_distribution_samples=min_distribution_samples,
            outlier_std_factor=outlier_std_factor,
        )
        self.detectors: dict[Any, _GrowingDistanceDetector] = {}
        self.blocked_identities: set[Any] = set()
        self.audit_only = bool(audit_only)

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_identities)

    @property
    def detector(self) -> _GrowingDistanceDetector:
        """The single-client detector (kept for callers of the old attribute)."""
        return self._detector_for("default")

    def _detector_for(self, identity: Any) -> _GrowingDistanceDetector:
        if identity not in self.detectors:
            self.detectors[identity] = _GrowingDistanceDetector(**self._detector_kwargs)
        return self.detectors[identity]

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        """Reject further queries from identities PRADA has already flagged."""
        identities = query_identities(context, len(batch))
        blocked = [index for index, identity in enumerate(identities) if identity in self.blocked_identities]
        if not blocked:
            # The engine reuses one context across batches; drop the previous
            # batch's indices so they are not attributed to this one.
            context.metadata.pop("prada_blocked_indices", None)
            return batch, context
        if self.audit_only:
            context.metadata["prada_blocked_indices"] = blocked
            return batch, context
        raise RuntimeError("Query blocked by PRADA defense.")

    def after_query(
        self, batch: torch.Tensor, outputs: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        """Inspect defended outputs and flag identities whose stream PRADA rejects."""
        identities = query_identities(context, len(batch))
        newly_blocked = False
        for identity, sample, prediction in zip(identities, batch, outputs, strict=True):
            if identity in self.blocked_identities:
                continue
            sample_array = sample.detach().cpu().numpy()
            target_class = self._prediction_to_class_id(prediction)
            if self._detector_for(identity).process_query(sample_array, target_class):
                self.blocked_identities.add(identity)
                newly_blocked = True
        if newly_blocked:
            if not self.audit_only:
                raise RuntimeError("PRADA detected a suspicious query distribution.")
            context.metadata["prada_blocked_indices"] = [
                index for index, identity in enumerate(identities) if identity in self.blocked_identities
            ]
        return outputs, context

    @staticmethod
    def _prediction_to_class_id(prediction: torch.Tensor) -> int:
        if prediction.ndim == 0:
            return int(prediction.item())
        return int(torch.argmax(prediction).item())
