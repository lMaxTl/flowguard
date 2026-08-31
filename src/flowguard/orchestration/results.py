from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QuerySummary:
    total_queries: int = 0
    distinct_inputs: int = 0
    average_batch_size: float = 0.0
    total_batches: int = 0
    history: Any = None


@dataclass(slots=True)
class AttackSummary:
    attack_name: str
    mode_name: str
    output_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationSummary:
    metrics: dict[str, float] = field(default_factory=dict)
    budget_sweeps: list[dict[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentResult:
    experiment_name: str
    query_summary: QuerySummary
    attack_summary: AttackSummary
    evaluation_summary: EvaluationSummary
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
