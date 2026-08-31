from __future__ import annotations

from flowguard.orchestration.results import EvaluationSummary


def build_evaluation_summary(
    metrics: dict[str, float],
    budget_sweeps: list[dict[str, float]] | None = None,
) -> EvaluationSummary:
    return EvaluationSummary(metrics=metrics, budget_sweeps=budget_sweeps or [])
