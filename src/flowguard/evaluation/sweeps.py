from __future__ import annotations

from typing import Callable


def budget_sweep(
    budgets: list[int],
    evaluator: Callable[[int], dict[str, float]],
) -> list[dict[str, float]]:
    results = []
    for budget in budgets:
        metrics = evaluator(budget)
        metrics["budget"] = float(budget)
        results.append(metrics)
    return results
