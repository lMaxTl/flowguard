"""Report how many queries each attack needs to reach a fidelity target.

Reads the ``query_curve`` field that
``scripts/evaluate_attack_defense_matrix_smoke.py`` stores per matrix cell (the
DisGUIDE-loop attacks record one point per epoch) and answers the question the
budget sweep would otherwise cost one run per budget to answer: at what query
count does the substitute first reach X% fidelity with the victim, and where
does it plateau.

Fidelity (agreement with the victim's top-1) rather than accuracy is the
headline: an extraction attacker is buying a copy of the victim's function, and
on a task where the victim is near-perfect the two numbers separate only in the
tail that matters.

Usage::

    python scripts/report_query_fidelity_benchmark.py \\
        --summary runs/mnist_benchmark/attack_defense_matrix_mnist-bench_summary.json \\
        --targets 80,90,95,99
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Attack key -> (row label, what the generator is penalized on). Ordering fixes
# the report order and mirrors Table "fully_adaptive" in the paper.
ROW_LABELS: tuple[tuple[str, str, str], ...] = (
    ("disguide", "baseline", "none (unconstrained DisGUIDE)"),
    ("flowguard_d1_latent_manifold", "D1", "C1"),
    ("flowguard_d2_procedural_natural", "D2", "C1"),
    ("flowguard_d3_adaptive_c1c2", "D3", "C1+C2"),
    ("flowguard_d4_adaptive_c1c3", "D4", "C1+C3"),
    ("flowguard_d5_adaptive_composite", "D5", "C1+C2+C3 (fused)"),
    ("flowguard_d6_adaptive_stateful", "D6", "C1-C5 (+ stateful)"),
)


def _queries_to_reach(
    curve: list[dict[str, Any]],
    *,
    field: str,
    target: float,
) -> float | None:
    """First query count at which ``field`` reaches ``target``.

    Returns ``None`` when the run never gets there. The curve is monotone only
    in expectation, so this reports the first crossing rather than requiring the
    value to stay above the target afterwards.
    """
    for point in curve:
        value = point.get(field)
        if value is not None and float(value) >= float(target):
            return float(point.get("queries", 0.0))
    return None


def _peak(curve: list[dict[str, Any]], *, field: str) -> tuple[float, float] | None:
    """Return ``(best_value, queries_at_best)`` for ``field``."""
    best: tuple[float, float] | None = None
    for point in curve:
        value = point.get(field)
        if value is None:
            continue
        candidate = (float(value), float(point.get("queries", 0.0)))
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def _format_queries(value: float | None) -> str:
    if value is None:
        return "never"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _load_curves(summary_path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    curves: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("results", []):
        curve = row.get("query_curve")
        if not curve:
            continue
        attack = str(row.get("attack"))
        # A cell is repeated per defense column; they are separate attack runs,
        # so keep the longest curve rather than silently mixing them.
        if attack not in curves or len(curve) > len(curves[attack]):
            curves[attack] = curve
    return curves


def _render(
    curves: dict[str, list[dict[str, Any]]],
    *,
    targets: list[float],
    field: str,
) -> str:
    header = ["attack", "adaptive to", "peak", "@queries"]
    header += [f"->{target:g}%" for target in targets]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    known = {key for key, _, _ in ROW_LABELS}
    ordered = list(ROW_LABELS) + [
        (key, key, "?") for key in sorted(curves) if key not in known
    ]

    for key, label, adaptive_to in ordered:
        curve = curves.get(key)
        if curve is None:
            continue
        best = _peak(curve, field=field)
        cells = [
            label,
            adaptive_to,
            f"{best[0]:.2f}%" if best else "n/a",
            _format_queries(best[1]) if best else "n/a",
        ]
        for target in targets:
            cells.append(
                _format_queries(_queries_to_reach(curve, field=field, target=target))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True,
                        help="Matrix summary JSON. Repeat to merge runs.")
    parser.add_argument("--targets", default="80,90,95,99",
                        help="Comma-separated fidelity targets in percent.")
    parser.add_argument("--field", default="surrogate_fidelity",
                        choices=["surrogate_fidelity", "surrogate_accuracy"])
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write the markdown report to.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    targets = [float(item) for item in str(args.targets).split(",") if item.strip()]

    curves: dict[str, list[dict[str, Any]]] = {}
    for path in args.summary:
        if not path.exists():
            raise FileNotFoundError(f"Summary not found: {path}")
        for attack, curve in _load_curves(path).items():
            if attack not in curves or len(curve) > len(curves[attack]):
                curves[attack] = curve

    if not curves:
        raise SystemExit(
            "No query_curve data in the summary. Only the DisGUIDE-loop attacks "
            "(disguide, D1-D6) record one."
        )

    metric = "fidelity" if args.field == "surrogate_fidelity" else "accuracy"
    table = _render(curves, targets=targets, field=args.field)
    report = (
        f"# Queries to reach a {metric} target\n\n"
        f"Peak {metric} and the first query count reaching each target.\n"
        f"'never' means the run ended below that target.\n\n"
        f"{table}\n"
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()
