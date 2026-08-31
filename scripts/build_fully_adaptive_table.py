"""Render the FlowGuard++-adaptive attacker results table from a matrix run.

Reads the ``attack_defense_matrix_*_summary.json`` produced by
``scripts/evaluate_attack_defense_matrix_smoke.py`` and writes
``paper/tables/fully_adaptive_results.tex``.

Cells with no completed run stay ``\\pending`` (a red "??" in the compiled PDF)
rather than being silently dropped or zero-filled, so a partially finished
sweep is visible in the paper instead of looking like a result.

Usage::

    python scripts/build_fully_adaptive_table.py \\
        --summary runs/adaptive_full/attack_defense_matrix_dist_summary.json \\
        --output paper/tables/fully_adaptive_results.tex

Pass ``--summary`` more than once to merge shards of a Slurm array run; later
files win on conflicting cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defense key -> column header. Matches the component labels used in
# Section 5 and in the main attack/defense matrix tables.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("flowpure", r"\textbf{C1}"),
    ("flowguard_integral", r"\textbf{C2}"),
    ("flow_matching", r"\textbf{C3}"),
    ("flowguard_userlevel", r"\textbf{C4}"),
    ("flowguard_composite", r"\textbf{Comp.}"),
)

# Attack key -> (row label, "adaptive to" cell). Order fixes the table order.
VELOCITY_ADAPTIVE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("flowguard_d1_latent_manifold", "D1 latent manifold", "C1"),
    ("flowguard_d2_procedural_natural", "D2 procedural natural", "C1"),
)

FULLY_ADAPTIVE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("flowguard_d3_adaptive_c1c2", "D3", "C1+C2"),
    ("flowguard_d4_adaptive_c1c3", "D4", "C1+C3"),
    ("flowguard_d5_adaptive_composite", "D5", "C1+C2+C3 (fused)"),
    ("flowguard_d6_adaptive_stateful", "D6", "C1--C5 (+ stateful)"),
)

PENDING = r"\pending"


def _load_cells(summary_paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map ``(attack, defense) -> result row``, later files overriding earlier."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row.get("error"):
                continue
            key = (str(row.get("attack")), str(row.get("defense")))
            cells[key] = row
    return cells


def _format_auroc(row: dict[str, Any] | None) -> str:
    if row is None or row.get("auroc") is None:
        return PENDING
    value = float(row["auroc"])
    # Sub-chance values are informative here (see Section 7.3), so keep the
    # extra digits that distinguish "barely misses" from "inverted".
    return f"{value:.4f}" if value < 0.01 else f"{value:.3f}"


def _format_accuracy(cells: dict[tuple[str, str], dict[str, Any]], attack: str) -> str:
    """Substitute accuracy, taken from the composite cell when available."""
    for defense in ("flowguard_composite", "flowpure"):
        row = cells.get((attack, defense))
        if row is not None and row.get("accuracy") is not None:
            return f"{float(row['accuracy']):.1f}\\%"
    return PENDING


def _render_row(
    cells: dict[tuple[str, str], dict[str, Any]],
    *,
    attack: str,
    label: str,
    adaptive_to: str,
) -> str:
    values = [_format_auroc(cells.get((attack, defense))) for defense, _ in COLUMNS]
    values.append(_format_accuracy(cells, attack))
    return f"{label} & {adaptive_to} &\n" + " & ".join(values) + r" \\"


def _render_table(
    cells: dict[tuple[str, str], dict[str, Any]],
    *,
    num_identities: int,
    query_budget: int,
) -> str:
    header = " &\n".join(
        [r"\textbf{Attack}", r"\textbf{Adaptive to}"]
        + [head for _, head in COLUMNS]
        + [r"\textbf{Sub.\ acc.}"]
    )
    velocity_rows = "\n\n".join(
        _render_row(cells, attack=attack, label=label, adaptive_to=adaptive)
        for attack, label, adaptive in VELOCITY_ADAPTIVE_ROWS
    )
    adaptive_rows = "\n\n".join(
        _render_row(cells, attack=attack, label=label, adaptive_to=adaptive)
        for attack, label, adaptive in FULLY_ADAPTIVE_ROWS
    )
    return f"""% GENERATED FILE - do not edit the numbers by hand.
%
% Rebuild with:
%   python scripts/build_fully_adaptive_table.py \\
%       --summary <matrix summary json> \\
%       --output paper/tables/fully_adaptive_results.tex
%
% Cells with no completed run render as \\pending (a red "??").
\\begin{{table*}}[!t]
\\caption{{
Detection performance against attackers adaptive to FlowGuard++ itself
($N={num_identities:,} $ Sybil identities; $K={query_budget:,} $ total queries).
D1 and D2 repeat the velocity-adaptive rows of Table~\\ref{{tab:distributed_attack_defense}}
for reference; D3--D6 add penalties on the components named in the
\\textbf{{Adaptive to}} column, following Eq.~\\ref{{eq:fully_adaptive_generator}}.
All detectors are calibrated on $50{{,}}000$ benign queries at a target \\gls{{FPR}}
of $5\\%$. Entries report \\gls{{AUROC}}; higher is better for the defender.
\\emph{{Sub.\\ acc.}} is the substitute's CIFAR-10 test accuracy after extraction,
which bounds how useful the evasion is to the attacker.
}}
\\label{{tab:fully_adaptive}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}ll ccccc c@{{}}}}
\\toprule
{header} \\\\
\\midrule

\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{Adaptive to single-point flow-velocity detection}}}} \\\\

{velocity_rows}

\\addlinespace
\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{Adaptive to FlowGuard++}}}} \\\\

{adaptive_rows}

\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, action="append", required=True,
        help="Matrix summary JSON. Repeat to merge Slurm-array shards.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "paper" / "tables" / "fully_adaptive_results.tex",
    )
    parser.add_argument("--num-identities", type=int, default=1250)
    parser.add_argument("--query-budget", type=int, default=50_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    missing = [path for path in args.summary if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Summary file(s) not found: {missing}")

    cells = _load_cells(list(args.summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _render_table(
            cells,
            num_identities=int(args.num_identities),
            query_budget=int(args.query_budget),
        ),
        encoding="utf-8",
    )

    wanted = [attack for attack, _, _ in VELOCITY_ADAPTIVE_ROWS + FULLY_ADAPTIVE_ROWS]
    filled = sum(
        1
        for attack in wanted
        for defense, _ in COLUMNS
        if (attack, defense) in cells
    )
    total = len(wanted) * len(COLUMNS)
    print(f"[table] wrote {args.output}")
    print(f"[table] {filled}/{total} cells populated; the rest render as \\pending")
    for attack in wanted:
        pending = [defense for defense, _ in COLUMNS if (attack, defense) not in cells]
        if pending:
            print(f"[table]   {attack}: missing {', '.join(pending)}")


if __name__ == "__main__":
    main()
