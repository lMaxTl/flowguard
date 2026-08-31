"""Trace every number in the paper's result tables back to a run summary.

Why this exists
---------------
``build_attack_defense_tables.py`` generates the matrices from run summaries and
refuses smoke-scale sources, but the committed ``paper/tables/*.tex`` files were
edited by hand afterwards. A hand-edited cell has no provenance: nothing checks
that it came from a run at the budget its caption claims, or from a run at all.

This script closes that gap from the other direction. It parses the published
tables, then searches every ``*summary*.json`` under ``runs/`` for a row whose
AUROC matches the published value. For each cell it reports the matching
sources and their query budget, and flags:

- cells whose only match comes from a run below ``--min-budget`` (the smoke-run
  defect: a caption claiming K=50,000 filled from a 256-query run);
- cells with no match anywhere in ``runs/`` (unreproducible from this repo);
- cells matched only by a run whose recorded attack/defense pair differs from
  the table position they occupy.

Usage::

    python scripts/audit_paper_table_provenance.py
    python scripts/audit_paper_table_provenance.py --markdown docs/table_provenance.md
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Column order of the CIFAR-10 matrices, matching the published header.
CIFAR_COLUMNS: tuple[str, ...] = (
    "PRADA", "FlowPure (C1)", "FDINet", "Int. (C2)", "Like. (C3)",
    "KS (C4)", "Label (C5)", "FlowGuard++", "FlowGuard++ (L)",
)
MNIST_COLUMNS: tuple[str, ...] = (
    "PRADA", "FlowPure (C1)", "FDINet", "Int. (C2)", "KS (C4)",
    "Label (C5)", "FlowGuard++",
)

# Column label -> defense key, row label -> attack key. A match is only
# evidence of provenance if it comes from the same (attack, defense) pair the
# cell occupies; an AUROC that happens to coincide with an unrelated pair says
# nothing about where the published number came from.
COLUMN_TO_DEFENSE: dict[str, str] = {
    "PRADA": "prada",
    "FlowPure (C1)": "flowpure",
    "FDINet": "fdinet",
    "Int. (C2)": "flowguard_integral",
    "Like. (C3)": "flow_matching",
    "KS (C4)": "flowguard_userlevel",
    "Label (C5)": "flowguard_labelhist",
    "FlowGuard++": "flowguard_composite",
    "FlowGuard++ (L)": "flowguard_composite_learned",
}

ROW_TO_ATTACK: dict[str, str] = {
    "PRADA attack": "prada",
    "MAZE": "maze",
    "DisGUIDE": "disguide",
    "Velocity-adaptive (Vel.)": "flowguard_d1_latent_manifold",
    "Trajectory-adaptive (Vel.+Traj.)": "flowguard_d3_adaptive_c1c2",
    "Typicality-adaptive (Vel.+Typ.)": "flowguard_d4_adaptive_c1c3",
    "Per-query composite-adaptive (PQ-Comp.)": "flowguard_d5_adaptive_composite",
    "Fully adaptive (Full-Comp.)": "flowguard_d6_adaptive_stateful",
}

CELL_RE = re.compile(r"(\d\.\d{3,4})\s*/\s*(\d\.\d{3,4}|--)")
ROW_LABEL_RE = re.compile(r"^([^&%\\][^&]*?)\s*&")


@dataclass
class Cell:
    table: str
    row: str
    column: str
    auroc: float
    f1: float | None
    matches: list[dict[str, Any]] = field(default_factory=list)


def _parse_table(path: Path, columns: tuple[str, ...]) -> list[Cell]:
    """Extract (row label, column, AUROC, F1) from a published table."""
    text = path.read_text(encoding="utf-8")
    body = text.split(r"\midrule", 1)[-1].split(r"\bottomrule", 1)[0]
    cells: list[Cell] = []
    for raw_row in body.split(r"\\"):
        row = raw_row.strip()
        if not row or row.startswith("%") or r"\multicolumn" in row or r"\addlinespace" in row:
            continue
        parts = [part.strip() for part in row.split("&")]
        if len(parts) < 2:
            continue
        label = re.sub(r"\\[a-zA-Z]+|[{}$]", "", parts[0]).strip()
        for index, part in enumerate(parts[1:]):
            if index >= len(columns):
                break
            match = CELL_RE.search(part)
            if not match:
                continue
            f1_text = match.group(2)
            cells.append(
                Cell(
                    table=path.name,
                    row=label,
                    column=columns[index],
                    auroc=float(match.group(1)),
                    f1=None if f1_text == "--" else float(f1_text),
                )
            )
    return cells


def _load_summaries(runs_root: Path) -> list[dict[str, Any]]:
    """Flatten every summary row, keeping the provenance of each."""
    records: list[dict[str, Any]] = []
    for path in sorted(runs_root.rglob("*summary*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
        for row in payload.get("results", []) or []:
            if not isinstance(row, dict) or not isinstance(row.get("auroc"), (int, float)):
                continue
            records.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "run_label": config.get("run_label"),
                    "query_budget": config.get("query_budget"),
                    "benign_query_budget": config.get("benign_query_budget"),
                    # Identity count is not written into the summary by the
                    # drivers, so a distributed run cannot be distinguished from
                    # a single-client one after the fact. Recorded here as the
                    # explicit None it is, rather than silently omitted.
                    "sybil_identities": config.get("sybil_identities"),
                    "attack": row.get("attack"),
                    "defense": row.get("defense"),
                    "auroc": float(row["auroc"]),
                    "f1": row.get("f1"),
                    "tpr": row.get("tpr"),
                    "fpr": row.get("fpr"),
                    "num_attack": row.get("num_attack"),
                    "num_benign": row.get("num_benign"),
                }
            )
    return records


def _match(cells: list[Cell], records: list[dict[str, Any]], tolerance: float) -> None:
    for cell in cells:
        defense = COLUMN_TO_DEFENSE.get(cell.column)
        attack = ROW_TO_ATTACK.get(cell.row)
        for record in records:
            # The tables print 3 decimals, so the tolerance only has to cover
            # rounding; anything looser would match unrelated runs.
            if abs(record["auroc"] - cell.auroc) > tolerance:
                continue
            same_pair = (
                (defense is None or record["defense"] == defense)
                and (attack is None or record["attack"] == attack)
            )
            record = {**record, "same_pair": same_pair}
            cell.matches.append(record)


def _classify(cell: Cell, min_budget: int) -> tuple[str, str]:
    if not cell.matches:
        return "UNMATCHED", "no run summary in runs/ reproduces this value"
    same_pair = [match for match in cell.matches if match.get("same_pair")]
    if not same_pair:
        return (
            "MISMATCHED-PAIR",
            "value occurs in runs/, but never for this attack x defense pair",
        )
    budgets = [
        int(match["query_budget"])
        for match in same_pair
        if isinstance(match.get("query_budget"), (int, float))
    ]
    if budgets and max(budgets) < min_budget:
        return (
            "SMOKE-ONLY",
            f"only matched by runs below the published budget (max query_budget={max(budgets)})",
        )
    return "OK", f"matched at query_budget={max(budgets)}" if budgets else "matched"


def _render_markdown(cells: list[Cell], min_budget: int) -> str:
    lines = [
        "# Provenance of published table values",
        "",
        "Generated by `scripts/audit_paper_table_provenance.py`. Each published cell is",
        "matched against every run summary under `runs/` by AUROC. `SMOKE-ONLY` means the",
        "only run reproducing that number ran below the budget the caption claims;",
        "`UNMATCHED` means no run in this repository produces it.",
        "",
    ]
    by_table: dict[str, list[Cell]] = {}
    for cell in cells:
        by_table.setdefault(cell.table, []).append(cell)
    for table, table_cells in by_table.items():
        lines += [f"## {table}", "", "| Row | Column | AUROC/F1 | Status | Source |", "|---|---|---|---|---|"]
        for cell in table_cells:
            status, reason = _classify(cell, min_budget)
            best = ""
            candidates = [m for m in cell.matches if m.get("same_pair")] or cell.matches
            if candidates:
                pick = max(candidates, key=lambda match: (match.get("query_budget") or 0))
                best = (
                    f"`{pick['path']}` ({pick['defense']} x {pick['attack']}, "
                    f"K={pick['query_budget']})"
                )
            f1 = "--" if cell.f1 is None else f"{cell.f1:.3f}"
            lines.append(
                f"| {cell.row} | {cell.column} | {cell.auroc:.3f}/{f1} | "
                f"**{status}** — {reason} | {best} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=PROJECT_ROOT / "paper" / "tables")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--min-budget", type=int, default=50_000)
    parser.add_argument("--tolerance", type=float, default=6e-4)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    sources = [
        (args.tables_dir / "attack_defense_matrix_results_single.tex", CIFAR_COLUMNS),
        (args.tables_dir / "attack_defense_matrix_results_multi.tex", CIFAR_COLUMNS),
        (args.tables_dir / "mnist_attack_defense_matrix_results.tex", MNIST_COLUMNS),
    ]
    cells: list[Cell] = []
    for path, columns in sources:
        if path.exists():
            cells.extend(_parse_table(path, columns))

    records = _load_summaries(args.runs_root)
    print(f"[audit] {len(cells)} published cells, {len(records)} summary rows")
    _match(cells, records, args.tolerance)

    counts = {"OK": 0, "SMOKE-ONLY": 0, "UNMATCHED": 0, "MISMATCHED-PAIR": 0}
    for cell in cells:
        status, reason = _classify(cell, args.min_budget)
        counts[status] += 1
        if status != "OK":
            print(f"[audit] {status:11s} {cell.table} | {cell.row} | {cell.column} | "
                  f"{cell.auroc:.3f} -- {reason}")
    print(f"[audit] summary: {counts}")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_render_markdown(cells, args.min_budget), encoding="utf-8")
        print(f"[audit] wrote {args.markdown}")


if __name__ == "__main__":
    main()
