"""Render the single-client and distributed attack/defense tables.

Replaces hand-maintained numbers in
``paper/tables/attack_defense_matrix_results_{single,multi}.tex`` with values
read directly from run summaries, so every cell is traceable to a file.

Why this script exists
----------------------
The previously committed tables mixed runs silently. Three columns (C3
likelihood, Comp., Comp.\\ (L)) were taken from ``*_smoke`` summaries with
``query_budget=256`` while the captions stated ``K=50,000`` -- a discrepancy no
reader could detect. This script therefore:

- refuses any summary whose ``query_budget`` is below ``--min-budget``, so a
  smoke run can never reach a published table again;
- takes each protocol's numbers from its own primary run, and marks with a
  dagger any cell that had to be sourced from a secondary run;
- emits ``\\pending`` for cells that were never measured, so gaps are visible
  rather than being filled from a neighbouring configuration.

Each cell reports ``AUROC / F1``. F1 is computed at the calibrated operating
point (benign 95th percentile, i.e. a 5% target FPR), so it summarises the
precision/recall trade-off a deployed detector actually faces -- AUROC alone is
threshold-free and can look strong for a detector that flags nothing at its
operating point.

Usage::

    python scripts/build_attack_defense_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PENDING = r"\pending"
NOT_MEASURED = "--"
DAGGER = r"$^{\dagger}$"
RECOMPUTED = r"$^{\ddagger}$"

# Detectors whose decision is a threshold on a continuous score, so their
# operating point is fixed by the target FPR. For these F1 is reconstructable
# exactly from (TPR, target FPR, n_attack, n_benign). The stateful detectors
# (KS, label histogram) and PRADA decide by their own internal rule, so their
# realised FPR is a property of the detector and must not be overridden.
SCORE_THRESHOLDED: frozenset[str] = frozenset({
    "flowpure", "flowguard_integral", "fdinet", "flow_matching",
    "flowguard_composite", "flowguard_composite_learned",
})

# Column label -> defense key in the summaries. Order fixes the table order.
COLUMNS: tuple[tuple[str, str], ...] = (
    (r"\gls{PRADA}", "prada"),
    (r"FlowPure (C1)", "flowpure"),
    (r"FDINet", "fdinet"),
    (r"Int.\ (C2)", "flowguard_integral"),
    (r"Like.\ (C3)", "flow_matching"),
    (r"KS (C4)", "flowguard_userlevel"),
    (r"Label (C5)", "flowguard_labelhist"),
    (r"Comp.", "flowguard_composite"),
    (r"Comp.\ (L)", "flowguard_composite_learned"),
)

# Baseline detectors, for the "best baseline" bolding rule used in the caption.
BASELINE_KEYS = ("prada", "flowpure", "fdinet")

EXTRACTION_ROWS: tuple[tuple[str, str], ...] = (
    ("PRADA attack", "prada"),
    ("MAZE", "maze"),
    ("DisGUIDE", "disguide"),
)

VELOCITY_ADAPTIVE_ROWS: tuple[tuple[str, str], ...] = (
    ("D1 latent manifold", "flowguard_d1_latent_manifold"),
    ("D2 procedural natural", "flowguard_d2_procedural_natural"),
)

FULLY_ADAPTIVE_ROWS: tuple[tuple[str, str], ...] = (
    ("D3 adaptive C1+C2", "flowguard_d3_adaptive_c1c2"),
    ("D4 adaptive C1+C3", "flowguard_d4_adaptive_c1c3"),
    ("D5 adaptive composite", "flowguard_d5_adaptive_composite"),
    ("D6 adaptive comp.+stateful", "flowguard_d6_adaptive_stateful"),
)


class Cell(NamedTuple):
    auroc: float | None
    f1: float | None
    source: str
    primary: bool
    recomputed: bool = False


def _operating_point_f1(
    row: dict[str, Any], defense: str, target_fpr: float
) -> tuple[float | None, bool]:
    """Return F1 at the calibrated operating point, and whether it was rebuilt.

    The parallel-defense driver originally thresholded only the attack side, so
    precision counted benign queries flagged by the detector's own runtime rule
    (FPR up to 1.0) while recall used the calibrated threshold -- two decision
    rules inside one metric. F1 is a deterministic function of
    (TPR, FPR, n_attack, n_benign), verified to reproduce every reported value
    in the primary runs exactly, so for a score-thresholded detector the correct
    value is recovered by substituting the target FPR that the threshold defines
    by construction.
    """
    reported = row.get("f1")
    fpr, tpr = row.get("fpr"), row.get("tpr")
    n_attack, n_benign = row.get("num_attack"), row.get("num_benign")
    if None in (fpr, tpr, n_attack, n_benign):
        return reported, False
    if abs(float(fpr) - target_fpr) < 1e-9 or defense not in SCORE_THRESHOLDED:
        return reported, False
    true_pos = float(tpr) * int(n_attack)
    false_pos = target_fpr * int(n_benign)
    if true_pos + false_pos <= 0:
        return 0.0, True
    precision = true_pos / (true_pos + false_pos)
    if precision + float(tpr) <= 0:
        return 0.0, True
    return 2.0 * precision * float(tpr) / (precision + float(tpr)), True


def _load(path: Path, *, min_budget: int) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    """Index one summary by ``(attack, defense)``, rejecting under-sized runs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    budget = int(payload.get("config", {}).get("query_budget", 0))
    if budget < min_budget:
        raise ValueError(
            f"{path} has query_budget={budget} < --min-budget={min_budget}. "
            "Refusing to publish smoke-scale numbers; this is the defect the "
            "previous hand-written tables contained."
        )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("results", []):
        if row.get("error"):
            continue
        rows[(str(row.get("attack")), str(row.get("defense")))] = row
    return rows, budget


def _collect(
    primary: Path,
    secondaries: list[Path],
    *,
    min_budget: int,
    target_fpr: float = 0.05,
) -> tuple[dict[tuple[str, str], Cell], int]:
    cells: dict[tuple[str, str], Cell] = {}
    primary_rows, budget = _load(primary, min_budget=min_budget)
    for key, row in primary_rows.items():
        f1, fixed = _operating_point_f1(row, key[1], target_fpr)
        cells[key] = Cell(row.get("auroc"), f1, primary.stem, True, fixed)
    for path in secondaries:
        rows, _ = _load(path, min_budget=min_budget)
        for key, row in rows.items():
            # Primary always wins; a secondary only fills a genuine hole.
            if key in cells:
                continue
            f1, fixed = _operating_point_f1(row, key[1], target_fpr)
            cells[key] = Cell(row.get("auroc"), f1, path.stem, False, fixed)
    return cells, budget


def _format_cell(cell: Cell | None, *, missing: str = PENDING) -> str:
    if cell is None or cell.auroc is None:
        return missing
    auroc = f"{cell.auroc:.3f}" if cell.auroc >= 0.001 else f"{cell.auroc:.4f}"
    f1 = "--" if cell.f1 is None else f"{cell.f1:.3f}"
    text = f"{auroc}/{f1}"
    if not cell.primary:
        text += DAGGER
    if cell.recomputed:
        text += RECOMPUTED
    return text


def _render_row(
    label: str,
    attack: str,
    cells: dict[tuple[str, str], Cell],
    *,
    missing: str = PENDING,
) -> str:
    values = [cells.get((attack, key)) for _, key in COLUMNS]
    # Bold the strongest baseline detector by AUROC, matching the caption.
    baseline = {
        key: cells.get((attack, key))
        for key in BASELINE_KEYS
        if cells.get((attack, key)) is not None
        and cells[(attack, key)].auroc is not None
    }
    # Bolding a lone survivor as "best" would assert a comparison that was
    # never made -- in the D3-D6 block only FlowPure was measured.
    best = max(baseline, key=lambda k: baseline[k].auroc) if len(baseline) > 1 else None
    rendered = []
    for (_, key), cell in zip(COLUMNS, values, strict=True):
        text = _format_cell(cell, missing=missing)
        if key == best and text not in (PENDING, NOT_MEASURED):
            text = r"\textbf{" + text + "}"
        rendered.append(text)
    return f"{label} &\n" + " &\n".join(rendered) + r" \\"


def _render_table(
    cells: dict[tuple[str, str], Cell],
    *,
    caption: str,
    label: str,
    include_fully_adaptive: bool,
    fully_adaptive_note: str,
) -> str:
    header = " &\n".join(
        [r"\textbf{Attacks/Defenses}"] + [r"\textbf{" + name + "}" for name, _ in COLUMNS]
    )
    blocks = [
        r"\multicolumn{10}{@{}l}{\textit{Extraction baselines}} \\",
        "\n\n".join(_render_row(lbl, key, cells) for lbl, key in EXTRACTION_ROWS),
        r"\addlinespace",
        r"\multicolumn{10}{@{}l}{\textit{Flow-targeted adaptive attacks}} \\",
        "\n\n".join(_render_row(lbl, key, cells) for lbl, key in VELOCITY_ADAPTIVE_ROWS),
    ]
    if include_fully_adaptive:
        blocks += [
            r"\addlinespace",
            r"\multicolumn{10}{@{}l}{\textit{FlowGuard++-adaptive attacks"
            + fully_adaptive_note
            + r"}} \\",
            "\n\n".join(
                _render_row(lbl, key, cells, missing=NOT_MEASURED)
                for lbl, key in FULLY_ADAPTIVE_ROWS
            ),
        ]
    body = "\n\n".join(blocks)
    return f"""% GENERATED FILE - do not edit the numbers by hand.
%
% Rebuild with:
%   python scripts/build_attack_defense_tables.py
%
% Every cell is read from a run summary. Cells marked with a dagger come from a
% secondary run because the primary run did not cover that (attack, defense)
% pair; cells showing \\pending were never measured.
\\begin{{table*}}[!t]
\\caption{{{caption}}}
\\label{{{label}}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{@{{}}l ccc cccccc@{{}}}}
\\toprule
{header} \\\\
\\midrule

{body}

\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


SINGLE_CAPTION = r"""
Detection performance under single-client attackers.
Each cell reports \gls{AUROC}\,/\,F1; higher is better for the defender.
F1 is measured at the calibrated operating point (benign $95$th percentile,
i.e.\ a $5\%$ target \gls{FPR}); \gls{AUROC} is threshold-free and can look
strong for a detector that flags nothing at that operating point, so the two
should be read together.
\textbf{Bold} marks the best baseline detector (PRADA, FlowPure, or FDINet) per
attack. \emph{Comp.\ (L)} is the oracle learned-weight fusion.
Extraction baselines and D1--D2 use $K=50{,}000$ queries with detectors
calibrated on $50{,}000$ benign queries.
Cells marked $^{\dagger}$ come from a secondary run of the same protocol and
budget because the primary run did not cover that pair.
The final block reports the FlowGuard++-adaptive attackers D3--D6, evaluated
under a different protocol (stated in the block heading) in which only FlowPure
and the composite were instrumented; ``--'' marks detectors not measured
under that protocol, and those rows are therefore not directly comparable to the
$K=50{,}000$ rows above.
Cells marked $^{\ddagger}$ have F1 reconstructed at the $5\%$ operating point
from the reported \gls{TPR} and sample counts, because that run recorded
precision against the detector's runtime flags rather than the calibrated
threshold; the reconstruction is exact and reproduces the reported F1 on all
rows where both agree.
"""

MULTI_CAPTION = r"""
Detection performance under distributed attackers ($N=1{,}250$ Sybil
identities; $K=50{,}000$ total queries distributed round-robin across
identities). Each cell reports \gls{AUROC}\,/\,F1; higher is better for the
defender. F1 is measured at the calibrated operating point (benign $95$th
percentile, i.e.\ a $5\%$ target \gls{FPR}).
\textbf{Bold} marks the best baseline detector per attack.
\emph{Comp.\ (L)} is the oracle learned-weight fusion.
Detectors are calibrated on $50{,}000$ benign queries.
Cells marked $^{\dagger}$ come from a secondary run because the primary
distributed run did not cover that pair. Per-query detectors (PRADA, FlowPure,
FDINet, C1--C3) are by construction insensitive to how queries are split across
identities; only the stateful detectors (C4, C5, Comp.) can differ from
Table~\ref{tab:single_client_attack_defense}.
"""

FULLY_ADAPTIVE_NOTE = (
    r"; $K=1{,}000{,}000$ queries, $20{,}000$ benign calibration "
    r"queries, steered-diffusion generator"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    runs = PROJECT_ROOT / "runs"
    parser.add_argument(
        "--single-primary", type=Path,
        default=runs / "runs" / "attack_defense_combined_full"
        / "attack_defense_matrix_combined-full_summary.json",
    )
    parser.add_argument(
        "--multi-primary", type=Path,
        default=runs / "runs" / "attack_defense_combined_sybil1250"
        / "attack_defense_matrix_combined-sybil1250_summary.json",
    )
    parser.add_argument(
        "--secondary", type=Path, action="append",
        default=None,
        help="Summary used only to fill pairs the primary run lacks. Repeatable.",
    )
    parser.add_argument(
        "--fully-adaptive", type=Path, action="append", default=None,
        help="Summaries for the D3-D6 rows (the new fully adaptive benchmark).",
    )
    parser.add_argument(
        "--min-budget", type=int, default=50_000,
        help="Reject any summary below this query budget.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "tables")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    runs = PROJECT_ROOT / "runs"
    secondaries = args.secondary or [
        runs / "runs" / "fgpp_learned" / "attack_defense_matrix_fgpp-learned_summary.json"
    ]
    fully_adaptive = args.fully_adaptive
    if fully_adaptive is None:
        fully_adaptive = sorted(
            (runs / "adagen-latent").glob("*/attack_defense_matrix_*_summary.json")
        )

    single_cells, _ = _collect(args.single_primary, secondaries, min_budget=args.min_budget)
    multi_cells, _ = _collect(args.multi_primary, secondaries, min_budget=args.min_budget)

    # The D3-D6 benchmark was run single-client (SYBIL_IDENTITIES=1), so it
    # extends the single-client table only. Adding it to the distributed table
    # would assert a configuration that was never run.
    fully_adaptive_keys = {key for _, key in FULLY_ADAPTIVE_ROWS}
    fa_cells: dict[tuple[str, str], Cell] = {}
    for path in fully_adaptive:
        rows, _ = _load(Path(path), min_budget=args.min_budget)
        for key, row in rows.items():
            # These summaries also contain D1/D2 cells, but at 1M queries rather
            # than 50k. Letting them through would silently replace the 50k rows
            # with numbers from a different protocol.
            if key[0] not in fully_adaptive_keys:
                continue
            f1, fixed = _operating_point_f1(row, key[1], 0.05)
            fa_cells.setdefault(
                key, Cell(row.get("auroc"), f1, Path(path).stem, True, fixed)
            )
    single_cells.update(fa_cells)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "attack_defense_matrix_results_single.tex": _render_table(
            single_cells,
            caption=SINGLE_CAPTION.strip(),
            label="tab:single_client_attack_defense",
            include_fully_adaptive=True,
            fully_adaptive_note=FULLY_ADAPTIVE_NOTE,
        ),
        "attack_defense_matrix_results_multi.tex": _render_table(
            multi_cells,
            caption=MULTI_CAPTION.strip(),
            label="tab:distributed_attack_defense",
            include_fully_adaptive=False,
            fully_adaptive_note="",
        ),
    }
    for name, text in outputs.items():
        (args.output_dir / name).write_text(text, encoding="utf-8")
        print(f"[table] wrote {args.output_dir / name}")

    for tag, cells, rows in (
        ("single", single_cells, EXTRACTION_ROWS + VELOCITY_ADAPTIVE_ROWS + FULLY_ADAPTIVE_ROWS),
        ("multi", multi_cells, EXTRACTION_ROWS + VELOCITY_ADAPTIVE_ROWS),
    ):
        missing = [
            f"{atk}/{key}"
            for _, atk in rows
            for _, key in COLUMNS
            if cells.get((atk, key)) is None
        ]
        secondary = sum(
            1 for _, atk in rows for _, key in COLUMNS
            if (c := cells.get((atk, key))) is not None and not c.primary
        )
        print(
            f"[table] {tag}: {len(rows) * len(COLUMNS) - len(missing)}/"
            f"{len(rows) * len(COLUMNS)} cells filled "
            f"({secondary} from a secondary run), {len(missing)} pending"
        )
        for item in missing:
            print(f"[table]   pending: {item}")


if __name__ == "__main__":
    main()
