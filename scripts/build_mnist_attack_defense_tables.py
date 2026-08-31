"""Render the MNIST attack/defense matrix and adaptive-stealing tables.

Companion to ``scripts/build_attack_defense_tables.py`` (which covers the
CIFAR-10 / VGG16-BN tables). MNIST does not yet have a full-budget,
multi-defense matrix run, so this script is explicit about the provenance and
scale of every number instead of dressing them up to look like the CIFAR-10
tables:

- ``paper/tables/mnist_attack_defense_matrix_results.tex`` is built from
  ``runs/mnist_full_matrix_smoke``, the only run that covers all attacks
  against all defenses on MNIST. Its query budget is 256 -- a smoke-test
  scale, an order of magnitude below the 50,000-query minimum the CIFAR-10
  builder enforces (``scripts/build_attack_defense_tables.py``). The caption
  states the budget explicitly and the table is not wired into the paper
  (nothing in ``paper/sections`` inputs it) until a full-budget matrix run
  replaces it.
- ``paper/tables/mnist_adaptive_stealing_results.tex`` mirrors
  ``tab:adaptive_stealing`` (Table~\\ref{tab:adaptive_stealing} in
  07_results.tex) using two large-budget, FlowPure-only runs
  (``runs/mnist_benchmark`` at 100k queries and ``runs/mnist_500k`` at 500k
  queries). These are genuinely large-budget runs, so this table is safe to
  cite, but the unmodified DisGUIDE baseline was never completed at this
  scale on MNIST (it errors in both source runs), so that row renders
  ``\\pending`` rather than being silently omitted.

Usage::

    python scripts/build_mnist_attack_defense_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PENDING = r"\pending"
NOT_MEASURED = "--"

# ---------------------------------------------------------------------------
# Full attack/defense matrix (smoke-scale, single source run).
# ---------------------------------------------------------------------------

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

BASELINE_KEYS = ("prada", "flowpure", "fdinet")

ROW_BLOCKS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Extraction baselines", (
        ("PRADA attack", "prada"),
        ("MAZE", "maze"),
        ("DisGUIDE", "disguide"),
    )),
    ("Transfer-set baselines", (
        ("Transfer (naive)", "transfer_naive"),
        ("Transfer (top-1)", "transfer_top1"),
        ("Transfer (S4L)", "transfer_s4l"),
        ("Transfer (smoothing)", "transfer_smoothing"),
    )),
    ("Query-side adaptive attacks", (
        ("A1 clean transfer", "flowguard_a1_clean_transfer"),
        ("A2 projected MAZE", "flowguard_a2_projected_maze"),
        ("A3 flow-blind", "flowguard_a3_flowblind"),
        ("A2+A3 combined", "flowguard_adaptive_adaptive"),
    )),
    ("Generator-side adaptive attacks (CEM)", (
        ("D1 latent manifold", "flowguard_d1_latent_manifold"),
        ("D2 procedural natural", "flowguard_d2_procedural_natural"),
        ("D3 projected latent", "flowguard_d3_projected_latent"),
        ("D4 rejection oracle", "flowguard_d4_rejection_oracle"),
    )),
)


class Cell(NamedTuple):
    auroc: float | None
    f1: float | None


def _load_matrix(path: Path) -> tuple[dict[tuple[str, str], Cell], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], Cell] = {}
    for row in payload.get("results", []):
        if row.get("error"):
            continue
        key = (str(row.get("attack")), str(row.get("defense")))
        cells[key] = Cell(row.get("auroc"), row.get("f1"))
    return cells, payload.get("config", {})


def _format_cell(cell: Cell | None) -> str:
    if cell is None:
        return NOT_MEASURED
    if cell.auroc is None:
        return PENDING
    auroc = f"{cell.auroc:.3f}" if cell.auroc >= 0.001 else f"{cell.auroc:.4f}"
    f1 = "--" if cell.f1 is None else f"{cell.f1:.3f}"
    return f"{auroc}/{f1}"


def _render_row(label: str, attack: str, cells: dict[tuple[str, str], Cell]) -> str:
    baseline = {
        key: cells[(attack, key)]
        for key in BASELINE_KEYS
        if cells.get((attack, key)) is not None and cells[(attack, key)].auroc is not None
    }
    best = max(baseline, key=lambda k: baseline[k].auroc) if len(baseline) > 1 else None
    rendered = []
    for _, key in COLUMNS:
        if key in ("flow_matching", "flowguard_composite_learned"):
            rendered.append(NOT_MEASURED)
            continue
        text = _format_cell(cells.get((attack, key)))
        if key == best and text not in (PENDING, NOT_MEASURED):
            text = r"\textbf{" + text + "}"
        rendered.append(text)
    return f"{label} &\n" + " &\n".join(rendered) + r" \\"


MATRIX_CAPTION = r"""
Detection performance under single-client attackers on an MNIST-trained
target (LeNet victim). Each cell reports \gls{AUROC}\,/\,F1; higher is better
for the defender. \textbf{Bold} marks the best baseline detector (PRADA,
FlowPure, or FDINet) per attack. \textbf{Caution:} this run uses a
$K{=}256$-query budget per attack with detectors calibrated on $256$ benign
queries (``mnist\_full\_matrix\_smoke''), roughly two orders of magnitude
below the $K{=}50{,}000$ budget used for the CIFAR-10 tables
(Table~\ref{tab:single_client_attack_defense}). It is the only run that
currently covers every attack against every MNIST defense and is included to
show the full qualitative landscape; the numbers should be treated as
indicative, not as a like-for-like comparison with the CIFAR-10 results,
until a full-budget MNIST matrix replaces it. Likelihood-typicality (C3) and
the learned-weight fusion (Comp.\ (L)) were not instrumented for MNIST and
render as ``--''. The \gls{PRADA} attack row draws its attack-side samples
from only $16$ queries (the attack's own iterative budget), so its column is
noisier than the others.
"""


def _render_matrix_table(cells: dict[tuple[str, str], Cell], config: dict[str, Any]) -> str:
    header = " &\n".join(
        [r"\textbf{Attacks/Defenses}"] + [r"\textbf{" + name + "}" for name, _ in COLUMNS]
    )
    blocks = []
    for title, rows in ROW_BLOCKS:
        blocks.append(r"\multicolumn{10}{@{}l}{\textit{" + title + r"}} \\")
        blocks.append("\n\n".join(_render_row(lbl, key, cells) for lbl, key in rows))
        blocks.append(r"\addlinespace")
    blocks.pop()  # drop trailing \addlinespace
    body = "\n\n".join(blocks)
    return f"""% GENERATED FILE - do not edit the numbers by hand.
%
% Rebuild with:
%   python scripts/build_mnist_attack_defense_tables.py
%
% Source: {config.get('run_label', '?')}, query_budget={config.get('query_budget', '?')},
% benign_query_budget={config.get('benign_query_budget', '?')}. This is a
% SMOKE-SCALE run -- see the caption caveat before citing these numbers.
\\begin{{table*}}[!t]
\\caption{{{MATRIX_CAPTION.strip()}}}
\\label{{tab:mnist_single_client_attack_defense}}
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


# ---------------------------------------------------------------------------
# Adaptive-stealing table (large-budget, FlowPure-only, two budgets).
# ---------------------------------------------------------------------------

STEALING_ROWS: tuple[tuple[str, str], ...] = (
    ("DisGUIDE (baseline)", "disguide"),
    ("D1 latent manifold", "flowguard_d1_latent_manifold"),
    ("D2 procedural natural", "flowguard_d2_procedural_natural"),
    ("D3 adaptive C1+C2", "flowguard_d3_adaptive_c1c2"),
    ("D4 adaptive C1+C3", "flowguard_d4_adaptive_c1c3"),
    ("D5 adaptive composite", "flowguard_d5_adaptive_composite"),
    ("D6 adaptive comp.+stateful", "flowguard_d6_adaptive_stateful"),
)


def _load_stealing(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("results", []):
        if str(row.get("defense")) != "flowpure":
            continue
        rows[str(row.get("attack"))] = row
    return rows, payload.get("config", {})


def _render_stealing_row(label: str, attack: str, rows: dict[str, dict[str, Any]], queries: int) -> str:
    row = rows.get(attack)
    if row is None or row.get("error"):
        return f"{label} & {queries:,} & {PENDING} & {PENDING} \\\\".replace(",", "{,}")
    acc = row.get("accuracy")
    fid = row.get("fidelity")
    acc_s = PENDING if acc is None else f"{float(acc):.2f}"
    fid_s = PENDING if fid is None else f"{float(fid):.2f}"
    q_s = f"{queries:,}".replace(",", "{,}")
    return f"{label} & {q_s} & {acc_s} & {fid_s} \\\\"


STEALING_CAPTION = r"""
Substitute-model quality after large-budget extraction against FlowPure on
MNIST (LeNet victim). FlowPure is calibrated on $2{,}048$ benign samples at
target \gls{FPR}${=}5\%$. Unlike the CIFAR-10 study
(Table~\ref{tab:adaptive_stealing}), fidelity does \emph{not} increase
monotonically with query budget here: every FlowGuard++ attack variant except
D1 and D6 scores \emph{lower} at $500$k queries than at $100$k, and D1 itself
drops from $63.3\%$ to $25.6\%$. We report both budgets rather than only the
larger one so this non-monotonicity is visible; it is consistent with
generator drift over a much longer run on a near-saturated, low-capacity
target (LeNet on MNIST) rather than with the budget scaling seen on
CIFAR-10/VGG16-BN. The unmodified DisGUIDE baseline errors out before
completion in both source runs (a batch-size mismatch in the fixed evaluation
harness) and is marked \pending rather than omitted.
"""


def _render_stealing_table(rows_100k: dict[str, dict[str, Any]], budget_100k: int,
                            rows_500k: dict[str, dict[str, Any]], budget_500k: int) -> str:
    block_100k = "\n\n".join(
        _render_stealing_row(lbl, key, rows_100k, budget_100k) for lbl, key in STEALING_ROWS
    )
    block_500k = "\n\n".join(
        _render_stealing_row(lbl, key, rows_500k, budget_500k) for lbl, key in STEALING_ROWS
    )
    return f"""% GENERATED FILE - do not edit the numbers by hand.
%
% Rebuild with:
%   python scripts/build_mnist_attack_defense_tables.py
%
% Sources: runs/mnist_benchmark (K={budget_100k:,}), runs/mnist_500k (K={budget_500k:,}).
% FlowPure-only (defense="flowpure"); DisGUIDE baseline errored in both source
% runs and renders as \\pending rather than being silently dropped.
\\begin{{table}}[!t]
\\caption{{{STEALING_CAPTION.strip()}}}
\\label{{tab:mnist_adaptive_stealing}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}l r rr@{{}}}}
\\toprule
\\textbf{{Attack}} & \\textbf{{Queries}} & \\textbf{{Substitute acc.\\ (\\%)}} & \\textbf{{Fidelity (\\%)}} \\\\
\\midrule
\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{$K=100{{,}}000$}}}} \\\\

{block_100k}

\\addlinespace
\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{$K=500{{,}}000$}}}} \\\\

{block_500k}

\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    runs = PROJECT_ROOT / "runs"
    parser.add_argument(
        "--matrix-summary", type=Path,
        default=runs / "mnist_full_matrix_smoke"
        / "attack_defense_matrix_mnist-full-matrix-smoke_summary.json",
    )
    parser.add_argument(
        "--stealing-100k", type=Path,
        default=runs / "mnist_benchmark" / "attack_defense_matrix_mnist-bench_summary.json",
    )
    parser.add_argument(
        "--stealing-500k", type=Path,
        default=runs / "mnist_500k" / "attack_defense_matrix_mnist-500k_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "tables")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix_cells, matrix_config = _load_matrix(args.matrix_summary)
    matrix_tex = _render_matrix_table(matrix_cells, matrix_config)
    matrix_path = args.output_dir / "mnist_attack_defense_matrix_results.tex"
    matrix_path.write_text(matrix_tex, encoding="utf-8")
    print(f"[table] wrote {matrix_path}")

    rows_100k, config_100k = _load_stealing(args.stealing_100k)
    rows_500k, config_500k = _load_stealing(args.stealing_500k)
    stealing_tex = _render_stealing_table(
        rows_100k, int(config_100k.get("query_budget", 0)),
        rows_500k, int(config_500k.get("query_budget", 0)),
    )
    stealing_path = args.output_dir / "mnist_adaptive_stealing_results.tex"
    stealing_path.write_text(stealing_tex, encoding="utf-8")
    print(f"[table] wrote {stealing_path}")

    missing = [
        f"{atk}/{key}"
        for _, rows in ROW_BLOCKS for _, atk in rows
        for _, key in COLUMNS
        if key not in ("flow_matching", "flowguard_composite_learned")
        and matrix_cells.get((atk, key)) is None
    ]
    print(f"[table] matrix: {len(missing)} cell(s) never measured (rendered as {PENDING})")
    for item in missing:
        print(f"[table]   pending: {item}")


if __name__ == "__main__":
    main()
