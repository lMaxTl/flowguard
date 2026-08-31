"""Render the MNIST D1 guidance-scale trade-off table and figure.

Reads ``runs/notebook/mnist_d1_guidance_search/search_summary.json`` -- a
sweep of the classifier-free-guidance scale used by the D1 (latent-manifold)
attack's DDPM latent steering, at a fixed budget of 12,000 queries per trial
-- and writes:

- ``paper/tables/mnist_guidance_tradeoff_results.tex``: the raw numbers.
- ``paper/figures/mnist_guidance_tradeoff.tex``: a dual-axis tikz/pgfplots
  figure of fidelity and detection rate against guidance scale, matching the
  native tikz style already used in ``paper/figures/auroc_vs_identities.tex``
  (this repo's other pgfplots figures are pre-rendered PNGs, not tikz).

Provenance note: the notebook cell that produced ``search_summary.json`` is
not preserved in this repository, so the exact defense/scoring rule behind
its ``score_mean`` column cannot be confirmed from run metadata. Its
magnitude (~350-430) matches the FlowPure velocity statistic used everywhere
else in this study (calibrated benign threshold ~406 in the other MNIST
runs), so the table and figure captions describe it as "FlowPure-scale"
rather than asserting it outright.

Usage::

    python scripts/build_mnist_guidance_tradeoff.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_trials(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trials = sorted(payload["all_trials"], key=lambda t: t["guidance_scale"])
    return trials, payload


def _fmt_scale(scale: float) -> str:
    return f"{scale:g}"


TABLE_CAPTION = r"""
Fidelity/evasion trade-off of the D1 (latent-manifold) attack as the
classifier-free-guidance scale of its DDPM latent steering is swept, at a
fixed budget of $12{,}000$ queries per trial (MNIST, LeNet victim). Detection
rate is measured at a calibrated operating point whose score magnitude
matches the FlowPure velocity statistic used elsewhere in this study; the
generating script is not preserved in the repository, so this is inferred
from score scale rather than confirmed defense metadata. \textbf{Bold} marks
the operating point with the lowest detection rate that still clears the
$90\%$ fidelity target. Scale $8.0$ reaches the lowest detection rate overall
but misses the target by $0.08$ points.
"""


def _render_table(trials: list[dict]) -> str:
    best_scale = min(
        (t for t in trials if t["meets_fidelity_target"]),
        key=lambda t: t["detection_rate"],
    )["guidance_scale"]
    rows = []
    for t in trials:
        scale = _fmt_scale(t["guidance_scale"])
        fid = f"{t['online_ensemble_fidelity']:.2f}"
        det = f"{100 * t['detection_rate']:.1f}"
        meets = "Yes" if t["meets_fidelity_target"] else r"\textcolor{red}{No}"
        line = f"{scale} & {fid} & {det} & {meets} \\\\"
        if t["guidance_scale"] == best_scale:
            line = f"\\textbf{{{scale}}} & \\textbf{{{fid}}} & \\textbf{{{det}}} & {meets} \\\\"
        rows.append(line)
    body = "\n".join(rows)
    return f"""% GENERATED FILE - do not edit the numbers by hand.
%
% Rebuild with:
%   python scripts/build_mnist_guidance_tradeoff.py
%
% Source: runs/notebook/mnist_d1_guidance_search/search_summary.json
\\begin{{table}}[t]
\\caption{{{TABLE_CAPTION.strip()}}}
\\label{{tab:mnist_guidance_tradeoff}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{@{{}}r rr c@{{}}}}
\\toprule
\\textbf{{Guidance scale}} & \\textbf{{Fidelity (\\%)}} & \\textbf{{Detection rate (\\%)}} & \\textbf{{Meets $90\\%$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


FIGURE_CAPTION = r"""
Substitute fidelity and detection rate of the D1 attack as its guidance
scale is swept ($K=12{,}000$ queries per trial, MNIST, LeNet victim). The two
metrics move together up to scale $7$--$7.5$: pushing the steering harder
both improves extraction and lowers detection, because guidance concentrates
queries near the decision boundary, which is simultaneously more informative
and less visually extreme than an unguided draw. Scale $8.0$ (hollow marker)
reaches the lowest detection rate of the sweep but overshoots into
under-fitting and drops fidelity just below the $90\%$ target (dashed line).
"""


def _render_figure(trials: list[dict]) -> str:
    fid_coords = " ".join(
        f"({t['guidance_scale']},{t['online_ensemble_fidelity']:.2f})" for t in trials
    )
    det_coords = " ".join(
        f"({t['guidance_scale']},{100 * t['detection_rate']:.2f})" for t in trials
    )
    fail_scale = next(t["guidance_scale"] for t in trials if not t["meets_fidelity_target"])
    fail_fid = next(t["online_ensemble_fidelity"] for t in trials if not t["meets_fidelity_target"])
    scales = sorted({t["guidance_scale"] for t in trials})
    xtick = ",".join(_fmt_scale(s) for s in scales)
    xmin, xmax = min(scales) - 0.5, max(scales) + 0.5
    return f"""% GENERATED FILE - do not edit the numbers by hand.
%
% Rebuild with:
%   python scripts/build_mnist_guidance_tradeoff.py
%
% Source: runs/notebook/mnist_d1_guidance_search/search_summary.json
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=\\linewidth,
    height=6.2cm,
    xlabel={{Guidance scale}},
    ylabel={{Substitute fidelity (\\%)}},
    xmin={xmin}, xmax={xmax},
    ymin=88, ymax=95,
    xtick={{{xtick}}},
    tick label style={{font=\\footnotesize}},
    label style={{font=\\footnotesize}},
    legend style={{font=\\scriptsize, at={{(0.02,0.02)}}, anchor=south west,
        fill opacity=0.85, draw opacity=1, text opacity=1}},
    grid=both,
    grid style={{gray!18}},
    axis y line*=left,
    axis x line*=bottom,
]
\\addplot[mark=*, color=blue!70!black, thick] coordinates {{{fid_coords}}};
\\addlegendentry{{Fidelity (\\%)}}
\\addplot[mark=none, dashed, color=gray, thick] coordinates {{({xmin},90)({xmax},90)}};
\\addlegendentry{{$90\\%$ target}}
\\addplot[mark=o, mark size=4pt, only marks, color=red!70!black, thick]
    coordinates {{({fail_scale},{fail_fid:.2f})}};
\\end{{axis}}
\\begin{{axis}}[
    width=\\linewidth,
    height=6.2cm,
    xmin={xmin}, xmax={xmax},
    ymin=0, ymax=65,
    ylabel={{Detection rate (\\%)}},
    tick label style={{font=\\footnotesize}},
    label style={{font=\\footnotesize}},
    legend style={{font=\\scriptsize, at={{(0.98,0.98)}}, anchor=north east,
        fill opacity=0.85, draw opacity=1, text opacity=1}},
    axis y line*=right,
    axis x line=none,
]
\\addplot[mark=square*, color=red!85!black, thick] coordinates {{{det_coords}}};
\\addlegendentry{{Detection rate (\\%)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{FIGURE_CAPTION.strip()}}}
\\label{{fig:mnist_guidance_tradeoff}}
\\end{{figure}}
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=PROJECT_ROOT / "runs" / "notebook" / "mnist_d1_guidance_search"
        / "search_summary.json",
    )
    parser.add_argument("--table-output", type=Path,
                         default=PROJECT_ROOT / "paper" / "tables"
                         / "mnist_guidance_tradeoff_results.tex")
    parser.add_argument("--figure-output", type=Path,
                         default=PROJECT_ROOT / "paper" / "figures"
                         / "mnist_guidance_tradeoff.tex")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    trials, _ = _load_trials(args.summary)

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.write_text(_render_table(trials), encoding="utf-8")
    print(f"[table] wrote {args.table_output}")

    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_output.write_text(_render_figure(trials), encoding="utf-8")
    print(f"[table] wrote {args.figure_output}")


if __name__ == "__main__":
    main()
