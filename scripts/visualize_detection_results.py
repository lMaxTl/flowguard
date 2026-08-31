from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Record:
    setting: str
    attack: str
    defense: str
    detection_rate: float
    tpr: float
    fpr: float
    precision: float
    f1: float
    macro_f1: float
    roc_auc: float


DATA: list[Record] = [
    Record("Single-client", "MAZE", "FDINet", 0.545, 0.545, 0.530, 0.507, 0.525, 0.507, 0.488),
    Record("Single-client", "MAZE", "PRADA", 0.840, 0.840, 0.000, 1.000, 0.913, 0.919, 0.920),
    Record("Single-client", "MAZE", "Flow Matching", 0.965, 0.965, 0.170, 0.850, 0.904, 0.897, 0.921),
    Record("Single-client", "DisGUIDE", "FDINet", 1.000, 1.000, 0.530, 0.644, 0.784, 0.712, 0.988),
    Record("Single-client", "DisGUIDE", "PRADA", 0.833, 0.833, 0.000, 1.000, 0.909, 0.918, 0.917),
    Record("Single-client", "DisGUIDE", "Flow Matching", 1.000, 1.000, 0.170, 0.850, 0.919, 0.913, 1.000),
    Record("Distributed (100 clients)", "MAZE", "FDINet", 0.500, 0.500, 0.530, 0.485, 0.493, 0.485, 0.473),
    Record("Distributed (100 clients)", "MAZE", "PRADA", 0.000, 0.000, 0.000, 0.000, 0.000, 0.333, 0.500),
    Record("Distributed (100 clients)", "MAZE", "Flow Matching", 0.965, 0.965, 0.170, 0.850, 0.904, 0.897, 0.922),
    Record("Distributed (100 clients)", "DisGUIDE", "FDINet", 1.000, 1.000, 0.530, 0.644, 0.784, 0.712, 0.989),
    Record("Distributed (100 clients)", "DisGUIDE", "PRADA", 0.000, 0.000, 0.000, 0.000, 0.000, 0.333, 0.500),
    Record("Distributed (100 clients)", "DisGUIDE", "Flow Matching", 1.000, 1.000, 0.170, 0.850, 0.919, 0.913, 1.000),
]

INDEX = {(r.setting, r.attack, r.defense): r for r in DATA}

SETTINGS = ["Single-client", "Distributed (100 clients)"]
ATTACKS = ["MAZE", "DisGUIDE"]
DEFENSES = ["FDINet", "PRADA", "Flow Matching"]
CONTEXTS = [
    ("Single-client", "MAZE"),
    ("Single-client", "DisGUIDE"),
    ("Distributed (100 clients)", "MAZE"),
    ("Distributed (100 clients)", "DisGUIDE"),
]
CONTEXT_LABELS = ["Single\nMAZE", "Single\nDisGUIDE", "Distributed\nMAZE", "Distributed\nDisGUIDE"]

COLORS = {
    "FDINet": "#7F7F7F",
    "PRADA": "#E69F00",
    "Flow Matching": "#009E73",
}
SETTING_MARKERS = {
    "Single-client": "o",
    "Distributed (100 clients)": "s",
}
HIGHLIGHT_DEFENSE = "Flow Matching"

METRIC_DIRECTIONS = [
    ("detection_rate", True),
    ("tpr", True),
    ("precision", True),
    ("f1", True),
    ("macro_f1", True),
    ("roc_auc", True),
    ("fpr", False),
]


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FCFCFC",
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
        }
    )


def get_record(setting: str, attack: str, defense: str) -> Record:
    return INDEX[(setting, attack, defense)]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_figure(
    fig: plt.Figure,
    path_stem: Path,
    formats: Sequence[str],
    transparent: bool,
) -> list[Path]:
    written: list[Path] = []
    for ext in formats:
        out_path = path_stem.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=transparent)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, transparent=transparent)
        written.append(out_path)
    plt.close(fig)
    return written


def plot_key_metrics(output_dir: Path, formats: Sequence[str], transparent: bool) -> list[Path]:
    metrics = [
        ("macro_f1", "Macro-F1"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    x = np.arange(len(CONTEXTS), dtype=float)
    bar_width = 0.23
    offsets = np.linspace(-bar_width, bar_width, len(DEFENSES))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True, constrained_layout=True)

    for metric_idx, (metric, title) in enumerate(metrics):
        ax = axes[metric_idx]
        metric_values: dict[str, list[float]] = {}
        for defense_idx, defense in enumerate(DEFENSES):
            values = [getattr(get_record(setting, attack, defense), metric) for setting, attack in CONTEXTS]
            metric_values[defense] = values
            bars = ax.bar(
                x + offsets[defense_idx],
                values,
                width=bar_width * 0.92,
                label=defense if metric_idx == 0 else None,
                color=COLORS[defense],
                alpha=1.0 if defense == HIGHLIGHT_DEFENSE else 0.82,
                edgecolor="#111111" if defense == HIGHLIGHT_DEFENSE else "none",
                linewidth=1.0 if defense == HIGHLIGHT_DEFENSE else 0.0,
                zorder=3,
            )

        for context_idx in range(len(CONTEXTS)):
            context_scores = np.array(
                [metric_values[defense][context_idx] for defense in DEFENSES],
                dtype=float,
            )
            best_idx = int(np.nanargmax(context_scores))
            best_value = float(context_scores[best_idx])
            x_pos = x[context_idx] + offsets[best_idx]
            ax.text(
                x_pos,
                best_value + 0.02,
                f"{best_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#1F1F1F",
            )

        ax.set_title(title, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(CONTEXT_LABELS, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if metric_idx == 0:
            ax.set_ylabel("Score")

    add_panel_label(axes[0], "A")
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.2,
    )

    return save_figure(fig, output_dir / "fig1_key_metrics", formats, transparent)


def plot_tpr_fpr_tradeoff(output_dir: Path, formats: Sequence[str], transparent: bool) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True, constrained_layout=True)

    for ax, attack in zip(axes, ATTACKS):
        for defense in DEFENSES:
            fpr_values: list[float] = []
            tpr_values: list[float] = []

            for setting in SETTINGS:
                rec = get_record(setting, attack, defense)
                fpr_values.append(rec.fpr)
                tpr_values.append(rec.tpr)

                ax.scatter(
                    rec.fpr,
                    rec.tpr,
                    s=84 if defense == HIGHLIGHT_DEFENSE else 64,
                    marker=SETTING_MARKERS[setting],
                    color=COLORS[defense],
                    edgecolor="#111111" if defense == HIGHLIGHT_DEFENSE else "white",
                    linewidth=1.0,
                    zorder=4,
                )

            ax.plot(
                fpr_values,
                tpr_values,
                color=COLORS[defense],
                linewidth=2.0 if defense == HIGHLIGHT_DEFENSE else 1.2,
                alpha=0.9 if defense == HIGHLIGHT_DEFENSE else 0.6,
                zorder=3,
            )

        ax.set_title(attack, pad=6)
        ax.set_xlim(-0.02, 0.56)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xlabel("FPR (lower is better)")
        ax.grid(True)
        ax.annotate(
            "better",
            xy=(0.02, 0.98),
            xytext=(0.18, 0.84),
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#333333"},
            fontsize=8,
            color="#333333",
        )

    axes[0].set_ylabel("TPR (higher is better)")
    add_panel_label(axes[0], "B")

    defense_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=COLORS[d],
            markeredgecolor="#111111" if d == HIGHLIGHT_DEFENSE else "white",
            label=d,
        )
        for d in DEFENSES
    ]
    setting_handles = [
        Line2D(
            [0],
            [0],
            marker=SETTING_MARKERS[s],
            linestyle="None",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="#444444",
            label=s,
        )
        for s in SETTINGS
    ]
    fig.legend(
        handles=defense_handles + setting_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    return save_figure(fig, output_dir / "fig2_tpr_vs_fpr", formats, transparent)


def compute_win_points() -> dict[str, float]:
    points = {defense: 0.0 for defense in DEFENSES}
    for setting, attack in CONTEXTS:
        for metric, higher_is_better in METRIC_DIRECTIONS:
            values = np.array([getattr(get_record(setting, attack, defense), metric) for defense in DEFENSES])
            best_value = np.max(values) if higher_is_better else np.min(values)
            winners = np.where(np.isclose(values, best_value, atol=1e-12))[0]
            split_point = 1.0 / float(len(winners))
            for winner_idx in winners:
                points[DEFENSES[int(winner_idx)]] += split_point
    return points


def plot_win_dominance(output_dir: Path, formats: Sequence[str], transparent: bool) -> list[Path]:
    points = compute_win_points()
    totals = [points[d] for d in DEFENSES]
    x = np.arange(len(DEFENSES))

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.9), constrained_layout=True)

    bars = ax.bar(
        x,
        totals,
        width=0.62,
        color=[COLORS[d] for d in DEFENSES],
        edgecolor=["#111111" if d == HIGHLIGHT_DEFENSE else "none" for d in DEFENSES],
        linewidth=[1.1 if d == HIGHLIGHT_DEFENSE else 0.0 for d in DEFENSES],
        zorder=3,
    )

    for bar, value in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.25,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DEFENSES, rotation=15, ha="right")
    ax.set_ylabel("Best-metric wins")
    ax.set_title("Overall dominance", pad=6)
    ax.set_ylim(0.0, max(totals) + 2.0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.97,
        "28 metric comparisons\n(ties split equally)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#333333",
    )
    add_panel_label(ax, "C")

    return save_figure(fig, output_dir / "fig3_overall_dominance", formats, transparent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality detection figures from fixed CIFAR-10 results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/detection_suite/figures/nature_style"),
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Use transparent background for exported figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    generated: list[Path] = []
    generated += plot_key_metrics(args.output_dir, args.formats, args.transparent)
    generated += plot_tpr_fpr_tradeoff(args.output_dir, args.formats, args.transparent)
    generated += plot_win_dominance(args.output_dir, args.formats, args.transparent)

    print("Generated figure files:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
