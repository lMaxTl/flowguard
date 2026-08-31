from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a paper-style teaser image for query-manifold OOD detection.",
    )
    parser.add_argument(
        "--output",
        default="runs/teaser/teaser_flow_matching_manifold.png",
        help="Output path for the teaser image (png/pdf supported by matplotlib).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=float, default=32.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=8.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _data_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ridge_center = 0.70 * np.sin(0.85 * x) - 0.12 * x
    ridge = np.exp(-0.5 * ((y - ridge_center) / 0.33) ** 2) * np.exp(-0.5 * (x / 2.3) ** 2)
    lobe_left = 0.55 * np.exp(-((x + 1.85) ** 2 / (2 * 0.78**2) + (y + 0.45) ** 2 / (2 * 0.95**2)))
    lobe_right = 0.45 * np.exp(-((x - 1.45) ** 2 / (2 * 0.95**2) + (y - 0.30) ** 2 / (2 * 0.85**2)))
    return ridge + lobe_left + lobe_right


def _sample_legitimate(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.uniform(-2.7, 2.7, size=n)
    manifold = 0.70 * np.sin(0.85 * x) - 0.12 * x
    y = manifold + rng.normal(loc=0.0, scale=0.16, size=n)
    return np.column_stack([x, y])


def _sample_synthetic_attacks(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.uniform(-3.0, 3.0, size=n)
    manifold = 0.70 * np.sin(0.85 * x) - 0.12 * x
    offset = np.sign(rng.normal(size=n)) * rng.uniform(0.95, 1.70, size=n)
    y = manifold + offset + rng.normal(loc=0.0, scale=0.18, size=n)
    return np.column_stack([x, y])


def _map_to_latent_space(
    rng: np.random.Generator,
    points: np.ndarray,
    *,
    attack: bool,
) -> np.ndarray:
    transform = np.asarray([[0.58, -0.20], [0.17, 0.62]], dtype=np.float64)
    latent = points @ transform.T
    latent += rng.normal(loc=0.0, scale=0.13, size=latent.shape)

    if attack:
        norms = np.linalg.norm(latent, axis=1, keepdims=True) + 1e-6
        target_radius = rng.uniform(2.25, 3.25, size=(len(latent), 1))
        latent = latent / norms * target_radius
    else:
        latent *= 0.78
    return latent


def _setup_axis(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float], title: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa6ad")
    ax.spines["bottom"].set_color("#9aa6ad")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def main() -> None:
    args = _build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 13,
            "axes.grid": False,
        }
    )

    legitimate = _sample_legitimate(rng, n=180)
    attacks = _sample_synthetic_attacks(rng, n=64)

    latent_legitimate = _map_to_latent_space(rng, legitimate, attack=False)
    latent_attacks = _map_to_latent_space(rng, attacks, attack=True)

    x = np.linspace(-3.5, 3.5, 400)
    y = np.linspace(-3.0, 3.0, 320)
    xx, yy = np.meshgrid(x, y)
    data_density = _data_density(xx, yy)

    z = np.linspace(-3.5, 3.5, 400)
    w = np.linspace(-3.5, 3.5, 320)
    zz, ww = np.meshgrid(z, w)
    normal_density = np.exp(-0.5 * (zz**2 + ww**2))

    manifold_cmap = LinearSegmentedColormap.from_list(
        "manifold_blue_teal",
        ["#f8fdff", "#d8eff5", "#abd8e5", "#7db5cf", "#4d8bb5", "#2b6f99"],
    )
    normal_cmap = LinearSegmentedColormap.from_list(
        "latent_normal",
        ["#ffffff", "#eaf0f3", "#d3dde5", "#b9c8d4", "#93a8b7", "#6f8798"],
    )

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(args.width, args.height),
        dpi=args.dpi,
        gridspec_kw={"wspace": 0.10},
    )

    _setup_axis(
        ax_left,
        xlim=(-3.4, 3.4),
        ylim=(-2.8, 2.8),
        title="Data Manifold (Query Space)",
    )
    _setup_axis(
        ax_right,
        xlim=(-3.4, 3.4),
        ylim=(-3.4, 3.4),
        title="Gaussian Latent Manifold (Backward ODE Map)",
    )

    contour_levels_left = np.linspace(float(data_density.min()), float(data_density.max()), 14)
    ax_left.contourf(xx, yy, data_density, levels=contour_levels_left, cmap=manifold_cmap, alpha=0.98)
    ax_left.contour(
        xx,
        yy,
        data_density,
        levels=contour_levels_left[::2],
        colors="#76a9bd",
        linewidths=0.75,
        alpha=0.55,
    )

    contour_levels_right = np.linspace(float(normal_density.min()), float(normal_density.max()), 14)
    ax_right.contourf(zz, ww, normal_density, levels=contour_levels_right, cmap=normal_cmap, alpha=0.95)
    ax_right.contour(
        zz,
        ww,
        normal_density,
        levels=contour_levels_right[::2],
        colors="#8ea1af",
        linewidths=0.70,
        alpha=0.60,
    )

    inlier_region = Circle(
        (0.0, 0.0),
        radius=2.0,
        fill=False,
        edgecolor="#415a69",
        linewidth=1.6,
        linestyle=(0, (5, 3)),
        alpha=0.95,
    )
    ax_right.add_patch(inlier_region)
    ax_right.text(
        -1.95,
        2.20,
        "In-distribution\nacceptance region",
        fontsize=12,
        color="#344b58",
        ha="left",
        va="bottom",
    )

    ax_left.scatter(
        legitimate[:, 0],
        legitimate[:, 1],
        s=28,
        c="#1f77b4",
        edgecolors="white",
        linewidths=0.55,
        alpha=0.96,
        zorder=4,
    )
    ax_left.scatter(
        attacks[:, 0],
        attacks[:, 1],
        s=64,
        facecolors="none",
        edgecolors="#d95f02",
        linewidths=1.6,
        alpha=0.95,
        zorder=5,
    )

    ax_right.scatter(
        latent_legitimate[:, 0],
        latent_legitimate[:, 1],
        s=24,
        c="#2f7fb8",
        edgecolors="white",
        linewidths=0.45,
        alpha=0.92,
        zorder=4,
    )
    ax_right.scatter(
        latent_attacks[:, 0],
        latent_attacks[:, 1],
        s=62,
        facecolors="none",
        edgecolors="#d95f02",
        linewidths=1.5,
        alpha=0.93,
        zorder=5,
    )

    for idx in rng.choice(len(legitimate), size=9, replace=False):
        connector = ConnectionPatch(
            xyA=(legitimate[idx, 0], legitimate[idx, 1]),
            coordsA=ax_left.transData,
            xyB=(latent_legitimate[idx, 0], latent_legitimate[idx, 1]),
            coordsB=ax_right.transData,
            color="#a5adb3",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )
        fig.add_artist(connector)

    for idx in rng.choice(len(attacks), size=7, replace=False):
        connector = ConnectionPatch(
            xyA=(attacks[idx, 0], attacks[idx, 1]),
            coordsA=ax_left.transData,
            xyB=(latent_attacks[idx, 0], latent_attacks[idx, 1]),
            coordsB=ax_right.transData,
            color="#b4bcc2",
            linewidth=0.75,
            linestyle="--",
            alpha=0.50,
            zorder=1,
        )
        fig.add_artist(connector)

    ax_left.text(
        -3.25,
        2.45,
        "A",
        fontsize=30,
        fontweight="bold",
        color="#1f1f1f",
        ha="left",
        va="top",
    )
    ax_right.text(
        -3.25,
        3.05,
        "B",
        fontsize=30,
        fontweight="bold",
        color="#1f1f1f",
        ha="left",
        va="top",
    )

    ax_left.text(
        -3.20,
        -2.55,
        "High-density ridge: legitimate traffic manifold",
        fontsize=12,
        color="#365f78",
        ha="left",
        va="bottom",
    )
    ax_right.text(
        -3.20,
        -3.20,
        "OOD attack queries remain outside the normal manifold",
        fontsize=12,
        color="#4e5d68",
        ha="left",
        va="bottom",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#1f77b4",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=9,
            label="Legitimate queries",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#d95f02",
            markeredgewidth=1.5,
            markersize=10,
            label="Synthetic attack queries",
        ),
        Line2D(
            [0],
            [0],
            color="#a5adb3",
            linewidth=1.0,
            label="Backward ODE flow trajectory",
        ),
        Line2D(
            [0],
            [0],
            color="#415a69",
            linewidth=1.4,
            linestyle=(0, (5, 3)),
            label="Latent in-distribution boundary",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=13,
        handlelength=1.8,
        columnspacing=1.5,
    )

    fig.suptitle(
        "Identity-Independent Detection via Flow Matching",
        y=1.06,
        fontsize=24,
        fontweight="bold",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated teaser image at: {output_path}")


if __name__ == "__main__":
    main()
