from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

BLUE = "#3B82F6"
ORANGE = "#F97316"
GRID = "#D9DEE5"


@dataclass(slots=True)
class QueryGroup:
    name: str
    query_features: np.ndarray
    latent_features: np.ndarray
    log_likelihood: np.ndarray


@dataclass(slots=True)
class FigureData:
    legitimate: QueryGroup
    maze: QueryGroup
    disguide: QueryGroup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 3-panel Nature-style figure for Flow Matching OOD detection.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/figures/flow_matching_ood",
        help="Directory where PDF/PNG are written.",
    )
    parser.add_argument(
        "--output-prefix",
        default="flow_matching_ood_three_panel",
        help="Filename prefix for the exported figure files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-legitimate", type=int, default=200)
    parser.add_argument("--num-maze", type=int, default=200)
    parser.add_argument("--num-disguide", type=int, default=200)
    parser.add_argument(
        "--projection",
        choices=["tsne", "pca"],
        default="tsne",
        help="2D projection used for Panel A/B.",
    )
    parser.add_argument(
        "--latent-view",
        choices=["first2", "tsne", "pca"],
        default="first2",
        help="2D view for Panel B latent space. 'first2' preserves radial OOD structure best.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Optional fixed threshold. If omitted, uses 10th percentile of legitimate log-likelihood.",
    )

    # Real-data hooks: if all three NPZs are provided and valid, placeholders are skipped.
    parser.add_argument("--legitimate-npz", default="", help="NPZ path with real legitimate query data.")
    parser.add_argument("--maze-npz", default="", help="NPZ path with real MAZE query data.")
    parser.add_argument("--disguide-npz", default="", help="NPZ path with real DisGUIDE query data.")
    return parser


def _load_group_npz(path: Path, *, name: str) -> QueryGroup:
    payload = np.load(path)
    required = ("query_features", "latent_features", "log_likelihood")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            f"{path} is missing required arrays: {missing}. "
            "Expected keys: query_features, latent_features, log_likelihood."
        )
    return QueryGroup(
        name=name,
        query_features=np.asarray(payload["query_features"], dtype=np.float64),
        latent_features=np.asarray(payload["latent_features"], dtype=np.float64),
        log_likelihood=np.asarray(payload["log_likelihood"], dtype=np.float64).reshape(-1),
    )


def load_real_data_if_available(args: argparse.Namespace) -> FigureData | None:
    npz_paths = {
        "legitimate": Path(args.legitimate_npz) if args.legitimate_npz else None,
        "maze": Path(args.maze_npz) if args.maze_npz else None,
        "disguide": Path(args.disguide_npz) if args.disguide_npz else None,
    }
    if any(path is None for path in npz_paths.values()):
        return None

    for path in npz_paths.values():
        assert path is not None
        if not path.exists():
            raise FileNotFoundError(f"Real-data NPZ not found: {path}")

    legitimate = _load_group_npz(npz_paths["legitimate"], name="Legitimate")
    maze = _load_group_npz(npz_paths["maze"], name="MAZE")
    disguide = _load_group_npz(npz_paths["disguide"], name="DisGUIDE")
    return FigureData(legitimate=legitimate, maze=maze, disguide=disguide)


def _make_manifold_features(
    rng: np.random.Generator,
    count: int,
    *,
    offset_curve: float,
    offset_vertical: float,
    noise_scale: float,
    projection: np.ndarray,
) -> np.ndarray:
    t = rng.uniform(-2.7, 2.7, size=count)
    latent_curve = np.column_stack(
        [
            t,
            np.sin(1.1 * t + offset_curve) + offset_vertical,
            np.cos(0.8 * t - 0.5 * offset_curve) + 0.35 * offset_vertical,
        ]
    )
    base = latent_curve @ projection
    return base + rng.normal(loc=0.0, scale=noise_scale, size=base.shape)


def _make_attack_latent_features(
    rng: np.random.Generator,
    count: int,
    *,
    dim: int,
    radius_low: float,
    radius_high: float,
    directional_jitter: float,
) -> np.ndarray:
    if dim < 2:
        raise ValueError("Attack latent simulation requires at least 2 dimensions.")

    angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radii = rng.uniform(radius_low, radius_high, size=count)
    first_two = np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None]
    first_two += rng.normal(loc=0.0, scale=0.10, size=first_two.shape)

    remaining_dim = dim - 2
    if remaining_dim > 0:
        remainder = rng.normal(loc=0.0, scale=directional_jitter, size=(count, remaining_dim))
        return np.concatenate([first_two, remainder], axis=1)
    return first_two


def simulate_placeholder_data(args: argparse.Namespace) -> FigureData:
    """
    Placeholder simulation to mimic the empirical behavior:
    - legitimate queries map close to latent Gaussian core and receive higher log-likelihood
    - attack queries (MAZE/DisGUIDE) map farther away and receive lower log-likelihood

    Replace this function with your real inference pipeline if desired.
    """
    rng = np.random.default_rng(args.seed)

    q_dim = 96
    z_dim = 24

    query_projection = rng.normal(loc=0.0, scale=0.72, size=(3, q_dim))

    legitimate_q = _make_manifold_features(
        rng,
        args.num_legitimate,
        offset_curve=0.0,
        offset_vertical=0.0,
        noise_scale=0.16,
        projection=query_projection,
    )
    maze_q = _make_manifold_features(
        rng,
        args.num_maze,
        offset_curve=0.45,
        offset_vertical=1.75,
        noise_scale=0.20,
        projection=query_projection,
    )
    disguide_q = _make_manifold_features(
        rng,
        args.num_disguide,
        offset_curve=-0.50,
        offset_vertical=-1.85,
        noise_scale=0.22,
        projection=query_projection,
    )

    legitimate_z = rng.normal(loc=0.0, scale=0.78, size=(args.num_legitimate, z_dim))
    maze_z = _make_attack_latent_features(
        rng,
        args.num_maze,
        dim=z_dim,
        radius_low=2.3,
        radius_high=3.6,
        directional_jitter=0.18,
    )
    disguide_z = _make_attack_latent_features(
        rng,
        args.num_disguide,
        dim=z_dim,
        radius_low=2.6,
        radius_high=4.1,
        directional_jitter=0.20,
    )

    # Simulated full log-likelihood values (including Jacobian correction).
    legitimate_ll = rng.normal(loc=-2350.0, scale=110.0, size=args.num_legitimate)
    maze_ll = rng.normal(loc=-3070.0, scale=120.0, size=args.num_maze)
    disguide_ll = rng.normal(loc=-3210.0, scale=125.0, size=args.num_disguide)

    return FigureData(
        legitimate=QueryGroup("Legitimate", legitimate_q, legitimate_z, legitimate_ll),
        maze=QueryGroup("MAZE", maze_q, maze_z, maze_ll),
        disguide=QueryGroup("DisGUIDE", disguide_q, disguide_z, disguide_ll),
    )


def load_or_simulate_data(args: argparse.Namespace) -> tuple[FigureData, bool]:
    real_data = load_real_data_if_available(args)
    if real_data is not None:
        return real_data, True
    return simulate_placeholder_data(args), False


def project_2d(features: np.ndarray, *, method: Literal["tsne", "pca"], seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(scaled)

    perplexity = float(max(10, min(40, (len(features) - 1) // 4)))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        max_iter=1200,
    )
    return tsne.fit_transform(scaled)


def project_latent_2d(
    latent_features: np.ndarray,
    *,
    method: Literal["first2", "tsne", "pca"],
    seed: int,
) -> np.ndarray:
    if method == "first2":
        if latent_features.shape[1] < 2:
            raise ValueError("latent_features must have at least 2 columns for --latent-view first2")
        return StandardScaler().fit_transform(latent_features[:, :2])
    if method == "pca":
        return project_2d(latent_features, method="pca", seed=seed)
    return project_2d(latent_features, method="tsne", seed=seed)


def _split_projection(projected: np.ndarray, sizes: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_legit, n_maze, n_disguide = sizes
    a = projected[:n_legit]
    b = projected[n_legit : n_legit + n_maze]
    c = projected[n_legit + n_maze : n_legit + n_maze + n_disguide]
    return a, b, c


def _ellipse_from_points(points_2d: np.ndarray, n_std: float = 2.0) -> tuple[np.ndarray, float, float, float]:
    mean = points_2d.mean(axis=0)
    cov = np.cov(points_2d.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    width = 2.0 * n_std * np.sqrt(max(eigenvalues[0], 1e-10))
    height = 2.0 * n_std * np.sqrt(max(eigenvalues[1], 1e-10))
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    return mean, width, height, angle


def _draw_scatter_groups(
    ax: plt.Axes,
    legitimate: np.ndarray,
    attacks: np.ndarray,
) -> None:
    ax.scatter(
        legitimate[:, 0],
        legitimate[:, 1],
        c=BLUE,
        s=15,
        alpha=0.60,
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        attacks[:, 0],
        attacks[:, 1],
        facecolors="none",
        edgecolors=ORANGE,
        s=20,
        alpha=0.72,
        linewidths=0.95,
        zorder=4,
    )


def plot_panel_a(
    ax: plt.Axes,
    *,
    legitimate_2d: np.ndarray,
    attacks_2d: np.ndarray,
    axis_label: str,
) -> None:
    sns.kdeplot(
        x=legitimate_2d[:, 0],
        y=legitimate_2d[:, 1],
        ax=ax,
        fill=True,
        levels=7,
        thresh=0.06,
        color=BLUE,
        alpha=0.16,
    )
    sns.kdeplot(
        x=legitimate_2d[:, 0],
        y=legitimate_2d[:, 1],
        ax=ax,
        fill=False,
        levels=7,
        thresh=0.06,
        color=BLUE,
        alpha=0.36,
        linewidths=0.85,
    )
    _draw_scatter_groups(ax, legitimate_2d, attacks_2d)
    ax.set_title("Query Space", fontsize=15, pad=8)
    ax.set_xlabel(f"{axis_label} 1", fontsize=15)
    ax.set_ylabel(f"{axis_label} 2", fontsize=15)


def plot_panel_b(
    ax: plt.Axes,
    *,
    legitimate_2d: np.ndarray,
    attacks_2d: np.ndarray,
    axis_label: str,
) -> None:
    _draw_scatter_groups(ax, legitimate_2d, attacks_2d)
    center, width, height, angle = _ellipse_from_points(legitimate_2d, n_std=2.0)
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor="#374151",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(ellipse)
    ax.annotate(
        "In-distribution\nacceptance region",
        xy=(center[0], center[1] - 0.5 * height),  # Pfeil zeigt auf Ellipsenrand
        xytext=(center[0] - 0.7 * width, center[1] - 0.8 * height),  # Text weit oben-links
        fontsize=15,
        color="#374151",
        fontstyle="italic",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#374151",
            lw=0.8,
            connectionstyle="arc3,rad=0.2",
        ),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#ffffff",
            edgecolor="#9CA3AF",
            alpha=0.8,
            linewidth=0.5,
        ),
        ha="center",
        va="top",
        zorder=10,
    )
    ax.set_title("Latent Space (Backward ODE)", fontsize=15, pad=8)
    ax.set_xlabel(f"{axis_label} 1", fontsize=15)
    ax.set_ylabel(f"{axis_label} 2", fontsize=15)


def plot_panel_c(
    ax: plt.Axes,
    *,
    legitimate_ll: np.ndarray,
    attack_ll: np.ndarray,
    tau: float,
) -> None:
    minimum = float(min(legitimate_ll.min(), attack_ll.min()))
    maximum = float(max(legitimate_ll.max(), attack_ll.max()))

    ax.axvspan(minimum, tau, color=ORANGE, alpha=0.11, lw=0.0, zorder=0)
    ax.axvspan(tau, maximum, color=BLUE, alpha=0.09, lw=0.0, zorder=0)

    sns.kdeplot(
        legitimate_ll,
        ax=ax,
        fill=True,
        color=BLUE,
        alpha=0.28,
        linewidth=1.0,
        label="Legitimate density",
    )
    sns.kdeplot(
        attack_ll,
        ax=ax,
        fill=True,
        color=ORANGE,
        alpha=0.28,
        linewidth=1.0,
        label="Attack density",
    )

    ax.axvline(tau, color="black", linestyle="--", linewidth=1.0)
    ymax = ax.get_ylim()[1]
    ax.text(tau, ymax * 0.96, "τ", ha="center", va="top", fontsize=15)

    ax.set_title("Log-Likelihood Decision", fontsize=15, pad=8)
    ax.set_xlabel(r"$\log p_1(x)$", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.text(
        0.5,
        -0.23,
        "Full log-likelihood incl. Jacobian correction",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        color="#4B5563",
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("#5B6573")
    ax.spines["bottom"].set_color("#5B6573")
    ax.grid(True, linewidth=0.45, alpha=0.18, color=GRID)
    ax.set_axisbelow(True)


def create_figure(
    data: FigureData,
    *,
    projection: str,
    latent_view: str,
    seed: int,
    tau: float,
    used_real_data: bool,
) -> plt.Figure:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.45,
        }
    )

    query_all = np.vstack([data.legitimate.query_features, data.maze.query_features, data.disguide.query_features])
    latent_all = np.vstack([data.legitimate.latent_features, data.maze.latent_features, data.disguide.latent_features])

    query_2d = project_2d(query_all, method=projection, seed=seed)
    latent_2d = project_latent_2d(latent_all, method=latent_view, seed=seed + 11)

    sizes = (len(data.legitimate.query_features), len(data.maze.query_features), len(data.disguide.query_features))
    legit_q, maze_q, disguide_q = _split_projection(query_2d, sizes)
    legit_z, maze_z, disguide_z = _split_projection(latent_2d, sizes)

    attack_q = np.vstack([maze_q, disguide_q])
    attack_z = np.vstack([maze_z, disguide_z])
    attack_ll = np.concatenate([data.maze.log_likelihood, data.disguide.log_likelihood], axis=0)

    axis_label_query = "t-SNE" if projection == "tsne" else "PCA"
    axis_label_latent = "z0" if latent_view == "first2" else ("t-SNE" if latent_view == "tsne" else "PCA")

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5), constrained_layout=False)
    fig.patch.set_facecolor("white")

    plot_panel_a(axes[0], legitimate_2d=legit_q, attacks_2d=attack_q, axis_label=axis_label_query)
    plot_panel_b(axes[1], legitimate_2d=legit_z, attacks_2d=attack_z, axis_label=axis_label_latent)
    plot_panel_c(axes[2], legitimate_ll=data.legitimate.log_likelihood, attack_ll=attack_ll, tau=tau)

    for index, ax in enumerate(axes):
        style_axes(ax)
        ax.text(
            -0.13,
            1.06,
            chr(ord("A") + index),
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="left",
        )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=6.5, label="Legitimate queries"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=ORANGE, markersize=7.0, label="Attack queries (MAZE + DisGUIDE)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Threshold τ"),
        Line2D([0], [0], color="#374151", linestyle=(0, (4, 3)), linewidth=1.0, label="2σ latent acceptance region"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=15,
        handlelength=1.9,
        columnspacing=1.6,
    )

    #fig.suptitle("Identity-Independent Detection via Flow Matching", fontsize=15, fontweight="bold", y=1.09)
    #subtitle = "Per-query evaluation · No identity information required · Resilient to Sybil attacks"
    #fig.text(0.5, -0.02, subtitle, ha="center", va="center", fontsize=15, color="#4B5563")

    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.20, wspace=0.22)
    return fig


def save_figure(fig: plt.Figure, *, output_dir: Path, output_prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_prefix}.png"
    pdf_path = output_dir / f"{output_prefix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    return png_path, pdf_path


def main() -> None:
    args = build_parser().parse_args()
    data, used_real_data = load_or_simulate_data(args)

    tau = float(args.tau) if args.tau is not None else float(np.quantile(data.legitimate.log_likelihood, 0.10))
    fig = create_figure(
        data,
        projection=args.projection,
        latent_view=args.latent_view,
        seed=args.seed,
        tau=tau,
        used_real_data=used_real_data,
    )

    png_path, pdf_path = save_figure(
        fig,
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
    )
    plt.close(fig)

    source_mode = "real CNF data" if used_real_data else "placeholder simulation"
    print(f"Generated figure using {source_mode}.")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
