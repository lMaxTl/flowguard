"""Score-space "landscape" plots for the FlowGuard++ composite detector.

Consumes the per-query score dumps written by
``scripts/evaluate_attack_defense_matrix_smoke.py --dump-composite-scores``
(one ``<attack>.npz`` per attack) and renders landscapes that make the
benign-vs-attack separation visible at a glance:

    <attack>_density_landscape.png   benign vs attack 2D density surfaces + difference
    <attack>_energy_landscape.png    detector anomaly-energy surface with query points
    attack_density_landscapes.png    every attack's 3D density surface side by side
    attack_component_heatmap.png     attack-vs-feature shift heatmap (relative to benign)
    landscape_grid.png               every attack's density contour over the benign one
    feature_ridges.png               per-feature density ridges (benign + each attack)

Two complementary readings of "landscape":

* density landscape -- where the query *mass* sits in a shared 2D projection of
  the feature space (a benign hill vs an attack hill in a different place);
* energy landscape -- the detector's equal-weight anomaly score as a continuous
  surface, with benign points in the low-energy basin and attack points pushed
  up the wall (this is the "loss landscape" analogue for the detector).

Both are built on a single shared PCA projection fit on the pooled
(benign + all attacks) features, so every panel lives in the same coordinates
and the attack landscapes are directly comparable to the one benign landscape.
The features are tail-compressed (signed log1p) before that projection: the C2
(t0) component is several orders of magnitude heavier tailed on attack traffic
than on benign traffic, and without the compression the pooled standardization
is driven by the attack tail, collapsing the whole benign distribution onto a
sub-pixel sliver. Density differences use one symmetric-log color scale shared
across panels, so panels stay quantitatively comparable and a single extreme
cell can no longer flatten every other panel to the neutral color.

Usage (local .venv):
    .\\.venv\\Scripts\\python.exe scripts/plot_score_landscapes.py \\
        --scores-dir runs/fgpp_learned/composite_scores_fgpp-learned
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (HPC compute nodes have no display)

import matplotlib.pyplot as plt  # noqa: E402
import mpl_toolkits.mplot3d  # noqa: E402, F401  (registers the '3d' projection)
import numpy as np  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BENIGN_COLOR = "#3B82F6"
ATTACK_COLOR = "#F97316"

# IEEEtran journal \textwidth on letter paper (matches figure* \linewidth in main.tex).
PAPER_TEXTWIDTH_IN = 7.1417
PAPER_ATTACK_LANDSCAPE_HEIGHT_IN = 2.75
PAPER_TITLE_FONT = 11
PAPER_LABEL_FONT = 10
PAPER_TICK_FONT = 9


def _apply_paper_style() -> None:
    """Matplotlib defaults sized for inclusion at \\linewidth in the IEEE paper."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": PAPER_LABEL_FONT,
            "axes.titlesize": PAPER_TITLE_FONT,
            "axes.labelsize": PAPER_LABEL_FONT,
            "xtick.labelsize": PAPER_TICK_FONT,
            "ytick.labelsize": PAPER_TICK_FONT,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _signed_log1p(matrix: np.ndarray) -> np.ndarray:
    """Monotone tail compression ``sign(x) * log1p(|x|)``.

    The C2 (``t0``) component is orders of magnitude heavier tailed on attack
    traffic than on benign traffic (benign median ~0, MAZE median ~1.8e4). Fit
    on the pooled features, a plain :class:`StandardScaler` therefore divides by
    an attack-driven scale and collapses the entire benign distribution onto a
    sub-pixel sliver, which makes the shared projection unreadable. Compressing
    the tails first keeps the ordering intact while putting benign and attack
    traffic in the same visible frame.
    """
    return np.sign(matrix) * np.log1p(np.abs(matrix))


def _inverse_signed_log1p(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_signed_log1p` (used to undo the energy-grid mapping)."""
    return np.sign(matrix) * np.expm1(np.abs(matrix))


def _tail_scale_ratio(benign: np.ndarray, attacks: list[np.ndarray]) -> float:
    """Worst-component ratio of the pooled spread to the benign spread.

    This is the quantity that decides whether a pooled standardization is safe.
    At ratio ~1 the groups share a scale and the raw projection is faithful; at
    ratio >> 1 the pooled scale is set by an attack tail and benign traffic is
    squeezed toward a point. Measured on CIFAR-10 (~7.8e3) versus MNIST (~5.6),
    which is why the choice cannot be hard-coded per dataset.
    """
    pooled = np.vstack([benign] + attacks)
    benign_std = benign.std(axis=0)
    safe = np.where(benign_std > 1e-12, benign_std, np.nan)
    ratio = np.asarray(pooled.std(axis=0)) / safe
    if not np.isfinite(ratio).any():
        return 1.0
    return float(np.nanmax(ratio))


@dataclass(slots=True)
class _DiffScale:
    """Symmetric-log color scale shared by every density-difference panel."""

    limit: float
    linthresh: float

    def norm(self) -> SymLogNorm:
        return SymLogNorm(
            linthresh=self.linthresh, linscale=0.5, vmin=-self.limit, vmax=self.limit, base=10
        )

    def levels(self, per_decade: int = 4) -> np.ndarray:
        decades = math.log10(self.limit / self.linthresh)
        steps = max(3, int(round(decades * per_decade)) + 1)
        positive = self.linthresh * np.logspace(0.0, decades, steps)
        positive[-1] = self.limit
        return np.concatenate([-positive[::-1], [0.0], positive])


def _diff_scale(diffs: list[np.ndarray], *, quantile: float = 0.90) -> _DiffScale | None:
    """Shared diverging scale for a set of density differences.

    ``limit`` is the true pooled maximum, so no panel silently clips its peak.
    ``linthresh`` (the width of the near-zero neutral band) is taken from a high
    quantile of |diff| so the log part of the scale spends its dynamic range on
    real structure instead of on KDE noise. Sharing both across panels is what
    makes the per-attack panels quantitatively comparable.
    """
    finite = [np.abs(diff).ravel() for diff in diffs if diff is not None]
    if not finite:
        return None
    magnitudes = np.concatenate(finite)
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    if magnitudes.size == 0:
        return None
    limit = float(magnitudes.max())
    if limit <= 0.0:
        return None
    linthresh = float(np.quantile(magnitudes, quantile))
    return _DiffScale(limit=limit, linthresh=float(np.clip(linthresh, limit * 1e-3, limit * 0.1)))


def _benign_contour_levels(density: np.ndarray) -> np.ndarray:
    """Quantile-spaced benign contours (linear levels degenerate on peaked KDEs)."""
    positive = density[density > 0.0]
    if positive.size < 8:
        return np.asarray([], dtype=float)
    return np.unique(np.quantile(positive, [0.55, 0.75, 0.90, 0.98]))


def _attack_panel_title(name: str) -> str:
    """Short attack label matching the paper figure caption."""
    lowered = name.lower()
    if "d1" in lowered or "latent_manifold" in lowered:
        return "Velocity-adaptive"
    if "d2" in lowered or "procedural_natural" in lowered:
        return "D2"
    if "d3" in lowered:
        return "D3"
    if "d4" in lowered:
        return "D4"
    if lowered == "maze" or lowered.endswith("_maze"):
        return "MAZE"
    if "disguide" in lowered:
        return "DisGUIDE"
    if "prada" in lowered:
        return "PRADA"
    return name.replace("_", " ")


def _style_3d_axis(ax: plt.Axes, *, show_zlabel: bool) -> None:
    ax.tick_params(axis="both", which="major", labelsize=PAPER_TICK_FONT, pad=2)
    if show_zlabel:
        ax.set_zlabel("density", fontsize=PAPER_LABEL_FONT, labelpad=6)
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-dir",
        required=True,
        help="Directory of <attack>.npz dumps (composite_scores_<run-label>) or a single .npz file.",
    )
    parser.add_argument("--outdir", default=None, help="Output dir (default: <scores-dir>/landscapes).")
    parser.add_argument("--attacks", default=None, help="Optional comma-separated subset of attack names.")
    parser.add_argument(
        "--feature-set",
        choices=["components", "raw"],
        default="components",
        help="'components' = standardized anomalies used for fusion; 'raw' = t0/integral/loglik.",
    )
    parser.add_argument(
        "--feature-compression",
        choices=["auto", "signed-log", "none"],
        default="auto",
        help=(
            "Tail compression applied before the shared scaler/PCA. 'auto' (default) applies "
            "signed-log1p only when the pooled spread exceeds the benign spread by more than "
            "--compression-threshold, which is the regime where pooled standardization collapses "
            "benign traffic to a sliver. 'signed-log' and 'none' force the choice."
        ),
    )
    parser.add_argument(
        "--compression-threshold",
        type=float,
        default=50.0,
        help="Pooled/benign spread ratio above which 'auto' turns on tail compression.",
    )
    parser.add_argument("--grid", type=int, default=120, help="Landscape grid resolution per axis.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=4000,
        help="Random subsample per group for KDE/scatter (speed + readability).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI (default: 300, paper-ready).")
    parser.add_argument(
        "--paper-width-in",
        type=float,
        default=PAPER_TEXTWIDTH_IN,
        help="Target print width in inches (IEEE figure* \\linewidth).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


@dataclass(slots=True)
class _Group:
    """A single attack dump reduced to the chosen feature set."""

    name: str
    feature_names: list[str]
    benign: np.ndarray  # (n_benign, k)
    attack: np.ndarray  # (n_attack, k)


def _resolve_npz_files(scores_dir: Path, attacks: list[str] | None) -> list[Path]:
    if scores_dir.is_file() and scores_dir.suffix == ".npz":
        return [scores_dir]
    files = sorted(scores_dir.glob("*.npz"))
    if attacks:
        wanted = set(attacks)
        files = [path for path in files if path.stem in wanted]
    return files


def _load_group(path: Path, feature_set: str) -> _Group | None:
    data = np.load(path)
    if feature_set == "components":
        names = [str(name) for name in data["component_names"].tolist()]
        benign = np.asarray(data["benign_components"], dtype=np.float64)
        attack = np.asarray(data["attack_components"], dtype=np.float64)
    else:
        aliases = [a for a in ("t0", "integral", "loglik") if f"benign_raw_{a}" in data]
        if not aliases:
            return None
        names = aliases
        benign = np.column_stack([np.asarray(data[f"benign_raw_{a}"], dtype=np.float64) for a in aliases])
        attack = np.column_stack([np.asarray(data[f"attack_raw_{a}"], dtype=np.float64) for a in aliases])
    benign = benign[np.isfinite(benign).all(axis=1)]
    attack = attack[np.isfinite(attack).all(axis=1)]
    if benign.shape[0] < 3 or attack.shape[0] < 3 or benign.shape[1] < 2:
        return None
    return _Group(name=path.stem, feature_names=names, benign=benign, attack=attack)


def _order_groups(groups: list[_Group]) -> list[_Group]:
    """Order attacks so the adaptive bypasses (D1-D4) render on the right.

    Baselines (prada, maze, disguide) keep their alphabetical order; any attack
    whose name carries an adaptive marker (d1..d4 / 'adaptive') is pushed last so
    the per-attack landscape rows read baseline -> adaptive left to right.
    """
    adaptive_markers = ("d1", "d2", "d3", "d4", "adaptive")

    def sort_key(group: _Group) -> tuple[int, str]:
        name = group.name.lower()
        is_adaptive = any(marker in name for marker in adaptive_markers)
        return (1 if is_adaptive else 0, name)

    return sorted(groups, key=sort_key)


def _subsample(matrix: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if matrix.shape[0] <= max_points:
        return matrix
    index = rng.choice(matrix.shape[0], size=max_points, replace=False)
    return matrix[index]


def _kde_surface(points2d: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray | None:
    """Gaussian-KDE density evaluated on a meshgrid; None if degenerate."""
    try:
        kde = gaussian_kde(points2d.T)
    except (np.linalg.LinAlgError, ValueError):
        return None
    positions = np.vstack([xx.ravel(), yy.ravel()])
    return kde(positions).reshape(xx.shape)


def _grid_axes(coords: np.ndarray, resolution: int, margin: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    lo = mins - margin * span
    hi = maxs + margin * span
    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    return np.meshgrid(xs, ys)


def _plot_density_landscape(
    group: _Group,
    benign2d: np.ndarray,
    attack2d: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    benign_density = _kde_surface(benign2d, xx, yy)
    attack_density = _kde_surface(attack2d, xx, yy)
    if benign_density is None or attack_density is None:
        return
    zmax = float(max(benign_density.max(), attack_density.max()))

    fig = plt.figure(figsize=(16, 4.6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2 = fig.add_subplot(1, 3, 3)

    for ax, density, title, cmap in (
        (ax0, benign_density, "benign", "Blues"),
        (ax1, attack_density, f"attack: {group.name}", "Oranges"),
    ):
        ax.plot_surface(xx, yy, density, cmap=cmap, linewidth=0, antialiased=True, vmin=0.0, vmax=zmax)
        ax.set_zlim(0.0, zmax)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("density")

    diff = attack_density - benign_density
    scale = _diff_scale([diff])
    if scale is None:
        contour = ax2.contourf(xx, yy, diff, levels=21, cmap="RdBu_r")
    else:
        contour = ax2.contourf(
            xx, yy, diff, levels=scale.levels(), cmap="RdBu_r", norm=scale.norm(), extend="neither"
        )
    ax2.set_title("attack - benign density")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    fig.colorbar(contour, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f"{group.name}: density landscape (shared projection)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_energy_landscape(
    group: _Group,
    benign2d: np.ndarray,
    attack2d: np.ndarray,
    benign_energy: np.ndarray,
    attack_energy: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    energy_grid: np.ndarray,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(xx, yy, energy_grid, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True)
    ax.scatter(
        benign2d[:, 0], benign2d[:, 1], benign_energy,
        color=BENIGN_COLOR, s=6, alpha=0.5, label="benign",
    )
    ax.scatter(
        attack2d[:, 0], attack2d[:, 1], attack_energy,
        color=ATTACK_COLOR, s=6, alpha=0.5, label="attack",
    )
    ax.set_title(f"{group.name}: detector anomaly-energy landscape")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("equal-weight anomaly energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_attack_density_landscapes(
    benign2d: np.ndarray,
    attacks2d: dict[str, np.ndarray],
    xx: np.ndarray,
    yy: np.ndarray,
    out_path: Path,
    *,
    dpi: int,
    paper_width_in: float,
) -> None:
    """One row of 3D attack-density surfaces sized for IEEE figure* \\linewidth."""
    if not attacks2d:
        return
    benign_density = _kde_surface(benign2d, xx, yy)
    if benign_density is None:
        return
    benign_peak = float(benign_density.max())

    densities: dict[str, np.ndarray] = {}
    for name, attack2d in attacks2d.items():
        surface = _kde_surface(attack2d, xx, yy)
        if surface is not None:
            densities[name] = surface
    if not densities:
        return

    names = list(densities.keys())
    fig = plt.figure(figsize=(paper_width_in, PAPER_ATTACK_LANDSCAPE_HEIGHT_IN))
    for index, name in enumerate(names):
        attack_density = densities[name]
        zmax = float(max(benign_peak, attack_density.max()))
        ax = fig.add_subplot(1, len(names), index + 1, projection="3d")
        ax.plot_surface(
            xx, yy, attack_density,
            cmap="Oranges", linewidth=0, antialiased=True, vmin=0.0, vmax=zmax,
        )
        ax.set_zlim(0.0, zmax)
        ax.set_title(_attack_panel_title(name), fontsize=PAPER_TITLE_FONT, pad=6)
        ax.set_xlabel("PC1", fontsize=PAPER_LABEL_FONT)
        ax.set_ylabel("PC2", fontsize=PAPER_LABEL_FONT)
        _style_3d_axis(ax, show_zlabel=index == 0)
        ax.view_init(elev=28, azim=-58)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.88, wspace=0.08)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_landscape_grid(
    benign2d: np.ndarray,
    attacks2d: dict[str, np.ndarray],
    xx: np.ndarray,
    yy: np.ndarray,
    out_path: Path,
    *,
    dpi: int,
    paper_width_in: float,
) -> None:
    """Shared-projection density-difference grid (one panel per attack).

    Every panel shows excess density (attack - benign) on a single symmetric-log
    color scale shared across panels, with the benign reference density drawn as
    contour lines on top.
    """
    benign_density = _kde_surface(benign2d, xx, yy)
    if benign_density is None or not attacks2d:
        return

    diffs: dict[str, np.ndarray] = {}
    for name, coords in attacks2d.items():
        attack_density = _kde_surface(coords, xx, yy)
        if attack_density is not None:
            diffs[name] = attack_density - benign_density
    if not diffs:
        return

    scale = _diff_scale(list(diffs.values()))
    if scale is None:
        return
    levels = scale.levels()
    norm = scale.norm()
    benign_levels = _benign_contour_levels(benign_density)

    names = list(diffs.keys())
    cols = max(1, len(names))
    height = 2.15
    fig, axes = plt.subplots(1, cols, figsize=(paper_width_in, height), squeeze=False)
    mappable = None
    for index, name in enumerate(names):
        ax = axes[0][index]
        mappable = ax.contourf(
            xx, yy, diffs[name], levels=levels, cmap="RdBu_r", norm=norm, extend="neither"
        )
        if benign_levels.size > 1:
            ax.contour(
                xx, yy, benign_density, levels=benign_levels,
                colors=BENIGN_COLOR, linewidths=0.8, alpha=0.95,
            )
        ax.set_title(_attack_panel_title(name), fontsize=PAPER_TICK_FONT, pad=3)
        ax.set_xlabel("PC1", fontsize=PAPER_TICK_FONT)
        ax.set_ylabel("PC2" if index == 0 else "", fontsize=PAPER_TICK_FONT)
        ax.tick_params(labelsize=PAPER_TICK_FONT - 1, length=2)
        ax.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.22, top=0.88, wspace=0.16)
    if mappable is not None:
        cax = fig.add_axes([0.915, 0.22, 0.012, 0.66])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.tick_params(labelsize=PAPER_TICK_FONT - 2, length=2)
        cbar.set_label(r"$\Delta\rho$ (symlog)", fontsize=PAPER_TICK_FONT - 1)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_feature_ridges(groups: list[_Group], out_path: Path, *, dpi: int) -> None:
    feature_names = groups[0].feature_names
    row_labels = ["benign"] + [group.name for group in groups]
    cols = len(feature_names)
    fig, axes = plt.subplots(1, cols, figsize=(4.6 * cols, 0.7 * len(row_labels) + 2.0), squeeze=False)
    for col, feature in enumerate(feature_names):
        ax = axes[0][col]
        # Benign is shared across files; use the first group's benign as reference.
        series = [groups[0].benign[:, col]] + [group.attack[:, col] for group in groups]
        finite = np.concatenate([values[np.isfinite(values)] for values in series])
        grid = np.linspace(np.quantile(finite, 0.01), np.quantile(finite, 0.99), 256)
        offset_step = 1.0
        for row, (label, values) in enumerate(zip(row_labels, series)):
            values = values[np.isfinite(values)]
            try:
                density = gaussian_kde(values)(grid)
            except (np.linalg.LinAlgError, ValueError):
                continue
            density = density / (density.max() or 1.0)
            base = row * offset_step
            color = BENIGN_COLOR if row == 0 else ATTACK_COLOR
            ax.fill_between(grid, base, base + density, color=color, alpha=0.55, linewidth=0)
            ax.plot(grid, base + density, color="black", linewidth=0.6, alpha=0.6)
            if col == 0:
                ax.text(grid[0], base + 0.1, label, fontsize=8, va="bottom")
        ax.set_yticks([])
        ax.set_title(feature)
        ax.set_xlabel("value")
    fig.suptitle("Per-feature density ridges (benign vs each attack)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_component_heatmap(groups: list[_Group], out_path: Path, *, dpi: int, paper_width_in: float) -> None:
    """Heatmap of attack feature shifts relative to benign.

    Cell value = (attack_mean - benign_mean) / benign_std for each feature.
    Positive means the attack tends to push that feature above benign; negative
    means below benign. This avoids projection artifacts and keeps every
    component directly interpretable.
    """
    if not groups:
        return
    feature_names = groups[0].feature_names
    benign = groups[0].benign
    benign_mean = np.nanmean(benign, axis=0)
    benign_std = np.nanstd(benign, axis=0)
    benign_std = np.where(benign_std > 1e-12, benign_std, 1.0)

    attack_names = [_attack_panel_title(group.name) for group in groups]
    shift_rows: list[np.ndarray] = []
    for group in groups:
        attack_mean = np.nanmean(group.attack, axis=0)
        shift_rows.append((attack_mean - benign_mean) / benign_std)
    shifts = np.vstack(shift_rows)

    abs_shifts = np.abs(shifts)
    # Robust color scaling so extreme outliers (e.g., MAZE t0) do not flatten
    # the rest of the grid into near-zero color.
    vmax = float(np.nanquantile(abs_shifts, 0.90))
    vmax = max(vmax, 3.0)
    vmin = -vmax
    display = np.clip(shifts, vmin, vmax)

    height = max(2.4, 0.48 * len(attack_names) + 1.2)
    fig, ax = plt.subplots(figsize=(paper_width_in, height))
    image = ax.imshow(display, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks(np.arange(len(attack_names)))
    ax.set_yticklabels(attack_names)
    ax.set_xlabel("FlowGuard++ component feature")
    ax.set_ylabel("Attack")
    ax.set_title("Component shift heatmap: attack mean relative to benign")

    for row in range(shifts.shape[0]):
        for col in range(shifts.shape[1]):
            value = float(shifts[row, col])
            text_color = "white" if abs(display[row, col]) > 0.55 * vmax else "black"
            label = f"{value:.1f}" if abs(value) >= 100 else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("z-shift (clipped colors) = (attack mean - benign mean) / benign std")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    _apply_paper_style()
    rng = np.random.default_rng(args.seed)
    scores_dir = Path(args.scores_dir)
    base = scores_dir.parent if scores_dir.is_file() else scores_dir
    outdir = Path(args.outdir) if args.outdir else base / "landscapes"
    outdir.mkdir(parents=True, exist_ok=True)
    attacks_filter = [token.strip() for token in args.attacks.split(",")] if args.attacks else None

    npz_files = _resolve_npz_files(scores_dir, attacks_filter)
    groups = [g for g in (_load_group(path, args.feature_set) for path in npz_files) if g is not None]
    if not groups:
        raise SystemExit(f"No usable score dumps under {scores_dir} (need >=3 queries, >=2 features).")
    groups = _order_groups(groups)

    # Shared projection: compress heavy tails, standardize, then PCA(2) on the
    # pooled benign + all attack features, so every landscape is drawn in one
    # comparable coordinate system. Without the compression the pooled scale is
    # set by the attack-side t0 tail and benign traffic collapses to a sliver.
    identity = lambda matrix: matrix  # noqa: E731
    tail_ratio = _tail_scale_ratio(groups[0].benign, [group.attack for group in groups])
    if args.feature_compression == "auto":
        use_compression = tail_ratio > float(args.compression_threshold)
    else:
        use_compression = args.feature_compression == "signed-log"
    print(
        f"[landscape] pooled/benign spread ratio: {tail_ratio:.4g} -> tail compression "
        f"{'on' if use_compression else 'off'} (--feature-compression {args.feature_compression})"
    )
    compress = _signed_log1p if use_compression else identity
    decompress = _inverse_signed_log1p if use_compression else identity

    pooled = compress(np.vstack([groups[0].benign] + [group.attack for group in groups]))
    scaler = StandardScaler().fit(pooled)
    pca = PCA(n_components=2, random_state=args.seed).fit(scaler.transform(pooled))

    def project(matrix: np.ndarray) -> np.ndarray:
        return pca.transform(scaler.transform(compress(matrix)))

    benign_full = groups[0].benign
    all_coords = np.vstack(
        [project(benign_full)] + [project(group.attack) for group in groups]
    )
    xx, yy = _grid_axes(all_coords, args.grid)

    # Energy grid: equal-weight anomaly (mean over features) reconstructed from
    # the 2D plane via the inverse projection -- the detector's decision surface.
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    reconstructed = decompress(scaler.inverse_transform(pca.inverse_transform(grid_points)))
    energy_grid = reconstructed.mean(axis=1).reshape(xx.shape)

    benign_sub = _subsample(benign_full, args.max_points, rng)
    benign2d = project(benign_sub)
    benign_energy = benign_sub.mean(axis=1)
    attacks2d: dict[str, np.ndarray] = {}

    for group in groups:
        attack_sub = _subsample(group.attack, args.max_points, rng)
        attack2d = project(attack_sub)
        attacks2d[group.name] = attack2d
        print(f"[landscape] {group.name}: benign={benign_sub.shape} attack={attack_sub.shape}")
        _plot_density_landscape(
            group, benign2d, attack2d, xx, yy,
            outdir / f"{group.name}_density_landscape.png", dpi=args.dpi,
        )
        _plot_energy_landscape(
            group, benign2d, attack2d, benign_energy, attack_sub.mean(axis=1),
            xx, yy, energy_grid,
            outdir / f"{group.name}_energy_landscape.png", dpi=args.dpi,
        )

    _plot_attack_density_landscapes(
        benign2d, attacks2d, xx, yy, outdir / "attack_density_landscapes.png",
        dpi=args.dpi, paper_width_in=args.paper_width_in,
    )
    _plot_component_heatmap(
        groups, outdir / "attack_component_heatmap.png", dpi=args.dpi, paper_width_in=args.paper_width_in
    )
    _plot_landscape_grid(
        benign2d, attacks2d, xx, yy, outdir / "landscape_grid.png", dpi=args.dpi, paper_width_in=args.paper_width_in
    )
    _plot_feature_ridges(groups, outdir / "feature_ridges.png", dpi=args.dpi)
    explained = ", ".join(f"{ratio:.2f}" for ratio in pca.explained_variance_ratio_)
    print(f"[landscape] PCA explained variance ratio (PC1, PC2): {explained}")
    print(f"[landscape] wrote figures for {len(groups)} attack(s) to {outdir}")


if __name__ == "__main__":
    main()
