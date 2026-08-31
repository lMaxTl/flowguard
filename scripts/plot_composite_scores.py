"""Post-hoc analysis plots for the FlowGuard++ composite detector.

This script consumes the per-query score dumps written by
``scripts/evaluate_attack_defense_matrix_smoke.py`` when it is run with
``--dump-composite-scores`` (or the sbatch wrapper with
``DUMP_COMPOSITE_SCORES=1``). Each dump is one ``<attack>.npz`` holding the
standardized component matrix used for fusion plus the raw t0 / integral /
log-likelihood signals, for both the benign reference stream and the attack
stream.

For every attack it produces:

    <attack>_component_distributions.png  standardized anomalies (benign vs attack)
    <attack>_loglik_typicality.png        raw log-likelihood histogram (typicality failure)
    <attack>_component_correlation.png     correlation matrix of standardized components
    <attack>_fusion_ablation.png           single-component + leave-one-out AUROC
    <attack>_learned_weights.png           logistic-regression component weights

and two cross-attack artifacts:

    auroc_overview.png                     per-component / fused AUROC heatmap
    composite_score_metrics.json           machine-readable copy of every number

Usage (local .venv):
    .\\.venv\\Scripts\\python.exe scripts/plot_composite_scores.py \\
        --scores-dir runs/fgpp_learned/composite_scores_fgpp-learned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (HPC compute nodes have no display)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

BENIGN_COLOR = "#3B82F6"
ATTACK_COLOR = "#F97316"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-dir",
        required=True,
        help=(
            "Directory of <attack>.npz dumps (composite_scores_<run-label>) "
            "or a single .npz file."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for figures (default: <scores-dir>/figures).",
    )
    parser.add_argument(
        "--attacks",
        default=None,
        help="Optional comma-separated subset of attack names to plot.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--bins", type=int, default=40, help="Histogram bin count.")
    parser.add_argument(
        "--target-fpr", type=float, default=0.05,
        help="Benign false-positive rate defining the operating threshold drawn on the raw-signal plots.",
    )
    return parser


def _resolve_npz_files(scores_dir: Path, attacks: list[str] | None) -> list[Path]:
    if scores_dir.is_file() and scores_dir.suffix == ".npz":
        return [scores_dir]
    files = sorted(scores_dir.glob("*.npz"))
    if attacks:
        wanted = set(attacks)
        files = [path for path in files if path.stem in wanted]
    return files


def _safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC that tolerates degenerate inputs (NaN / single class)."""
    finite = np.isfinite(scores)
    if not finite.all():
        scores = np.where(finite, scores, np.nanmean(scores[finite]) if finite.any() else 0.0)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _fused_score(components: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Equal-weight fusion = mean over (selected) standardized component columns."""
    selected = components if mask is None else components[:, mask]
    if selected.shape[1] == 0:
        return np.zeros(selected.shape[0], dtype=np.float64)
    return selected.mean(axis=1)


class _AttackScores:
    """Loaded score dump for a single attack."""

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.attack = path.stem
        self.names: list[str] = [str(name) for name in data["component_names"].tolist()]
        self.benign_components: np.ndarray = np.asarray(data["benign_components"], dtype=np.float64)
        self.attack_components: np.ndarray = np.asarray(data["attack_components"], dtype=np.float64)
        self.raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for alias in ("t0", "integral", "loglik", "composite"):
            benign_key = f"benign_raw_{alias}"
            attack_key = f"attack_raw_{alias}"
            if benign_key in data and attack_key in data:
                self.raw[alias] = (
                    np.asarray(data[benign_key], dtype=np.float64),
                    np.asarray(data[attack_key], dtype=np.float64),
                )

    @property
    def features(self) -> np.ndarray:
        return np.vstack([self.benign_components, self.attack_components])

    @property
    def labels(self) -> np.ndarray:
        n_benign = self.benign_components.shape[0]
        n_attack = self.attack_components.shape[0]
        return np.concatenate(
            [np.zeros(n_benign, dtype=np.int64), np.ones(n_attack, dtype=np.int64)]
        )


def _plot_component_distributions(scores: _AttackScores, out_path: Path, *, bins: int, dpi: int) -> None:
    names = scores.names
    n = len(names)
    cols = min(3, n) or 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
    labels = scores.labels
    for index, name in enumerate(names):
        ax = axes[index // cols][index % cols]
        benign = scores.benign_components[:, index]
        attack = scores.attack_components[:, index]
        edges = np.histogram_bin_edges(
            np.concatenate([benign, attack])[np.isfinite(np.concatenate([benign, attack]))],
            bins=bins,
        )
        ax.hist(benign, bins=edges, color=BENIGN_COLOR, alpha=0.6, density=True, label="benign")
        ax.hist(attack, bins=edges, color=ATTACK_COLOR, alpha=0.6, density=True, label="attack")
        auroc = _safe_auroc(scores.features[:, index], labels)
        ax.set_title(f"{name}  (AUROC={auroc:.3f})")
        ax.set_xlabel("standardized anomaly (z)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    for index in range(n, rows * cols):
        axes[index // cols][index % cols].axis("off")
    fig.suptitle(f"{scores.attack}: per-component standardized anomalies")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_loglik_typicality(scores: _AttackScores, out_path: Path, *, bins: int, dpi: int) -> float | None:
    if "loglik" not in scores.raw:
        return None
    benign, attack = scores.raw["loglik"]
    benign = benign[np.isfinite(benign)]
    attack = attack[np.isfinite(attack)]
    if benign.size == 0 or attack.size == 0:
        return None
    mu = float(benign.mean())
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    edges = np.histogram_bin_edges(np.concatenate([benign, attack]), bins=bins)
    ax.hist(benign, bins=edges, color=BENIGN_COLOR, alpha=0.6, density=True, label="benign")
    ax.hist(attack, bins=edges, color=ATTACK_COLOR, alpha=0.6, density=True, label="attack")
    ax.axvline(mu, color="black", linestyle="--", linewidth=1.0, label=r"benign mean $\mu_L$")
    # Fraction of attack queries that sit on the *higher-likelihood* side of
    # the benign mean: the typicality failure a one-sided -log p(x) detector misses.
    above = float(np.mean(attack > mu))
    ax.set_title(
        f"{scores.attack}: raw log-likelihood\n"
        f"{above * 100:.1f}% of attack queries have log p(x) > benign mean"
    )
    ax.set_xlabel(r"$\log p_1(x)$")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return above


def _plot_correlation(scores: _AttackScores, out_path: Path, *, dpi: int) -> None:
    features = scores.features
    if features.shape[1] < 2:
        return
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(features, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    names = scores.names
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.2 * len(names) + 1.5))
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{scores.attack}: component correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_fusion_ablation(scores: _AttackScores, out_path: Path, *, dpi: int) -> dict[str, float]:
    names = scores.names
    features = scores.features
    labels = scores.labels
    single = {name: _safe_auroc(features[:, idx], labels) for idx, name in enumerate(names)}
    full = _safe_auroc(_fused_score(features), labels)
    leave_one_out: dict[str, float] = {}
    for idx, name in enumerate(names):
        mask = np.ones(len(names), dtype=bool)
        mask[idx] = False
        leave_one_out[name] = _safe_auroc(_fused_score(features, mask), labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positions = np.arange(len(names))
    axes[0].bar(positions, [single[name] for name in names], color=BENIGN_COLOR)
    axes[0].axhline(full, color="black", linestyle="--", linewidth=1.0, label=f"full fusion={full:.3f}")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Single-component AUROC")
    axes[0].legend(fontsize=8)

    drops = [full - leave_one_out[name] for name in names]
    axes[1].bar(positions, drops, color=ATTACK_COLOR)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_ylabel("AUROC drop when removed")
    axes[1].set_title("Leave-one-out contribution")

    fig.suptitle(f"{scores.attack}: fusion ablation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "single_component_auroc": single,
        "full_fusion_auroc": full,
        "leave_one_out_auroc": leave_one_out,
    }


def _plot_learned_weights(scores: _AttackScores, out_path: Path, *, dpi: int) -> dict[str, float]:
    features = scores.features
    labels = scores.labels
    if len(np.unique(labels)) < 2:
        return {}
    finite_features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    model = LogisticRegression(max_iter=1000)
    model.fit(finite_features, labels)
    weights = {name: float(w) for name, w in zip(scores.names, model.coef_.ravel())}

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    positions = np.arange(len(scores.names))
    colors = [BENIGN_COLOR if weights[name] >= 0 else ATTACK_COLOR for name in scores.names]
    ax.bar(positions, [weights[name] for name in scores.names], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(scores.names, rotation=45, ha="right")
    ax.set_ylabel(r"logistic-regression weight $\alpha$")
    ax.set_title(f"{scores.attack}: learned component weights")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return weights


RAW_SIGNAL_LABELS = {
    "t0": "FlowPure score  ||v(t=0,x)||^2",
    "integral": "trajectory integral score",
    "loglik": "log-likelihood  log p(x)",
    "composite": "composite anomaly score",
}


def _plot_raw_signal_grid(
    all_scores: list[_AttackScores],
    alias: str,
    out_path: Path,
    *,
    bins: int,
    dpi: int,
    target_fpr: float = 0.05,
) -> dict[str, dict]:
    """Raw single-signal detector view: benign vs attack, one panel per attack.

    The composite fuses several components, which hides *which* signal a given
    attack actually defeats. This plots one raw signal on its own, with the
    benign-calibrated operating threshold drawn in, so a chance-level column in
    the defense matrix can be read off directly: an attack whose histogram sits
    on top of the benign one is matching that statistic.
    """
    usable = [s for s in all_scores if alias in s.raw]
    if not usable:
        return {}

    pooled = []
    for s in usable:
        benign, attack = s.raw[alias]
        pooled.append(benign[np.isfinite(benign)])
        pooled.append(attack[np.isfinite(attack)])
    flat = np.concatenate(pooled)
    if flat.size == 0:
        return {}
    lo, hi = np.percentile(flat, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(flat.min()), float(flat.max()) or 1.0
    edges = np.linspace(lo, hi, bins + 1)

    stats: dict[str, dict] = {}
    cols = len(usable)
    fig, axes = plt.subplots(1, cols, figsize=(3.1 * cols, 3.0), squeeze=False, sharex=True)
    for index, s in enumerate(usable):
        ax = axes[0][index]
        benign, attack = s.raw[alias]
        benign = benign[np.isfinite(benign)]
        attack = attack[np.isfinite(attack)]
        # FlowPure flags high scores, so the operating threshold is the benign
        # upper quantile at the target false-positive rate.
        threshold = float(np.quantile(benign, 1.0 - target_fpr))
        tpr = float(np.mean(attack >= threshold))
        labels = np.concatenate([np.zeros(benign.size), np.ones(attack.size)])
        values = np.concatenate([benign, attack])
        auroc = float(roc_auc_score(labels, values)) if benign.size and attack.size else float("nan")
        stats[s.attack] = {"auroc": auroc, "tpr_at_target_fpr": tpr, "threshold": threshold}

        ax.hist(benign, bins=edges, color=BENIGN_COLOR, alpha=0.6, density=True, label="benign")
        ax.hist(attack, bins=edges, color=ATTACK_COLOR, alpha=0.6, density=True, label="attack")
        ax.axvline(
            threshold, color="black", linestyle="--", linewidth=1.0,
            label=f"threshold (FPR={target_fpr:.2f})",
        )
        ax.set_title(f"{s.attack}\nAUROC={auroc:.3f}  TPR={tpr:.3f}", fontsize=9)
        ax.set_xlabel(RAW_SIGNAL_LABELS.get(alias, alias), fontsize=8)
        ax.tick_params(labelsize=7)
        if index == 0:
            ax.set_ylabel("density", fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle(
        f"Raw {alias} signal on its own (benign vs attack), shared x-axis", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return stats


def _plot_auroc_overview(per_attack: dict[str, dict], names: list[str], out_path: Path, *, dpi: int) -> None:
    attacks = list(per_attack.keys())
    if not attacks or not names:
        return
    columns = names + ["fused"]
    grid = np.full((len(attacks), len(columns)), np.nan, dtype=np.float64)
    for row, attack in enumerate(attacks):
        single = per_attack[attack].get("single_component_auroc", {})
        for col, name in enumerate(names):
            grid[row, col] = single.get(name, np.nan)
        grid[row, -1] = per_attack[attack].get("full_fusion_auroc", np.nan)

    fig, ax = plt.subplots(figsize=(1.3 * len(columns) + 2, 0.8 * len(attacks) + 2))
    image = ax.imshow(grid, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(attacks)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(attacks)
    for row in range(len(attacks)):
        for col in range(len(columns)):
            value = grid[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    ax.set_title("Per-component vs fused AUROC by attack")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    scores_dir = Path(args.scores_dir)
    outdir = Path(args.outdir) if args.outdir else (
        scores_dir.parent / "figures" if scores_dir.is_file() else scores_dir / "figures"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    attacks_filter = [token.strip() for token in args.attacks.split(",")] if args.attacks else None

    npz_files = _resolve_npz_files(scores_dir, attacks_filter)
    if not npz_files:
        raise SystemExit(f"No .npz score dumps found under {scores_dir}")

    metrics: dict[str, dict] = {}
    component_names: list[str] = []
    all_scores: list[_AttackScores] = []
    for path in npz_files:
        scores = _AttackScores(path)
        all_scores.append(scores)
        component_names = component_names or scores.names
        print(f"[plot] {scores.attack}: benign={scores.benign_components.shape} attack={scores.attack_components.shape}")
        _plot_component_distributions(
            scores, outdir / f"{scores.attack}_component_distributions.png", bins=args.bins, dpi=args.dpi
        )
        typicality_above = _plot_loglik_typicality(
            scores, outdir / f"{scores.attack}_loglik_typicality.png", bins=args.bins, dpi=args.dpi
        )
        _plot_correlation(scores, outdir / f"{scores.attack}_component_correlation.png", dpi=args.dpi)
        ablation = _plot_fusion_ablation(
            scores, outdir / f"{scores.attack}_fusion_ablation.png", dpi=args.dpi
        )
        weights = _plot_learned_weights(
            scores, outdir / f"{scores.attack}_learned_weights.png", dpi=args.dpi
        )
        metrics[scores.attack] = {
            **ablation,
            "learned_weights": weights,
            "frac_attack_loglik_above_benign_mean": typicality_above,
            "num_benign": int(scores.benign_components.shape[0]),
            "num_attack": int(scores.attack_components.shape[0]),
        }

    for alias, filename in (
        ("t0", "flowpure_t0_scores.png"),
        ("integral", "trajectory_integral_scores.png"),
        ("loglik", "loglik_scores.png"),
    ):
        raw_stats = _plot_raw_signal_grid(
            all_scores, alias, outdir / filename,
            bins=args.bins, dpi=args.dpi, target_fpr=float(args.target_fpr),
        )
        for attack_name, values in raw_stats.items():
            metrics.setdefault(attack_name, {}).setdefault("raw_signal", {})[alias] = values

    _plot_auroc_overview(metrics, component_names, outdir / "auroc_overview.png", dpi=args.dpi)
    metrics_path = outdir / "composite_score_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[plot] wrote {len(npz_files)} attack(s) of figures to {outdir}")
    print(f"[plot] wrote metrics {metrics_path}")


if __name__ == "__main__":
    main()
