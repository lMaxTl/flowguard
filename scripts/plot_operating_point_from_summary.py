"""Paper figure: operating-point detection metrics from combined-full summary.

Uses only metrics already present in
``attack_defense_matrix_combined-full_summary.json`` (no new experiments):

- TPR / FNR / accepted-attack fraction at the *deployed* threshold
- AUROC vs TPR contrast (shows why AUROC alone is insufficient)
- Surrogate fidelity of the evaluated attack stream (passive labels)

For score-based defenses calibrated at target FPR=0.05 (FlowPure, FDINet,
Integral), the stored ``tpr`` is TPR at that ~5% FPR operating point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ATTACKS: list[tuple[str, str]] = [
    ("prada", "PRADA"),
    ("maze", "MAZE"),
    ("disguide", "DisGUIDE"),
    ("flowguard_d1_latent_manifold", "D1"),
    ("flowguard_d2_procedural_natural", "D2"),
]

# Paper-facing defense order (exclude Like./Comp.(L); Comp incomplete for D1/D2).
DEFENSES: list[tuple[str, str]] = [
    ("prada", "PRADA"),
    ("flowpure", "FlowPure"),
    ("fdinet", "FDINet"),
    ("flowguard_integral", "Int."),
    ("flowguard_userlevel", "KS"),
    ("flowguard_labelhist", "Label"),
    ("flowguard_composite", "Comp."),
]

# Defenses whose threshold was calibrated to target FPR=0.05 in this run.
SCORE_CALIBRATED_5PCT = {
    "flowpure",
    "fdinet",
    "flowguard_integral",
    "flowguard_composite",
}


def _index_results(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["defense"], r["attack"]): r for r in results}


def _load(summary_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return _index_results(payload["results"])


def _metric_matrix(
    index: dict[tuple[str, str], dict[str, Any]],
    key: str,
) -> np.ndarray:
    """Shape: (n_defenses, n_attacks); NaN when missing."""
    mat = np.full((len(DEFENSES), len(ATTACKS)), np.nan, dtype=float)
    for i, (def_key, _) in enumerate(DEFENSES):
        for j, (atk_key, _) in enumerate(ATTACKS):
            row = index.get((def_key, atk_key))
            if row is None or row.get(key) is None:
                continue
            mat[i, j] = float(row[key])
    return mat


def _accepted_fraction(index: dict[tuple[str, str], dict[str, Any]]) -> np.ndarray:
    """Fraction of attack queries not blocked = FN / num_attack = 1 - TPR."""
    tpr = _metric_matrix(index, "tpr")
    return 1.0 - tpr


def _accepted_count(index: dict[tuple[str, str], dict[str, Any]]) -> np.ndarray:
    mat = np.full((len(DEFENSES), len(ATTACKS)), np.nan, dtype=float)
    for i, (def_key, _) in enumerate(DEFENSES):
        for j, (atk_key, _) in enumerate(ATTACKS):
            row = index.get((def_key, atk_key))
            if row is None or row.get("fn") is None:
                continue
            mat[i, j] = float(row["fn"])
    return mat


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str = "{:.2f}",
    cbar_label: str,
) -> None:
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([lab for _, lab in ATTACKS])
    ax.set_yticks(range(len(DEFENSES)))
    ax.set_yticklabels([lab for _, lab in DEFENSES])
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                text = "—"
                color = "0.4"
            else:
                text = fmt.format(val)
                mid = 0.5 * (vmin + vmax)
                color = "white" if val >= mid else "black"
                if cmap == "RdYlGn" and 0.35 < val < 0.65:
                    color = "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)


def plot_operating_point_figure(
    index: dict[tuple[str, str], dict[str, Any]],
    out_path: Path,
) -> None:
    """Four-panel figure aimed at Major Comment 3."""
    _style()
    tpr = _metric_matrix(index, "tpr")
    auroc = _metric_matrix(index, "auroc")
    accepted_frac = _accepted_fraction(index)
    _ = _accepted_count(index)
    _ = 1.0 - tpr  # FNR == accepted_frac for binary detection

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))

    # (a) TPR at deployed threshold — primary rebuttal metric
    _heatmap(
        axes[0, 0],
        tpr,
        title="(a) Attack-query block rate (TPR @ deployed thr.)",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        cbar_label="TPR",
    )
    axes[0, 0].set_xlabel("Attack")
    axes[0, 0].set_ylabel("Defense")

    # (b) Accepted attack labels (what the thief still gets)
    _heatmap(
        axes[0, 1],
        accepted_frac,
        title="(b) Fraction of attack queries accepted (= FNR)",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Accepted fraction",
    )
    axes[0, 1].set_xlabel("Attack")
    axes[0, 1].set_ylabel("Defense")

    # (c) AUROC vs TPR scatter — reviewer narrative
    ax = axes[1, 0]
    markers = {"prada": "o", "maze": "s", "disguide": "^", "d1": "D", "d2": "v"}
    attack_short = {
        "prada": "prada",
        "maze": "maze",
        "disguide": "disguide",
        "flowguard_d1_latent_manifold": "d1",
        "flowguard_d2_procedural_natural": "d2",
    }
    colors = {
        "prada": "#4C78A8",
        "maze": "#F58518",
        "disguide": "#54A24B",
        "d1": "#E45756",
        "d2": "#B279A2",
    }
    for j, (atk_key, atk_lab) in enumerate(ATTACKS):
        xs: list[float] = []
        ys: list[float] = []
        for i, (_def_key, _) in enumerate(DEFENSES):
            a = auroc[i, j]
            t = tpr[i, j]
            if np.isnan(a) or np.isnan(t):
                continue
            xs.append(a)
            ys.append(t)
        short = attack_short[atk_key]
        ax.scatter(
            xs,
            ys,
            s=42,
            marker=markers[short],
            color=colors[short],
            label=atk_lab,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1, label="AUROC = TPR")
    ks_d2 = index.get(("flowguard_userlevel", "flowguard_d2_procedural_natural"))
    if ks_d2 and ks_d2.get("auroc") is not None:
        ax.annotate(
            "KS on D2:\nAUROC≈0.02, TPR≈0.98",
            xy=(float(ks_d2["auroc"]), float(ks_d2["tpr"])),
            xytext=(0.35, 0.55),
            fontsize=7,
            arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.8},
            color="0.2",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("AUROC (threshold-free)")
    ax.set_ylabel("TPR @ deployed threshold")
    ax.set_title("(c) AUROC does not imply operating-point TPR")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.grid(True, alpha=0.25, lw=0.5)

    # (d) Grouped bars for D1/D2 block rate — paper's critical adaptive attacks
    ax = axes[1, 1]
    d1_idx = 3
    d2_idx = 4
    x = np.arange(len(DEFENSES))
    width = 0.38
    d1_vals = tpr[:, d1_idx]
    d2_vals = tpr[:, d2_idx]
    bars1 = ax.bar(
        x - width / 2,
        np.nan_to_num(d1_vals, nan=0.0),
        width,
        label="D1",
        color="#E45756",
        edgecolor="white",
        linewidth=0.4,
    )
    bars2 = ax.bar(
        x + width / 2,
        np.nan_to_num(d2_vals, nan=0.0),
        width,
        label="D2",
        color="#B279A2",
        edgecolor="white",
        linewidth=0.4,
    )
    for bars, vals in ((bars1, d1_vals), (bars2, d2_vals)):
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                bar.set_facecolor("0.85")
                bar.set_hatch("///")
                bar.set_height(0.02)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in DEFENSES], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("TPR @ deployed threshold")
    ax.set_title("(d) Adaptive attacks D1/D2: block rate by detector")
    ax.axhline(0.05, color="0.5", ls=":", lw=1, label="chance @ 5% FPR")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    fig.suptitle(
        "Detection at the deployed operating point "
        "(K=50k; target FPR=5% for score detectors)",
        fontsize=11,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "Source: attack_defense_matrix_combined-full_summary.json. "
        "FlowPure/FDINet/Int./Comp. thresholds calibrated on 50k benign queries "
        "at FPR=5%. Comp.×D1/D2 missing (run incomplete). Hatched = unavailable.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

    csv_path = out_path.with_name(out_path.stem + "_table.csv")
    lines = [
        "defense,attack,auroc,tpr_deployed,fnr_deployed,fpr_deployed,"
        "accepted_attack_queries,num_attack,calibrated_at_5pct_fpr,"
        "surrogate_accuracy,surrogate_fidelity"
    ]
    for def_key, def_lab in DEFENSES:
        for atk_key, atk_lab in ATTACKS:
            row = index.get((def_key, atk_key))
            if row is None:
                lines.append(
                    f"{def_lab},{atk_lab},,,,,,,,{str(def_key in SCORE_CALIBRATED_5PCT).lower()},,"
                )
                continue
            auroc_v = row.get("auroc")
            tpr_v = row.get("tpr")
            fpr_v = row.get("fpr")
            fn = row.get("fn")
            n_atk = row.get("num_attack")
            lines.append(
                ",".join(
                    [
                        def_lab,
                        atk_lab,
                        "" if auroc_v is None else f"{auroc_v:.6f}",
                        "" if tpr_v is None else f"{tpr_v:.6f}",
                        "" if tpr_v is None else f"{1.0 - float(tpr_v):.6f}",
                        "" if fpr_v is None else f"{fpr_v:.6f}",
                        "" if fn is None else str(fn),
                        "" if n_atk is None else str(n_atk),
                        str(def_key in SCORE_CALIBRATED_5PCT).lower(),
                        "" if row.get("accuracy") is None else f"{row['accuracy']:.4f}",
                        "" if row.get("fidelity") is None else f"{row['fidelity']:.4f}",
                    ]
                )
            )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")
    print(f"Wrote {csv_path}")
    print("\nD2 operating-point snapshot (block / accept):")
    for def_key, def_lab in DEFENSES:
        row = index.get((def_key, "flowguard_d2_procedural_natural"))
        if row is None:
            print(f"  {def_lab:10s}  MISSING")
            continue
        print(
            f"  {def_lab:10s}  AUROC={row['auroc']:.3f}  "
            f"TPR={row['tpr']:.3f}  FNR={1 - row['tpr']:.3f}  "
            f"accepted={row['fn']}/{row['num_attack']}  "
            f"FPR={row['fpr']:.3f}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "runs/runs/attack_defense_combined_full/"
            "attack_defense_matrix_combined-full_summary.json"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/runs/attack_defense_combined_full/figures"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    index = _load(args.summary)
    out = args.outdir / "operating_point_detection_prada_maze_disguide_d1_d2.png"
    plot_operating_point_figure(index, out)


if __name__ == "__main__":
    main()
