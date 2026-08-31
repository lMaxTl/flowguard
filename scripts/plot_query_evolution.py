"""Plot how each attack's queries evolve over the query budget.

Companion to Fig.~\\ref{fig:d1-d2-queries}: one row per attack, one column per
query milestone, showing the images the victim actually received at that point
in the run. Reads the ``query_snapshots/`` directory that
``scripts/evaluate_attack_defense_matrix_smoke.py`` writes when
``--query-snapshot-milestones`` is set.

The figure is the qualitative counterpart to the fidelity table: it shows
*why* a generator family succeeds or fails in a domain, which a scalar AUROC
cannot.

Usage::

    python scripts/plot_query_evolution.py \\
        --snapshot-dir runs/mnist_benchmark/query_snapshots \\
        --output paper/images/mnist_query_evolution.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Attack key -> row label. Anything not listed still plots, labelled by key.
ROW_LABELS: dict[str, str] = {
    "disguide": "Baseline\n(DisGUIDE)",
    "flowguard_d1_latent_manifold": "D1\nlatent manifold",
    "flowguard_d2_procedural_natural": "D2\nprocedural",
    "flowguard_d3_adaptive_c1c2": "D3\nC1+C2",
    "flowguard_d4_adaptive_c1c3": "D4\nC1+C3",
    "flowguard_d5_adaptive_composite": "D5\ncomposite",
    "flowguard_d6_adaptive_stateful": "D6\ncomposite+stateful",
}
ROW_ORDER: list[str] = list(ROW_LABELS)


def _attack_from_filename(path: Path) -> str:
    """Recover the attack key from a ``<defense>__<attack>.pt`` snapshot name."""
    stem = path.stem
    return stem.split("__", 1)[1] if "__" in stem else stem


def _load_snapshots(snapshot_dir: Path) -> dict[str, list[dict[str, Any]]]:
    payloads: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(snapshot_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
        if not snapshots:
            continue
        attack = _attack_from_filename(path)
        # One file per (defense, attack) cell; they are independent runs, so
        # prefer whichever covers the most milestones.
        if attack not in payloads or len(snapshots) > len(payloads[attack]):
            payloads[attack] = snapshots
    return payloads


def _tile(images: np.ndarray, *, grid: int) -> np.ndarray:
    """Arrange the first ``grid**2`` images into a single square mosaic."""
    count, channels, height, width = images.shape
    wanted = grid * grid
    if count < wanted:
        pad = np.zeros((wanted - count, channels, height, width), dtype=images.dtype)
        images = np.concatenate([images, pad], axis=0)
    images = images[:wanted]
    rows = [
        np.concatenate(list(images[row * grid : (row + 1) * grid]), axis=2)
        for row in range(grid)
    ]
    mosaic = np.concatenate(rows, axis=1)
    return np.transpose(mosaic, (1, 2, 0))


def _format_queries(value: int) -> str:
    """Compact label for a realized query count (not the milestone threshold)."""
    if value >= 1_000_000:
        return f"{round(value / 1_000_000)}M"
    if value >= 10_000:
        return f"{round(value / 1_000)}k"
    return str(int(value))


def _snapshot_query_count(snapshot: dict[str, Any]) -> int:
    """Actual victim-query count when the snapshot was taken.

    Milestones are thresholds (``queries >= milestone``), so labelling a cell
    with the milestone alone is misleading: the first panel is typically the
    first full batch (e.g. 64), not ``1`` query.
    """
    if "queries" in snapshot and snapshot["queries"] is not None:
        return max(1, int(snapshot["queries"]))
    return max(1, int(snapshot["milestone"]))


def _output_paths(output: Path, formats: list[str]) -> list[Path]:
    """Expand a base path into one path per requested format."""
    return [output.with_suffix(f".{fmt.lstrip('.')}") for fmt in formats]


def _render_single(
    snapshots: list[dict[str, Any]],
    *,
    attack: str,
    output: Path,
    formats: list[str],
    grid: int,
    dpi: int,
) -> None:
    """Write a one-row figure for a single attack, for viewing in isolation."""
    figure, axes = plt.subplots(
        1, len(snapshots), figsize=(1.9 * len(snapshots), 2.3), squeeze=False
    )
    for index, snapshot in enumerate(sorted(snapshots, key=lambda s: int(s["milestone"]))):
        axis = axes[0][index]
        axis.set_xticks([])
        axis.set_yticks([])
        mosaic = np.clip(_tile(np.asarray(snapshot["images"]), grid=grid), 0.0, 1.0)
        if mosaic.shape[-1] == 1:
            axis.imshow(mosaic[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(mosaic)
        axis.set_title(
            f"{_format_queries(_snapshot_query_count(snapshot))} queries",
            fontsize=9,
        )
    label = ROW_LABELS.get(attack, attack).replace("\n", " - ")
    figure.suptitle(label, fontsize=10)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in _output_paths(output, formats):
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[figure] wrote {path}")
    plt.close(figure)


def _render(
    payloads: dict[str, list[dict[str, Any]]],
    *,
    output: Path,
    formats: list[str],
    grid: int,
    dpi: int,
    title: str | None,
) -> None:
    attacks = [key for key in ROW_ORDER if key in payloads]
    attacks += [key for key in sorted(payloads) if key not in ROW_LABELS]
    if not attacks:
        raise SystemExit("No snapshots to plot.")

    milestones = sorted({int(s["milestone"]) for key in attacks for s in payloads[key]})
    # Column titles use the realized query count, not the milestone threshold.
    milestone_query_labels: dict[int, int] = {}
    for key in attacks:
        for snapshot in payloads[key]:
            milestone = int(snapshot["milestone"])
            realized = _snapshot_query_count(snapshot)
            previous = milestone_query_labels.get(milestone)
            if previous is None or realized > previous:
                milestone_query_labels[milestone] = realized

    figure, axes = plt.subplots(
        len(attacks),
        len(milestones),
        figsize=(1.9 * len(milestones), 1.9 * len(attacks)),
        squeeze=False,
    )

    for row_index, attack in enumerate(attacks):
        by_milestone = {int(s["milestone"]): s for s in payloads[attack]}
        for column_index, milestone in enumerate(milestones):
            axis = axes[row_index][column_index]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_edgecolor("#999999")
                spine.set_linewidth(0.6)

            snapshot = by_milestone.get(milestone)
            if snapshot is None:
                # The run ended before this milestone: leave the cell empty
                # rather than repeating the last available image, which would
                # read as "the generator stopped changing".
                axis.text(
                    0.5, 0.5, "not\nreached",
                    ha="center", va="center", fontsize=7, color="#999999",
                    transform=axis.transAxes,
                )
                axis.set_facecolor("#f5f5f5")
            else:
                mosaic = _tile(np.asarray(snapshot["images"]), grid=grid)
                mosaic = np.clip(mosaic, 0.0, 1.0)
                if mosaic.shape[-1] == 1:
                    axis.imshow(mosaic[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
                else:
                    axis.imshow(mosaic)

            if row_index == 0:
                axis.set_title(
                    f"{_format_queries(milestone_query_labels[milestone])} queries",
                    fontsize=9,
                )
            if column_index == 0:
                axis.set_ylabel(
                    ROW_LABELS.get(attack, attack), fontsize=8, rotation=0,
                    ha="right", va="center", labelpad=42,
                )

    if title:
        figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97 if title else 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in _output_paths(output, formats):
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[figure] wrote {path}")
    plt.close(figure)
    print(f"[figure] {len(attacks)} attacks x {len(milestones)} milestones")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, action="append", required=True,
                        help="Directory of *.pt snapshots. Repeat to merge runs.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "runs" / "figures" / "query_evolution.png")
    parser.add_argument("--grid", type=int, default=3,
                        help="Mosaic side length; grid**2 images shown per cell.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title", default=None)
    parser.add_argument("--formats", default="png,pdf",
                        help="Comma-separated output formats (e.g. 'png,pdf').")
    parser.add_argument("--per-attack", action="store_true",
                        help="Also write one figure per attack alongside the grid.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    payloads: dict[str, list[dict[str, Any]]] = {}
    for directory in args.snapshot_dir:
        if not directory.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {directory}")
        for attack, snapshots in _load_snapshots(directory).items():
            if attack not in payloads or len(snapshots) > len(payloads[attack]):
                payloads[attack] = snapshots
    formats = [item.strip() for item in str(args.formats).split(",") if item.strip()]
    _render(payloads, output=args.output, formats=formats, grid=int(args.grid),
            dpi=int(args.dpi), title=args.title)
    if args.per_attack:
        for attack, snapshots in payloads.items():
            _render_single(
                snapshots,
                attack=attack,
                output=args.output.with_name(f"{args.output.stem}_{attack}"),
                formats=formats,
                grid=int(args.grid),
                dpi=int(args.dpi),
            )


if __name__ == "__main__":
    main()
