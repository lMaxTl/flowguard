"""Helpers for data-free bypass (D1/D2/D4) paper visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from torchvision import transforms as tv_transforms

from defenses import datasets as legacy_datasets

from flowguard.attacks.adaptive import high_frequency_energy, total_variation
from flowguard.attacks.base import AttackRunContext, AttackRunResult, AttackRunner
from flowguard.attacks.bypass import CemIterationSnapshot
from flowguard.attacks.disguide_bypass import DisguideTraceSnapshot
from flowguard.experiments.factory import build_experiment_spec
from flowguard.experiments.spec import AttackKind
from flowguard.defenses.query.flowpure import FlowPureQueryDefense
from flowguard.querying.engine import QueryEngine
from flowguard.serving.model_loader import load_legacy_model
from flowguard.serving.target_service import TargetService

from flowguard_notebook_utils import NATURE, apply_nature_style, to_cifar_image_array


@dataclass(frozen=True, slots=True)
class BypassNotebookConfig:
    """Shared paths and budgets for the bypass visualization notebook."""

    project_root: Path
    target_checkpoint_dir: Path
    flow_checkpoint: Path
    diffusion_model_id: str = "google/ddpm-cifar10-32"
    diffusion_scheduler: str = "ddim"
    diffusion_steps: int = 15
    device: str = "cuda"
    query_budget: int = 384
    population_size: int = 24
    elite_fraction: float = 0.20
    cem_latent_dim: int = 128
    cem_selector: str = "entropy"
    cem_lambda_detector: float = 1.0
    target_fpr: float = 0.05
    benign_calibration_queries: int = 128
    cem_trace_max_images: int = 8
    figure_dir: Path | None = None


def default_notebook_config(project_root: Path, *, device: str | None = None) -> BypassNotebookConfig:
    """Return default checkpoint paths used by other flow-matching notebooks."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return BypassNotebookConfig(
        project_root=project_root,
        target_checkpoint_dir=project_root
        / "runs"
        / "notebook"
        / "training-victim-cifar10-vgg16_bn-nodefense"
        / "target_model",
        flow_checkpoint=project_root
        / "runs"
        / "flow_matching"
        / "cifar10_flowpure_pgd"
        / "checkpoint_latest.pt",
        device=resolved_device,
        figure_dir=project_root / "runs" / "notebook" / "figures",
    )


def d1_disguide_extra(
    config: BypassNotebookConfig,
    *,
    diffusion_train_steps: int = 1,
    diffusion_steering_scale: float = 1.0,
) -> dict[str, Any]:
    """DisGUIDE knobs for D1: frozen DDPM + trainable latent steering."""
    extra = disguide_bypass_common_extra(config)
    extra.update(
        {
            "generator_hf_model_id": str(config.diffusion_model_id),
            "diffusion_scheduler": str(config.diffusion_scheduler),
            "diffusion_steps": int(config.diffusion_steps),
            "diffusion_use_safetensors": True,
            "diffusion_train_steps": int(diffusion_train_steps),
            "diffusion_steering_scale": float(diffusion_steering_scale),
            "g_iter": int(diffusion_train_steps),
        }
    )
    return extra


def d2_disguide_extra(config: BypassNotebookConfig) -> dict[str, Any]:
    """DisGUIDE knobs for D2: procedural generator (no pretrained DDPM)."""
    extra = disguide_bypass_common_extra(config)
    extra.update(
        {
            "procedural_grid_size": 4,
            "procedural_fourier_modes": 3,
            "procedural_blobs": 4,
        }
    )
    return extra


def disguide_bypass_common_extra(config: BypassNotebookConfig) -> dict[str, Any]:
    """DisGUIDE-style loop knobs for D1/D2 visualization runs."""
    return {
        "ensemble_size": 2,
        "latent_dim": int(config.cem_latent_dim),
        "g_iter": 1,
        "d_iter": 1,
        "rep_iter": 1,
        "replay_size": 10_000,
        "lambda_div": 0.2,
        "loss": "l1",
        "epoch_itrs": 4,
        "scheduler": "none",
        "lr_student": 0.03,
        "lr_generator": 1e-4,
        "velocity_regularizer_weight": float(config.cem_lambda_detector),
        "surrogate_checkpoint_path": str(config.flow_checkpoint),
        "velocity_regularizer_device": str(config.device),
        "disable_pbar": False,
    }


def cem_common_extra(config: BypassNotebookConfig) -> dict[str, Any]:
    """CEM knobs for D4 rejection-oracle visualization runs."""
    return {
        "engine": "cem",
        "population_size": int(config.population_size),
        "elite_fraction": float(config.elite_fraction),
        "latent_dim": int(config.cem_latent_dim),
        "selector": str(config.cem_selector),
        "lambda_detector": float(config.cem_lambda_detector),
        "artifact_sample_size": int(config.query_budget),
        "generator_hf_model_id": str(config.diffusion_model_id),
        "diffusion_scheduler": str(config.diffusion_scheduler),
        "diffusion_steps": int(config.diffusion_steps),
        "diffusion_use_safetensors": True,
    }


def build_bypass_spec(
    config: BypassNotebookConfig,
    *,
    name: str,
    attack_kind: AttackKind,
    attack_mode: str,
    attack_extra: dict[str, Any],
    projection_name: str = "none",
    projection_strength: float = 0.1,
    projection_steps: int = 10,
    attack_batch_size: int = 16,
) -> Any:
    """Build an experiment spec for a single bypass attack run."""
    extra = dict(attack_extra)
    if projection_name != "none":
        extra["projection"] = {
            "name": projection_name,
            "strength": float(projection_strength),
            "steps": int(projection_steps),
            "differentiable": False,
        }
    return build_experiment_spec(
        name=name,
        dataset="CIFAR10",
        target_architecture="vgg16_bn",
        substitute_architecture="vgg16_bn",
        attack_kind=attack_kind,
        attack_mode=attack_mode,
        query_dataset="CIFAR10",
        query_budget=int(config.query_budget),
        target_checkpoint_dir=str(config.target_checkpoint_dir),
        target_device=config.device,
        substitute_device=config.device,
        attack_batch_size=int(attack_batch_size),
        attack_extra=extra,
        training_batch_size=16,
        epochs=1,
        query_download=True,
        dataset_download=False,
        verbose=False,
    )


def build_flowpure_defense(
    config: BypassNotebookConfig,
    *,
    velocity_threshold: float,
    audit_only: bool,
) -> FlowPureQueryDefense:
    """Construct a FlowPure query defense for notebook experiments."""
    return FlowPureQueryDefense(
        fm_checkpoint_path=str(config.flow_checkpoint),
        dataset_name="CIFAR10",
        device=config.device,
        velocity_threshold=float(velocity_threshold),
        inputs_normalized=True,
        audit_only=bool(audit_only),
    )


def build_query_engine(
    config: BypassNotebookConfig,
    *,
    velocity_threshold: float,
    audit_only: bool,
) -> QueryEngine:
    """Victim API with an optional blocking FlowPure defense."""
    loaded = load_legacy_model(str(config.target_checkpoint_dir), device=config.device)
    defense = build_flowpure_defense(
        config,
        velocity_threshold=velocity_threshold,
        audit_only=audit_only,
    )
    return QueryEngine(
        TargetService(loaded),
        query_defenses=[defense],
    )


def _flowpure_score_from_engine(engine: QueryEngine, query: torch.Tensor) -> float:
    """Return the FlowPure score recorded on the last query batch."""
    batch = query.unsqueeze(0) if query.ndim == 3 else query
    engine.query_batch(batch.float())
    if not engine.history.records:
        return 0.0
    metadata = engine.history.records[-1].metadata
    scores = metadata.get("flowpure_scores")
    if isinstance(scores, list) and scores:
        return float(scores[0])
    return 0.0


def calibrate_flowpure_threshold(
    engine: QueryEngine,
    *,
    num_queries: int = 128,
    target_fpr: float = 0.05,
    download: bool = True,
) -> float:
    """Calibrate FlowPure threshold from benign CIFAR-10 train samples."""
    dataset = legacy_datasets.CIFAR10(
        train=True,
        transform=legacy_datasets.modelfamily_to_transforms["cifar"]["test"],
        download=download,
    )
    scores: list[float] = []
    for index in range(min(num_queries, len(dataset))):
        tensor, _ = dataset[index]
        scores.append(_flowpure_score_from_engine(engine, tensor))
    if not scores:
        raise RuntimeError("No benign scores collected for FlowPure calibration.")
    quantile = float(np.quantile(np.asarray(scores, dtype=np.float64), 1.0 - float(target_fpr)))
    return quantile


def load_benign_reference_images(
    *,
    count: int = 8,
    download: bool = True,
) -> list[np.ndarray]:
    """Load denormalized CIFAR-10 test images for comparison rows."""
    dataset = legacy_datasets.CIFAR10(
        train=False,
        transform=tv_transforms.ToTensor(),
        download=download,
    )
    images: list[np.ndarray] = []
    for index in range(min(count, len(dataset))):
        tensor, _ = dataset[index]
        images.append(to_cifar_image_array(tensor))
    return images


def run_traced_attack(
    runner_cls: Type[AttackRunner],
    *,
    spec: Any,
    engine: QueryEngine,
    output_dir: Path,
    trace_max_images: int = 8,
) -> tuple[AttackRunResult, list[CemIterationSnapshot]]:
    """Run a CEM bypass attack and return the iteration trace (D4)."""
    if str(spec.attack.extra.get("engine", "disguide")).lower() != "cem":
        raise ValueError(
            "run_traced_attack requires attack_extra['engine']='cem' "
            "(use run_traced_disguide_attack for D1/D2)."
        )
    cem_trace: list[CemIterationSnapshot] = []
    runner = runner_cls(query_engine=engine)
    result = runner.run(
        AttackRunContext(
            experiment=spec,
            output_dir=str(output_dir),
            metadata={
                "cem_trace": cem_trace,
                "cem_trace_max_images": int(trace_max_images),
            },
        )
    )
    return result, cem_trace


def run_traced_disguide_attack(
    runner_cls: Type[AttackRunner],
    *,
    spec: Any,
    engine: QueryEngine,
    output_dir: Path,
    trace_max_images: int = 8,
) -> tuple[AttackRunResult, list[DisguideTraceSnapshot]]:
    """Run a DisGUIDE-style D1/D2 attack and return the step trace."""
    disguide_trace: list[DisguideTraceSnapshot] = []
    disguide_metrics: list[dict[str, float]] = []
    runner = runner_cls(query_engine=engine)
    result = runner.run(
        AttackRunContext(
            experiment=spec,
            output_dir=str(output_dir),
            metadata={
                "disguide_trace": disguide_trace,
                "disguide_trace_max_images": int(trace_max_images),
                "disguide_metrics": disguide_metrics,
            },
        )
    )
    return result, disguide_trace


def load_cem_trace(path: Path) -> list[CemIterationSnapshot]:
    """Load a CEM iteration trace written by a bypass attack run."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_disguide_trace(path: Path) -> list[DisguideTraceSnapshot]:
    """Load a DisGUIDE step trace written by a bypass attack run."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _disguide_snapshot_at(
    trace: Sequence[DisguideTraceSnapshot],
    step: int,
) -> DisguideTraceSnapshot | None:
    for snapshot in trace:
        if snapshot.step == step:
            return snapshot
    if not trace:
        return None
    if step <= 0:
        return trace[0]
    if step >= len(trace):
        return trace[-1]
    return trace[step - 1]


def _snapshot_at(trace: Sequence[CemIterationSnapshot], iteration: int) -> CemIterationSnapshot | None:
    for snapshot in trace:
        if snapshot.iteration == iteration:
            return snapshot
    return trace[iteration - 1] if 0 < iteration <= len(trace) else None


def plot_image_row(
    images: Sequence[np.ndarray],
    *,
    titles: Sequence[str] | None = None,
    suptitle: str | None = None,
    border_colors: Sequence[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a single row of RGB images."""
    apply_nature_style()
    count = len(images)
    if count == 0:
        raise ValueError("images must not be empty.")
    fig, axes = plt.subplots(1, count, figsize=(1.9 * count, 2.2))
    axes_list = np.atleast_1d(axes).reshape(-1)
    for axis, image, index in zip(axes_list, images, range(len(images)), strict=True):
        axis.imshow(np.clip(to_cifar_image_array(image), 0.0, 1.0))
        axis.set_xticks([])
        axis.set_yticks([])
        if titles is not None and index < len(titles):
            axis.set_title(titles[index], fontsize=9)
        if border_colors is not None and index < len(border_colors):
            for spine in axis.spines.values():
                spine.set_edgecolor(border_colors[index])
                spine.set_linewidth(2.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_disguide_evolution_grid(
    trace: Sequence[DisguideTraceSnapshot],
    benign_images: Sequence[np.ndarray],
    *,
    step_labels: Sequence[int] | None = None,
    images_per_row: int = 6,
    save_path: Path | None = None,
) -> plt.Figure:
    """Three-row figure: early queries, late queries, benign reference."""
    apply_nature_style()
    if not trace:
        raise ValueError("trace must not be empty.")
    if step_labels is None:
        step_labels = [trace[0].step, trace[len(trace) // 2].step, trace[-1].step]

    rows: list[tuple[str, list[np.ndarray]]] = []
    first = _disguide_snapshot_at(trace, step_labels[0])
    mid = _disguide_snapshot_at(trace, step_labels[1])
    last = _disguide_snapshot_at(trace, step_labels[2])
    if first is not None:
        rows.append((f"Step {first.step} queries", first.query_images[:images_per_row]))
    if mid is not None:
        rows.append((f"Step {mid.step} queries", mid.query_images[:images_per_row]))
    elif last is not None:
        rows.append((f"Step {last.step} queries", last.query_images[:images_per_row]))
    rows.append(("Benign CIFAR-10", list(benign_images[:images_per_row])))

    fig, axes = plt.subplots(len(rows), 1, figsize=(1.8 * images_per_row, 2.3 * len(rows)))
    axes_rows = np.atleast_1d(axes).reshape(-1)
    for axis, (label, images) in zip(axes_rows, rows, strict=True):
        axis.set_title(label, loc="left", fontsize=11)
        axis.axis("off")
        if not images:
            axis.text(0.5, 0.5, "no samples", ha="center", va="center")
            continue
        mosaic = np.concatenate(
            [to_cifar_image_array(image) for image in images],
            axis=1,
        )
        axis.imshow(np.clip(mosaic, 0.0, 1.0))
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_disguide_training_curve(
    trace: Sequence[DisguideTraceSnapshot],
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot surrogate accuracy and query count over DisGUIDE steps."""
    apply_nature_style()
    if not trace:
        raise ValueError("trace must not be empty.")
    steps = [snapshot.step for snapshot in trace]
    queries = [snapshot.queries for snapshot in trace]
    accuracies = [
        snapshot.surrogate_accuracy
        for snapshot in trace
        if snapshot.surrogate_accuracy is not None
    ]
    accuracy_steps = [
        snapshot.step
        for snapshot in trace
        if snapshot.surrogate_accuracy is not None
    ]
    fig, axis_left = plt.subplots(figsize=(7.0, 3.6))
    axis_right = axis_left.twinx()
    axis_left.plot(steps, queries, color=NATURE["blue"], marker="o", label="Queries")
    if accuracies:
        axis_right.plot(
            accuracy_steps,
            accuracies,
            color=NATURE["green"],
            marker="s",
            label="Surrogate accuracy (%)",
        )
    axis_left.set_xlabel("DisGUIDE step")
    axis_left.set_ylabel("Total queries")
    axis_right.set_ylabel("Surrogate accuracy (%)")
    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        frameon=False,
        loc="center right",
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_disguide_detector_entropy_scatter(
    trace: Sequence[DisguideTraceSnapshot],
    *,
    steps: Sequence[int] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter detector score vs label entropy for selected DisGUIDE steps."""
    apply_nature_style()
    if steps is None:
        step = max(1, len(trace) // 4)
        steps = [snapshot.step for snapshot in trace[::step]][:4]
    snapshots = [panel for panel in (_disguide_snapshot_at(trace, step) for step in steps) if panel is not None]
    columns = len(snapshots)
    fig, axes = plt.subplots(1, columns, figsize=(3.4 * columns, 3.2), sharey=True)
    axes_list = np.atleast_1d(axes).reshape(-1)
    for axis, snapshot in zip(axes_list, snapshots, strict=True):
        scores = np.asarray(snapshot.detector_scores, dtype=np.float64)
        entropy = np.asarray(snapshot.query_entropy, dtype=np.float64)
        blocked = np.asarray(snapshot.query_blocked, dtype=bool)
        axis.scatter(scores[blocked], entropy[blocked], c=NATURE["red"], s=36, alpha=0.85)
        axis.scatter(scores[~blocked], entropy[~blocked], c=NATURE["green"], s=42, alpha=0.9)
        axis.set_title(f"Step {snapshot.step}")
        axis.set_xlabel("Detector score")
    axes_list[0].set_ylabel("Label entropy")
    fig.legend(
        handles=[
            Line2D([0], [0], color=NATURE["green"], marker="o", linestyle="", label="Accepted"),
            Line2D([0], [0], color=NATURE["red"], marker="o", linestyle="", label="Blocked"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_d1_d2_disguide_query_grid(
    d1_trace: Sequence[DisguideTraceSnapshot],
    d2_trace: Sequence[DisguideTraceSnapshot],
    *,
    iterations: Sequence[int] = (1, 16),
    save_path: Path | None = None,
) -> plt.Figure:
    """2x2 panel: one query each for D1/D2 at early and late DisGUIDE steps."""
    apply_nature_style()
    panels: list[tuple[str, Sequence[DisguideTraceSnapshot]]] = [
        ("D1", d1_trace),
        ("D2", d2_trace),
    ]
    fig, axes = plt.subplots(2, len(iterations), figsize=(2.4 * len(iterations), 2.5 * 2))
    axes_grid = np.atleast_2d(axes)
    for row_index, (attack_label, trace) in enumerate(panels):
        for column_index, step in enumerate(iterations):
            snapshot = _disguide_snapshot_at(trace, step)
            if snapshot is None:
                raise ValueError(f"No DisGUIDE snapshot for {attack_label} step {step}.")
            axis = axes_grid[row_index, column_index]
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"step {step}", fontsize=9)
            if column_index == 0:
                axis.set_ylabel(attack_label, fontsize=9, rotation=0, labelpad=24, va="center")
            if snapshot.query_images:
                image = snapshot.query_images[0]
                axis.imshow(np.clip(to_cifar_image_array(image), 0.0, 1.0))
            else:
                axis.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
                axis.set_axis_off()
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_attack_evolution_grid(
    trace: Sequence[CemIterationSnapshot],
    benign_images: Sequence[np.ndarray],
    *,
    iteration_labels: Sequence[int] | None = None,
    images_per_row: int = 6,
    save_path: Path | None = None,
) -> plt.Figure:
    """Three-row figure: early population, late elite, benign reference (CEM)."""
    apply_nature_style()
    if not trace:
        raise ValueError("trace must not be empty.")
    if iteration_labels is None:
        iteration_labels = [trace[0].iteration, trace[len(trace) // 2].iteration, trace[-1].iteration]

    rows: list[tuple[str, list[np.ndarray]]] = []
    first = _snapshot_at(trace, iteration_labels[0])
    mid = _snapshot_at(trace, iteration_labels[1])
    last = _snapshot_at(trace, iteration_labels[2])
    if first is not None:
        rows.append((f"Iter {first.iteration} population", first.population_queries[:images_per_row]))
    if mid is not None and mid.elite_queries:
        rows.append((f"Iter {mid.iteration} elite", mid.elite_queries[:images_per_row]))
    elif last is not None and last.elite_queries:
        rows.append((f"Iter {last.iteration} elite", last.elite_queries[:images_per_row]))
    if last is not None and last.population_queries:
        rows.append((f"Iter {last.iteration} population", last.population_queries[:images_per_row]))
    rows.append(("Benign CIFAR-10", list(benign_images[:images_per_row])))

    fig, axes = plt.subplots(len(rows), 1, figsize=(1.8 * images_per_row, 2.3 * len(rows)))
    axes_rows = np.atleast_1d(axes).reshape(-1)
    for axis, (label, images) in zip(axes_rows, rows, strict=True):
        axis.set_title(label, loc="left", fontsize=11)
        axis.axis("off")
        if not images:
            axis.text(0.5, 0.5, "no samples", ha="center", va="center")
            continue
        mosaic = np.concatenate(
            [to_cifar_image_array(image) for image in images],
            axis=1,
        )
        axis.imshow(np.clip(mosaic, 0.0, 1.0))
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_d1_d2_attack_query_grid(
    d1_trace: Sequence[CemIterationSnapshot],
    d2_trace: Sequence[CemIterationSnapshot],
    *,
    iterations: Sequence[int] = (1, 16),
    save_path: Path | None = None,
) -> plt.Figure:
    """2x2 panel: one attack query each for D1/D2 at early and late CEM iterations."""
    apply_nature_style()
    panels: list[tuple[str, Sequence[CemIterationSnapshot]]] = [
        ("D1", d1_trace),
        ("D2", d2_trace),
    ]

    fig, axes = plt.subplots(2, len(iterations), figsize=(2.4 * len(iterations), 2.5 * 2))
    axes_grid = np.atleast_2d(axes)
    for row_index, (attack_label, trace) in enumerate(panels):
        for column_index, iteration in enumerate(iterations):
            snapshot = _snapshot_at(trace, iteration)
            if snapshot is None:
                raise ValueError(f"No CEM snapshot for {attack_label} iteration {iteration}.")
            axis = axes_grid[row_index, column_index]
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"iter {iteration}", fontsize=9)
            if column_index == 0:
                axis.set_ylabel(attack_label, fontsize=9, rotation=0, labelpad=24, va="center")
            if snapshot.population_queries:
                image = snapshot.population_queries[0]
                axis.imshow(np.clip(to_cifar_image_array(image), 0.0, 1.0))
            else:
                axis.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
                axis.set_axis_off()

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cem_iteration_strip(
    trace: Sequence[CemIterationSnapshot],
    *,
    iterations: Sequence[int],
    images_per_panel: int = 8,
    save_path: Path | None = None,
) -> plt.Figure:
    """Show accepted (green) vs rejected (red) queries across selected iterations."""
    apply_nature_style()
    panels = [_snapshot_at(trace, iteration) for iteration in iterations]
    panels = [panel for panel in panels if panel is not None]
    if not panels:
        raise ValueError("No trace snapshots matched the requested iterations.")

    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * images_per_panel, 2.6))
    axes_list = np.atleast_1d(axes).reshape(-1)
    for axis, snapshot in zip(axes_list, panels, strict=True):
        axis.set_title(f"Iter {snapshot.iteration}", fontsize=11)
        axis.axis("off")
        for index, image in enumerate(snapshot.population_queries[:images_per_panel]):
            sub = axis.inset_axes(
                [
                    (index % 4) * 0.24,
                    0.52 - (index // 4) * 0.48,
                    0.22,
                    0.42,
                ]
            )
            sub.imshow(np.clip(to_cifar_image_array(image), 0.0, 1.0))
            sub.set_xticks([])
            sub.set_yticks([])
            accepted = (
                snapshot.population_accepted[index]
                if index < len(snapshot.population_accepted)
                else False
            )
            color = NATURE["green"] if accepted else NATURE["red"]
            for spine in sub.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)
    legend_handles = [
        Line2D([0], [0], color=NATURE["green"], lw=3, label="Accepted"),
        Line2D([0], [0], color=NATURE["red"], lw=3, label="Rejected"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_detector_entropy_scatter(
    trace: Sequence[CemIterationSnapshot],
    *,
    iterations: Sequence[int] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Small-multiple scatter: detector score vs label entropy."""
    apply_nature_style()
    if iterations is None:
        step = max(1, len(trace) // 4)
        iterations = [snapshot.iteration for snapshot in trace[::step]][:4]
    snapshots = [panel for panel in (_snapshot_at(trace, it) for it in iterations) if panel is not None]
    columns = len(snapshots)
    fig, axes = plt.subplots(1, columns, figsize=(3.4 * columns, 3.2), sharey=True)
    axes_list = np.atleast_1d(axes).reshape(-1)
    for axis, snapshot in zip(axes_list, snapshots, strict=True):
        scores = np.asarray(snapshot.population_detector_score, dtype=np.float64)
        entropy = np.asarray(snapshot.population_entropy, dtype=np.float64)
        accepted = np.asarray(snapshot.population_accepted, dtype=bool)
        axis.scatter(
            scores[~accepted],
            entropy[~accepted],
            c=NATURE["red"],
            s=36,
            alpha=0.85,
            label="Rejected",
        )
        axis.scatter(
            scores[accepted],
            entropy[accepted],
            c=NATURE["green"],
            s=42,
            alpha=0.9,
            label="Accepted",
        )
        axis.set_title(f"Iter {snapshot.iteration}")
        axis.set_xlabel("Detector score")
    axes_list[0].set_ylabel("Label entropy")
    fig.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_acceptance_timeseries(
    trace: Sequence[CemIterationSnapshot],
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot acceptance rate and mean detector score over CEM iterations."""
    apply_nature_style()
    iterations = [snapshot.iteration for snapshot in trace]
    acceptance_rates = [
        float(snapshot.accepted) / float(max(1, snapshot.submitted)) for snapshot in trace
    ]
    mean_scores = [
        float(np.mean(snapshot.population_detector_score))
        if snapshot.population_detector_score
        else 0.0
        for snapshot in trace
    ]
    fig, axis_left = plt.subplots(figsize=(7.0, 3.6))
    axis_right = axis_left.twinx()
    axis_left.plot(iterations, acceptance_rates, color=NATURE["green"], marker="o", label="Accept rate")
    axis_right.plot(iterations, mean_scores, color=NATURE["blue"], marker="s", label="Mean detector score")
    axis_left.set_xlabel("CEM iteration")
    axis_left.set_ylabel("Accepted / submitted")
    axis_right.set_ylabel("Mean FlowPure score")
    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        frameon=False,
        loc="center right",
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def _feature_vector(image_chw: np.ndarray, detector_score: float, entropy: float) -> np.ndarray:
    tensor = torch.as_tensor(image_chw, dtype=torch.float32).unsqueeze(0)
    return np.asarray(
        [
            float(detector_score),
            float(entropy),
            float(total_variation(tensor).item()),
            float(high_frequency_energy(tensor).item()),
        ],
        dtype=np.float64,
    )


def collect_trace_feature_points(
    trace: Sequence[CemIterationSnapshot],
    *,
    early_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrices for early vs late CEM queries."""
    early_features: list[np.ndarray] = []
    late_features: list[np.ndarray] = []

    for snapshot in trace:
        is_early = snapshot.iteration <= early_iterations
        for index, image in enumerate(snapshot.population_queries):
            entropy = (
                snapshot.population_entropy[index]
                if index < len(snapshot.population_entropy)
                else 0.0
            )
            score = (
                snapshot.population_detector_score[index]
                if index < len(snapshot.population_detector_score)
                else 0.0
            )
            accepted = (
                snapshot.population_accepted[index]
                if index < len(snapshot.population_accepted)
                else False
            )
            feature = _feature_vector(image, score, entropy)
            if is_early:
                early_features.append(feature)
            elif accepted:
                late_features.append(feature)

    if not late_features and trace:
        for snapshot in trace[-3:]:
            for image in snapshot.elite_queries:
                late_features.append(_feature_vector(image, score=0.0, entropy=0.0))

    if not early_features or not late_features:
        raise ValueError("Need query features in both early and late CEM iterations for PCA panels.")
    return np.stack(early_features, axis=0), np.stack(late_features, axis=0)


def _schematic_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ridge_center = 0.70 * np.sin(0.85 * x) - 0.12 * x
    ridge = np.exp(-0.5 * ((y - ridge_center) / 0.33) ** 2) * np.exp(-0.5 * (x / 2.3) ** 2)
    lobe_left = 0.55 * np.exp(-((x + 1.85) ** 2 / (2 * 0.78**2) + (y + 0.45) ** 2 / (2 * 0.95**2)))
    lobe_right = 0.45 * np.exp(-((x - 1.45) ** 2 / (2 * 0.95**2) + (y - 0.30) ** 2 / (2 * 0.85**2)))
    return ridge + lobe_left + lobe_right


def _subsample_rows(features: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if features.shape[0] <= count:
        return features
    indices = rng.choice(features.shape[0], size=count, replace=False)
    return features[indices]


def plot_manifold_teaser(
    trace: Sequence[CemIterationSnapshot],
    benign_features: np.ndarray,
    *,
    early_iterations: int = 2,
    samples_per_group: int | None = 32,
    save_path: Path | None = None,
    seed: int = 7,
) -> plt.Figure:
    """Schematic background + measured PCA with shared axes and balanced sampling.

    Each subplot uses identical axis limits so the decorative manifold background
    does not appear to shrink or shift. Groups are subsampled to equal counts when
    ``samples_per_group`` is set.
    """
    apply_nature_style()
    early, late = collect_trace_feature_points(trace, early_iterations=early_iterations)
    rng = np.random.default_rng(seed)

    group_features: dict[str, np.ndarray] = {
        "benign": benign_features,
        "early_cem": early,
        "late_accepted": late,
    }
    if samples_per_group is not None:
        target = int(samples_per_group)
        group_features = {
            key: _subsample_rows(values, target, rng)
            for key, values in group_features.items()
        }

    features = np.concatenate(
        [group_features["benign"], group_features["early_cem"], group_features["late_accepted"]],
        axis=0,
    )
    projection = PCA(n_components=2, random_state=seed).fit_transform(features)

    benign_n = group_features["benign"].shape[0]
    early_n = group_features["early_cem"].shape[0]
    late_n = group_features["late_accepted"].shape[0]
    slices = {
        "benign": projection[:benign_n],
        "early_cem": projection[benign_n : benign_n + early_n],
        "late_accepted": projection[benign_n + early_n :],
    }

    padding = 0.35
    x_min = float(projection[:, 0].min() - padding)
    x_max = float(projection[:, 0].max() + padding)
    y_min = float(projection[:, 1].min() - padding)
    y_max = float(projection[:, 1].max() + padding)

    x = np.linspace(x_min, x_max, 320)
    y = np.linspace(y_min, y_max, 260)
    xx, yy = np.meshgrid(x, y)
    density = _schematic_density(xx, yy)
    cmap = LinearSegmentedColormap.from_list(
        "manifold_blue_teal",
        ["#f8fdff", "#d8eff5", "#abd8e5", "#7db5cf", "#4d8bb5", "#2b6f99"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharex=True, sharey=True)
    panel_specs = [
        ("benign", f"Benign reference (n={benign_n})"),
        ("early_cem", f"Early CEM (n={early_n})"),
        ("late_accepted", f"Late accepted (n={late_n})"),
    ]
    for axis, (group_key, title) in zip(axes, panel_specs, strict=True):
        points = slices[group_key]
        axis.contourf(xx, yy, density, levels=12, cmap=cmap, alpha=0.95)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=36,
            c=NATURE["green"],
            edgecolors="white",
            linewidths=0.4,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks([])
        axis.set_yticks([])

    axes[0].set_ylabel("PCA-2")
    for axis in axes:
        axis.set_xlabel("PCA-1")

    fig.suptitle(
        "Measured query features in a shared PCA space (schematic background only)",
        fontsize=12,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Equal subsampling per group when possible. Higher spread ≠ off-manifold; "
        "use detector-score time series for defense drift. Background is not data-driven.",
        ha="center",
        fontsize=9,
        color=NATURE["gray"],
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_manifold_pca_combined(
    trace: Sequence[CemIterationSnapshot],
    benign_features: np.ndarray,
    *,
    early_iterations: int = 2,
    samples_per_group: int | None = 32,
    save_path: Path | None = None,
    seed: int = 7,
) -> plt.Figure:
    """Single-panel PCA: all groups overlaid (clearer than three separate autoscale panels)."""
    apply_nature_style()
    early, late = collect_trace_feature_points(trace, early_iterations=early_iterations)
    rng = np.random.default_rng(seed)
    groups = {
        "Benign": benign_features,
        "Early CEM": early,
        "Late accepted": late,
    }
    if samples_per_group is not None:
        target = int(samples_per_group)
        groups = {name: _subsample_rows(values, target, rng) for name, values in groups.items()}

    names: list[str] = []
    stacked: list[np.ndarray] = []
    for name, values in groups.items():
        names.extend([name] * values.shape[0])
        stacked.append(values)
    features = np.concatenate(stacked, axis=0)
    projection = PCA(n_components=2, random_state=seed).fit_transform(features)

    color_map = {
        "Benign": NATURE["blue"],
        "Early CEM": NATURE["orange"],
        "Late accepted": NATURE["green"],
    }
    fig, axis = plt.subplots(figsize=(6.5, 5.0))
    for name, color in color_map.items():
        mask = np.asarray(names) == name
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=42,
            alpha=0.85,
            c=color,
            edgecolors="white",
            linewidths=0.4,
            label=f"{name} (n={int(mask.sum())})",
        )
    axis.set_xlabel("PCA-1")
    axis.set_ylabel("PCA-2")
    axis.set_title("Shared PCA of measured query features", fontsize=12)
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def collect_manifold_benign_features(
    config: BypassNotebookConfig,
    *,
    count: int = 64,
    download: bool = True,
    velocity_threshold: float | None = None,
    use_train_split: bool = False,
) -> np.ndarray:
    """Feature matrix for benign CIFAR queries with FlowPure scores (audit-only).

    Scoring uses ``audit_only=True`` so benign samples are never blocked while
    collecting detector features for manifold plots. Do not pass the D4 blocking
    ``QueryEngine`` here.
    """
    score_engine = build_query_engine(
        config,
        velocity_threshold=float(velocity_threshold if velocity_threshold is not None else 1e9),
        audit_only=True,
    )
    dataset = legacy_datasets.CIFAR10(
        train=use_train_split,
        transform=legacy_datasets.modelfamily_to_transforms["cifar"]["test"],
        download=download,
    )
    rows: list[np.ndarray] = []
    for index in range(min(count, len(dataset))):
        tensor, _ = dataset[index]
        score = _flowpure_score_from_engine(score_engine, tensor)
        rows.append(_feature_vector(tensor.numpy(), score, entropy=0.0))
    return np.stack(rows, axis=0)


# Backward-compatible alias (prefer collect_manifold_benign_features in notebooks).
benign_feature_matrix = collect_manifold_benign_features
