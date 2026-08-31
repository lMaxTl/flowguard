from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib.colors import ListedColormap
from plotly.colors import qualitative
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from defenses import datasets as legacy_datasets
from flowguard.experiments.spec import AttackKind, ExperimentSpec
from flowguard.orchestration.results import ExperimentResult
from flowguard.serving.model_loader import load_legacy_model
from flowguard.training.checkpoints import load_checkpoint

CLASS_PALETTE = (
    qualitative.Plotly
    + qualitative.Safe
    + qualitative.Vivid
    + qualitative.Bold
    + qualitative.Dark24
    + qualitative.Light24
)


@dataclass(slots=True)
class AttackVisualizationRun:
    spec: ExperimentSpec
    result: ExperimentResult
    target_model: torch.nn.Module
    substitute_model: torch.nn.Module
    query_records: list[dict[str, Any]]


def _coerce_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu()


def _flatten_tensor(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1)


def _predict_classes(
    model: torch.nn.Module,
    samples: Sequence[torch.Tensor],
    batch_size: int = 512,
) -> np.ndarray:
    if not samples:
        return np.array([], dtype=int)

    device = next(model.parameters()).device
    model.eval()
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            batch = torch.stack(
                [sample.detach().cpu().float() for sample in samples[start:end]],
                dim=0,
            ).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.array([], dtype=int)


def _pad_embedding(embedding: np.ndarray, dims: int) -> np.ndarray:
    if embedding.shape[1] >= dims:
        return embedding[:, :dims]
    padding = np.zeros((embedding.shape[0], dims - embedding.shape[1]), dtype=embedding.dtype)
    return np.hstack([embedding, padding])


def _fit_projection(
    tensors: Sequence[torch.Tensor],
    projection_method: str,
    projection_dims: int,
    seed: int,
) -> tuple[object | None, np.ndarray]:
    vectors = [_flatten_tensor(tensor) for tensor in tensors]
    if not vectors:
        return None, np.empty((0, projection_dims), dtype=np.float32)

    stacked_vectors = np.vstack(vectors).astype(np.float32, copy=False)
    sample_count, feature_count = stacked_vectors.shape
    normalized_method = projection_method.lower()
    if sample_count == 1:
        return None, np.zeros((1, projection_dims), dtype=np.float32)

    if normalized_method == "tsne" and sample_count >= 3:
        perplexity = max(1, min(30, sample_count - 1))
        reducer = TSNE(
            n_components=projection_dims,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        return None, reducer.fit_transform(stacked_vectors)

    if normalized_method == "umap" and sample_count >= 3:
        reducer = UMAP(
            n_components=projection_dims,
            n_neighbors=max(2, min(15, sample_count - 1)),
            random_state=seed,
        )
        embedding = reducer.fit_transform(stacked_vectors)
        return reducer, _pad_embedding(np.asarray(embedding, dtype=np.float32), projection_dims)

    if normalized_method not in {"pca", "tsne", "umap"}:
        raise ValueError(
            f"Unsupported projection_method '{projection_method}'. "
            "Expected one of: 'pca', 'tsne', 'umap'."
        )

    components = max(1, min(projection_dims, sample_count, feature_count))
    reducer = PCA(n_components=components, random_state=seed)
    embedding = reducer.fit_transform(stacked_vectors)
    return reducer, _pad_embedding(embedding, projection_dims)


def _fit_split_projection(
    background_tensors: Sequence[torch.Tensor],
    query_tensors: Sequence[torch.Tensor],
    projection_method: str,
    projection_dims: int,
    seed: int,
) -> tuple[object | None, np.ndarray, np.ndarray]:
    projector, embedding = _fit_projection(
        [*background_tensors, *query_tensors],
        projection_method=projection_method,
        projection_dims=projection_dims,
        seed=seed,
    )
    background_count = len(background_tensors)
    return projector, embedding[:background_count], embedding[background_count:]


def _predict_grid_classes(
    model: torch.nn.Module,
    flat_vectors: np.ndarray,
    input_shape: tuple[int, ...],
    batch_size: int = 512,
) -> np.ndarray:
    if len(flat_vectors) == 0:
        return np.array([], dtype=int)

    device = next(model.parameters()).device
    model.eval()
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(flat_vectors), batch_size):
            end = min(start + batch_size, len(flat_vectors))
            batch = torch.tensor(flat_vectors[start:end], dtype=torch.float32).reshape(-1, *input_shape).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.array([], dtype=int)


def _load_dataset(dataset_name: str, *, train: bool, download: bool):
    family = legacy_datasets.dataset_to_modelfamily[dataset_name]
    transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
    return legacy_datasets.__dict__[dataset_name](
        train=train,
        transform=transform,
        download=download,
    )


def _query_pool_size(spec: ExperimentSpec, result: ExperimentResult) -> int:
    metadata = result.attack_summary.metadata
    if "query_pool_size" in metadata:
        return int(metadata["query_pool_size"])
    if spec.attack.query_transfer_set_size is not None:
        return int(spec.attack.query_transfer_set_size)
    return int(spec.attack.query_budget)


def _prepare_background_samples(
    run: AttackVisualizationRun,
    max_background_samples: int | None,
    seed: int,
) -> tuple[list[torch.Tensor], np.ndarray, int]:
    dataset = _load_dataset(
        run.spec.dataset.name,
        train=True,
        download=run.spec.dataset.download,
    )
    total_available = len(dataset)
    selected_count = min(_query_pool_size(run.spec, run.result), total_available)
    if selected_count == 0:
        selected_count = total_available if max_background_samples is None else min(total_available, max_background_samples)
    selected_indices = np.arange(selected_count)

    if max_background_samples is not None and selected_count > max_background_samples:
        rng = np.random.default_rng(seed)
        selected_indices = np.sort(rng.choice(selected_indices, size=max_background_samples, replace=False))

    selected_samples = [dataset[int(index)] for index in selected_indices]
    background_tensors = [sample for sample, _ in selected_samples]
    background_labels = np.array([int(label) for _, label in selected_samples], dtype=int)
    fallback_class_count = int(background_labels.max()) + 1 if len(background_labels) else 1
    class_count = len(getattr(dataset, "classes", [])) or max(fallback_class_count, 1)
    return background_tensors, background_labels, class_count


def _plot_model_projection(
    axis: plt.Axes,
    title: str,
    model: torch.nn.Module,
    projector: object | None,
    background_embedding: np.ndarray,
    background_labels: np.ndarray,
    query_embedding: np.ndarray,
    class_count: int,
    input_shape: tuple[int, ...],
    projection_method: str,
    projection_dims: int,
    grid_resolution: int,
) -> Any:
    show_decision_regions = (
        projection_method == "pca"
        and projection_dims == 2
        and projector is not None
        and hasattr(projector, "inverse_transform")
        and len(background_embedding) > 0
    )

    point_cmap = plt.get_cmap("tab20", max(class_count, 1))
    region_cmap = ListedColormap(point_cmap(np.arange(max(class_count, 1))))

    if show_decision_regions:
        x_min, x_max = background_embedding[:, 0].min() - 0.8, background_embedding[:, 0].max() + 0.8
        y_min, y_max = background_embedding[:, 1].min() - 0.8, background_embedding[:, 1].max() + 0.8
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
        )
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        reconstructed = projector.inverse_transform(grid_points)
        grid_predictions = _predict_grid_classes(
            model=model,
            flat_vectors=reconstructed,
            input_shape=input_shape,
        ).reshape(grid_x.shape)
        axis.contourf(
            grid_x,
            grid_y,
            grid_predictions,
            levels=np.arange(class_count + 1) - 0.5,
            cmap=region_cmap,
            alpha=0.18,
            antialiased=True,
        )
    else:
        title = f"{title} (no decision regions for {projection_method}/{projection_dims}D)"

    scatter = axis.scatter(
        background_embedding[:, 0] if len(background_embedding) else [],
        background_embedding[:, 1] if len(background_embedding) else [],
        c=background_labels if len(background_labels) else None,
        cmap=point_cmap,
        s=22,
        alpha=0.72,
        marker="o",
        edgecolors="none",
        label="Selected train samples",
    )
    axis.scatter(
        query_embedding[:, 0] if len(query_embedding) else [],
        query_embedding[:, 1] if len(query_embedding) else [],
        c="#d62728",
        s=24,
        marker="x",
        linewidths=0.8,
        alpha=0.9,
        label="Attack queries",
    )
    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    return scatter


def _build_hover_row(
    record: dict[str, Any],
    target_prediction: int,
    substitute_prediction: int,
) -> list[int]:
    return [
        int(record["query_index"]),
        int(record.get("client_id", -1)),
        int(target_prediction),
        int(substitute_prediction),
    ]


def _stretched_original_point(
    original_point: np.ndarray,
    query_point: np.ndarray,
    min_visible_connector_distance: float,
) -> tuple[np.ndarray, bool]:
    delta = original_point - query_point
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return original_point, False
    if distance >= min_visible_connector_distance:
        return original_point, True
    direction = delta / distance
    return query_point + direction * min_visible_connector_distance, True


def _records_from_transferset(entries: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            query_tensor = _coerce_tensor(entry.get("query_x", entry.get("x")))
            original_tensor = _coerce_tensor(entry.get("original_x", query_tensor))
            label = entry.get("label")
            original_label = entry.get("original_label", label)
            query_index = int(entry.get("query_index", index))
            client_id = int(entry.get("client_id", -1))
        else:
            if not isinstance(entry, (tuple, list)) or len(entry) == 0:
                raise TypeError("Unsupported transfer-set sample format for visualization")
            query_tensor = _coerce_tensor(entry[0])
            original_tensor = query_tensor.clone()
            label = entry[1] if len(entry) > 1 else None
            original_label = label
            query_index = index
            client_id = -1

        records.append(
            {
                "query_x": query_tensor,
                "original_x": original_tensor,
                "label": _coerce_tensor(label) if label is not None else None,
                "original_label": _coerce_tensor(original_label) if original_label is not None else None,
                "query_index": query_index,
                "client_id": client_id,
            }
        )
    return records


def _load_query_records(result: ExperimentResult) -> list[dict[str, Any]]:
    attack_dir = Path(result.attack_summary.output_dir) if result.attack_summary.output_dir else None
    metadata = result.attack_summary.metadata

    candidate_paths: list[Path] = []
    for key in ("visualization_records_path", "transferset_path"):
        value = metadata.get(key)
        if value:
            candidate_paths.append(Path(value))
    if attack_dir is not None:
        candidate_paths.extend(
            [
                attack_dir / "visualization_records.pt",
                attack_dir / "transferset.pickle",
            ]
        )

    seen: set[Path] = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        payload = load_checkpoint(path)
        if isinstance(payload, Sequence):
            return _records_from_transferset(payload)

    raise FileNotFoundError("Could not locate visualization records for this experiment result")


def load_attack_visualization_run(
    spec: ExperimentSpec,
    result: ExperimentResult,
    *,
    target_model: torch.nn.Module | None = None,
    substitute_model: torch.nn.Module | None = None,
) -> AttackVisualizationRun:
    if target_model is None:
        target_checkpoint_dir = result.artifacts.get("target_model")
        if not target_checkpoint_dir:
            raise ValueError("ExperimentResult is missing the target_model artifact")
        target_model = load_legacy_model(target_checkpoint_dir, device=spec.target_model.device).model

    if substitute_model is None:
        substitute_checkpoint_dir = result.artifacts.get("substitute_model")
        if not substitute_checkpoint_dir:
            raise ValueError("ExperimentResult is missing the substitute_model artifact")
        substitute_model = load_legacy_model(
            substitute_checkpoint_dir,
            device=spec.substitute_model.device,
        ).model

    return AttackVisualizationRun(
        spec=spec,
        result=result,
        target_model=target_model,
        substitute_model=substitute_model,
        query_records=_load_query_records(result),
    )


def plot_attack_queries_projection(
    run: AttackVisualizationRun,
    *,
    projection_method: str = "pca",
    projection_dims: int = 2,
    max_background_samples: int | None = 1500,
    grid_resolution: int = 160,
    seed: int = 42,
) -> plt.Figure:
    if projection_dims != 2:
        raise ValueError("This visualization currently supports only 2D projections")

    background_tensors, background_labels, class_count = _prepare_background_samples(
        run=run,
        max_background_samples=max_background_samples,
        seed=seed,
    )
    query_tensors = [record["query_x"] for record in run.query_records]
    if not query_tensors:
        raise ValueError("No attack queries available for visualization")

    projector, background_embedding, query_embedding = _fit_split_projection(
        background_tensors=background_tensors,
        query_tensors=query_tensors,
        projection_method=projection_method,
        projection_dims=projection_dims,
        seed=seed,
    )

    input_shape = tuple(background_tensors[0].shape if background_tensors else query_tensors[0].shape)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    scatter = _plot_model_projection(
        axis=axes[0],
        title="Target decision space",
        model=run.target_model,
        projector=projector,
        background_embedding=background_embedding,
        background_labels=background_labels,
        query_embedding=query_embedding,
        class_count=class_count,
        input_shape=input_shape,
        projection_method=projection_method,
        projection_dims=projection_dims,
        grid_resolution=grid_resolution,
    )
    _plot_model_projection(
        axis=axes[1],
        title="Stolen substitute decision space",
        model=run.substitute_model,
        projector=projector,
        background_embedding=background_embedding,
        background_labels=background_labels,
        query_embedding=query_embedding,
        class_count=class_count,
        input_shape=input_shape,
        projection_method=projection_method,
        projection_dims=projection_dims,
        grid_resolution=grid_resolution,
    )

    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("Class label")
    figure.suptitle("Attack queries and selected training samples in projection space", fontsize=14)
    return figure


def build_attack_query_animation(
    run: AttackVisualizationRun,
    *,
    projection_method: str = "pca",
    projection_dims: int = 2,
    max_background_samples: int | None = 1200,
    show_only_synthetic_prada_queries: bool = True,
    min_visible_connector_distance: float = 0.6,
    max_frames_count: int | None = None,
    start_frame: int = 1,
    seed: int = 42,
) -> go.Figure:
    if projection_dims != 2:
        raise ValueError("The interactive animation currently supports only 2D projections")

    background_tensors, background_labels, _ = _prepare_background_samples(
        run=run,
        max_background_samples=max_background_samples,
        seed=seed,
    )
    records = list(run.query_records)
    if show_only_synthetic_prada_queries and run.spec.attack.kind == AttackKind.PRADA:
        synthetic_only = [
            record
            for record in records
            if torch.norm(record["query_x"] - record["original_x"]).item() > 1e-6
        ]
        if synthetic_only:
            records = synthetic_only

    if not records:
        raise ValueError("No attack queries available. Run the attack cell first.")
    if start_frame < 1:
        raise ValueError("start_frame must be >= 1")
    if start_frame > len(records):
        raise ValueError(f"start_frame ({start_frame}) exceeds available frames ({len(records)})")
    if max_frames_count is not None and max_frames_count < 1:
        raise ValueError("max_frames_count must be >= 1 when provided")

    start_index = start_frame - 1
    end_index = len(records) if max_frames_count is None else min(len(records), start_index + max_frames_count)
    records = records[start_index:end_index]

    original_tensors = [record["original_x"] for record in records]
    query_tensors = [record["query_x"] for record in records]
    _, embedding = _fit_projection(
        [*background_tensors, *original_tensors, *query_tensors],
        projection_method=projection_method,
        projection_dims=projection_dims,
        seed=seed,
    )
    background_count = len(background_tensors)
    query_count = len(records)
    background_embedding = embedding[:background_count]
    original_embedding = embedding[background_count : background_count + query_count]
    query_embedding = embedding[background_count + query_count :]

    target_original_predictions = _predict_classes(run.target_model, original_tensors)
    substitute_original_predictions = _predict_classes(run.substitute_model, original_tensors)
    target_query_predictions = _predict_classes(run.target_model, query_tensors)
    substitute_query_predictions = _predict_classes(run.substitute_model, query_tensors)

    point_colors = [CLASS_PALETTE[int(label) % len(CLASS_PALETTE)] for label in background_labels]
    title_suffix = (
        " (synthetic PRADA queries)"
        if show_only_synthetic_prada_queries and run.spec.attack.kind == AttackKind.PRADA
        else ""
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=background_embedding[:, 0] if len(background_embedding) else [],
            y=background_embedding[:, 1] if len(background_embedding) else [],
            mode="markers",
            name="Selected train samples",
            marker={"size": 7, "color": point_colors, "opacity": 0.72},
            customdata=np.column_stack([background_labels]) if len(background_labels) else None,
            hovertemplate="Train class %{customdata[0]}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="Past attack queries",
            marker={"size": 7, "color": "rgba(255,0,0,0.35)", "symbol": "x"},
            hovertemplate="Past query<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="Original sample",
            marker={
                "size": 12,
                "color": "rgba(230,70,70,0.98)",
                "line": {"color": "rgba(230,70,70,0.98)", "width": 2.8},
                "symbol": "circle-open",
            },
            customdata=[],
            hovertemplate=(
                "Original sample<br>"
                "Query %{customdata[0]}<br>"
                "Client %{customdata[1]}<br>"
                "Target pred %{customdata[2]}<br>"
                "Stolen pred %{customdata[3]}<br>"
                "x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Original to query",
            line={"color": "rgba(230,70,70,0.95)", "width": 3, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers+text",
            name="Current attack query",
            marker={
                "size": 12,
                "color": "red",
                "symbol": "x",
                "line": {"color": "darkred", "width": 1.2},
            },
            text=[],
            textposition="top center",
            customdata=[],
            hovertemplate=(
                "Attack query %{text}<br>"
                "Client %{customdata[1]}<br>"
                "Target pred %{customdata[2]}<br>"
                "Stolen pred %{customdata[3]}<br>"
                "x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"
            ),
        )
    )

    frames: list[go.Frame] = []
    for frame_index, record in enumerate(records, start=1):
        past_query_points = query_embedding[: frame_index - 1]
        current_original = original_embedding[frame_index - 1]
        current_query = query_embedding[frame_index - 1]
        display_original, show_connector = _stretched_original_point(
            original_point=current_original,
            query_point=current_query,
            min_visible_connector_distance=min_visible_connector_distance,
        )
        original_hover = [
            _build_hover_row(
                record,
                target_original_predictions[frame_index - 1],
                substitute_original_predictions[frame_index - 1],
            )
        ] if show_connector else []
        query_hover = [
            _build_hover_row(
                record,
                target_query_predictions[frame_index - 1],
                substitute_query_predictions[frame_index - 1],
            )
        ]

        frames.append(
            go.Frame(
                name=str(frame_index),
                data=[
                    go.Scatter(
                        x=past_query_points[:, 0] if len(past_query_points) else [],
                        y=past_query_points[:, 1] if len(past_query_points) else [],
                        mode="markers",
                        marker={"size": 7, "color": "rgba(255,0,0,0.35)", "symbol": "x"},
                    ),
                    go.Scatter(
                        x=[display_original[0]] if show_connector else [],
                        y=[display_original[1]] if show_connector else [],
                        mode="markers",
                        marker={
                            "size": 12,
                            "color": "rgba(230,70,70,0.98)",
                            "line": {"color": "rgba(230,70,70,0.98)", "width": 2.8},
                            "symbol": "circle-open",
                        },
                        customdata=original_hover,
                    ),
                    go.Scatter(
                        x=[display_original[0], current_query[0]] if show_connector else [],
                        y=[display_original[1], current_query[1]] if show_connector else [],
                        mode="lines",
                        line={"color": "rgba(230,70,70,0.95)", "width": 3, "dash": "dot"},
                    ),
                    go.Scatter(
                        x=[current_query[0]],
                        y=[current_query[1]],
                        mode="markers+text",
                        text=[str(record["query_index"])],
                        textposition="top center",
                        marker={
                            "size": 12,
                            "color": "red",
                            "symbol": "x",
                            "line": {"color": "darkred", "width": 1.2},
                        },
                        customdata=query_hover,
                    ),
                ],
                traces=[1, 2, 3, 4],
            )
        )

    figure.frames = frames
    for trace_index, replacement in zip((1, 2, 3, 4), frames[0].data, strict=True):
        figure.data[trace_index].update(replacement)

    figure.update_layout(
        title=f"Attack query order in {projection_method.upper()} space{title_suffix}",
        xaxis={"title": "Component 1"},
        yaxis={"title": "Component 2"},
        height=720,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": str(record["query_index"]),
                        "method": "animate",
                        "args": [[str(step_index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for step_index, record in enumerate(records, start=1)
                ],
                "currentvalue": {"prefix": "Query index: "},
            }
        ],
    )
    return figure


__all__ = [
    "AttackVisualizationRun",
    "build_attack_query_animation",
    "load_attack_visualization_run",
    "plot_attack_queries_projection",
]
