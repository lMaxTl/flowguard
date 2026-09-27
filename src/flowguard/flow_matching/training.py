from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from torchvision.utils import save_image

from flowguard.flow_matching.config import (
    FlowMatchingDatasetConfig,
    FlowMatchingTrainingConfig,
    get_model_config,
    resolve_dataset_config,
)
from flowguard.flow_matching.data import (
    build_flow_matching_dataset,
    build_training_dataloader,
)
from flowguard.flow_matching.ema import EMA
from flowguard.flow_matching.tabular_model import TabularFlowModel
from flowguard.flow_matching.unet import UNetModel
from flowguard.training.checkpoints import load_checkpoint, save_metadata


@dataclass(slots=True)
class FlowMatchingCheckpoint:
    """Structured view of a saved flow matching checkpoint."""

    model: torch.nn.Module
    training_config: FlowMatchingTrainingConfig
    dataset_config: FlowMatchingDatasetConfig
    checkpoint_path: Path
    epoch: int


class VelocityFieldWrapper(ModelWrapper):
    """Adapter that exposes the local UNet with the `flow_matching` solver API."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        class_conditioned: bool,
    ) -> None:
        super().__init__(model)
        self.class_conditioned = class_conditioned

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = torch.full((x.shape[0],), float(t.item()), device=x.device, dtype=x.dtype)
        extras: dict[str, torch.Tensor] = {}
        if self.class_conditioned and label is not None:
            extras["label"] = label
        return self.model(x, t, extras)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    dataset_config: FlowMatchingDatasetConfig,
    *,
    use_ema: bool,
    ema_decay: float,
    feature_dim: int | None = None,
) -> torch.nn.Module:
    """Build the FM model for the requested dataset preset."""
    model_config = get_model_config(dataset_config, feature_dim=feature_dim)
    if dataset_config.dataset_type == "tabular":
        model = TabularFlowModel(**model_config)
    else:
        model = UNetModel(**model_config)
    if use_ema:
        return EMA(model=model, decay=ema_decay)
    return model


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if isinstance(model, EMA) else model


def _extract_ema_wrapper(model: torch.nn.Module) -> EMA | None:
    """Return the underlying EMA wrapper for eager or compiled models."""
    if isinstance(model, EMA):
        return model
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, EMA):
        return original
    return None


def _strip_key_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _load_model_state_dict_compat(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    """Load checkpoint weights while handling compiled-model key prefixes."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        # torch.compile can persist keys prefixed by "_orig_mod."; strip once and retry.
        prefixed = any(key.startswith("_orig_mod.") for key in state_dict)
        if not prefixed:
            raise

        stripped_state_dict = _strip_key_prefix(state_dict, "_orig_mod.")
        try:
            model.load_state_dict(stripped_state_dict)
            return
        except RuntimeError:
            raise exc


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    average_loss: float,
    training_config: FlowMatchingTrainingConfig,
    dataset_config: FlowMatchingDatasetConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "average_loss": average_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_config": training_config.to_dict(),
        "dataset_config": dataset_config.to_dict(),
        **(extra or {}),
    }


def save_flow_matching_checkpoint(
    output_dir: str | Path,
    filename: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    average_loss: float,
    training_config: FlowMatchingTrainingConfig,
    dataset_config: FlowMatchingDatasetConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a training checkpoint and return its path.

    Written to a temporary file and renamed, so a job killed mid-write leaves
    the previous checkpoint intact instead of a truncated one.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / filename
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            average_loss=average_loss,
            training_config=training_config,
            dataset_config=dataset_config,
            extra=extra,
        ),
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def write_done_marker(output_dir: str | Path, **fields: Any) -> Path:
    """Mark a training run as finished (read by scripts/schedule_experiments.py)."""
    import json

    marker = Path(output_dir) / "DONE.json"
    marker.write_text(json.dumps(fields, indent=2, default=str), encoding="utf-8")
    return marker


def load_flow_matching_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> FlowMatchingCheckpoint:
    """Load a previously trained flow matching model checkpoint."""
    payload = load_checkpoint(checkpoint_path, map_location=device)
    training_config = FlowMatchingTrainingConfig(**payload["training_config"])
    dataset_config = FlowMatchingDatasetConfig(**payload["dataset_config"])
    model = build_model(
        dataset_config,
        use_ema=training_config.use_ema,
        ema_decay=training_config.ema_decay,
    )
    _load_model_state_dict_compat(model, payload["model_state_dict"])
    model = model.to(torch.device(device))
    model.eval()
    return FlowMatchingCheckpoint(
        model=model,
        training_config=training_config,
        dataset_config=dataset_config,
        checkpoint_path=Path(checkpoint_path),
        epoch=int(payload["epoch"]),
    )


def estimate_log_likelihood(
    checkpoint: FlowMatchingCheckpoint,
    x_1: torch.Tensor,
    *,
    device: str | torch.device,
    step_size: float | None = 0.05,
    method: str = "midpoint",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    exact_divergence: bool = False,
) -> torch.Tensor:
    """Estimate log-likelihood of FM-space inputs using reverse-time ODE integration."""
    torch_device = torch.device(device)
    checkpoint.model = checkpoint.model.to(torch_device)
    checkpoint.model.eval()
    velocity_model = VelocityFieldWrapper(
        model=checkpoint.model,
        class_conditioned=checkpoint.dataset_config.class_conditioned,
    )
    solver = ODESolver(velocity_model=velocity_model)

    def _standard_gaussian_log_prob(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(start_dim=1)
        dimension = flat.shape[1]
        return -0.5 * (
            dimension * math.log(2.0 * math.pi) + flat.pow(2).sum(dim=1)
        )

    _, log_likelihood = solver.compute_likelihood(
        x_1=x_1.to(torch_device),
        log_p0=_standard_gaussian_log_prob,
        step_size=step_size,
        method=method,
        atol=atol,
        rtol=rtol,
        exact_divergence=exact_divergence,
        enable_grad=False,
    )
    return log_likelihood


@torch.no_grad()
def sample_images(
    model: torch.nn.Module,
    dataset_config: FlowMatchingDatasetConfig,
    *,
    output_path: str | Path,
    device: str | torch.device,
    sample_count: int,
    ode_method: str,
    step_size: float,
    seed: int,
) -> Path:
    """Generate a snapshot artifact from a trained FM model."""
    set_random_seed(seed)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    labels: torch.Tensor | None = None
    if dataset_config.class_conditioned:
        if dataset_config.num_classes is None:
            raise ValueError("Conditional sampling requires a finite number of classes.")
        labels = torch.randint(
            low=0,
            high=dataset_config.num_classes,
            size=(sample_count,),
            device=torch_device,
        )

    if dataset_config.dataset_type == "tabular":
        feature_dim = dataset_config.feature_dim
        if feature_dim is None:
            raise ValueError("Tabular sampling requires dataset_config.feature_dim to be set.")
        x_init = torch.randn(sample_count, feature_dim, device=torch_device)
    else:
        if dataset_config.image_size is None:
            raise ValueError("Image sampling requires dataset_config.image_size to be set.")
        # Channel count comes from the architecture preset, not a constant:
        # grayscale presets (mnist) build a 1-channel UNet.
        x_init = torch.randn(
            sample_count,
            int(get_model_config(dataset_config)["in_channels"]),
            dataset_config.image_size,
            dataset_config.image_size,
            device=torch_device,
        )
    velocity_model = VelocityFieldWrapper(
        model=model,
        class_conditioned=dataset_config.class_conditioned,
    )
    solver = ODESolver(velocity_model=velocity_model)
    generated = solver.sample(
        x_init=x_init,
        time_grid=torch.tensor([0.0, 1.0], device=torch_device),
        method=ode_method,
        step_size=step_size,
        label=labels,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_config.dataset_type == "tabular":
        torch.save(generated.detach().cpu(), output_path)
    else:
        generated = torch.clamp(generated * 0.5 + 0.5, min=0.0, max=1.0)
        save_image(generated, output_path, nrow=max(1, int(sample_count**0.5)))
    return output_path


def _infer_feature_dim(dataset: Any) -> int | None:
    if hasattr(dataset, "feature_dim"):
        value = dataset.feature_dim
        if value is not None:
            return int(value)
    if hasattr(dataset, "features"):
        features = dataset.features
        if isinstance(features, torch.Tensor) and features.ndim == 2:
            return int(features.shape[1])
    first_item = dataset[0]
    if isinstance(first_item, tuple):
        sample = first_item[0]
    else:
        sample = first_item
    if isinstance(sample, torch.Tensor) and sample.ndim == 1:
        return int(sample.shape[0])
    return None


def _enrich_dataset_config_from_dataset(
    dataset_config: FlowMatchingDatasetConfig,
    dataset: Any,
) -> FlowMatchingDatasetConfig:
    if dataset_config.dataset_type != "tabular":
        return dataset_config
    feature_dim = dataset_config.feature_dim
    if feature_dim is None:
        feature_dim = _infer_feature_dim(dataset)
    num_classes = dataset_config.num_classes
    if num_classes is None and hasattr(dataset, "num_classes"):
        candidate = dataset.num_classes
        if candidate is not None:
            num_classes = int(candidate)
    if feature_dim is None:
        raise ValueError("Failed to infer tabular feature dimension from dataset.")
    return replace(dataset_config, feature_dim=int(feature_dim), num_classes=num_classes)


def _normalize_training_samples(
    samples: torch.Tensor,
    dataset_config: FlowMatchingDatasetConfig,
) -> torch.Tensor:
    if dataset_config.dataset_type == "tabular":
        return samples
    return samples * 2.0 - 1.0


def train_flow_matching_model(
    training_config: FlowMatchingTrainingConfig,
) -> FlowMatchingCheckpoint:
    """Train a Gaussian-source flow matching model (x_0 ~ N(0, I), x_1 = data).

    Resumes from ``checkpoint_latest.pt`` when ``training_config.resume`` is set,
    so re-submitting the same command after a wall-clock kill continues the run.
    """
    set_random_seed(training_config.seed)
    dataset_config = resolve_dataset_config(training_config.dataset, training_config.data_path)
    output_dir = training_config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "checkpoint_latest.pt"
    device = torch.device(training_config.device)

    if training_config.resume and (output_dir / "DONE.json").exists() and latest_path.exists():
        print(f"[FlowMatching] {output_dir} is already complete; nothing to do.")
        return load_flow_matching_checkpoint(latest_path, device=device)

    dataset = build_flow_matching_dataset(
        dataset_config,
        download=training_config.download,
    )
    dataset_config = _enrich_dataset_config_from_dataset(dataset_config, dataset)
    save_metadata(output_dir, "fm_training_config.json", training_config.to_dict())
    save_metadata(output_dir, "fm_dataset_config.json", dataset_config.to_dict())

    dataloader = build_training_dataloader(
        dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
    )
    steps_per_epoch = max(1, len(dataloader))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(
        dataset_config,
        use_ema=training_config.use_ema,
        ema_decay=training_config.ema_decay,
        feature_dim=dataset_config.feature_dim,
    ).to(device)

    start_epoch = 0
    global_step = 0
    resume_payload: dict[str, Any] | None = None
    if training_config.resume and latest_path.exists():
        resume_payload = load_checkpoint(latest_path, map_location=device)
        # Load before torch.compile: the compat loader strips "_orig_mod.".
        _load_model_state_dict_compat(model, resume_payload["model_state_dict"])
        completed = bool(resume_payload.get("epoch_complete", True))
        start_epoch = int(resume_payload["epoch"]) + (1 if completed else 0)
        global_step = int(resume_payload.get("global_step", start_epoch * steps_per_epoch))
        print(f"[FlowMatching] resuming from {latest_path}: epoch={start_epoch} step={global_step}")

    ema_wrapper = _extract_ema_wrapper(model)

    if device.type == 'cuda':
        print("[FlowMatching] Compiling model for CUDA... (this takes a minute initially)")
        model = torch.compile(model)
    else:
        print(f"[FlowMatching] Skipping torch.compile() for device {device.type} to avoid backend errors.")
    if ema_wrapper is None:
        ema_wrapper = _extract_ema_wrapper(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
    )
    if resume_payload is not None and "optimizer_state_dict" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    path = CondOTProbPath()
    latest_checkpoint_path: Path | None = latest_path if latest_path.exists() else None

    max_steps = int(training_config.max_steps) if training_config.max_steps else None
    total_epochs = training_config.epochs
    if max_steps is not None:
        total_epochs = max(total_epochs, -(-max_steps // steps_per_epoch))
    step_checkpoint_every = max(0, int(training_config.checkpoint_every_steps))

    def _save(epoch_index: int, average: float, *, complete: bool) -> Path:
        return save_flow_matching_checkpoint(
            output_dir,
            "checkpoint_latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch_index,
            average_loss=average,
            training_config=training_config,
            dataset_config=dataset_config,
            extra={"global_step": global_step, "epoch_complete": complete},
        )

    def _budget_spent() -> bool:
        return max_steps is not None and global_step >= max_steps

    for epoch in range(start_epoch, total_epochs):
        if _budget_spent():
            break
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for samples, labels in dataloader:
            # non_blocking=True is only safe when the source tensor lives in
            # pinned memory. build_training_dataloader sets pin_memory=False, so
            # the loader can release the batch buffer while the async copy is
            # still in flight; on MPS that silently lands garbage in the batch
            # (pixels ~1e27 in what should be a [0, 1] image) instead of raising,
            # and the loss is NaN within a few iterations. Copy synchronously
            # unless the source really is pinned.
            samples = samples.to(device, non_blocking=samples.is_pinned())
            labels = labels.to(device, non_blocking=labels.is_pinned())
            conditioning: dict[str, torch.Tensor] = {}
            if dataset_config.class_conditioned and torch.rand(1).item() >= training_config.class_drop_prob:
                conditioning["label"] = labels

            samples = _normalize_training_samples(samples, dataset_config)
            noise = torch.randn_like(samples)
            t = torch.rand(samples.shape[0], device=device)
            path_sample = path.sample(x_0=noise, x_1=samples, t=t)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda_amp):
                prediction = model(path_sample.x_t, t, conditioning)
                loss = F.mse_loss(prediction, path_sample.dx_t)

            # GradScaler("cuda") is a no-op when disabled, but on MPS it can still
            # leave non-finite optimizer state across a full epoch. Keep the AMP
            # path CUDA-only and use a plain step elsewhere.
            if use_cuda_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if ema_wrapper is not None:
                ema_wrapper.update_ema()

            epoch_loss += float(loss.detach().item())
            num_batches += 1
            global_step += 1

            if step_checkpoint_every and global_step % step_checkpoint_every == 0:
                latest_checkpoint_path = _save(
                    epoch, epoch_loss / max(1, num_batches), complete=False
                )
            if _budget_spent():
                break

        average_loss = epoch_loss / max(1, num_batches)
        latest_checkpoint_path = _save(epoch, average_loss, complete=num_batches >= steps_per_epoch)

        if (epoch + 1) % training_config.checkpoint_every == 0:
            save_flow_matching_checkpoint(
                output_dir,
                f"checkpoint_epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                average_loss=average_loss,
                training_config=training_config,
                dataset_config=dataset_config,
                extra={"global_step": global_step, "epoch_complete": True},
            )

        if training_config.sample_every > 0 and (
            (epoch + 1) % training_config.sample_every == 0
            or epoch == total_epochs - 1
            or _budget_spent()
        ):
            sample_extension = "pt" if dataset_config.dataset_type == "tabular" else "png"
            sample_images(
                model,
                dataset_config,
                output_path=output_dir / "samples" / f"epoch_{epoch + 1:04d}.{sample_extension}",
                device=device,
                sample_count=training_config.sample_count,
                ode_method=training_config.sample_ode_method,
                step_size=training_config.sample_step_size,
                seed=training_config.seed + epoch + 1,
            )

        print(
            f"[FlowMatching] epoch={epoch + 1}/{total_epochs} step={global_step} "
            f"loss={average_loss:.6f}",
            flush=True,
        )

    if latest_checkpoint_path is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    write_done_marker(output_dir, global_step=global_step, max_steps=max_steps, kind="gaussian_cnf")
    final_model = load_flow_matching_checkpoint(latest_checkpoint_path, device=device)
    final_model.model = final_model.model.to(device)
    return final_model
