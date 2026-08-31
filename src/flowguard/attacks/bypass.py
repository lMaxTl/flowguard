from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler, UNet2DModel
from torch.utils.data import DataLoader

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.attacks.adaptive import (
    ProjectionConfig,
    apply_query_projection,
    color_stat_penalty,
    high_frequency_energy,
    total_variation,
)
from flowguard.attacks.base import AttackRunContext, AttackRunner, AttackRunResult
from flowguard.attacks.maze import ConvGenerator
from flowguard.querying.engine import QueryEngine


@dataclass(slots=True)
class CemAttackConfig:
    """Configuration shared by data-free FlowPure-bypass CEM attacks."""

    population_size: int
    elite_fraction: float
    latent_dim: int
    selector: str
    lambda_detector: float
    accepted_only: bool
    projection: ProjectionConfig
    procedural_grid_size: int
    procedural_fourier_modes: int
    procedural_blobs: int
    lambda_tv: float
    lambda_freq: float
    lambda_color: float
    artifact_sample_size: int
    substitute_train_steps: int
    substitute_lr: float
    eval_interval: int


@dataclass(slots=True)
class QueryCandidate:
    """One submitted candidate and its victim response."""

    index: int
    query: torch.Tensor
    probabilities: torch.Tensor
    detector_score: float
    accepted: bool


@dataclass(slots=True)
class CemIterationSnapshot:
    """One CEM iteration recorded for visualization or paper figures.

    Attributes:
        iteration: 1-based iteration index.
        submitted: Queries submitted in this iteration.
        accepted: Accepted (non-blocked) queries in this iteration.
        blocked: Blocked queries in this iteration.
        mean_norm: L2 norm of the CEM mean vector after the update (0 if skipped).
        std_norm: Mean L2 norm of the CEM std vector after the update (0 if skipped).
        population_queries: Subset of decoded queries sent this iteration (CHW numpy).
        population_accepted: Whether each population query was accepted.
        population_entropy: Shannon entropy of victim soft labels per population item.
        population_detector_score: Detector score per population item.
        elite_queries: Elite queries selected for the CEM update (CHW numpy).
    """

    iteration: int
    submitted: int
    accepted: int
    blocked: int
    mean_norm: float
    std_norm: float
    population_queries: list[np.ndarray]
    population_accepted: list[bool]
    population_entropy: list[float]
    population_detector_score: list[float]
    elite_queries: list[np.ndarray]


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp_min(1e-8)
    return -torch.sum(probabilities * probabilities.log(), dim=1)


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1))


def _evaluate_substitute_accuracy(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            predictions = logits.argmax(dim=1)
            target_ids = targets.long()
            correct += int((predictions.cpu() == target_ids.cpu()).sum().item())
            total += int(inputs.size(0))
    return 100.0 * correct / max(total, 1)


def _as_clip_tensors(
    clip_min: np.ndarray,
    clip_max: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    min_t = torch.as_tensor(clip_min, device=device, dtype=dtype).view(1, -1, 1, 1)
    max_t = torch.as_tensor(clip_max, device=device, dtype=dtype).view(1, -1, 1, 1)
    return min_t, max_t


def procedural_natural_images(
    parameters: torch.Tensor,
    *,
    channels: int,
    height: int,
    width: int,
    clip_min: np.ndarray,
    clip_max: np.ndarray,
    grid_size: int = 4,
    fourier_modes: int = 3,
    num_blobs: int = 4,
) -> torch.Tensor:
    """Decode a compact natural-statistics parameter vector to normalized images.

    The parameterization combines a bicubic color-control grid, low-frequency
    Fourier components, and alpha-blended blobs. The output follows the
    repository's normalized model-input convention.
    """

    if parameters.ndim != 2:
        raise ValueError(f"Expected (B, D) parameters; got {tuple(parameters.shape)}.")
    batch_size = parameters.shape[0]
    device = parameters.device
    dtype = parameters.dtype
    cursor = 0

    grid_values = parameters[:, cursor : cursor + channels * grid_size * grid_size]
    cursor += channels * grid_size * grid_size
    grid = torch.sigmoid(grid_values.view(batch_size, channels, grid_size, grid_size))
    pixels = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False)

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    for mode_index in range(max(0, int(fourier_modes))):
        coeffs = parameters[:, cursor : cursor + channels * 2].view(batch_size, channels, 2, 1, 1)
        cursor += channels * 2
        frequency = float(mode_index + 1)
        basis = torch.stack(
            [
                torch.sin(2.0 * math.pi * frequency * xx),
                torch.cos(2.0 * math.pi * frequency * yy),
            ],
            dim=0,
        ).view(1, 1, 2, height, width)
        pixels = pixels + 0.12 * torch.sum(torch.tanh(coeffs) * basis, dim=2)

    for _ in range(max(0, int(num_blobs))):
        blob_params = parameters[:, cursor : cursor + 3 + channels]
        cursor += 3 + channels
        center_x = torch.sigmoid(blob_params[:, 0]).view(batch_size, 1, 1, 1)
        center_y = torch.sigmoid(blob_params[:, 1]).view(batch_size, 1, 1, 1)
        sigma = (0.04 + 0.20 * torch.sigmoid(blob_params[:, 2])).view(batch_size, 1, 1, 1)
        color = torch.sigmoid(blob_params[:, 3 : 3 + channels]).view(batch_size, channels, 1, 1)
        mask = torch.exp(-((xx - center_x).pow(2) + (yy - center_y).pow(2)) / (2.0 * sigma.pow(2)))
        pixels = pixels * (1.0 - 0.25 * mask) + color * (0.25 * mask)

    pixels = pixels.clamp(0.0, 1.0)
    min_t, max_t = _as_clip_tensors(clip_min, clip_max, device=device, dtype=dtype)
    return min_t + pixels * (max_t - min_t)


def procedural_parameter_dim(
    *,
    channels: int,
    grid_size: int = 4,
    fourier_modes: int = 3,
    num_blobs: int = 4,
) -> int:
    """Return the strict procedural parameter-vector size."""

    return (
        channels * grid_size * grid_size
        + max(0, int(fourier_modes)) * channels * 2
        + max(0, int(num_blobs)) * (3 + channels)
    )


class DiffusionLatentGenerator(torch.nn.Module):
    """Latent-to-image generator backed by a diffusers DDPM checkpoint.

    This adapter allows D1 to consume model IDs like
    ``google/ddpm-cifar10-32`` while preserving the CEM latent API expected by
    the attack runner.
    """

    def __init__(
        self,
        *,
        model_id_or_path: str,
        latent_dim: int,
        image_size: int = 32,
        scheduler_name: str = "ddim",
        num_inference_steps: int = 20,
        device: torch.device,
        use_safetensors: bool = True,
        latent_noise_mode: str = "projection",
        noise_projection_seed: int = 0,
    ) -> None:
        super().__init__()
        self.model_id_or_path = str(model_id_or_path)
        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)
        self.num_inference_steps = max(1, int(num_inference_steps))
        self.device = device
        mode = str(latent_noise_mode).lower()
        if mode not in {"projection", "tile"}:
            raise ValueError(
                f"Unsupported latent_noise_mode '{latent_noise_mode}'. "
                "Choose 'projection' (Gaussian-like x_T, correct) or 'tile' "
                "(legacy periodic x_T, kept only to reproduce older runs)."
            )
        self.latent_noise_mode = mode
        self.noise_projection_seed = int(noise_projection_seed)
        self._noise_projection: torch.Tensor | None = None
        self.unet = UNet2DModel.from_pretrained(
            self.model_id_or_path,
            use_safetensors=bool(use_safetensors),
        ).to(device)
        self.unet.eval()
        # The UNet's downsampling path is fixed by its pretrained weights (a
        # fixed number of stride-2 stages), so decoding at any resolution
        # other than what it was trained at can break its skip connections --
        # e.g. 28 (MNIST's native size, if a caller defaults image_size to
        # the victim's shape) doesn't divide evenly through 32's stages
        # (28->14->7->3.5) and crashes inside the UNet's upsample blocks. D1's
        # whole premise is a *generic*, victim-agnostic natural-image prior
        # reused unchanged across datasets, so the decode resolution must
        # always be the checkpoint's own native size; callers that need a
        # different size resize the decoded output afterward instead.
        native_size = getattr(getattr(self.unet, "config", None), "sample_size", None)
        native_size = int(native_size) if native_size is not None else int(image_size)
        if int(image_size) != native_size:
            warnings.warn(
                f"DiffusionLatentGenerator: requested image_size={image_size} does not "
                f"match '{self.model_id_or_path}''s native sample_size={native_size}; "
                "decoding at the native size and leaving resizing to the caller.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.image_size = native_size
        scheduler_key = str(scheduler_name).lower()
        scheduler_map = {
            "ddpm": DDPMScheduler,
            "ddim": DDIMScheduler,
            "pndm": PNDMScheduler,
        }
        if scheduler_key not in scheduler_map:
            raise ValueError(
                f"Unsupported diffusion scheduler '{scheduler_name}'. "
                "Choose from: ddpm, ddim, pndm."
            )
        try:
            self.scheduler = scheduler_map[scheduler_key].from_pretrained(
                self.model_id_or_path,
                subfolder="scheduler",
            )
        except OSError:
            self.scheduler = scheduler_map[scheduler_key].from_pretrained(
                self.model_id_or_path,
            )

    def _latent_to_noise(self, latent: torch.Tensor) -> torch.Tensor:
        """Map a compact latent to the DDPM's initial state ``x_T``.

        A diffusion model's reverse process assumes ``x_T ~ N(0, I)`` with
        independent pixels. The ``tile`` mode repeats the latent to fill the
        image, which makes ``x_T`` exactly periodic (a 128-d latent is tiled 24x
        for 3x32x32) and therefore not a sample from that prior at all: on
        CIFAR-10 the decode collapses to saturated barcode stripes and the
        resulting queries carry no class information, pinning the extracted
        model at chance accuracy. ``projection`` instead applies a fixed random
        Gaussian map, so every pixel is marginally ~N(0, 1) and spatially
        decorrelated while the output stays a differentiable function of the
        latent.
        """
        if latent.ndim != 2:
            raise ValueError(f"Expected latent shape (B, D); got {tuple(latent.shape)}.")
        batch_size = latent.shape[0]
        # Channel count comes from the loaded checkpoint, not a constant: the
        # same adapter has to serve 3-channel CIFAR DDPMs and 1-channel
        # grayscale ones.
        channels = int(self.unet.config.in_channels)
        target_elements = channels * self.image_size * self.image_size

        if self.latent_noise_mode == "tile":
            repeats = (target_elements + latent.shape[1] - 1) // latent.shape[1]
            expanded = latent.repeat(1, repeats)[:, :target_elements]
            noise = expanded.reshape(batch_size, channels, self.image_size, self.image_size)
            return noise.to(device=latent.device, dtype=torch.float32)

        projection = self._noise_projection
        if projection is None or projection.shape != (latent.shape[1], target_elements):
            generator = torch.Generator(device="cpu").manual_seed(self.noise_projection_seed)
            projection = torch.randn(
                latent.shape[1], target_elements, generator=generator
            ) / math.sqrt(float(latent.shape[1]))
            self._noise_projection = projection.to(latent.device)
            projection = self._noise_projection
        elif projection.device != latent.device:
            self._noise_projection = projection.to(latent.device)
            projection = self._noise_projection
        noise = (latent.to(torch.float32) @ projection).reshape(
            batch_size, channels, self.image_size, self.image_size
        )
        return noise

    @staticmethod
    def _apply_step_guidance(
        sample: torch.Tensor,
        *,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """One classifier-guidance-style correction, local to this step.

        Computes the gradient of ``guidance_fn`` (a detector score, to be
        minimized) with respect to the *current* sample only -- via a
        throwaway autograd graph that never touches the frozen UNet -- and
        subtracts it as a constant nudge. Because the nudge is detached
        before being combined with ``sample``, this does not require (or
        add) any gradient path through the denoising network itself, and it
        does not break ``sample``'s existing gradient path back to the
        initial latent (the correction is just a constant offset from the
        caller's point of view). This is what makes per-step guidance cheap
        relative to differentiating through the whole unrolled trajectory:
        one small forward/backward through the detector per step, not
        through the (much larger) UNet.

        Wrapped in ``torch.enable_grad()`` because callers may invoke the
        whole decode (and thus this method) from inside an outer
        ``torch.no_grad()`` block -- e.g. the D1 loop's black-box query step,
        which generates fakes without gradients since none are needed for
        querying. Without locally re-enabling grad tracking here, ``score``
        would carry no graph back to ``probe`` regardless of
        ``requires_grad_(True)``, and ``torch.autograd.grad`` would fail.
        """
        with torch.enable_grad():
            probe = sample.detach().requires_grad_(True)
            pixels_01 = torch.clamp((probe + 1.0) * 0.5, min=0.0, max=1.0)
            score = guidance_fn(pixels_01)
            grad = torch.autograd.grad(score.sum(), probe)[0]
        return sample - guidance_scale * grad.detach()

    def _decode_pixels(
        self,
        latent: torch.Tensor,
        *,
        differentiable: bool,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        guidance_scale: float = 0.0,
        guidance_start_frac: float = 0.0,
    ) -> torch.Tensor:
        sample = self._latent_to_noise(latent)
        self.scheduler.set_timesteps(self.num_inference_steps, device=latent.device)
        timesteps = list(self.scheduler.timesteps)
        total_steps = max(1, len(timesteps))
        for step_index, timestep in enumerate(timesteps):
            model_input = self.scheduler.scale_model_input(sample, timestep)
            if differentiable:
                with torch.no_grad():
                    noise_prediction = self.unet(model_input, timestep).sample
            else:
                noise_prediction = self.unet(model_input, timestep).sample
            sample = self.scheduler.step(noise_prediction, timestep, sample).prev_sample
            if (
                guidance_fn is not None
                and guidance_scale > 0.0
                and (step_index + 1) / total_steps >= guidance_start_frac
            ):
                sample = self._apply_step_guidance(
                    sample, guidance_fn=guidance_fn, guidance_scale=guidance_scale
                )
        return torch.clamp((sample + 1.0) * 0.5, min=0.0, max=1.0)

    @torch.inference_mode()
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decode_pixels(latent, differentiable=False)

    def decode_differentiable(
        self,
        latent: torch.Tensor,
        *,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        guidance_scale: float = 0.0,
        guidance_start_frac: float = 0.0,
    ) -> torch.Tensor:
        """Decode latents to ``[0, 1]`` pixels with gradients through the latent path.

        ``guidance_fn``, when given, is applied as a classifier-guidance-style
        correction at every denoising step (weighted by ``guidance_scale``,
        active for the trajectory fraction from ``guidance_start_frac`` to 1),
        rather than only shaping the *starting* latent once before the frozen
        multi-step decode runs unchanged. Continuous, trajectory-wide
        correction is the standard way detector/classifier guidance is used
        in diffusion sampling; a single upfront latent perturbation is a much
        weaker proxy for it since the frozen decode after that point has no
        further opportunity to respond to the detector.
        """
        return self._decode_pixels(
            latent,
            differentiable=True,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
            guidance_start_frac=guidance_start_frac,
        )


# Warn once per process when _query_candidates treats a query-time crash as an
# ordinary "blocked" query (see the try/except there). A RuntimeError from the
# victim model is far more often a real bug -- e.g. a shape/channel mismatch
# between the generator's output and the victim's input -- than an actual
# defense rejection, and folding it into "blocked" can silently zero out an
# entire run's transfer set (every query "blocks") with no crash to flag it,
# surfacing only later as an unrelated-looking IndexError on an empty list.
_WARNED_QUERY_RUNTIME_ERROR: set[str] = set()


class _CemDataFreeAttackRunner(AttackRunner):
    """Base CEM loop used by the data-free FlowPure bypass attacks."""

    name = "data_free_cem"
    proposal_kind = "latent"
    force_accepted_only = False

    def __init__(self, query_engine: QueryEngine, **_: Any) -> None:
        self.query_engine = query_engine

    def _build_config(self, context: AttackRunContext) -> CemAttackConfig:
        extra = dict(context.experiment.attack.extra)
        projection_raw = extra.get("projection", {}) or {}
        return CemAttackConfig(
            population_size=max(1, int(extra.get("population_size", context.experiment.attack.batch_size))),
            elite_fraction=min(max(float(extra.get("elite_fraction", 0.20)), 1e-3), 1.0),
            latent_dim=max(4, int(extra.get("latent_dim", 128))),
            selector=str(extra.get("selector", "entropy")).lower(),
            lambda_detector=float(extra.get("lambda_detector", extra.get("lambda_flow", 1.0))),
            accepted_only=bool(extra.get("accepted_only", self.force_accepted_only)),
            projection=ProjectionConfig(
                name=str(projection_raw.get("name", "none")),
                strength=float(projection_raw.get("strength", 0.1)),
                steps=int(projection_raw.get("steps", 10)),
                differentiable=bool(projection_raw.get("differentiable", False)),
            ),
            procedural_grid_size=max(2, int(extra.get("procedural_grid_size", 4))),
            procedural_fourier_modes=max(0, int(extra.get("procedural_fourier_modes", 3))),
            procedural_blobs=max(0, int(extra.get("procedural_blobs", 4))),
            lambda_tv=float(extra.get("lambda_tv", 0.0)),
            lambda_freq=float(extra.get("lambda_freq", 0.0)),
            lambda_color=float(extra.get("lambda_color", 0.0)),
            artifact_sample_size=max(0, int(extra.get("artifact_sample_size", 0))),
            substitute_train_steps=max(0, int(extra.get("substitute_train_steps", 1))),
            substitute_lr=float(extra.get("substitute_lr", context.experiment.training.lr)),
            eval_interval=max(0, int(extra.get("eval_interval", 5))),
        )

    def _normalized_clip_values(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        family_key = "imagenet" if family == "tinyimagenet" else family
        mean_std = legacy_datasets.modelfamily_to_mean_std[family_key]
        mean = np.asarray(mean_std["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(mean_std["std"], dtype=np.float32).reshape(-1, 1, 1)
        return (0.0 - mean) / std, (1.0 - mean) / std

    def _build_queryset(self, context: AttackRunContext):
        dataset_name = context.experiment.attack.query_dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
        return legacy_datasets.__dict__[dataset_name](
            train=True,
            transform=transform,
            download=context.experiment.attack.query_dataset.download,
        )

    def _build_eval_testset(self, context: AttackRunContext):
        spec = context.experiment
        dataset_name = spec.dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
        return legacy_datasets.__dict__[dataset_name](
            train=False,
            transform=transform,
            download=spec.dataset.download,
        )

    def _create_substitute(self, context: AttackRunContext, *, num_classes: int, device: torch.device):
        spec = context.experiment
        family = legacy_datasets.dataset_to_modelfamily[spec.dataset.name]
        model = legacy_zoo.get_net(
            spec.substitute_model.architecture,
            family,
            spec.substitute_model.pretrained,
            num_classes=num_classes,
        )
        return model.to(device)

    def _train_substitute_on_elites(
        self,
        substitute: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        queried: list[QueryCandidate],
        elite_indices: list[int],
        *,
        steps: int,
    ) -> float:
        if not elite_indices or steps <= 0:
            return 0.0
        device = next(substitute.parameters()).device
        queries = torch.stack([queried[index].query for index in elite_indices]).to(device)
        labels = torch.stack([queried[index].probabilities for index in elite_indices]).to(device).float()
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        loss_sum = 0.0
        substitute.train()
        for _ in range(steps):
            optimizer.zero_grad()
            logits = substitute(queries)
            loss = _soft_cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        return loss_sum / max(steps, 1)

    def _write_metrics(self, output_dir: Path, metrics: list[dict[str, float | int]]) -> None:
        if not metrics:
            return
        fieldnames = list(metrics[0].keys())
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _build_generator(
        self,
        context: AttackRunContext,
        *,
        channels: int,
        height: int,
        width: int,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        config: CemAttackConfig,
        device: torch.device,
    ) -> torch.nn.Module:
        checkpoint_path = context.experiment.attack.extra.get("generator")
        hf_model_id = context.experiment.attack.extra.get("generator_hf_model_id")
        if hf_model_id:
            return DiffusionLatentGenerator(
                model_id_or_path=str(hf_model_id),
                latent_dim=config.latent_dim,
                image_size=height,
                scheduler_name=str(context.experiment.attack.extra.get("diffusion_scheduler", "ddim")),
                num_inference_steps=int(context.experiment.attack.extra.get("diffusion_steps", 20)),
                device=device,
                use_safetensors=bool(context.experiment.attack.extra.get("diffusion_use_safetensors", True)),
            )
        if checkpoint_path:
            loaded = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            if isinstance(loaded, torch.nn.Module):
                return loaded.to(device).eval()
            if isinstance(loaded, dict) and isinstance(loaded.get("model"), torch.nn.Module):
                return loaded["model"].to(device).eval()
            raise ValueError("generator checkpoint must contain a torch.nn.Module or {'model': module}.")
        return ConvGenerator(
            latent_dim=config.latent_dim,
            channels=channels,
            image_shape=(height, width),
            clip_min=clip_min,
            clip_max=clip_max,
        ).to(device).eval()

    def _parameter_dim(self, *, channels: int, config: CemAttackConfig) -> int:
        if self.proposal_kind == "procedural":
            return procedural_parameter_dim(
                channels=channels,
                grid_size=config.procedural_grid_size,
                fourier_modes=config.procedural_fourier_modes,
                num_blobs=config.procedural_blobs,
            )
        return config.latent_dim

    def _decode_parameters(
        self,
        parameters: torch.Tensor,
        *,
        generator: torch.nn.Module,
        channels: int,
        height: int,
        width: int,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        config: CemAttackConfig,
    ) -> torch.Tensor:
        if self.proposal_kind == "procedural":
            return procedural_natural_images(
                parameters,
                channels=channels,
                height=height,
                width=width,
                clip_min=clip_min,
                clip_max=clip_max,
                grid_size=config.procedural_grid_size,
                fourier_modes=config.procedural_fourier_modes,
                num_blobs=config.procedural_blobs,
            )
        with torch.no_grad():
            decoded = generator(parameters)
        images = decoded[0] if isinstance(decoded, tuple) else decoded
        if isinstance(generator, DiffusionLatentGenerator):
            min_t, max_t = _as_clip_tensors(
                clip_min,
                clip_max,
                device=images.device,
                dtype=images.dtype,
            )
            images = min_t + images * (max_t - min_t)
        if images.shape[-2:] != (height, width):
            images = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
        if images.shape[1] != channels:
            # A frozen external generator (e.g. the pretrained CIFAR-10
            # DDPM used by D1/D3) is a *generic* natural-image prior reused
            # unchanged across victims, so its channel count is not
            # guaranteed to match the victim's (e.g. 3-channel CIFAR-10
            # output against a 1-channel MNIST victim). Left unreconciled,
            # querying the victim with this raises inside
            # query_engine.query_batch, which _query_candidates' generic
            # `except RuntimeError` treats as an ordinary "blocked" query --
            # every single query in the run silently "blocks", producing an
            # empty transfer set instead of a loud failure. Same class of
            # fix as DisguideBypassAttackRunner._adapt_to_substitute_domain;
            # a no-op whenever the generator already matches the victim's
            # domain (e.g. CIFAR-10 against a CIFAR-10 victim).
            if channels == 1:
                images = images.mean(dim=1, keepdim=True)
            elif images.shape[1] == 1:
                images = images.repeat(1, channels, 1, 1)
            elif images.shape[1] > channels:
                images = images[:, :channels]
            else:
                repeats = -(-channels // images.shape[1])  # ceil division
                images = images.repeat(1, repeats, 1, 1)[:, :channels]
        return images

    def _detector_metadata(self) -> tuple[bool, float]:
        if not self.query_engine.history.records:
            return False, 0.0
        metadata = self.query_engine.history.records[-1].metadata
        raw_flags = metadata.get("flowpure_blocked") or metadata.get("flow_matching_blocked")
        blocked = bool(raw_flags[0]) if isinstance(raw_flags, list) and raw_flags else False
        raw_scores = (
            metadata.get("flowpure_scores")
            or metadata.get("flowguard_integral_scores")
            or metadata.get("queries_blocked")
        )
        score = float(raw_scores[0]) if isinstance(raw_scores, list) and raw_scores else 0.0
        return blocked, score

    def _query_candidates(
        self,
        candidates: torch.Tensor,
        *,
        config: CemAttackConfig,
        submitted: int,
        query_budget: int,
    ) -> tuple[list[QueryCandidate], int, int, int]:
        accepted: list[QueryCandidate] = []
        blocked_queries = 0
        submitted_queries = 0
        metadata = {
            "fdinet_gt": 1,
            "fdinet_source": self.name,
            "fdinet_mode": self.proposal_kind,
        }
        for index, candidate in enumerate(candidates):
            if submitted + submitted_queries >= query_budget:
                break
            query = candidate.unsqueeze(0)
            submitted_queries += 1
            try:
                result = self.query_engine.query_batch(query, metadata=metadata)
            except RuntimeError as exc:
                if self.name not in _WARNED_QUERY_RUNTIME_ERROR:
                    _WARNED_QUERY_RUNTIME_ERROR.add(self.name)
                    warnings.warn(
                        f"{self.name}: query_batch raised {exc!r} and is being "
                        "counted as a blocked query. If this happens on every "
                        "query, the run will silently collect zero samples "
                        "(surfacing later as an unrelated IndexError on an "
                        "empty transfer set) rather than failing loudly here -- "
                        "check whether this is a real defense rejection or a "
                        "shape/channel mismatch between the generator's output "
                        "and the victim's input.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                blocked_queries += 1
                continue
            blocked, detector_score = self._detector_metadata()
            if blocked:
                blocked_queries += 1
                if config.accepted_only:
                    continue
            probabilities = result.payload.detach().cpu()
            if probabilities.ndim == 1:
                probabilities = probabilities.unsqueeze(0)
            accepted.append(
                QueryCandidate(
                    index=index,
                    query=candidate.detach().cpu(),
                    probabilities=probabilities[0],
                    detector_score=detector_score,
                    accepted=not blocked,
                )
            )
        return accepted, submitted_queries, len(accepted), blocked_queries

    def _selection_scores(
        self,
        queried: list[QueryCandidate],
        *,
        class_counts: torch.Tensor,
        config: CemAttackConfig,
    ) -> torch.Tensor:
        probabilities = torch.stack([item.probabilities for item in queried], dim=0)
        selector = config.selector
        if selector == "low_detector":
            base_score = torch.zeros(len(queried), dtype=torch.float32)
        elif selector == "class_rarity":
            labels = probabilities.argmax(dim=1)
            base_score = 1.0 / torch.sqrt(class_counts[labels].to(torch.float32) + 1.0)
        else:
            base_score = _entropy(probabilities)
        detector = torch.tensor([item.detector_score for item in queried], dtype=torch.float32)
        if detector.numel() > 1 and float(detector.std(unbiased=False).item()) > 1e-8:
            detector = (detector - detector.mean()) / detector.std(unbiased=False)
        return base_score - float(config.lambda_detector) * detector

    def _regularizer_penalty(self, queries: torch.Tensor, config: CemAttackConfig) -> float:
        if self.proposal_kind != "procedural" or len(queries) == 0:
            return 0.0
        penalty = torch.tensor(0.0)
        if config.lambda_tv > 0.0:
            penalty = penalty + config.lambda_tv * total_variation(queries)
        if config.lambda_freq > 0.0:
            penalty = penalty + config.lambda_freq * high_frequency_energy(queries)
        if config.lambda_color > 0.0:
            penalty = penalty + config.lambda_color * color_stat_penalty(queries)
        return float(penalty.detach().cpu().item())

    def _candidate_entropy(self, probabilities: torch.Tensor) -> float:
        entropy = _entropy(probabilities.unsqueeze(0) if probabilities.ndim == 1 else probabilities)
        return float(entropy[0].item())

    def _tensor_to_trace_image(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().float().numpy()

    def _append_cem_trace(
        self,
        context: AttackRunContext,
        *,
        iteration: int,
        submitted: int,
        accepted: int,
        blocked: int,
        projected_queries: torch.Tensor,
        queried: list[QueryCandidate],
        elite_indices: list[int],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        trace = context.metadata.get("cem_trace")
        if not isinstance(trace, list):
            return

        max_images = max(1, int(context.metadata.get("cem_trace_max_images", 8)))
        population_queries: list[np.ndarray] = []
        population_accepted: list[bool] = []
        population_entropy: list[float] = []
        population_detector_score: list[float] = []

        for item in queried[:max_images]:
            population_queries.append(self._tensor_to_trace_image(item.query))
            population_accepted.append(bool(item.accepted))
            population_entropy.append(self._candidate_entropy(item.probabilities))
            population_detector_score.append(float(item.detector_score))

        if not population_queries and projected_queries.shape[0] > 0:
            for row in projected_queries[:max_images]:
                population_queries.append(self._tensor_to_trace_image(row))
                population_accepted.append(False)
                population_entropy.append(0.0)
                population_detector_score.append(0.0)

        elite_queries = [
            self._tensor_to_trace_image(queried[index].query)
            for index in elite_indices
            if 0 <= index < len(queried)
        ][:max_images]

        trace.append(
            CemIterationSnapshot(
                iteration=int(iteration),
                submitted=int(submitted),
                accepted=int(accepted),
                blocked=int(blocked),
                mean_norm=float(mean.norm().item()),
                std_norm=float(std.norm().item()) if std.numel() else 0.0,
                population_queries=population_queries,
                population_accepted=population_accepted,
                population_entropy=population_entropy,
                population_detector_score=population_detector_score,
                elite_queries=elite_queries,
            )
        )

    def _save_cem_trace(self, context: AttackRunContext, output_dir: Path) -> str | None:
        trace = context.metadata.get("cem_trace")
        if not isinstance(trace, list) or not trace:
            return None
        trace_path = output_dir / "cem_trace.pt"
        torch.save(trace, trace_path)
        return str(trace_path)

    def run(self, context: AttackRunContext) -> AttackRunResult:
        spec = context.experiment
        config = self._build_config(context)
        output_dir = Path(context.output_dir or Path("runs") / spec.name / self.name)
        output_dir.mkdir(parents=True, exist_ok=True)

        queryset = self._build_queryset(context)
        sample_input = queryset[0][0]
        channels, height, width = sample_input.shape
        device = torch.device(spec.substitute_model.device)
        clip_min, clip_max = self._normalized_clip_values(spec.attack.query_dataset.name)
        generator = self._build_generator(
            context,
            channels=channels,
            height=height,
            width=width,
            clip_min=clip_min,
            clip_max=clip_max,
            config=config,
            device=device,
        )

        parameter_dim = self._parameter_dim(channels=channels, config=config)
        mean = torch.zeros(parameter_dim, device=device)
        std = torch.ones(parameter_dim, device=device)
        class_counts = torch.zeros(self.query_engine.target_service.num_classes, dtype=torch.long)
        transferset: list[tuple[np.ndarray, torch.Tensor]] = []
        submitted_queries = 0
        accepted_queries = 0
        blocked_queries = 0
        iteration = 0
        regularizer_penalties: list[float] = []
        projection_distortions: list[float] = []
        fft_before_values: list[float] = []
        fft_after_values: list[float] = []
        metrics: list[dict[str, float | int]] = []

        substitute: torch.nn.Module | None = None
        substitute_optimizer: torch.optim.Optimizer | None = None
        eval_testset = None
        if config.substitute_train_steps > 0:
            num_classes = self.query_engine.target_service.num_classes
            substitute = self._create_substitute(context, num_classes=num_classes, device=device)
            substitute_optimizer = torch.optim.SGD(
                substitute.parameters(),
                lr=config.substitute_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            eval_testset = self._build_eval_testset(context)

        while submitted_queries < spec.attack.query_budget:
            iteration += 1
            batch_size = min(config.population_size, spec.attack.query_budget - submitted_queries)
            parameters = mean + std.clamp_min(1e-3) * torch.randn(batch_size, parameter_dim, device=device)
            raw_queries = self._decode_parameters(
                parameters,
                generator=generator,
                channels=channels,
                height=height,
                width=width,
                clip_min=clip_min,
                clip_max=clip_max,
                config=config,
            )
            projected_queries = apply_query_projection(raw_queries, config.projection)
            if projected_queries.shape == raw_queries.shape:
                projection_distortions.append(
                    float((projected_queries - raw_queries).flatten(start_dim=1).norm(p=2, dim=1).mean().item())
                )
                fft_before_values.append(float(high_frequency_energy(raw_queries).detach().cpu().item()))
                fft_after_values.append(float(high_frequency_energy(projected_queries).detach().cpu().item()))
            regularizer_penalties.append(self._regularizer_penalty(projected_queries.detach().cpu(), config))

            queried, submitted, accepted, blocked = self._query_candidates(
                projected_queries.detach().cpu(),
                config=config,
                submitted=submitted_queries,
                query_budget=spec.attack.query_budget,
            )
            submitted_queries += submitted
            accepted_queries += accepted
            blocked_queries += blocked
            if not queried:
                self._append_cem_trace(
                    context,
                    iteration=iteration,
                    submitted=submitted,
                    accepted=accepted,
                    blocked=blocked,
                    projected_queries=projected_queries.detach().cpu(),
                    queried=[],
                    elite_indices=[],
                    mean=mean,
                    std=std,
                )
                std = std * 1.25
                continue

            scores = self._selection_scores(queried, class_counts=class_counts, config=config)
            elite_count = max(1, int(math.ceil(len(queried) * config.elite_fraction)))
            elite_indices = torch.topk(scores, k=elite_count).indices.tolist()
            source_indices = [queried[index].index for index in elite_indices]
            elite_parameters = parameters[source_indices].detach()
            mean = elite_parameters.mean(dim=0)
            std = elite_parameters.std(dim=0, unbiased=False).clamp_min(1e-3)
            self._append_cem_trace(
                context,
                iteration=iteration,
                submitted=submitted,
                accepted=accepted,
                blocked=blocked,
                projected_queries=projected_queries.detach().cpu(),
                queried=queried,
                elite_indices=elite_indices,
                mean=mean,
                std=std,
            )

            for elite_index in elite_indices:
                item = queried[elite_index]
                label = item.probabilities.detach().cpu().float()
                label_id = int(label.argmax().item())
                class_counts[label_id] += 1
                if config.artifact_sample_size <= 0 or len(transferset) < config.artifact_sample_size:
                    transferset.append((item.query.numpy(), label))

            if substitute is not None and substitute_optimizer is not None and elite_indices:
                train_loss = self._train_substitute_on_elites(
                    substitute,
                    substitute_optimizer,
                    queried,
                    elite_indices,
                    steps=config.substitute_train_steps,
                )
                should_eval = (
                    eval_testset is not None
                    and config.eval_interval > 0
                    and iteration % config.eval_interval == 0
                )
                if should_eval:
                    surrogate_accuracy = _evaluate_substitute_accuracy(
                        substitute,
                        eval_testset,
                        batch_size=max(1, spec.training.batch_size),
                        device=device,
                    )
                    metrics.append({
                        "iteration": iteration,
                        "queries": float(submitted_queries),
                        "substitute_loss": float(train_loss),
                        "surrogate_accuracy": float(surrogate_accuracy),
                    })

        if substitute is not None and eval_testset is not None:
            surrogate_accuracy = _evaluate_substitute_accuracy(
                substitute,
                eval_testset,
                batch_size=max(1, spec.training.batch_size),
                device=device,
            )
            if not metrics or metrics[-1].get("queries") != float(submitted_queries):
                metrics.append({
                    "iteration": iteration,
                    "queries": float(submitted_queries),
                    "substitute_loss": 0.0,
                    "surrogate_accuracy": float(surrogate_accuracy),
                })
            checkpoint = {
                "epoch": iteration,
                "arch": substitute.__class__,
                "state_dict": substitute.state_dict(),
                "best_acc": max(
                    (float(row["surrogate_accuracy"]) for row in metrics),
                    default=float(surrogate_accuracy),
                ),
                "optimizer": {},
                "created_on": self.name,
            }
            torch.save(checkpoint, output_dir / "checkpoint.pth.tar")

        transferset_path = output_dir / "transferset.pickle"
        with transferset_path.open("wb") as handle:
            torch.save(transferset, handle)
        params_payload = {
            "attack_kind": spec.attack.kind.value,
            "mode": spec.attack.mode.name,
            "proposal_kind": self.proposal_kind,
            "selector": config.selector,
            "optimizer": "cem",
            "budget": spec.attack.query_budget,
            "population_size": config.population_size,
            "elite_fraction": config.elite_fraction,
            "submitted_queries": submitted_queries,
            "accepted_queries": accepted_queries,
            "blocked_queries": blocked_queries,
            "stored_samples": len(transferset),
            "projection": {
                "name": config.projection.name,
                "strength": config.projection.strength,
                "steps": config.projection.steps,
                "differentiable": config.projection.differentiable,
            },
            "projection_distortion": float(np.mean(projection_distortions)) if projection_distortions else 0.0,
            "fft_high_frequency_before_projection": float(np.mean(fft_before_values)) if fft_before_values else 0.0,
            "fft_high_frequency_after_projection": float(np.mean(fft_after_values)) if fft_after_values else 0.0,
            "procedural_regularizer_penalty": float(np.mean(regularizer_penalties)) if regularizer_penalties else 0.0,
        }
        with (output_dir / "params_transfer.json").open("w", encoding="utf-8") as handle:
            json.dump(params_payload, handle, indent=2)
        self._write_metrics(output_dir, metrics)
        cem_trace_path = self._save_cem_trace(context, output_dir)
        result_metadata: dict[str, Any] = {
            "num_samples": len(transferset),
            "transferset_path": str(transferset_path),
            "actual_queries": submitted_queries,
            "accepted_queries": accepted_queries,
            "blocked_queries": blocked_queries,
        }
        if cem_trace_path is not None:
            result_metadata["cem_trace_path"] = cem_trace_path
            trace = context.metadata.get("cem_trace")
            if isinstance(trace, list):
                result_metadata["cem_trace_iterations"] = len(trace)
        return AttackRunResult(
            attack_name=self.name,
            mode_name=spec.attack.mode.name,
            output_dir=str(output_dir),
            metadata=result_metadata,
        )


class CemLatentManifoldAttackRunner(_CemDataFreeAttackRunner):
    """Legacy CEM attack over a compact generator latent manifold."""

    name = "latent_manifold"
    proposal_kind = "latent"


class CemProceduralNaturalAttackRunner(_CemDataFreeAttackRunner):
    """Legacy CEM attack over strict natural-statistics procedural parameters."""

    name = "procedural_natural"
    proposal_kind = "procedural"


def _uses_cem_engine(context: AttackRunContext) -> bool:
    return str(context.experiment.attack.extra.get("engine", "disguide")).lower() == "cem"


class RejectionOracleAttackRunner(_CemDataFreeAttackRunner):
    """Detector-oracle attack that trains only on accepted queries."""

    name = "rejection_oracle"
    proposal_kind = "latent"
    force_accepted_only = True

    def _build_config(self, context: AttackRunContext) -> CemAttackConfig:
        config = super()._build_config(context)
        proposal = str(context.experiment.attack.extra.get("proposal", "latent")).lower()
        self.proposal_kind = "procedural" if proposal == "procedural" else "latent"
        config.accepted_only = True
        return config


def __getattr__(name: str):
    if name in {"LatentManifoldAttackRunner", "ProceduralNaturalAttackRunner"}:
        from flowguard.attacks import disguide_bypass

        return getattr(disguide_bypass, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
