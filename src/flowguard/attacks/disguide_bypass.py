"""DisGUIDE training loop with D1/D2 generators and optional flow-matching regularization."""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from defenses import datasets as legacy_datasets
from flowguard.attacks.adaptive import (
    AdaptiveAttackConfig,
    GeneratorCollapseMonitor,
    SurrogateVelocityRegularizer,
    apply_query_projection,
    batch_quantile_detector_penalty,
    distribution_matching_penalty,
    label_distribution_penalty,
    normalized_detector_penalty,
)
from flowguard.attacks.base import AttackRunContext, AttackRunResult
from flowguard.attacks.bypass import (
    DiffusionLatentGenerator,
    _as_clip_tensors,
    procedural_natural_images,
    procedural_parameter_dim,
)
from flowguard.attacks.disguide import (
    DisguideAttackRunner,
    DisguideConfig,
    SubstituteEnsemble,
    _evaluate_accuracy,
    _evaluate_fidelity,
    _PlateauStopper,
    _ReplayBuffer,
    _student_loss,
)
from flowguard.attacks.maze import ConvGenerator
from flowguard.attacks.score_distillation import DistilledCompositeScore

GeneratorKind = Literal["latent", "procedural"]

# Warn once per process when an adaptive penalty silently degrades to its
# unnormalized form; the loss is evaluated once per generator step, so an
# unguarded warning would fire thousands of times per cell.
_WARNED_RAW_PENALTY: set[str] = set()


def _format_duration(seconds: float) -> str:
    """Format seconds as ``H:MM:SS``; hours are not wrapped at 24."""
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


@dataclass(slots=True)
class DisguideTraceSnapshot:
    """One DisGUIDE inner-loop step recorded for visualization."""

    step: int
    epoch: int
    queries: int
    generator_loss: float
    query_images: list[np.ndarray]
    query_entropy: list[float]
    detector_scores: list[float]
    query_blocked: list[bool]
    surrogate_accuracy: float | None = None


@dataclass(slots=True)
class DisguideBypassConfig(DisguideConfig):
    # Defaults are required because the base class now ends with defaulted
    # fields; every one of these is always passed explicitly by _build_config.
    procedural_grid_size: int = 4
    procedural_fourier_modes: int = 3
    procedural_blobs: int = 4
    adaptive: AdaptiveAttackConfig | None = None
    generator_hf_model_id: str | None = None
    diffusion_scheduler: str = "ddim"
    diffusion_steps: int = 15
    diffusion_use_safetensors: bool = True
    diffusion_steering_scale: float = 1.0
    # Continuous per-step classifier-guidance correction inside the frozen
    # decode, in addition to the one-shot trainable latent steering above.
    # 0.0 (default) reproduces the original steering-only behavior exactly.
    diffusion_guidance_scale: float = 0.0
    # Fraction of the trajectory (from t=1 toward t=0) before which guidance
    # is skipped. Early steps mostly settle coarse structure; guiding only
    # the later fraction avoids fighting the model over class-relevant
    # content before it exists, while still catching most of the
    # low-level/textural signal a velocity-style detector keys on.
    diffusion_guidance_start_frac: float = 0.5
    # Anti-collapse: reinitialize the generator (not the ensemble) when its
    # queries have stayed near-constant for too long. Off by default.
    collapse_detection: bool = False
    collapse_pixel_std_threshold: float = 0.05
    collapse_patience_queries: int = 20_000
    collapse_min_queries: int = 5_000


class SteeredDiffusionGenerator(nn.Module):
    """Frozen DDPM decode with a trainable latent steering map for DisGUIDE-D1.

    Optionally also applies classifier-guidance-style detector steering at
    every denoising step (see ``DiffusionLatentGenerator.decode_differentiable``),
    rather than relying solely on the one-shot latent nudge from ``steering``.
    The latent steering map still exists and is still trained -- it is what
    shapes *which region* of the generator's manifold gets explored for
    ensemble disagreement -- but it can only act once, before a frozen
    multi-step decode that has no further opportunity to respond to the
    detector. Per-step guidance corrects the trajectory continuously instead,
    which is the standard way detector/classifier guidance is used in
    diffusion sampling.
    """

    def __init__(
        self,
        *,
        model_id_or_path: str,
        latent_dim: int,
        image_size: int,
        scheduler_name: str,
        num_inference_steps: int,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        device: torch.device,
        use_safetensors: bool = True,
        steering_scale: float = 1.0,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        guidance_scale: float = 0.0,
        guidance_start_frac: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.steering_scale = float(steering_scale)
        self.guidance_fn = guidance_fn
        self.guidance_scale = float(guidance_scale)
        self.guidance_start_frac = float(guidance_start_frac)
        self.steering = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=True),
            nn.Tanh(),
        ).to(device)
        nn.init.zeros_(self.steering[0].weight)
        nn.init.zeros_(self.steering[0].bias)
        self.decoder = DiffusionLatentGenerator(
            model_id_or_path=model_id_or_path,
            latent_dim=self.latent_dim,
            image_size=image_size,
            scheduler_name=scheduler_name,
            num_inference_steps=num_inference_steps,
            device=device,
            use_safetensors=use_safetensors,
        )
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        min_t, max_t = _as_clip_tensors(
            clip_min,
            clip_max,
            device=device,
            dtype=torch.float32,
        )
        self.register_buffer("clip_min", min_t)
        self.register_buffer("clip_max", max_t)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_steered = z + self.steering_scale * self.steering(z)
        images_01 = self.decoder.decode_differentiable(
            z_steered,
            guidance_fn=self.guidance_fn,
            guidance_scale=self.guidance_scale,
            guidance_start_frac=self.guidance_start_frac,
        )
        return self.clip_min + images_01 * (self.clip_max - self.clip_min)


class ProceduralGenerator(nn.Module):
    """Maps latent noise to procedural-natural images (D2)."""

    def __init__(
        self,
        *,
        latent_dim: int,
        channels: int,
        height: int,
        width: int,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        grid_size: int,
        fourier_modes: int,
        num_blobs: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.grid_size = grid_size
        self.fourier_modes = fourier_modes
        self.num_blobs = num_blobs
        self._clip_min = clip_min
        self._clip_max = clip_max
        param_dim = procedural_parameter_dim(
            channels=channels,
            grid_size=grid_size,
            fourier_modes=fourier_modes,
            num_blobs=num_blobs,
        )
        self.mapping = nn.Linear(latent_dim, param_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        params = self.mapping(z)
        return procedural_natural_images(
            params,
            channels=self.channels,
            height=self.height,
            width=self.width,
            clip_min=self._clip_min,
            clip_max=self._clip_max,
            grid_size=self.grid_size,
            fourier_modes=self.fourier_modes,
            num_blobs=self.num_blobs,
        )


class DisguideBypassAttackRunner(DisguideAttackRunner):
    """DisGUIDE loop with ConvGenerator (D1) or ProceduralGenerator (D2)."""

    generator_kind: GeneratorKind = "latent"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._velocity_regularizer: SurrogateVelocityRegularizer | None = None
        # Separate CNF for the C3 typicality term; see
        # SurrogateVelocityRegularizer.composite_score for why the velocity and
        # likelihood components cannot share one checkpoint.
        self._likelihood_regularizer: SurrogateVelocityRegularizer | None = None
        self._distilled_score: DistilledCompositeScore | None = None
        self._benign_score_reference: torch.Tensor | None = None
        self._benign_label_histogram: torch.Tensor | None = None

    def _log(self, message: str, *, enabled: bool) -> None:
        if enabled:
            print(f"[{self.name}] {message}")

    def _build_config(self, context: AttackRunContext) -> DisguideBypassConfig:
        base = super()._build_config(context)
        extra = dict(context.experiment.attack.extra)
        adaptive = AdaptiveAttackConfig.from_extra(extra)
        diffusion_train_steps = extra.get("diffusion_train_steps")
        g_iter = int(diffusion_train_steps) if diffusion_train_steps is not None else base.g_iter
        return DisguideBypassConfig(
            **{
                field.name: (
                    g_iter if field.name == "g_iter" and diffusion_train_steps is not None
                    else getattr(base, field.name)
                )
                for field in DisguideConfig.__dataclass_fields__.values()
            },
            procedural_grid_size=max(2, int(extra.get("procedural_grid_size", 4))),
            procedural_fourier_modes=max(0, int(extra.get("procedural_fourier_modes", 3))),
            procedural_blobs=max(0, int(extra.get("procedural_blobs", 4))),
            adaptive=adaptive if adaptive.is_active() else None,
            generator_hf_model_id=(
                str(extra["generator_hf_model_id"])
                if extra.get("generator_hf_model_id")
                else None
            ),
            diffusion_scheduler=str(extra.get("diffusion_scheduler", "ddim")),
            diffusion_steps=max(1, int(extra.get("diffusion_steps", 15))),
            diffusion_use_safetensors=bool(extra.get("diffusion_use_safetensors", True)),
            diffusion_steering_scale=float(extra.get("diffusion_steering_scale", 1.0)),
            diffusion_guidance_scale=float(extra.get("diffusion_guidance_scale", 0.0)),
            diffusion_guidance_start_frac=float(extra.get("diffusion_guidance_start_frac", 0.5)),
            collapse_detection=bool(extra.get("collapse_detection", False)),
            collapse_pixel_std_threshold=float(extra.get("collapse_pixel_std_threshold", 0.05)),
            collapse_patience_queries=max(0, int(extra.get("collapse_patience_queries", 20_000))),
            collapse_min_queries=max(0, int(extra.get("collapse_min_queries", 5_000))),
        )

    def _build_generator(
        self,
        *,
        config: DisguideBypassConfig,
        channels: int,
        height: int,
        width: int,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        device: torch.device,
    ) -> nn.Module:
        if self.generator_kind == "procedural":
            return ProceduralGenerator(
                latent_dim=config.latent_dim,
                channels=channels,
                height=height,
                width=width,
                clip_min=clip_min,
                clip_max=clip_max,
                grid_size=config.procedural_grid_size,
                fourier_modes=config.procedural_fourier_modes,
                num_blobs=config.procedural_blobs,
            ).to(device)
        if config.generator_hf_model_id:
            guidance_fn = self._build_detector_guidance_fn(config=config, channels=channels)
            return SteeredDiffusionGenerator(
                model_id_or_path=config.generator_hf_model_id,
                latent_dim=config.latent_dim,
                image_size=height,
                scheduler_name=config.diffusion_scheduler,
                num_inference_steps=config.diffusion_steps,
                clip_min=clip_min,
                clip_max=clip_max,
                device=device,
                use_safetensors=config.diffusion_use_safetensors,
                steering_scale=config.diffusion_steering_scale,
                guidance_fn=guidance_fn,
                guidance_scale=config.diffusion_guidance_scale if guidance_fn is not None else 0.0,
                guidance_start_frac=config.diffusion_guidance_start_frac,
            ).to(device)
        return ConvGenerator(
            latent_dim=config.latent_dim,
            channels=channels,
            image_shape=(height, width),
            clip_min=clip_min,
            clip_max=clip_max,
        ).to(device)

    def _build_detector_guidance_fn(
        self, *, config: DisguideBypassConfig, channels: int
    ) -> Callable[[torch.Tensor], torch.Tensor] | None:
        """Build the per-denoising-step guidance callback, or None if unconfigured.

        Reuses the same calibrated velocity regularizer and benign stats as
        the existing C1 loss term (_flow_matching_loss), so guidance and the
        outer loss push in a mathematically consistent direction -- just at
        different points in the generation process (continuously along the
        trajectory here, vs only through the loss gradient at the end there).
        """
        if config.diffusion_guidance_scale <= 0.0:
            return None
        regularizer = self._velocity_regularizer
        adaptive = config.adaptive
        if regularizer is None or adaptive is None or not adaptive.has_benign_stats:
            warnings.warn(
                "diffusion_guidance_scale > 0 but no calibrated surrogate "
                "velocity regularizer is available (need both "
                "surrogate_checkpoint_path and benign_stats); per-step "
                "detector guidance is disabled for this run.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        benign_mean = adaptive.benign_stats.velocity_mean
        benign_std = adaptive.benign_stats.velocity_std

        def guidance_fn(pixels_01: torch.Tensor) -> torch.Tensor:
            # Reconcile the decoder's native channel count with the
            # surrogate CNF's domain (e.g. 3-channel CIFAR-10 output against
            # a 1-channel MNIST-trained CNF). No spatial resize is needed
            # here: SurrogateVelocityRegularizer.prepare() already resizes
            # internally to its own expected resolution.
            native_shape = (int(pixels_01.shape[-2]), int(pixels_01.shape[-1]))
            adapted = self._adapt_to_substitute_domain(
                pixels_01, channels=channels, shape=native_shape
            )
            raw = regularizer.velocity_score(adapted.to(regularizer.device))
            z_score = (raw - benign_mean) / (abs(benign_std) + 1e-8)
            return z_score.to(pixels_01.device)

        return guidance_fn

    def _generate_fake(
        self,
        generator: nn.Module,
        z: torch.Tensor,
        *,
        channels: int,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        output = generator(z)
        if isinstance(output, tuple):
            output = output[0]
        return self._adapt_to_substitute_domain(output, channels=channels, shape=shape)

    def _adapt_to_substitute_domain(
        self,
        fake: torch.Tensor,
        *,
        channels: int,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        """Reconcile the generator's raw output with the substitute ensemble's input domain.

        A frozen external generator (e.g. the pretrained CIFAR-10 diffusion
        prior used by D1) is a *generic* natural-image prior reused unchanged
        across victims, so it is not guaranteed to already emit the same
        channel count or resolution as the victim being attacked. Left
        unadapted, that mismatch reaches the substitute ensemble -- built for
        the victim's own domain -- and crashes (or, for a legacy LeNet-style
        head that flattens with a hardcoded ``view(-1, C*H*W)``, corrupts the
        batch dimension instead of failing loudly). ConvGenerator and
        ProceduralGenerator already build their output in the victim's exact
        domain, and a CIFAR-10 DDPM against a CIFAR-10 victim already matches,
        so this is a no-op in every case that worked before this method
        existed; it only activates for a genuine cross-domain generator such
        as D1 on MNIST.
        """
        adapted = fake
        if adapted.shape[-2:] != shape:
            adapted = F.interpolate(adapted, size=shape, mode="bilinear", align_corners=False)
        source_channels = adapted.shape[1]
        if source_channels != channels:
            if channels == 1:
                adapted = adapted.mean(dim=1, keepdim=True)
            elif source_channels == 1:
                adapted = adapted.repeat(1, channels, 1, 1)
            elif source_channels > channels:
                adapted = adapted[:, :channels]
            else:
                repeats = -(-channels // source_channels)  # ceil division
                adapted = adapted.repeat(1, repeats, 1, 1)[:, :channels]
        return adapted

    def _reinit_generator(
        self,
        generator: nn.Module,
        optimizer_generator: torch.optim.Optimizer,
        *,
        config: DisguideBypassConfig,
        channels: int,
        height: int,
        width: int,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        device: torch.device,
    ) -> None:
        """Reinitialize the generator's weights in place after a collapse.

        Rebuilds a fresh generator of the same kind/shape and copies its
        (freshly initialized) weights into the existing module, so the caller
        keeps its reference and the ensemble/replay buffer/optimizer objects
        stay untouched. The generator optimizer's Adam moments are cleared
        too, since they were fit to the collapsed trajectory and would
        otherwise immediately pull the fresh weights back toward it.
        """
        fresh = self._build_generator(
            config=config,
            channels=channels,
            height=height,
            width=width,
            clip_min=clip_min,
            clip_max=clip_max,
            device=device,
        )
        generator.load_state_dict(fresh.state_dict())
        optimizer_generator.state.clear()

    def _build_testset(self, context: AttackRunContext, *, image_size: int):
        spec = context.experiment
        dataset_name = spec.dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
        return legacy_datasets.__dict__[dataset_name](
            train=False,
            transform=transform,
            download=spec.dataset.download,
        )

    def _prepare_teacher_inputs(
        self,
        fake: torch.Tensor,
        *,
        dataset_name: str,
        victim_shape: tuple[int, int],
    ) -> torch.Tensor:
        teacher_inputs = fake
        if teacher_inputs.shape[-2:] != victim_shape:
            teacher_inputs = F.interpolate(
                teacher_inputs,
                size=victim_shape,
                mode="bilinear",
                align_corners=False,
            )
        return teacher_inputs

    def _denormalize_to_pixel_space(self, batch: torch.Tensor, dataset_name: str) -> torch.Tensor:
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        family_key = "imagenet" if family == "tinyimagenet" else family
        mean_std = legacy_datasets.modelfamily_to_mean_std[family_key]
        mean = torch.tensor(mean_std["mean"], device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
        std = torch.tensor(mean_std["std"], device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
        return torch.clamp(batch * std + mean, 0.0, 1.0)

    def _flow_matching_loss(
        self,
        fake: torch.Tensor,
        *,
        config: DisguideBypassConfig,
        dataset_name: str,
        queries: int = 0,
    ) -> torch.Tensor:
        """Component C1 term: penalty on the surrogate ``t=0`` velocity score.

        Uses the standardized hinge whenever benign statistics are available,
        either the explicit ``detector_benign_*`` values or the calibrated
        ``benign_stats`` the FlowGuard++-adaptive attackers carry, so C1 has the
        same threshold-agnostic form as the C2/C3 terms. Without any
        calibration it falls back to the raw mean score, which is the original
        D1/D2 behavior.

        Two anti-collapse knobs apply here (see ``AdaptiveAttackConfig``): the
        weight is scaled by ``adaptive.weight_scale(queries)`` for the
        warm-up/ramp schedule, and ``detector_penalty_mode="batch_quantile"``
        hinges only the batch's worst quantile instead of every sample, so a
        diverse batch is not punished for having any high-scoring outliers.
        """
        adaptive = config.adaptive
        reg = self._velocity_regularizer
        if adaptive is None or reg is None or adaptive.velocity_regularizer_weight <= 0.0:
            return torch.tensor(0.0, device=fake.device)
        weight_scale = adaptive.weight_scale(queries)
        if weight_scale <= 0.0:
            return torch.tensor(0.0, device=fake.device)
        pixel = self._denormalize_to_pixel_space(fake, dataset_name)
        scores = reg.velocity_score(pixel.to(reg.device))
        use_batch_quantile = adaptive.detector_penalty_mode == "batch_quantile"
        penalty_fn = batch_quantile_detector_penalty if use_batch_quantile else normalized_detector_penalty
        penalty_kwargs = (
            {"batch_quantile": adaptive.detector_batch_quantile} if use_batch_quantile else {}
        )
        if adaptive.detector_benign_mean is not None and adaptive.detector_benign_std is not None:
            penalty = penalty_fn(
                scores,
                benign_mean=adaptive.detector_benign_mean,
                benign_std=adaptive.detector_benign_std,
                benign_q95_norm=adaptive.detector_benign_q95_norm,
                **penalty_kwargs,
            )
        elif adaptive.has_benign_stats:
            penalty = penalty_fn(
                scores,
                benign_mean=adaptive.benign_stats.velocity_mean,
                benign_std=adaptive.benign_stats.velocity_std,
                benign_q95_norm=adaptive.benign_stats.velocity_band,
                **penalty_kwargs,
            )
        else:
            if "c1" not in _WARNED_RAW_PENALTY:
                _WARNED_RAW_PENALTY.add("c1")
                warnings.warn(
                    "C1 generator penalty is falling back to the raw "
                    "mean ||v(0, x)||^2 because no benign velocity statistics "
                    "were supplied. Benign scores on the CIFAR-10 FlowPure CNF "
                    "are ~2.5e-3, so at lambda=1 this term is orders of "
                    "magnitude below the extraction loss and the attack is "
                    "effectively non-adaptive to FlowPure. Pass 'benign_stats' "
                    "(or 'detector_benign_mean'/'detector_benign_std') to get "
                    "the benign-normalized hinge, or scale "
                    "velocity_regularizer_weight to the raw score magnitude.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            penalty = scores.mean()
        return weight_scale * adaptive.velocity_regularizer_weight * penalty.to(fake.device)

    def _composite_detector_loss(
        self,
        fake: torch.Tensor,
        *,
        config: DisguideBypassConfig,
        dataset_name: str,
        probabilities: torch.Tensor | None = None,
        queries: int = 0,
    ) -> torch.Tensor:
        """FlowGuard++-aware terms of the generator objective.

        Implements the remaining terms of

            L_G = L_dis + l1*z0 + l2*z_int + l3*s_typ + l4*T_state + l5*T_label

        on the attacker's own surrogate CNF, with each term switched on by its
        weight. ``l1`` is handled by :meth:`_flow_matching_loss`.

        The per-component hinges and the fused-risk hinge are *both* applied
        when their weights are set, because the deployed detector uses the two
        differently: it blocks on an OR over per-component thresholds and ranks
        on the fused risk. An attacker that only pushed the fused risk down
        could still trip an individual component threshold, so suppressing both
        is what "adaptive to the composite" has to mean here. Components are
        computed once and shared between the two hinges, so covering both costs
        no extra ODE solves.
        """
        adaptive = config.adaptive
        reg = self._velocity_regularizer
        zero = torch.zeros((), device=fake.device, dtype=fake.dtype)
        if adaptive is None or not adaptive.uses_composite_objective():
            return zero
        if reg is None and self._distilled_score is None:
            return zero
        # Warm-up/ramp schedule (see AdaptiveAttackConfig.weight_scale): applied
        # uniformly to every term below, so the whole composite objective comes
        # online gradually rather than constraining the generator from step 0.
        weight_scale = adaptive.weight_scale(queries)
        if weight_scale <= 0.0:
            return zero
        use_batch_quantile = adaptive.detector_penalty_mode == "batch_quantile"
        batch_quantile = adaptive.detector_batch_quantile

        score_device = (
            reg.device
            if reg is not None
            else next(self._distilled_score.parameters()).device
        )
        pixel = self._denormalize_to_pixel_space(fake, dataset_name).to(score_device)
        stats = adaptive.benign_stats
        loss = zero

        needs_integral = (
            adaptive.integral_regularizer_weight > 0.0
            or adaptive.composite_regularizer_weight > 0.0
            or adaptive.distribution_regularizer_weight > 0.0
        )
        needs_typicality = (
            adaptive.likelihood_regularizer_weight > 0.0
            or adaptive.composite_regularizer_weight > 0.0
        )

        integral_scores: torch.Tensor | None = None
        typicality_scores: torch.Tensor | None = None
        risk: torch.Tensor | None = None

        if self._distilled_score is not None:
            # Cheap path: one forward/backward through the distilled regressor
            # instead of a double-backward per ODE step. Used for the
            # high-budget runs where the exact path is not affordable. The head
            # already predicts |z| for the likelihood, so the two-sided nature
            # of the typicality score is baked into the regression target.
            components = self._distilled_score.components(pixel)
            integral_scores = components["integral"]
            typicality_scores = components["typicality"]
            if adaptive.composite_regularizer_weight > 0.0:
                risk = self._distilled_score.risk(pixel)
        else:
            if adaptive.composite_regularizer_weight > 0.0:
                risk, components = reg.composite_score(
                    pixel,
                    stats=stats,
                    weights=adaptive.composite_weights,
                    integral_num_steps=adaptive.integral_num_steps,
                    likelihood_num_steps=adaptive.likelihood_num_steps,
                    hutchinson_samples=adaptive.likelihood_hutchinson_samples,
                    likelihood_regularizer=self._likelihood_regularizer,
                )
                integral_scores = components.get("integral")
                typicality_scores = components.get("likelihood")
            else:
                if needs_integral:
                    integral = reg.trajectory_integral_score(
                        pixel, num_steps=adaptive.integral_num_steps
                    )
                    integral_scores = (integral - stats.integral_mean) / (
                        abs(stats.integral_std) + 1e-8
                    )
                if needs_typicality:
                    log_likelihood = (self._likelihood_regularizer or reg).log_likelihood(
                        pixel,
                        num_steps=adaptive.likelihood_num_steps,
                        hutchinson_samples=adaptive.likelihood_hutchinson_samples,
                    )
                    typicality_scores = (
                        log_likelihood - stats.likelihood_mean
                    ).abs() / (abs(stats.likelihood_std) + 1e-8)

        # Fused-risk hinge: what the defender ranks on.
        if risk is not None:
            loss = loss + adaptive.composite_regularizer_weight * torch.relu(
                risk - stats.composite_band
            ).pow(2).mean().to(fake.device)

        # Per-component hinges: what the defender blocks on. In batch_quantile
        # mode only the batch's worst quantile is hinged (see
        # batch_quantile_detector_penalty), leaving the rest of the batch free
        # to vary instead of pulling every sample toward the same safe point.
        if adaptive.integral_regularizer_weight > 0.0 and integral_scores is not None:
            if use_batch_quantile:
                integral_penalty = torch.relu(
                    torch.quantile(integral_scores, batch_quantile) - stats.integral_band
                ).pow(2)
            else:
                integral_penalty = torch.relu(
                    integral_scores - stats.integral_band
                ).pow(2).mean()
            loss = loss + adaptive.integral_regularizer_weight * integral_penalty.to(fake.device)
        if adaptive.likelihood_regularizer_weight > 0.0 and typicality_scores is not None:
            if use_batch_quantile:
                typicality_penalty = torch.relu(
                    torch.quantile(typicality_scores, batch_quantile) - stats.typicality_band
                ).pow(2)
            else:
                typicality_penalty = torch.relu(
                    typicality_scores - stats.typicality_band
                ).pow(2).mean()
            loss = loss + adaptive.likelihood_regularizer_weight * typicality_penalty.to(fake.device)

        # C4: keep the batch's score *distribution* close to benign so the
        # defender's windowed two-sample test has nothing to fire on.
        if adaptive.distribution_regularizer_weight > 0.0 and self._benign_score_reference is not None:
            if integral_scores is not None:
                batch_scores = integral_scores
            elif reg is not None:
                velocity = reg.velocity_score(pixel)
                batch_scores = (velocity - stats.velocity_mean) / (
                    abs(stats.velocity_std) + 1e-8
                )
            else:
                batch_scores = None
            if batch_scores is not None:
                loss = loss + adaptive.distribution_regularizer_weight * distribution_matching_penalty(
                    batch_scores,
                    self._benign_score_reference,
                ).to(fake.device)

        # C5: match the benign label/entropy profile of the predictions the
        # queries induce, using the substitute ensemble as a victim proxy.
        if adaptive.label_regularizer_weight > 0.0 and probabilities is not None:
            loss = loss + adaptive.label_regularizer_weight * label_distribution_penalty(
                probabilities,
                benign_label_histogram=self._benign_label_histogram,
                benign_entropy_mean=stats.entropy_mean,
                benign_entropy_std=stats.entropy_std,
            ).to(fake.device)

        return weight_scale * loss

    def _query_projection(self, batch: torch.Tensor, config: DisguideBypassConfig) -> torch.Tensor:
        adaptive = config.adaptive
        if adaptive is None or adaptive.projection.name == "none":
            return batch
        return apply_query_projection(batch, adaptive.projection)

    def _ensemble_probabilities(
        self, ensemble: SubstituteEnsemble, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Mean soft vote of the substitute ensemble (proxy for victim answers)."""
        logits = torch.stack(
            [ensemble(inputs, idx=index) for index in range(ensemble.size())], dim=1
        )
        return F.softmax(logits, dim=2).mean(dim=1)

    def _disagreement_loss_with_flow(
        self,
        ensemble: SubstituteEnsemble,
        fake: torch.Tensor,
        *,
        config: DisguideBypassConfig,
        dataset_name: str,
        queries: int = 0,
    ) -> torch.Tensor:
        loss_g = self._disagreement_loss(ensemble, fake, lambda_div=config.lambda_div)
        loss_g = loss_g + self._flow_matching_loss(
            fake, config=config, dataset_name=dataset_name, queries=queries
        )
        adaptive = config.adaptive
        if adaptive is not None and adaptive.uses_composite_objective():
            probabilities = (
                self._ensemble_probabilities(ensemble, fake)
                if adaptive.label_regularizer_weight > 0.0
                else None
            )
            loss_g = loss_g + self._composite_detector_loss(
                fake,
                config=config,
                dataset_name=dataset_name,
                probabilities=probabilities,
                queries=queries,
            )
        return loss_g

    def _softmax_entropy_list(self, probabilities: torch.Tensor) -> list[float]:
        probs = probabilities.clamp_min(1e-8)
        entropy = -torch.sum(probs * probs.log(), dim=1)
        return [float(value) for value in entropy.detach().cpu()]

    def _flowpure_metadata_for_batch(self, batch_size: int) -> tuple[list[bool], list[float]]:
        if not self.query_engine.history.records:
            return [False] * batch_size, [0.0] * batch_size
        metadata = self.query_engine.history.records[-1].metadata
        raw_blocked = metadata.get("flowpure_blocked") or metadata.get("flow_matching_blocked") or []
        raw_scores = (
            metadata.get("flowpure_scores")
            or metadata.get("flowguard_integral_scores")
            or metadata.get("queries_blocked")
            or []
        )
        blocked = [
            bool(raw_blocked[index]) if index < len(raw_blocked) else False
            for index in range(batch_size)
        ]
        scores = [
            float(raw_scores[index]) if index < len(raw_scores) else 0.0
            for index in range(batch_size)
        ]
        return blocked, scores

    def _maybe_append_disguide_trace(
        self,
        context: AttackRunContext,
        *,
        step: int,
        epoch: int,
        queries: int,
        generator_loss: float,
        query_batch: torch.Tensor,
        victim_softmax: torch.Tensor,
        surrogate_accuracy: float | None = None,
    ) -> None:
        trace = context.metadata.get("disguide_trace")
        if not isinstance(trace, list):
            return
        max_images = max(1, int(context.metadata.get("disguide_trace_max_images", 8)))
        blocked, scores = self._flowpure_metadata_for_batch(len(query_batch))
        entropies = self._softmax_entropy_list(victim_softmax)
        trace.append(
            DisguideTraceSnapshot(
                step=int(step),
                epoch=int(epoch),
                queries=int(queries),
                generator_loss=float(generator_loss),
                query_images=[
                    query_batch[index].detach().cpu().float().numpy()
                    for index in range(min(len(query_batch), max_images))
                ],
                query_entropy=entropies[:max_images],
                detector_scores=scores[:max_images],
                query_blocked=blocked[:max_images],
                surrogate_accuracy=surrogate_accuracy,
            )
        )

    # -- mid-run checkpointing --------------------------------------------
    #
    # High-budget adaptive runs (20M queries) do not fit in a single Slurm
    # allocation. The matrix harness resumes at (defense, attack) granularity,
    # which is useless here: a cell that cannot finish inside the wall clock
    # would restart from zero forever. These helpers snapshot the full training
    # state so an interrupted cell continues where it stopped.

    @staticmethod
    def _checkpoint_path(output_dir: Path) -> Path:
        return output_dir / "attack_checkpoint.pt"

    def _save_attack_checkpoint(
        self,
        path: Path,
        *,
        generator: nn.Module,
        ensemble: SubstituteEnsemble,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_student: torch.optim.Optimizer,
        generator_scheduler,
        student_scheduler,
        replay_buffer: _ReplayBuffer,
        replay_records: list,
        metrics: list[dict[str, float]],
        epoch: int,
        global_step: int,
        queries_used: int,
        query_index: int,
    ) -> None:
        payload = {
            "generator": generator.state_dict(),
            "ensemble": ensemble.state_dict(),
            "optimizer_generator": optimizer_generator.state_dict(),
            "optimizer_student": optimizer_student.state_dict(),
            "generator_scheduler": (
                generator_scheduler.state_dict() if generator_scheduler is not None else None
            ),
            "student_scheduler": (
                student_scheduler.state_dict() if student_scheduler is not None else None
            ),
            "replay_buffer": {
                "inputs": replay_buffer._inputs,
                "labels": replay_buffer._labels,
                "size": replay_buffer._size,
                "head": replay_buffer._head,
            },
            "replay_records": list(replay_records),
            "metrics": list(metrics),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "queries_used": int(queries_used),
            "query_index": int(query_index),
            # Stored on CPU so a reload with map_location=cuda cannot hand
            # set_rng_state_all a GPU tensor, which it rejects.
            "torch_rng_state": torch.get_rng_state().cpu(),
            "cuda_rng_state": (
                [state.cpu() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None
            ),
        }
        # Write to a temporary file and rename, so a job killed mid-write
        # leaves the previous checkpoint intact rather than a truncated one.
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _load_attack_checkpoint(
        self,
        path: Path,
        *,
        generator: nn.Module,
        ensemble: SubstituteEnsemble,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_student: torch.optim.Optimizer,
        generator_scheduler,
        student_scheduler,
        replay_buffer: _ReplayBuffer,
        device: torch.device,
    ) -> dict:
        payload = torch.load(path, map_location=device, weights_only=False)
        generator.load_state_dict(payload["generator"])
        ensemble.load_state_dict(payload["ensemble"])
        optimizer_generator.load_state_dict(payload["optimizer_generator"])
        optimizer_student.load_state_dict(payload["optimizer_student"])
        if generator_scheduler is not None and payload.get("generator_scheduler"):
            generator_scheduler.load_state_dict(payload["generator_scheduler"])
        if student_scheduler is not None and payload.get("student_scheduler"):
            student_scheduler.load_state_dict(payload["student_scheduler"])

        buffer_state = payload.get("replay_buffer") or {}
        replay_buffer._inputs = buffer_state.get("inputs")
        replay_buffer._labels = buffer_state.get("labels")
        replay_buffer._size = int(buffer_state.get("size", 0))
        replay_buffer._head = int(buffer_state.get("head", 0))

        # RNG restore is a reproducibility nicety, not correctness-critical: the
        # generator, ensemble, optimizers and replay buffer above are what carry
        # the run. It is therefore never allowed to abort a resume. It previously
        # did exactly that -- torch.load(map_location=cuda) puts the saved CUDA
        # RNG states on the GPU, while set_rng_state_all requires CPU
        # ByteTensors, so every resume died with "RNG state must be a
        # torch.ByteTensor" and the caller's cleanup then deleted the checkpoint.
        try:
            torch_state = payload.get("torch_rng_state")
            if torch_state is not None:
                torch.set_rng_state(torch_state.cpu().to(torch.uint8))
            cuda_state = payload.get("cuda_rng_state")
            if cuda_state is not None and torch.cuda.is_available():
                states = [state.cpu().to(torch.uint8) for state in cuda_state]
                # A checkpoint written on a multi-GPU node must not break a
                # resume on a single-GPU one.
                available = torch.cuda.device_count()
                if len(states) > available:
                    states = states[:available]
                if states:
                    torch.cuda.set_rng_state_all(states)
        except Exception as error:  # noqa: BLE001 - never lose a run over RNG.
            warnings.warn(
                f"Could not restore RNG state from {path}: {error!r}. Resuming "
                "with fresh RNG; the restored model/optimizer state is intact, "
                "so only exact sample-for-sample reproducibility is affected.",
                RuntimeWarning,
                stacklevel=2,
            )
        return payload

    def _maybe_snapshot_queries(
        self,
        snapshots: list[dict[str, Any]],
        *,
        milestones: list[int],
        queries: int,
        query_batch: torch.Tensor,
        dataset_name: str,
        max_images: int,
    ) -> None:
        """Record the queries in flight the first time each milestone is passed.

        Captures what the victim actually receives, in ``[0, 1]`` pixel space, so
        a figure can show how the generator's output evolves over the query
        budget. Milestones are consumed in order and each fires once; the batch
        that crosses one is kept whole rather than interpolating, so the images
        are real queries rather than a reconstruction.
        """
        if not milestones or query_batch.numel() == 0:
            return
        while milestones and queries >= milestones[0]:
            milestone = milestones.pop(0)
            images = self._denormalize_to_pixel_space(
                query_batch[: max(1, int(max_images))].detach(), dataset_name
            )
            snapshots.append(
                {
                    "milestone": int(milestone),
                    "queries": int(queries),
                    "images": images.cpu().float().numpy(),
                }
            )

    def _save_disguide_trace(self, context: AttackRunContext, output_dir: Path) -> str | None:
        trace = context.metadata.get("disguide_trace")
        if not isinstance(trace, list) or not trace:
            return None
        trace_path = output_dir / "disguide_trace.pt"
        torch.save(trace, trace_path)
        return str(trace_path)

    def run(self, context: AttackRunContext) -> AttackRunResult:
        spec = context.experiment
        config = self._build_config(context)
        self.history_records = []
        output_dir = Path(context.output_dir or Path("runs") / spec.name / self.name)
        output_dir.mkdir(parents=True, exist_ok=True)

        victim_testset = self._build_victim_testset(context)
        victim_sample = victim_testset[0][0]
        _, victim_height, victim_width = victim_sample.shape
        victim_shape = (victim_height, victim_width)

        channels = victim_sample.shape[0]
        image_height = int(spec.attack.extra.get("image_size", victim_height))
        image_width = image_height
        clip_min, clip_max = self._normalized_clip_values(spec.dataset.name)
        attack_device = torch.device(spec.substitute_model.device)
        num_classes = self.query_engine.target_service.num_classes

        adaptive = config.adaptive
        self._velocity_regularizer = None
        self._likelihood_regularizer = None
        self._distilled_score = None
        self._benign_score_reference = None
        self._benign_label_histogram = None
        if adaptive is not None and adaptive.distilled_surrogate_path:
            self._distilled_score = DistilledCompositeScore.load(
                adaptive.distilled_surrogate_path,
                device=adaptive.velocity_regularizer_device,
            )
        if adaptive is not None and adaptive.surrogate_checkpoint_path and (
            adaptive.velocity_regularizer_weight > 0.0 or adaptive.uses_composite_objective()
        ):
            self._velocity_regularizer = SurrogateVelocityRegularizer(
                adaptive.surrogate_checkpoint_path,
                device=adaptive.velocity_regularizer_device,
            )
            # Load a second CNF for the typicality term only when it differs;
            # otherwise reuse the first so the common single-checkpoint case
            # costs no extra memory.
            likelihood_path = adaptive.likelihood_surrogate_checkpoint_path
            if likelihood_path and str(likelihood_path) != str(
                adaptive.surrogate_checkpoint_path
            ):
                self._likelihood_regularizer = SurrogateVelocityRegularizer(
                    str(likelihood_path),
                    device=adaptive.velocity_regularizer_device,
                )
            if adaptive.benign_reference_scores:
                self._benign_score_reference = torch.tensor(
                    adaptive.benign_reference_scores,
                    dtype=torch.float32,
                    device=attack_device,
                )
            if adaptive.benign_label_histogram:
                self._benign_label_histogram = torch.tensor(
                    adaptive.benign_label_histogram,
                    dtype=torch.float32,
                    device=attack_device,
                )

        ensemble = self._create_substitute_ensemble(context, num_classes, config.ensemble_size)
        generator = self._build_generator(
            config=config,
            channels=channels,
            height=image_height,
            width=image_width,
            clip_min=clip_min,
            clip_max=clip_max,
            device=attack_device,
        )
        blackbox = self._build_blackbox(context)
        testset = self._build_testset(context, image_size=image_height)

        collapse_monitor = (
            GeneratorCollapseMonitor(
                pixel_std_threshold=config.collapse_pixel_std_threshold,
                patience_queries=config.collapse_patience_queries,
                min_queries=config.collapse_min_queries,
            )
            if config.collapse_detection
            else None
        )

        target_accuracy = _evaluate_accuracy(
            self.query_engine.target_service.model,
            victim_testset,
            batch_size=max(1, spec.training.batch_size),
            device=self.query_engine.target_service.device,
        )

        # Query-evolution snapshots (opt-in via the attack recipe). Sorted and
        # de-duplicated so the milestone list is consumed monotonically.
        snapshots: list[dict[str, Any]] = []
        snapshot_milestones = sorted(
            {
                max(1, int(value))
                for value in spec.attack.extra.get("query_snapshot_milestones", []) or []
            }
        )
        snapshot_images = max(1, int(spec.attack.extra.get("query_snapshot_images", 8)))

        plateau_stopper = _PlateauStopper(
            patience_queries=config.plateau_patience_queries,
            min_delta=config.plateau_min_delta,
            min_queries=config.plateau_min_queries,
            smoothing_window=config.plateau_smoothing_window,
        )
        early_stop_reason: str | None = None
        stop_reason: str | None = None

        cost_per_iteration = spec.attack.batch_size * config.d_iter
        number_epochs = max(
            1,
            spec.attack.query_budget // max(cost_per_iteration * config.epoch_itrs, 1) + 1,
        )

        optimizer_student = torch.optim.SGD(
            ensemble.parameters(),
            lr=config.lr_student,
            weight_decay=5e-4,
            momentum=0.9,
        )
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=config.lr_generator)

        steps = sorted([int(s * number_epochs) for s in config.scheduler_steps])
        student_scheduler = None
        generator_scheduler = None
        if config.scheduler == "multistep":
            student_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_student, steps, config.scheduler_scale,
            )
            generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_generator, steps, config.scheduler_scale,
            )
        elif config.scheduler == "cosine":
            student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_student, max(1, number_epochs),
            )
            generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_generator, max(1, number_epochs),
            )

        replay_buffer = _ReplayBuffer(
            max_size=max(config.replay_size, spec.attack.batch_size),
            batch_size=spec.attack.batch_size,
            device=attack_device,
        )

        # Bounded transfer-record store. Each record carries the query, a copy
        # of it, and two label vectors, so an uncapped 20M-query run needs
        # hundreds of GB. The cap keeps the most recent records, which are the
        # ones produced by the converged generator.
        replay_records: deque[dict] | list[dict] = (
            deque(maxlen=config.artifact_sample_size)
            if config.artifact_sample_size > 0
            else []
        )
        metrics: list[dict[str, float]] = []
        # Tracked separately from len(replay_records): once the deque saturates
        # its length stops growing, but query indices must stay monotonic.
        query_index = 0

        flow_weight = 0.0 if adaptive is None else adaptive.velocity_regularizer_weight
        adaptivity = "none" if adaptive is None else adaptive.adaptivity_label()
        self._log(
            (
                f"Starting DisGUIDE-style loop budget={spec.attack.query_budget}, "
                f"generator={self.generator_kind}, ensemble={config.ensemble_size}, "
                f"batch={spec.attack.batch_size}, g_iter={config.g_iter}, d_iter={config.d_iter}, "
                f"rep_iter={config.rep_iter}, lambda_div={config.lambda_div}, flow_weight={flow_weight}, "
                f"adaptive_to={adaptivity}"
            ),
            enabled=config.verbose,
        )

        stop_requested = False
        global_step = 0
        start_epoch = 1
        checkpoint_path = self._checkpoint_path(output_dir)
        last_checkpoint_queries = 0

        if config.resume_from_checkpoint and checkpoint_path.exists():
            payload = self._load_attack_checkpoint(
                checkpoint_path,
                generator=generator,
                ensemble=ensemble,
                optimizer_generator=optimizer_generator,
                optimizer_student=optimizer_student,
                generator_scheduler=generator_scheduler,
                student_scheduler=student_scheduler,
                replay_buffer=replay_buffer,
                device=attack_device,
            )
            replay_records.extend(payload.get("replay_records", []))
            metrics.extend(payload.get("metrics", []))
            global_step = int(payload.get("global_step", 0))
            query_index = int(payload.get("query_index", 0))
            start_epoch = max(1, int(payload.get("epoch", 1)))
            # Restoring the spent-query count is what makes the budget
            # accounting continue rather than restart: _remaining_budget reads
            # blackbox.call_count.
            blackbox.call_count = int(payload.get("queries_used", 0))
            last_checkpoint_queries = int(blackbox.call_count)
            self._log(
                f"Resumed from {checkpoint_path}: epoch={start_epoch} "
                f"queries={blackbox.call_count}/{spec.attack.query_budget}",
                enabled=config.verbose,
            )

        # Rate and ETA cover only this process, so a resumed job does not
        # count the queries restored from the checkpoint as its own work.
        loop_started = time.perf_counter()
        loop_start_queries = int(blackbox.call_count)

        for epoch in range(start_epoch, number_epochs + 1):
            if stop_requested:
                break

            progress = tqdm(
                range(config.epoch_itrs),
                ncols=80,
                disable=not config.progress_enabled,
                leave=False,
                desc=f"Epoch {epoch}/{number_epochs}",
            )

            for _iteration in progress:
                if self._remaining_budget(context, blackbox) < cost_per_iteration:
                    stop_requested = True
                    break

                generator.train()
                ensemble.eval()
                g_loss_sum = 0.0
                for _ in range(config.g_iter):
                    optimizer_generator.zero_grad()
                    z = torch.randn(spec.attack.batch_size, config.latent_dim, device=attack_device)
                    fake = self._generate_fake(generator, z, channels=channels, shape=victim_shape)
                    loss_g = self._disagreement_loss_with_flow(
                        ensemble,
                        fake,
                        config=config,
                        dataset_name=spec.dataset.name,
                        queries=int(blackbox.call_count),
                    )
                    loss_g.backward()
                    optimizer_generator.step()
                    g_loss_sum += loss_g.item()

                generator.eval()
                ensemble.train()
                s_loss_sum = 0.0
                last_fake: torch.Tensor | None = None
                last_original: torch.Tensor | None = None

                for _ in range(config.d_iter):
                    optimizer_student.zero_grad()
                    with torch.no_grad():
                        z = torch.randn(spec.attack.batch_size, config.latent_dim, device=attack_device)
                        fake = self._generate_fake(
                            generator, z, channels=channels, shape=victim_shape
                        ).detach()

                    query_batch = self._query_projection(fake, config)
                    teacher_inputs = self._prepare_teacher_inputs(
                        query_batch,
                        dataset_name=spec.dataset.name,
                        victim_shape=victim_shape,
                    )
                    queried_inputs, defended_outputs, original_outputs = self._query_target(
                        context, blackbox, teacher_inputs, return_origin=True,
                    )
                    if len(queried_inputs) == 0:
                        stop_requested = True
                        break

                    fake = fake[: len(queried_inputs)]
                    original_outputs = original_outputs[: len(queried_inputs)]
                    teacher_logits = self._postprocess_teacher_logits(defended_outputs, config)
                    replay_buffer.update(fake, teacher_logits)
                    last_fake = fake
                    last_original = original_outputs

                    if collapse_monitor is not None:
                        pixel_std = float(fake.detach().float().std())
                        if collapse_monitor.observe(
                            pixel_std=pixel_std, queries=int(blackbox.call_count)
                        ):
                            self._log(
                                f"Generator collapse detected (pixel_std={pixel_std:.4f} "
                                f"at {int(blackbox.call_count)} queries) -- reinitializing "
                                f"generator (reinit #{collapse_monitor.reinit_count})",
                                enabled=config.verbose,
                            )
                            self._reinit_generator(
                                generator,
                                optimizer_generator,
                                config=config,
                                channels=channels,
                                height=image_height,
                                width=image_width,
                                clip_min=clip_min,
                                clip_max=clip_max,
                                device=attack_device,
                            )

                    for idx in range(ensemble.size()):
                        s_logit = ensemble(fake, idx=idx)
                        loss_s = _student_loss(s_logit, teacher_logits, config.loss_type)
                        loss_s.backward()
                        s_loss_sum += loss_s.item()
                    optimizer_student.step()

                    replay_records.extend(
                        self._records_from_batch(
                            inputs=fake,
                            labels=teacher_logits,
                            original_labels=original_outputs,
                            starting_query_index=query_index,
                        )
                    )
                    query_index += len(fake)
                    self._maybe_snapshot_queries(
                        snapshots,
                        milestones=snapshot_milestones,
                        queries=int(blackbox.call_count),
                        query_batch=query_batch[: len(queried_inputs)],
                        dataset_name=spec.dataset.name,
                        max_images=snapshot_images,
                    )

                if stop_requested:
                    break

                if last_fake is not None and last_original is not None:
                    global_step += 1
                    self._maybe_append_disguide_trace(
                        context,
                        step=global_step,
                        epoch=epoch,
                        queries=int(blackbox.call_count),
                        generator_loss=g_loss_sum / max(config.g_iter, 1),
                        query_batch=last_fake,
                        victim_softmax=last_original,
                    )

                if replay_buffer.can_sample():
                    for _ in range(config.rep_iter):
                        optimizer_student.zero_grad()
                        rep_inputs, rep_labels = replay_buffer.sample()
                        for idx in range(ensemble.size()):
                            s_logit = ensemble(rep_inputs, idx=idx)
                            loss_s = _student_loss(s_logit, rep_labels, config.loss_type)
                            loss_s.backward()
                        optimizer_student.step()

                total_queries = int(blackbox.call_count)
                progress.set_description(
                    f"queries={total_queries} g={g_loss_sum / max(config.g_iter, 1):.3f}"
                )
                # Printed regardless of --verbose: the tqdm bar and the
                # per-epoch line are both off in Slurm runs, so without this
                # a monitor cannot tell progress from a hang via `tail -n1`.
                if global_step % config.log_interval == 0:
                    elapsed = time.perf_counter() - loop_started
                    rate = (total_queries - loop_start_queries) / max(elapsed, 1e-9)
                    remaining = max(0, int(spec.attack.query_budget) - total_queries)
                    eta = _format_duration(remaining / rate) if rate > 0 else "?"
                    print(
                        f"[{self.name}] progress queries={total_queries}/"
                        f"{spec.attack.query_budget} "
                        f"({100.0 * total_queries / max(spec.attack.query_budget, 1):.1f}%) "
                        f"epoch={epoch}/{number_epochs} elapsed={_format_duration(elapsed)} "
                        f"rate={rate:.1f}q/s eta={eta}",
                        flush=True,
                    )

                if (
                    config.checkpoint_every_queries > 0
                    and total_queries - last_checkpoint_queries
                    >= config.checkpoint_every_queries
                ):
                    self._save_attack_checkpoint(
                        checkpoint_path,
                        generator=generator,
                        ensemble=ensemble,
                        optimizer_generator=optimizer_generator,
                        optimizer_student=optimizer_student,
                        generator_scheduler=generator_scheduler,
                        student_scheduler=student_scheduler,
                        replay_buffer=replay_buffer,
                        replay_records=replay_records,
                        metrics=metrics,
                        # The in-progress epoch, resumed as-is. Its scheduler
                        # step happens at the end of the epoch and has not run
                        # yet, so re-entering it is consistent; the query budget
                        # (not the epoch counter) decides when the run stops.
                        epoch=epoch,
                        global_step=global_step,
                        queries_used=total_queries,
                        query_index=query_index,
                    )
                    last_checkpoint_queries = total_queries
                    self._log(
                        f"Checkpointed at {total_queries} queries -> {checkpoint_path}",
                        enabled=config.verbose,
                    )

            if student_scheduler is not None:
                student_scheduler.step()
            if generator_scheduler is not None:
                generator_scheduler.step()

            surrogate_accuracy = _evaluate_accuracy(
                ensemble,
                testset,
                batch_size=max(1, spec.training.batch_size),
                device=attack_device,
            )
            # Fidelity alongside accuracy so the per-epoch history doubles as a
            # queries-to-fidelity curve; the runbook reads the two together to
            # tell "detection stopped a working attack" from "the adaptive
            # penalties broke the attack on their own".
            surrogate_fidelity = _evaluate_fidelity(
                ensemble,
                self.query_engine.target_service.model,
                victim_testset,
                batch_size=max(1, spec.training.batch_size),
                device=attack_device,
                victim_device=self.query_engine.target_service.device,
            )
            metrics.append({
                "epoch": float(epoch),
                "queries": float(blackbox.call_count),
                "surrogate_accuracy": float(surrogate_accuracy),
                "surrogate_fidelity": float(surrogate_fidelity),
                "normalized_accuracy": float(surrogate_accuracy / max(target_accuracy, 1e-8)),
            })
            self._log(
                f"Epoch {epoch}: queries={int(blackbox.call_count)} "
                f"sur_acc={surrogate_accuracy:.2f}% sur_fid={surrogate_fidelity:.2f}%",
                enabled=config.verbose,
            )
            stop_reason = plateau_stopper.update(
                value=float(surrogate_fidelity), queries=int(blackbox.call_count)
            )
            trace = context.metadata.get("disguide_trace")
            if isinstance(trace, list) and trace:
                trace[-1] = DisguideTraceSnapshot(
                    step=trace[-1].step,
                    epoch=trace[-1].epoch,
                    queries=trace[-1].queries,
                    generator_loss=trace[-1].generator_loss,
                    query_images=trace[-1].query_images,
                    query_entropy=trace[-1].query_entropy,
                    detector_scores=trace[-1].detector_scores,
                    query_blocked=trace[-1].query_blocked,
                    surrogate_accuracy=float(surrogate_accuracy),
                )
            metrics_store = context.metadata.get("disguide_metrics")
            if isinstance(metrics_store, list):
                metrics_store.append(dict(metrics[-1]))

            if stop_reason is not None:
                early_stop_reason = stop_reason
                print(
                    f"[{self.name}] stopping early at {int(blackbox.call_count)} "
                    f"queries -- {stop_reason}"
                )
                break

        self._save_artifacts(
            context,
            output_dir=output_dir,
            ensemble=ensemble,
            replay_records=list(replay_records),
            config=config,
            metrics=metrics,
        )
        # The run completed, so the mid-run checkpoint is now dead weight (it
        # can be tens of GB with the replay records). Removing it also stops a
        # re-submission of the same cell from resuming a finished attack.
        if config.checkpoint_every_queries > 0 and checkpoint_path.exists():
            checkpoint_path.unlink()

        disguide_trace_path = self._save_disguide_trace(context, output_dir)
        result_metadata = {
            "num_samples": len(replay_records),
            "query_pool_size": 0,
            "transferset_path": str(output_dir / "transferset.pickle"),
            "visualization_records_path": str(output_dir / "visualization_records.pt"),
            "actual_queries": int(blackbox.call_count),
            "target_accuracy": float(target_accuracy),
            "ensemble_size": config.ensemble_size,
            "generator_kind": self.generator_kind,
            "adaptive_to_components": adaptivity,
            "early_stopped": early_stop_reason is not None,
            "early_stop_reason": early_stop_reason,
        }
        if adaptive is not None:
            result_metadata["adaptive_weights"] = {
                "velocity": adaptive.velocity_regularizer_weight,
                "integral": adaptive.integral_regularizer_weight,
                "likelihood": adaptive.likelihood_regularizer_weight,
                "composite": adaptive.composite_regularizer_weight,
                "distribution": adaptive.distribution_regularizer_weight,
                "label": adaptive.label_regularizer_weight,
            }
        if disguide_trace_path is not None:
            result_metadata["disguide_trace_path"] = disguide_trace_path
            trace = context.metadata.get("disguide_trace")
            if isinstance(trace, list):
                result_metadata["disguide_trace_steps"] = len(trace)
        if snapshots:
            snapshot_path = output_dir / "query_snapshots.pt"
            torch.save(
                {
                    "attack": self.name,
                    "generator_kind": self.generator_kind,
                    "dataset": spec.dataset.name,
                    "snapshots": snapshots,
                },
                snapshot_path,
            )
            result_metadata["query_snapshots_path"] = str(snapshot_path)
        # Always return the per-epoch history. It was previously exported only
        # when a caller had pre-seeded context.metadata["disguide_metrics"] with
        # a list, which nothing does, so the queries-vs-accuracy/fidelity curve
        # was silently dropped on every run.
        result_metadata["disguide_metrics"] = [dict(point) for point in metrics]

        return AttackRunResult(
            attack_name=self.name,
            mode_name=spec.attack.mode.name,
            output_dir=str(output_dir),
            metadata=result_metadata,
        )


class LatentManifoldAttackRunner(DisguideBypassAttackRunner):
    """D1: DisGUIDE loop with ConvGenerator (+ optional flow-matching loss)."""

    generator_kind: GeneratorKind = "latent"
    name = "latent_manifold"

    def __init__(self, query_engine, distributed_coordinator=None) -> None:
        super().__init__(query_engine, distributed_coordinator=distributed_coordinator)
        from flowguard.attacks.bypass import CemLatentManifoldAttackRunner

        self._cem_runner = CemLatentManifoldAttackRunner(
            query_engine,
            distributed_coordinator=distributed_coordinator,
        )

    def run(self, context: AttackRunContext) -> AttackRunResult:
        from flowguard.attacks.bypass import _uses_cem_engine

        if _uses_cem_engine(context):
            return self._cem_runner.run(context)
        return super().run(context)


class ProceduralNaturalAttackRunner(DisguideBypassAttackRunner):
    """D2: DisGUIDE loop with ProceduralGenerator (+ optional flow-matching loss)."""

    generator_kind: GeneratorKind = "procedural"
    name = "procedural_natural"

    def __init__(self, query_engine, distributed_coordinator=None) -> None:
        super().__init__(query_engine, distributed_coordinator=distributed_coordinator)
        from flowguard.attacks.bypass import CemProceduralNaturalAttackRunner

        self._cem_runner = CemProceduralNaturalAttackRunner(
            query_engine,
            distributed_coordinator=distributed_coordinator,
        )

    def run(self, context: AttackRunContext) -> AttackRunResult:
        from flowguard.attacks.bypass import _uses_cem_engine

        if _uses_cem_engine(context):
            return self._cem_runner.run(context)
        return super().run(context)
