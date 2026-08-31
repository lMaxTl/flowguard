"""Adaptive-attack primitives against FlowPure-style velocity detectors.

Implements two pieces of machinery used by the PhD research plan:

- ``apply_query_denoising``: Projects query tensors onto a smooth
  natural-image manifold to suppress the high-frequency structure that drives
  ``||v(t=0, x)||^2`` in the paper-faithful FlowPure detector. Used by the
  ProjectedMAZE attack (plan section A2).

- ``SurrogateVelocityRegularizer``: Wraps a loaded CNF checkpoint (the
  attacker's own proxy) and exposes a differentiable mean squared velocity
  score at ``t=0``. The MAZE generator adds this as a regularizer so its
  samples look low-velocity to any transferably trained CNF. Used by the
  FlowBlind attack (plan section A3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from flowguard.flow_matching.training import (
    VelocityFieldWrapper,
    load_flow_matching_checkpoint,
)


def _standard_gaussian_log_prob(x: torch.Tensor) -> torch.Tensor:
    """Log density of a standard Gaussian, matching the defender's base measure."""
    flat = x.flatten(start_dim=1)
    dimension = flat.shape[1]
    return -0.5 * (dimension * math.log(2.0 * math.pi) + flat.pow(2).sum(dim=1))


@dataclass(slots=True)
class CompositeWeights:
    """Per-component weights the attacker assumes the defender uses.

    Defaults mirror the deployable FlowGuard++ configuration (equal weights over
    standardized components). An attacker that does not know the true weights can
    still attack the equal-weight composite, which is the published default.
    """

    velocity: float = 1.0
    integral: float = 1.0
    typicality: float = 1.0


@dataclass(slots=True)
class SurrogateBenignStats:
    """Benign calibration statistics the attacker estimates on its own data.

    These are deliberately the attacker's *own* estimates, computed with the
    attacker's surrogate CNF over a pool of benign-looking images. They are not
    the defender's calibration constants, so the attack does not assume
    knowledge of the deployed operating point.
    """

    velocity_mean: float = 0.0
    velocity_std: float = 1.0
    integral_mean: float = 0.0
    integral_std: float = 1.0
    likelihood_mean: float = 0.0
    likelihood_std: float = 1.0
    velocity_band: float = 0.0
    integral_band: float = 0.0
    typicality_band: float = 0.0
    composite_band: float = 0.0
    entropy_mean: float | None = None
    entropy_std: float | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> SurrogateBenignStats:
        if not payload:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{key: float(value) for key, value in payload.items()
                      if key in known and value is not None})


@dataclass(slots=True)
class DenoiseConfig:
    """Configuration for query denoising.

    Attributes:
        kind: ``"none"``, ``"gaussian_blur"``, ``"box_blur"``, or ``"pixel_soft"``.
        sigma: Gaussian blur std (kind=gaussian_blur) in pixel units.
        kernel_size: Odd kernel size used for blur kernels.
        mix: Final output is ``mix * denoised + (1 - mix) * input`` so the
            attacker can trade off information retention vs. defense evasion.
    """

    kind: str = "none"
    sigma: float = 0.5
    kernel_size: int = 5
    mix: float = 1.0


@dataclass(slots=True)
class ProjectionConfig:
    """Configuration for the attack-side manifold projection stage."""

    name: str = "none"
    strength: float = 0.1
    steps: int = 10
    differentiable: bool = False


def _gaussian_kernel_2d(*, sigma: float, kernel_size: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    half = kernel_size // 2
    axis = torch.arange(-half, half + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-(axis.pow(2)) / (2.0 * max(sigma, 1e-6) ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    kernel = gauss_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel


def _box_kernel_2d(*, kernel_size: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    value = 1.0 / float(kernel_size * kernel_size)
    kernel = torch.full(
        (channels, 1, kernel_size, kernel_size),
        value,
        device=device,
        dtype=dtype,
    )
    return kernel


def apply_query_denoising(batch: torch.Tensor, config: DenoiseConfig) -> torch.Tensor:
    """Apply the attacker-side denoising transform to a query batch.

    Input is expected as a ``(B, C, H, W)`` tensor in whatever normalization
    the caller uses; the operation is a pixel-space smoothing so it commutes
    with CIFAR-style per-channel normalization (smoothing in normalized space
    is equivalent to smoothing in [0,1] space up to the affine constants).
    Returns a tensor with the same shape and normalization.
    """
    if config.kind == "none" or config.mix <= 0.0:
        return batch

    if batch.ndim != 4:
        raise ValueError(f"Expected a (B, C, H, W) tensor; got shape {tuple(batch.shape)}.")

    channels = batch.shape[1]
    kind = config.kind.lower()

    if kind == "gaussian_blur":
        kernel = _gaussian_kernel_2d(
            sigma=float(config.sigma),
            kernel_size=int(config.kernel_size),
            channels=channels,
            device=batch.device,
            dtype=batch.dtype,
        )
    elif kind == "box_blur":
        kernel = _box_kernel_2d(
            kernel_size=int(config.kernel_size),
            channels=channels,
            device=batch.device,
            dtype=batch.dtype,
        )
    elif kind == "pixel_soft":
        # Soft pixel projection: weighted average between bilinear downsample
        # then upsample. Cheap approximation of a low-frequency natural-image
        # projection (a smaller effective receptive field than Gaussian blur).
        original_shape = batch.shape[-2:]
        target = (max(4, original_shape[0] // 2), max(4, original_shape[1] // 2))
        downsampled = F.interpolate(batch, size=target, mode="bilinear", align_corners=False)
        upsampled = F.interpolate(downsampled, size=original_shape, mode="bilinear", align_corners=False)
        mix = float(config.mix)
        return mix * upsampled + (1.0 - mix) * batch
    else:
        raise ValueError(f"Unsupported denoise kind '{config.kind}'.")

    pad = kernel.shape[-1] // 2
    padded = F.pad(batch, (pad, pad, pad, pad), mode="reflect")
    denoised = F.conv2d(padded, kernel, groups=channels)
    mix = float(config.mix)
    return mix * denoised + (1.0 - mix) * batch


def apply_query_projection(batch: torch.Tensor, config: ProjectionConfig) -> torch.Tensor:
    """Project generated queries before sending them to the victim.

    The current implementation keeps the projection family intentionally
    simple and deterministic. Diffusion/autoencoder/CNF projections can be
    wired in by callers that own those checkpoints; the built-in options cover
    the non-parametric baselines used in the FlowPure bypass ablations.
    """

    name = config.name.lower()
    if name == "none" or config.strength <= 0.0:
        return batch
    if name in {"gaussian_blur", "diffusion_denoise", "autoencoder", "cnf_purify"}:
        denoise = DenoiseConfig(
            kind="gaussian_blur",
            sigma=max(float(config.strength), 1e-3) * 5.0,
            kernel_size=max(3, int(config.steps) | 1),
            mix=min(max(float(config.strength), 0.0), 1.0),
        )
        return apply_query_denoising(batch, denoise)
    if name == "pixel_soft":
        denoise = DenoiseConfig(kind="pixel_soft", mix=min(max(float(config.strength), 0.0), 1.0))
        return apply_query_denoising(batch, denoise)
    raise ValueError(f"Unsupported projection name '{config.name}'.")


def total_variation(batch: torch.Tensor) -> torch.Tensor:
    """Return isotropic total-variation energy for a batch."""

    if batch.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W); got {tuple(batch.shape)}.")
    horizontal = torch.mean(torch.abs(batch[:, :, :, 1:] - batch[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(batch[:, :, 1:, :] - batch[:, :, :-1, :]))
    return horizontal + vertical


def high_frequency_energy(batch: torch.Tensor, *, cutoff_ratio: float = 0.35) -> torch.Tensor:
    """Return mean FFT energy outside a centered low-frequency disk."""

    if batch.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W); got {tuple(batch.shape)}.")
    gray = batch.to(torch.float32).mean(dim=1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    power = spectrum.abs().pow(2)
    height, width = gray.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=batch.device),
        torch.arange(width, device=batch.device),
        indexing="ij",
    )
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = torch.sqrt((yy - center_y).pow(2) + (xx - center_x).pow(2))
    cutoff = float(cutoff_ratio) * min(height, width)
    mask = radius > cutoff
    return power[:, mask].mean()


def color_stat_penalty(
    batch: torch.Tensor,
    *,
    target_mean: tuple[float, ...] = (0.4914, 0.4822, 0.4465),
    target_std: tuple[float, ...] = (0.2023, 0.1994, 0.2010),
) -> torch.Tensor:
    """Penalize deviation from benign CIFAR-like color statistics."""

    if batch.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W); got {tuple(batch.shape)}.")
    channels = batch.shape[1]
    mean_t = torch.tensor(target_mean[:channels], device=batch.device, dtype=batch.dtype)
    std_t = torch.tensor(target_std[:channels], device=batch.device, dtype=batch.dtype)
    observed_mean = batch.mean(dim=(0, 2, 3))
    observed_std = batch.std(dim=(0, 2, 3), unbiased=False)
    return F.mse_loss(observed_mean, mean_t) + F.mse_loss(observed_std, std_t)


def normalized_detector_penalty(
    scores: torch.Tensor,
    *,
    benign_mean: float,
    benign_std: float,
    benign_q95_norm: float,
) -> torch.Tensor:
    """Benign-normalized FlowPure surrogate loss from the implementation spec."""

    score_norm = (scores - float(benign_mean)) / (float(benign_std) + 1e-8)
    return torch.relu(score_norm - float(benign_q95_norm)).pow(2).mean()


def batch_quantile_detector_penalty(
    scores: torch.Tensor,
    *,
    benign_mean: float,
    benign_std: float,
    benign_q95_norm: float,
    batch_quantile: float = 0.9,
) -> torch.Tensor:
    """Batch-level counterpart of :func:`normalized_detector_penalty`.

    ``normalized_detector_penalty`` hinges *every sample* against the benign
    band, which asks each generated query to individually look typical. That
    is a much narrower target than "the stream looks benign on average" (what
    a windowed/user-level detector actually scores), and it collapses the
    generator onto whatever single point most cheaply satisfies the hinge for
    every sample at once -- typically something close to the CNF's mode.

    This version only penalizes the ``batch_quantile``-th worst sample in the
    batch. As long as that fraction of the batch stays under the benign band,
    the rest of the batch is free to vary, which is what leaves room for the
    disagreement/diversity terms to keep exploring.
    """
    score_norm = (scores - float(benign_mean)) / (float(benign_std) + 1e-8)
    quantile_value = torch.quantile(score_norm, float(batch_quantile))
    return torch.relu(quantile_value - float(benign_q95_norm)).pow(2)


def two_sided_typicality_penalty(
    log_likelihood: torch.Tensor,
    *,
    benign_mean: float,
    benign_std: float,
    benign_band: float = 0.0,
) -> torch.Tensor:
    """Penalty on the defender's two-sided likelihood-typicality score.

    Mirrors ``s_typ(x) = |log p(x) - mu_L| / sigma_L``: the attacker is punished
    for deviating from the benign log-likelihood band in *either* direction,
    which is what makes this objective qualitatively harder than driving a
    one-sided velocity score toward zero. ``benign_band`` is the attacker's own
    estimate of the tolerated |z| (e.g. the benign 95th percentile of the
    typicality statistic), so the penalty is threshold-agnostic.
    """
    z_score = (log_likelihood - float(benign_mean)) / (abs(float(benign_std)) + 1e-8)
    return torch.relu(z_score.abs() - float(benign_band)).pow(2).mean()


def batch_quantile_typicality_penalty(
    log_likelihood: torch.Tensor,
    *,
    benign_mean: float,
    benign_std: float,
    benign_band: float = 0.0,
    batch_quantile: float = 0.9,
) -> torch.Tensor:
    """Batch-level counterpart of :func:`two_sided_typicality_penalty`.

    Same rationale as :func:`batch_quantile_detector_penalty`: only the
    ``batch_quantile``-th worst |z-score| in the batch is hinged, so the rest
    of the batch keeps room to vary rather than every sample being pulled
    toward the single most "typical" point under the surrogate CNF.
    """
    z_score = (log_likelihood - float(benign_mean)) / (abs(float(benign_std)) + 1e-8)
    quantile_value = torch.quantile(z_score.abs(), float(batch_quantile))
    return torch.relu(quantile_value - float(benign_band)).pow(2)


def distribution_matching_penalty(
    scores: torch.Tensor,
    reference: torch.Tensor,
    *,
    moment_weight: float = 1.0,
    quantile_weight: float = 1.0,
) -> torch.Tensor:
    """Differentiable surrogate for the defender's two-sample KS statistic (C4).

    The KS statistic itself is a supremum over a step function and carries no
    usable gradient. We therefore minimize a differentiable upper-bound proxy:
    the 1-D Wasserstein distance between the batch's score sample and the benign
    reference (computed by matching sorted quantiles), plus a mean/std moment
    match. Driving these to zero drives the KS statistic toward zero as well,
    because two 1-D samples with matching quantile functions have matching
    empirical CDFs.

    Args:
        scores: ``(B,)`` differentiable per-query scores for the current batch.
        reference: ``(M,)`` benign reference scores (no gradient required).
        moment_weight: Weight on the mean/std matching term.
        quantile_weight: Weight on the sorted-quantile (Wasserstein-1) term.
    """
    if scores.numel() == 0 or reference.numel() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)

    reference = reference.detach().to(device=scores.device, dtype=scores.dtype)

    penalty = torch.zeros((), device=scores.device, dtype=scores.dtype)
    if moment_weight != 0.0:
        moment = (scores.mean() - reference.mean()).pow(2) + (
            scores.std(unbiased=False) - reference.std(unbiased=False)
        ).pow(2)
        penalty = penalty + float(moment_weight) * moment
    if quantile_weight != 0.0:
        # Match the batch's order statistics against the reference quantile
        # function evaluated at the batch's plotting positions.
        count = scores.shape[0]
        sorted_scores, _ = torch.sort(scores)
        positions = (torch.arange(count, device=scores.device, dtype=scores.dtype) + 0.5) / count
        reference_sorted, _ = torch.sort(reference)
        indices = torch.clamp(
            (positions * reference_sorted.shape[0]).long(),
            max=reference_sorted.shape[0] - 1,
        )
        penalty = penalty + float(quantile_weight) * (
            sorted_scores - reference_sorted[indices]
        ).abs().mean()
    return penalty


def label_distribution_penalty(
    probabilities: torch.Tensor,
    *,
    benign_label_histogram: torch.Tensor | None = None,
    benign_entropy_mean: float | None = None,
    benign_entropy_std: float | None = None,
    histogram_weight: float = 1.0,
    entropy_weight: float = 1.0,
) -> torch.Tensor:
    """Differentiable surrogate for the prediction-side monitor (C5).

    The attacker cannot see the victim's returned label histogram before it
    queries, but it can read the *substitute ensemble's* soft votes, which
    approximate the victim's predictions ever more closely as extraction
    proceeds. This penalty pushes the induced soft-label histogram toward the
    benign reference and the induced prediction entropy toward the benign
    entropy band, directly opposing the class-sweep and high-entropy signatures
    that C5 keys on.

    Args:
        probabilities: ``(B, C)`` softmax probabilities from the attacker's
            substitute (a differentiable proxy for the victim's answers).
        benign_label_histogram: ``(C,)`` benign top-1 label distribution.
        benign_entropy_mean: Benign mean prediction entropy (nats).
        benign_entropy_std: Benign entropy standard deviation.
        histogram_weight: Weight on the soft label-histogram match.
        entropy_weight: Weight on the entropy match.
    """
    penalty = torch.zeros((), device=probabilities.device, dtype=probabilities.dtype)
    if benign_label_histogram is not None and histogram_weight != 0.0:
        reference = benign_label_histogram.detach().to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        reference = reference / reference.sum().clamp_min(1e-12)
        soft_histogram = probabilities.mean(dim=0)
        soft_histogram = soft_histogram / soft_histogram.sum().clamp_min(1e-12)
        # Total variation, matching the defender's histogram statistic.
        penalty = penalty + float(histogram_weight) * 0.5 * (
            soft_histogram - reference
        ).abs().sum()
    if benign_entropy_mean is not None and entropy_weight != 0.0:
        clamped = probabilities.clamp_min(1e-8)
        entropy = -(clamped * clamped.log()).sum(dim=1)
        scale = abs(float(benign_entropy_std or 1.0)) + 1e-8
        penalty = penalty + float(entropy_weight) * (
            (entropy.mean() - float(benign_entropy_mean)) / scale
        ).pow(2)
    return penalty


class SurrogateVelocityRegularizer:
    """Differentiable surrogate-CNF velocity score for adaptive query generation.

    Loads a FlowPure-style CNF checkpoint from ``checkpoint_path`` and exposes
    ``velocity_score(x)`` which returns ``||v(t=0, x)||^2`` per sample. The
    operation is fully differentiable in ``x`` (gradients flow back into the
    attacker's generator).

    The input convention matches ``FlowPureQueryDefense._prepare_inputs``:
    the attacker must pass denormalized tensors in ``[0, 1]`` with shape
    ``(B, C, H, W)``. Internally we apply the same ``* 2 - 1`` scaling as the
    defense so the surrogate score is a good proxy for the deployed score.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str | torch.device = "cuda",
        train_mode: bool = False,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        self.flow_checkpoint = load_flow_matching_checkpoint(
            checkpoint_path,
            device=self.device,
        )
        self.velocity_model = VelocityFieldWrapper(
            model=self.flow_checkpoint.model,
            class_conditioned=self.flow_checkpoint.dataset_config.class_conditioned,
        )
        self._set_mode(train_mode=train_mode)

    def _set_mode(self, *, train_mode: bool) -> None:
        if train_mode:
            self.flow_checkpoint.model.train()
        else:
            self.flow_checkpoint.model.eval()
        for parameter in self.flow_checkpoint.model.parameters():
            parameter.requires_grad_(False)

    @property
    def image_size(self) -> int:
        return int(self.flow_checkpoint.dataset_config.image_size)

    def prepare(self, batch_01: torch.Tensor) -> torch.Tensor:
        """Scale a ``[0, 1]`` tensor to the ``[-1, 1]`` space the CNF expects."""
        if batch_01.ndim != 4:
            raise ValueError(
                f"Expected (B, C, H, W) tensor in [0,1]; got shape {tuple(batch_01.shape)}."
            )
        x = batch_01.to(self.device, dtype=torch.float32)
        expected = self.image_size
        if x.shape[-2:] != (expected, expected):
            x = F.interpolate(x, size=(expected, expected), mode="bilinear", align_corners=False)
        return x * 2.0 - 1.0

    def velocity_score(self, batch_01: torch.Tensor) -> torch.Tensor:
        """Return per-sample ``||v(t=0, x)||^2``. Differentiable in ``batch_01``."""
        fm_inputs = self.prepare(batch_01)
        t_zero = torch.zeros((fm_inputs.shape[0],), device=self.device, dtype=fm_inputs.dtype)
        velocity = self.velocity_model(fm_inputs, t_zero)
        return velocity.flatten(start_dim=1).pow(2).sum(dim=1)

    def trajectory_integral_score(
        self,
        batch_01: torch.Tensor,
        *,
        num_steps: int = 8,
    ) -> torch.Tensor:
        """Return the surrogate trajectory-integral energy, differentiable in ``batch_01``.

        Mirrors ``FlowGuardCompositeDefense._flow_scores``: a forward Euler solve
        from ``t=0`` to ``t=1`` accumulating ``||v(t, x_t)||^2 dt``. Unlike the
        defender's version this keeps the autograd graph, so a generator can be
        trained against component C2 rather than only against the ``t=0`` score.

        The direction has to match the defender's exactly, otherwise the attacker
        optimizes a different statistic than the one it is being scored on.
        """
        fm_inputs = self.prepare(batch_01)
        steps = max(1, int(num_steps))
        time_grid = torch.linspace(0.0, 1.0, steps + 1, device=self.device, dtype=fm_inputs.dtype)
        x_current = fm_inputs
        integral = torch.zeros(fm_inputs.shape[0], device=self.device, dtype=fm_inputs.dtype)
        for step_index in range(steps):
            t_current = time_grid[step_index]
            dt = time_grid[step_index + 1] - t_current
            t_batch = torch.full(
                (x_current.shape[0],),
                float(t_current.item()),
                device=self.device,
                dtype=x_current.dtype,
            )
            velocity = self.velocity_model(x_current, t_batch)
            integral = integral + velocity.flatten(start_dim=1).pow(2).sum(dim=1) * float(dt.item())
            x_current = x_current + velocity * dt
        return integral

    def log_likelihood(
        self,
        batch_01: torch.Tensor,
        *,
        num_steps: int = 8,
        hutchinson_samples: int = 1,
        exact_divergence: bool = False,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return a differentiable estimate of ``log p_1(x)`` under the surrogate CNF.

        Implements the instantaneous change-of-variables
        ``log p_1(x_1) = log p_0(x_0) - int_0^1 tr(d v / d x) dt`` with an explicit
        Euler solve from ``t=1`` to ``t=0``. The divergence is estimated with the
        Hutchinson trace estimator using Rademacher probes, and every step keeps
        ``create_graph=True`` so the whole estimate is differentiable with respect
        to ``batch_01``.

        This is the piece the library's ``ODESolver.compute_likelihood`` cannot
        provide: it detaches the velocity and divergence inside its dynamics
        function, so its output carries no gradient back to the query. Recomputing
        it here is what lets an attacker optimize against component C3.

        Args:
            batch_01: ``(B, C, H, W)`` tensor in ``[0, 1]``.
            num_steps: Euler steps for the reverse solve. More steps track the
                defender's discretization more closely at linear cost.
            hutchinson_samples: Number of Rademacher probes averaged per step.
            exact_divergence: Compute the exact trace by looping over input
                dimensions. Accurate but ``O(d)`` backward passes; only usable
                for tiny inputs or verification.
            create_graph: Keep the second-order graph so the result can be
                backpropagated into a generator. Set ``False`` for scoring or
                calibration, where only the value is needed: the divergence
                still requires a first-order backward per step, but nothing is
                retained, which cuts peak memory by roughly ``num_steps``.
        """
        fm_inputs = self.prepare(batch_01)
        steps = max(1, int(num_steps))
        time_grid = torch.linspace(1.0, 0.0, steps + 1, device=self.device, dtype=fm_inputs.dtype)

        probes: list[torch.Tensor] = []
        if not exact_divergence:
            probes = [
                (torch.rand_like(fm_inputs) < 0.5).to(fm_inputs.dtype) * 2.0 - 1.0
                for _ in range(max(1, int(hutchinson_samples)))
            ]

        x_current = fm_inputs
        log_det = torch.zeros(fm_inputs.shape[0], device=self.device, dtype=fm_inputs.dtype)
        for step_index in range(steps):
            t_current = time_grid[step_index]
            dt = time_grid[step_index + 1] - t_current
            t_batch = torch.full(
                (x_current.shape[0],),
                float(t_current.item()),
                device=self.device,
                dtype=x_current.dtype,
            )
            with torch.enable_grad():
                x_with_grad = x_current
                if not x_with_grad.requires_grad:
                    x_with_grad = x_with_grad.detach().requires_grad_(True)
                velocity = self.velocity_model(x_with_grad, t_batch)
                divergence = self._divergence(
                    velocity,
                    x_with_grad,
                    probes=probes,
                    exact=exact_divergence,
                    create_graph=create_graph,
                )
            if not create_graph:
                velocity = velocity.detach()
                divergence = divergence.detach()
            # d(log_det)/dt = div, integrated from 1 to 0 (dt is negative), so
            # log_det = -int_0^1 div dt, matching ODESolver.compute_likelihood.
            log_det = log_det + divergence * dt
            x_current = x_current + velocity * dt

        return _standard_gaussian_log_prob(x_current) + log_det

    @staticmethod
    def _divergence(
        velocity: torch.Tensor,
        inputs: torch.Tensor,
        *,
        probes: list[torch.Tensor],
        exact: bool,
        create_graph: bool = True,
    ) -> torch.Tensor:
        if exact:
            flat_velocity = velocity.flatten(start_dim=1)
            divergence = torch.zeros(
                velocity.shape[0], device=velocity.device, dtype=velocity.dtype
            )
            for dimension in range(flat_velocity.shape[1]):
                grad = torch.autograd.grad(
                    flat_velocity[:, dimension].sum(),
                    inputs,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
                divergence = divergence + grad.flatten(start_dim=1)[:, dimension]
            return divergence

        estimates = []
        for index, probe in enumerate(probes):
            velocity_dot_probe = (
                velocity.flatten(start_dim=1) * probe.flatten(start_dim=1)
            ).sum(dim=1)
            grad = torch.autograd.grad(
                velocity_dot_probe.sum(),
                inputs,
                create_graph=create_graph,
                retain_graph=create_graph or index + 1 < len(probes),
            )[0]
            estimates.append(
                (grad.flatten(start_dim=1) * probe.flatten(start_dim=1)).sum(dim=1)
            )
        return torch.stack(estimates, dim=0).mean(dim=0)

    def composite_score(
        self,
        batch_01: torch.Tensor,
        *,
        stats: SurrogateBenignStats,
        weights: CompositeWeights,
        integral_num_steps: int = 8,
        likelihood_num_steps: int = 8,
        hutchinson_samples: int = 1,
        likelihood_regularizer: SurrogateVelocityRegularizer | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the attacker's replica of the defender's fused risk ``R(x)``.

        Mirrors Eq. (12) of the defense: a weighted sum of one-sided standardized
        velocity components and the two-sided likelihood-typicality score, all
        computed on the attacker's own surrogate CNF and standardized with the
        attacker's own benign statistics.

        ``likelihood_regularizer`` supplies a second CNF for the typicality
        term. It is needed whenever this regularizer holds a FlowPure-family
        checkpoint: that flow's velocity is the detection signal, but its source
        distribution is perturbed images rather than ``N(0, I)``, so ``log p(x)``
        under it does not measure benign density and the typicality term would
        optimize an undefined quantity. Defaults to ``self``, correct only when
        this checkpoint is itself Gaussian-source.

        Returns the fused per-query risk and the individual standardized
        components so callers can log or re-weight them.
        """
        likelihood_source = likelihood_regularizer or self
        components: dict[str, torch.Tensor] = {}
        risk = torch.zeros(batch_01.shape[0], device=self.device, dtype=torch.float32)

        if weights.velocity != 0.0:
            velocity = self.velocity_score(batch_01)
            z_velocity = (velocity - stats.velocity_mean) / (abs(stats.velocity_std) + 1e-8)
            components["t0"] = z_velocity
            risk = risk + float(weights.velocity) * z_velocity
        if weights.integral != 0.0:
            integral = self.trajectory_integral_score(batch_01, num_steps=integral_num_steps)
            z_integral = (integral - stats.integral_mean) / (abs(stats.integral_std) + 1e-8)
            components["integral"] = z_integral
            risk = risk + float(weights.integral) * z_integral
        if weights.typicality != 0.0:
            log_likelihood = likelihood_source.log_likelihood(
                batch_01,
                num_steps=likelihood_num_steps,
                hutchinson_samples=hutchinson_samples,
            )
            typicality = (
                log_likelihood - stats.likelihood_mean
            ).abs() / (abs(stats.likelihood_std) + 1e-8)
            components["likelihood"] = typicality
            components["log_likelihood"] = log_likelihood
            risk = risk + float(weights.typicality) * typicality
        return risk, components


def _default_cifar_mean_std() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )


def denormalize_cifar(batch: torch.Tensor) -> torch.Tensor:
    """Convert a CIFAR-normalized tensor back to ``[0, 1]`` pixel space."""
    mean, std = _default_cifar_mean_std()
    mean_t = torch.tensor(mean, device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    return torch.clamp(batch * std_t + mean_t, min=0.0, max=1.0)


def normalize_cifar(batch_01: torch.Tensor) -> torch.Tensor:
    """Convert a ``[0, 1]`` tensor to CIFAR-normalized input space."""
    mean, std = _default_cifar_mean_std()
    mean_t = torch.tensor(mean, device=batch_01.device, dtype=batch_01.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=batch_01.device, dtype=batch_01.dtype).view(1, -1, 1, 1)
    return (batch_01 - mean_t) / std_t


@dataclass(slots=True)
class AdaptiveAttackConfig:
    """Composite config covering adaptive-attack knobs consumed by MAZE."""

    denoise: DenoiseConfig
    projection: ProjectionConfig
    surrogate_checkpoint_path: str | None = None
    # Second CNF for the C3 typicality term. See composite_score for why one
    # checkpoint cannot serve both the velocity and likelihood components.
    likelihood_surrogate_checkpoint_path: str | None = None
    velocity_regularizer_weight: float = 0.0
    velocity_regularizer_device: str = "cuda"
    velocity_regularizer_is_pixel_space: bool = True
    record_surrogate_scores: bool = True
    detector_benign_mean: float | None = None
    detector_benign_std: float | None = None
    detector_benign_q95_norm: float = 0.0

    # --- Fully adaptive objective (FlowGuard++-aware) -----------------------
    # Each weight switches on one term of the attacker loss
    #   L_G = L_extract + l1*z0 + l2*z_int + l3*s_typ + l4*T_state + l5*T_label
    # so a single runner covers the C1-only attacker (D1/D2) up to an attacker
    # adaptive to the full composite plus its distributional tests.
    integral_regularizer_weight: float = 0.0
    likelihood_regularizer_weight: float = 0.0
    composite_regularizer_weight: float = 0.0
    distribution_regularizer_weight: float = 0.0
    label_regularizer_weight: float = 0.0
    integral_num_steps: int = 8
    likelihood_num_steps: int = 8
    likelihood_hutchinson_samples: int = 1
    likelihood_batch_chunk: int = 0
    composite_weights: CompositeWeights = field(default_factory=CompositeWeights)
    benign_stats: SurrogateBenignStats = field(default_factory=SurrogateBenignStats)
    # True only when calibrated statistics were actually supplied. The default
    # SurrogateBenignStats is a neutral placeholder, so callers must not treat
    # it as a real calibration.
    has_benign_stats: bool = False
    benign_reference_scores: list[float] | None = None
    benign_label_histogram: list[float] | None = None
    distilled_surrogate_path: str | None = None

    # --- Anti-collapse knobs -------------------------------------------------
    # The evasion terms above are an *attractive* force (pull toward a benign
    # statistical region), unlike DisGUIDE's purely *repulsive* disagreement
    # objective. Applied at full strength from step 0, that attractor can win
    # before the disagreement/diversity terms establish a diverse query
    # distribution, and the generator collapses onto whatever single point
    # most cheaply satisfies every hinge at once. These knobs are opt-in
    # (defaults reproduce the original unscheduled, per-sample behavior).
    #
    # Warm-up + linear ramp on every evasion weight: 0 until `warmup_queries`,
    # then linearly 0 -> 1 over the next `ramp_queries`, then 1 thereafter.
    warmup_queries: int = 0
    ramp_queries: int = 0
    # "per_sample" (original) hinges every sample against the benign band.
    # "batch_quantile" only hinges the batch's `detector_batch_quantile`-th
    # worst sample, leaving the rest of the batch free to vary.
    detector_penalty_mode: str = "per_sample"
    detector_batch_quantile: float = 0.9

    def weight_scale(self, queries_used: int) -> float:
        """Warm-up/ramp multiplier applied to every evasion-loss weight."""
        if queries_used < self.warmup_queries:
            return 0.0
        if self.ramp_queries <= 0:
            return 1.0
        progress = (queries_used - self.warmup_queries) / float(self.ramp_queries)
        return float(min(1.0, max(0.0, progress)))

    def uses_composite_objective(self) -> bool:
        """True when any FlowGuard++-aware term beyond the ``t=0`` score is active."""
        return any(
            weight > 0.0
            for weight in (
                self.integral_regularizer_weight,
                self.likelihood_regularizer_weight,
                self.composite_regularizer_weight,
                self.distribution_regularizer_weight,
                self.label_regularizer_weight,
            )
        )

    def adaptivity_label(self) -> str:
        """Short description of which detector components this attacker targets."""
        targeted: list[str] = []
        if self.velocity_regularizer_weight > 0.0 or self.composite_regularizer_weight > 0.0:
            targeted.append("C1")
        if self.integral_regularizer_weight > 0.0 or self.composite_regularizer_weight > 0.0:
            targeted.append("C2")
        if self.likelihood_regularizer_weight > 0.0 or self.composite_regularizer_weight > 0.0:
            targeted.append("C3")
        if self.distribution_regularizer_weight > 0.0:
            targeted.append("C4")
        if self.label_regularizer_weight > 0.0:
            targeted.append("C5")
        return "+".join(dict.fromkeys(targeted)) if targeted else "none"

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> AdaptiveAttackConfig:
        denoise_raw = extra.get("denoise", {}) or {}
        denoise_cfg = DenoiseConfig(
            kind=str(denoise_raw.get("kind", "none")),
            sigma=float(denoise_raw.get("sigma", 0.5)),
            kernel_size=int(denoise_raw.get("kernel_size", 5)),
            mix=float(denoise_raw.get("mix", 1.0)),
        )
        projection_raw = extra.get("projection", {}) or {}
        projection_cfg = ProjectionConfig(
            name=str(projection_raw.get("name", "none")),
            strength=float(projection_raw.get("strength", 0.1)),
            steps=int(projection_raw.get("steps", 10)),
            differentiable=bool(projection_raw.get("differentiable", False)),
        )
        return cls(
            denoise=denoise_cfg,
            projection=projection_cfg,
            surrogate_checkpoint_path=extra.get("surrogate_checkpoint_path"),
            likelihood_surrogate_checkpoint_path=extra.get(
                "likelihood_surrogate_checkpoint_path"
            ),
            velocity_regularizer_weight=float(extra.get("velocity_regularizer_weight", 0.0)),
            velocity_regularizer_device=str(extra.get("velocity_regularizer_device", "cuda")),
            velocity_regularizer_is_pixel_space=bool(
                extra.get("velocity_regularizer_is_pixel_space", True)
            ),
            record_surrogate_scores=bool(extra.get("record_surrogate_scores", True)),
            detector_benign_mean=(
                float(extra["detector_benign_mean"])
                if extra.get("detector_benign_mean") is not None
                else None
            ),
            detector_benign_std=(
                float(extra["detector_benign_std"])
                if extra.get("detector_benign_std") is not None
                else None
            ),
            detector_benign_q95_norm=float(extra.get("detector_benign_q95_norm", 0.0)),
            integral_regularizer_weight=float(extra.get("integral_regularizer_weight", 0.0)),
            likelihood_regularizer_weight=float(extra.get("likelihood_regularizer_weight", 0.0)),
            composite_regularizer_weight=float(extra.get("composite_regularizer_weight", 0.0)),
            distribution_regularizer_weight=float(
                extra.get("distribution_regularizer_weight", 0.0)
            ),
            label_regularizer_weight=float(extra.get("label_regularizer_weight", 0.0)),
            integral_num_steps=int(extra.get("integral_num_steps", 8)),
            likelihood_num_steps=int(extra.get("likelihood_num_steps", 8)),
            likelihood_hutchinson_samples=int(extra.get("likelihood_hutchinson_samples", 1)),
            likelihood_batch_chunk=int(extra.get("likelihood_batch_chunk", 0)),
            composite_weights=CompositeWeights(
                **{
                    key: float(value)
                    for key, value in (extra.get("composite_weights") or {}).items()
                    if key in CompositeWeights.__dataclass_fields__
                }
            ),
            benign_stats=SurrogateBenignStats.from_mapping(extra.get("benign_stats")),
            has_benign_stats=bool(extra.get("benign_stats")),
            benign_reference_scores=(
                [float(value) for value in extra["benign_reference_scores"]]
                if extra.get("benign_reference_scores")
                else None
            ),
            benign_label_histogram=(
                [float(value) for value in extra["benign_label_histogram"]]
                if extra.get("benign_label_histogram")
                else None
            ),
            distilled_surrogate_path=extra.get("distilled_surrogate_path"),
            warmup_queries=int(extra.get("warmup_queries", 0)),
            ramp_queries=int(extra.get("ramp_queries", 0)),
            detector_penalty_mode=str(extra.get("detector_penalty_mode", "per_sample")),
            detector_batch_quantile=float(extra.get("detector_batch_quantile", 0.9)),
        )

    def is_active(self) -> bool:
        return (
            self.denoise.kind != "none"
            or self.projection.name != "none"
            or (
                self.surrogate_checkpoint_path
                and (
                    self.velocity_regularizer_weight > 0.0
                    or self.uses_composite_objective()
                )
            )
        )


def calibrate_surrogate_stats(
    regularizer: SurrogateVelocityRegularizer,
    benign_batches_01: list[torch.Tensor],
    *,
    integral_num_steps: int = 8,
    likelihood_num_steps: int = 8,
    hutchinson_samples: int = 1,
    band_quantile: float = 0.95,
    composite_weights: CompositeWeights | None = None,
    likelihood_regularizer: SurrogateVelocityRegularizer | None = None,
) -> tuple[SurrogateBenignStats, dict[str, list[float]]]:
    """Estimate the attacker's benign reference statistics on its own image pool.

    The attacker needs ``(mu, sigma)`` per component to standardize its penalty,
    and a tolerated band (a benign upper quantile) so the penalty only fires
    outside the benign range. All of it is computed from benign-looking images
    the attacker owns; nothing here uses the defender's calibration set or
    thresholds.

    Returns the statistics plus the raw per-component score columns, which the
    caller can reuse as the benign reference sample for the distributional term.
    """
    weights = composite_weights or CompositeWeights()
    # Velocity and integral come from the detection flow; the likelihood must
    # come from a Gaussian-source flow or the statistic is undefined. Same
    # object unless the caller splits them.
    likelihood_source = likelihood_regularizer or regularizer
    columns: dict[str, list[float]] = {"t0": [], "integral": [], "log_likelihood": []}
    for batch in benign_batches_01:
        batch = batch.to(regularizer.device)
        with torch.no_grad():
            columns["t0"].extend(regularizer.velocity_score(batch).detach().cpu().tolist())
            columns["integral"].extend(
                regularizer.trajectory_integral_score(
                    batch, num_steps=integral_num_steps
                ).detach().cpu().tolist()
            )
        # The likelihood needs grad enabled internally for the divergence probe,
        # but calibration only needs the value, so we skip the second-order graph.
        log_likelihood = likelihood_source.log_likelihood(
            batch.to(likelihood_source.device).detach(),
            num_steps=likelihood_num_steps,
            hutchinson_samples=hutchinson_samples,
            create_graph=False,
        )
        columns["log_likelihood"].extend(log_likelihood.detach().cpu().tolist())

    velocity = np.asarray(columns["t0"], dtype=np.float64)
    integral = np.asarray(columns["integral"], dtype=np.float64)
    likelihood = np.asarray(columns["log_likelihood"], dtype=np.float64)

    stats = SurrogateBenignStats(
        velocity_mean=float(velocity.mean()),
        velocity_std=float(velocity.std() or 1.0),
        integral_mean=float(integral.mean()),
        integral_std=float(integral.std() or 1.0),
        likelihood_mean=float(likelihood.mean()),
        likelihood_std=float(likelihood.std() or 1.0),
    )
    z_velocity = (velocity - stats.velocity_mean) / (abs(stats.velocity_std) + 1e-8)
    z_integral = (integral - stats.integral_mean) / (abs(stats.integral_std) + 1e-8)
    typicality = np.abs(likelihood - stats.likelihood_mean) / (abs(stats.likelihood_std) + 1e-8)
    composite = (
        weights.velocity * z_velocity
        + weights.integral * z_integral
        + weights.typicality * typicality
    )
    stats.velocity_band = float(np.quantile(z_velocity, band_quantile))
    stats.integral_band = float(np.quantile(z_integral, band_quantile))
    stats.typicality_band = float(np.quantile(typicality, band_quantile))
    stats.composite_band = float(np.quantile(composite, band_quantile))
    columns["composite"] = composite.tolist()
    return stats, columns


def compute_surrogate_batch_score(
    regularizer: SurrogateVelocityRegularizer,
    batch_normalized: torch.Tensor,
    *,
    dataset_is_cifar: bool = True,
) -> torch.Tensor:
    """Helper: given a CIFAR-normalized tensor, return the surrogate velocity scores."""
    batch_01 = denormalize_cifar(batch_normalized) if dataset_is_cifar else batch_normalized
    return regularizer.velocity_score(batch_01)


@dataclass(slots=True)
class GeneratorCollapseMonitor:
    """Detects a sustained generator/label collapse and asks for a reinit.

    DisGUIDE's own limitations section anticipates exactly this failure mode
    ("DisGUIDE could theoretically get stuck in local minima... A possible
    solution is to retrain, given the training non-determinism.") -- this is
    that mechanism, wired to fire automatically rather than requiring a human
    to notice a dead run and resubmit it.

    Tracks the per-batch query pixel std. A run below ``pixel_std_threshold``
    for ``patience_queries`` in a row (after an initial ``min_queries`` grace
    period, so the generator has had a chance to move at all) is judged
    collapsed, and :meth:`observe` returns ``True`` exactly once per episode.
    """

    pixel_std_threshold: float = 0.05
    patience_queries: int = 20_000
    min_queries: int = 5_000

    _below_since: int | None = field(default=None, init=False, repr=False)
    reinit_count: int = field(default=0, init=False)

    def observe(self, *, pixel_std: float, queries: int) -> bool:
        if queries < self.min_queries:
            return False
        if pixel_std >= self.pixel_std_threshold:
            self._below_since = None
            return False
        if self._below_since is None:
            self._below_since = queries
            return False
        if queries - self._below_since >= self.patience_queries:
            self._below_since = None
            self.reinit_count += 1
            return True
        return False


def numpy_frequency_profile(batch: torch.Tensor) -> np.ndarray:
    """Return the radially-averaged FFT magnitude of a ``(B, C, H, W)`` tensor.

    Useful for the ProjectedMAZE frequency-analysis plot (plan experiment E2.2).
    """
    if batch.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W); got {tuple(batch.shape)}.")
    tensor = batch.detach().to(torch.float32).cpu()
    gray = tensor.mean(dim=1)
    fft = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    magnitude = fft.abs().numpy()
    b, h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
    max_radius = int(radius.max()) + 1
    profile = np.zeros((b, max_radius), dtype=np.float64)
    counts = np.zeros(max_radius, dtype=np.int64)
    np.add.at(counts, radius.ravel(), 1)
    for item_index in range(b):
        np.add.at(profile[item_index], radius.ravel(), magnitude[item_index].ravel())
    counts = np.maximum(counts, 1)
    return profile / counts
