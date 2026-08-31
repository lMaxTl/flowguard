"""Distilled differentiable surrogate of the FlowGuard++ composite score.

Backpropagating through the exact likelihood ODE (see
:meth:`~flowguard.attacks.adaptive.SurrogateVelocityRegularizer.log_likelihood`)
costs one double-backward per solver step, which bounds the batch size and the
step count an attacker can afford inside a generator loop. This module provides
the cheap alternative the threat model actually permits: the attacker distills
its own CNF-derived component scores into a small feed-forward network, then
optimizes its generator against that network's output at the cost of a single
forward and backward pass.

The distillation uses only the attacker's own resources:

- the attacker's surrogate CNF, which the threat model already grants,
- an unlabeled pool of benign-looking images plus samples from the attacker's
  own generator, so the regression covers the region of input space the
  generator actually visits.

Nothing here reads the defender's model, calibration set, or thresholds. The
resulting network is an *approximation* of the defender's score, which is the
realistic setting: the attacker optimizes a proxy and the residual mismatch
between proxy and deployed detector is part of what the evaluation measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowguard.attacks.adaptive import (
    CompositeWeights,
    SurrogateBenignStats,
    SurrogateVelocityRegularizer,
)

COMPONENT_NAMES: tuple[str, ...] = ("t0", "integral", "typicality")


class ScoreSurrogateNet(nn.Module):
    """Small CNN regressing the three standardized FlowGuard++ components.

    Outputs are in standardized units (benign z-scores), so the composite risk
    is a plain weighted sum of the head and no further calibration is needed at
    attack time.
    """

    def __init__(self, *, in_channels: int = 3, width: int = 64) -> None:
        super().__init__()
        # Retained so a checkpoint can rebuild the same architecture; a plain
        # state_dict load would otherwise silently require the caller to
        # remember the width it trained with.
        self.in_channels = int(in_channels)
        self.width = int(width)

        def norm(channels: int) -> nn.GroupNorm:
            # GroupNorm requires channels to divide evenly into groups, which
            # breaks for the narrow widths used in tests.
            return nn.GroupNorm(min(8, channels), channels)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1),
            norm(width),
            nn.SiLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            norm(width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            norm(width * 2),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, stride=2, padding=1),
            norm(width * 2),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, len(COMPONENT_NAMES)),
        )

    def forward(self, x_01: torch.Tensor) -> torch.Tensor:
        """Map a ``[0, 1]`` image batch to ``(B, 3)`` standardized components."""
        features = self.features(x_01 * 2.0 - 1.0)
        pooled = features.mean(dim=(2, 3))
        return self.head(pooled)


@dataclass(slots=True)
class DistillationReport:
    """Fit quality of the distilled surrogate, per component."""

    num_samples: int
    epochs: int
    final_loss: float
    component_rmse: dict[str, float]
    component_correlation: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "component_rmse": dict(self.component_rmse),
            "component_correlation": dict(self.component_correlation),
        }


class DistilledCompositeScore(nn.Module):
    """Differentiable stand-in for the defender's per-query composite risk."""

    def __init__(
        self,
        network: ScoreSurrogateNet,
        *,
        weights: CompositeWeights | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.weights = weights or CompositeWeights()

    def components(self, x_01: torch.Tensor) -> dict[str, torch.Tensor]:
        predicted = self.network(x_01)
        return {name: predicted[:, index] for index, name in enumerate(COMPONENT_NAMES)}

    def risk(self, x_01: torch.Tensor) -> torch.Tensor:
        """Return the fused per-query risk, differentiable in ``x_01``."""
        predicted = self.network(x_01)
        coefficients = torch.tensor(
            [self.weights.velocity, self.weights.integral, self.weights.typicality],
            device=predicted.device,
            dtype=predicted.dtype,
        )
        return (predicted * coefficients).sum(dim=1)

    forward = risk

    def save(self, path: str | Path, *, stats: SurrogateBenignStats | None = None) -> None:
        payload: dict[str, object] = {
            "state_dict": self.network.state_dict(),
            "component_names": list(COMPONENT_NAMES),
            "architecture": {
                "in_channels": self.network.in_channels,
                "width": self.network.width,
            },
            "weights": {
                "velocity": self.weights.velocity,
                "integral": self.weights.integral,
                "typicality": self.weights.typicality,
            },
        }
        if stats is not None:
            payload["benign_stats"] = {
                field: getattr(stats, field) for field in stats.__dataclass_fields__
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cuda") -> DistilledCompositeScore:
        payload = torch.load(str(path), map_location=device, weights_only=False)
        network = ScoreSurrogateNet(**payload.get("architecture", {}))
        network.load_state_dict(payload["state_dict"])
        network.to(device).eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        weights = CompositeWeights(**payload.get("weights", {}))
        return cls(network, weights=weights)


@torch.no_grad()
def _standardize_targets(
    regularizer: SurrogateVelocityRegularizer,
    batch_01: torch.Tensor,
    *,
    stats: SurrogateBenignStats,
    integral_num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = regularizer.velocity_score(batch_01)
    integral = regularizer.trajectory_integral_score(batch_01, num_steps=integral_num_steps)
    z_velocity = (velocity - stats.velocity_mean) / (abs(stats.velocity_std) + 1e-8)
    z_integral = (integral - stats.integral_mean) / (abs(stats.integral_std) + 1e-8)
    return z_velocity, z_integral


def build_distillation_targets(
    regularizer: SurrogateVelocityRegularizer,
    image_batches: list[torch.Tensor],
    *,
    stats: SurrogateBenignStats,
    integral_num_steps: int = 8,
    likelihood_num_steps: int = 8,
    hutchinson_samples: int = 4,
    likelihood_regularizer: SurrogateVelocityRegularizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score an image pool with the surrogate CNF to build the regression set.

    Runs entirely without the second-order graph, so it is roughly as cheap as
    the defender's own scoring pass and can cover tens of thousands of images.

    Returns:
        ``(images, targets)`` with ``targets`` holding the standardized
        ``(z0, z_int, s_typ)`` triples on the CPU.
    """
    # The typicality target must come from the same flow the attack will
    # optimize against, and that has to be a Gaussian-source CNF: a
    # FlowPure-family checkpoint has no standard-Gaussian p_0, so log p(x)
    # under it is not a density. Distilling an undefined target would hide the
    # problem behind a well-fitted regressor.
    likelihood_source = likelihood_regularizer or regularizer
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in image_batches:
        batch = batch.to(regularizer.device, dtype=torch.float32).clamp(0.0, 1.0)
        z_velocity, z_integral = _standardize_targets(
            regularizer, batch, stats=stats, integral_num_steps=integral_num_steps
        )
        log_likelihood = likelihood_source.log_likelihood(
            batch.to(likelihood_source.device),
            num_steps=likelihood_num_steps,
            hutchinson_samples=hutchinson_samples,
            create_graph=False,
        ).detach().to(batch.device)
        typicality = (log_likelihood - stats.likelihood_mean).abs() / (
            abs(stats.likelihood_std) + 1e-8
        )
        images.append(batch.detach().cpu())
        targets.append(
            torch.stack([z_velocity, z_integral, typicality], dim=1).detach().cpu()
        )
    return torch.cat(images, dim=0), torch.cat(targets, dim=0)


def train_score_surrogate(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    width: int = 64,
    weights: CompositeWeights | None = None,
    seed: int = 0,
) -> tuple[DistilledCompositeScore, DistillationReport]:
    """Fit :class:`ScoreSurrogateNet` to CNF-derived component scores.

    Args:
        images: ``(N, C, H, W)`` tensor in ``[0, 1]``.
        targets: ``(N, 3)`` standardized ``(z0, z_int, s_typ)`` targets.
        device: Training device.
        epochs: Passes over the regression set.
        batch_size: Minibatch size.
        learning_rate: Adam learning rate.
        width: Base channel width of the surrogate CNN.
        weights: Component weights the attacker assumes the defender fuses with.
        seed: Torch seed for reproducible fits.

    Returns:
        The trained differentiable score and a fit-quality report. The
        per-component correlation is the number to check before trusting an
        attack built on this surrogate: a low correlation means the attacker is
        optimizing noise, and the attack should fall back to the exact path.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    network = ScoreSurrogateNet(in_channels=images.shape[1], width=width).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    count = images.shape[0]
    final_loss = float("nan")
    for _epoch in range(max(1, int(epochs))):
        permutation = torch.randperm(count)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, count, batch_size):
            indices = permutation[start : start + batch_size]
            batch_images = images[indices].to(device)
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(network(batch_images), batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        final_loss = epoch_loss / max(batches, 1)

    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            predictions.append(network(images[start : start + batch_size].to(device)).cpu())
    predicted = torch.cat(predictions, dim=0)

    rmse: dict[str, float] = {}
    correlation: dict[str, float] = {}
    for index, name in enumerate(COMPONENT_NAMES):
        error = predicted[:, index] - targets[:, index]
        rmse[name] = float(error.pow(2).mean().sqrt())
        stacked = torch.stack([predicted[:, index], targets[:, index]], dim=0)
        if float(stacked.std(dim=1).min()) > 1e-8:
            correlation[name] = float(torch.corrcoef(stacked)[0, 1])
        else:
            correlation[name] = float("nan")

    report = DistillationReport(
        num_samples=int(count),
        epochs=int(epochs),
        final_loss=float(final_loss),
        component_rmse=rmse,
        component_correlation=correlation,
    )
    return DistilledCompositeScore(network, weights=weights), report
