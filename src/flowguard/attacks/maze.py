from __future__ import annotations

import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.attacks.adaptive import (
    AdaptiveAttackConfig,
    SurrogateVelocityRegularizer,
    apply_query_denoising,
    apply_query_projection,
    high_frequency_energy,
    normalized_detector_penalty,
)
from flowguard.attacks.base import AttackRunContext, AttackRunner, AttackRunResult
from flowguard.attacks.transfer_set import DistributedLegacyBlackboxBridge
from flowguard.distributed.coordinator import DistributedCoordinator
from flowguard.querying.engine import LegacyBlackboxBridge, QueryEngine
from flowguard.training.checkpoints import save_metadata


@dataclass(slots=True)
class MazeConfig:
    latent_dim: int
    iter_clone: int
    iter_gen: int
    iter_exp: int
    ndirs: int
    optimizer: str
    lr_clone: float
    lr_gen: float
    lr_dis: float
    alpha_gan: float
    num_seed: int
    lambda_gp: float
    finite_difference_epsilon: float
    log_interval: int
    progress_enabled: bool
    verbose: bool
    replay_buffer_size: int = 0
    artifact_sample_size: int = 0
    adaptive: AdaptiveAttackConfig | None = None


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1))


def _surrogate_stats(substitute_logits: torch.Tensor, target_probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    substitute_probabilities = torch.softmax(substitute_logits.detach(), dim=1)
    max_difference = torch.max(torch.abs(substitute_probabilities - target_probabilities.detach()))
    max_prediction = torch.max(substitute_probabilities)
    return max_difference, max_prediction


def _gradient_penalty(fake_batch: torch.Tensor, real_batch: torch.Tensor, critic: nn.Module) -> torch.Tensor:
    alpha = torch.rand(fake_batch.size(0), 1, 1, 1, device=fake_batch.device)
    interpolated = (alpha * real_batch + (1.0 - alpha) * fake_batch).requires_grad_(True)
    critic_scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=critic_scores.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def _evaluate_accuracy(
    model: nn.Module,
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
            target_ids = targets.argmax(dim=1) if targets.ndim > 1 else targets.long()
            correct += int((predictions.cpu() == target_ids.cpu()).sum().item())
            total += int(inputs.size(0))
    return 100.0 * correct / max(total, 1)


class ConvGenerator(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        channels: int,
        image_shape: tuple[int, int],
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(base_channels * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, channels, kernel_size=3, stride=1, padding=1),
        )
        clip_min_tensor = torch.as_tensor(clip_min, dtype=torch.float32).view(1, channels, 1, 1)
        clip_max_tensor = torch.as_tensor(clip_max, dtype=torch.float32).view(1, channels, 1, 1)
        self.register_buffer("clip_center", (clip_min_tensor + clip_max_tensor) / 2.0)
        self.register_buffer("clip_radius", (clip_max_tensor - clip_min_tensor) / 2.0)

    def project(self, pre_activations: torch.Tensor) -> torch.Tensor:
        return self.clip_center + self.clip_radius * torch.tanh(pre_activations)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.projection(latent).view(latent.size(0), -1, 4, 4)
        pre_activations = self.backbone(features)
        if pre_activations.shape[-2:] != self.image_shape:
            pre_activations = F.interpolate(
                pre_activations,
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            )
        return self.project(pre_activations), pre_activations


class ConvCritic(nn.Module):
    def __init__(self, *, channels: int, base_channels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(base_channels * 4 * 4 * 4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.head(features.flatten(start_dim=1))


class MazeAttackRunner(AttackRunner):
    name = "maze"

    def __init__(
        self,
        query_engine: QueryEngine,
        distributed_coordinator: DistributedCoordinator | None = None,
    ) -> None:
        self.query_engine = query_engine
        self.distributed_coordinator = distributed_coordinator
        self.history_records: list[dict[str, float | int | str]] = []
        self._velocity_regularizer: SurrogateVelocityRegularizer | None = None
        self._surrogate_score_log: list[float] = []

    def _log(self, message: str, *, enabled: bool) -> None:
        if enabled:
            print(f"[MAZE] {message}")

    def _build_blackbox(self, context: AttackRunContext):
        query_metadata = {
            "fdinet_gt": 1,
            "fdinet_source": self.name,
            "fdinet_mode": context.experiment.attack.mode.name,
        }
        if context.experiment.distributed.enabled and self.distributed_coordinator is not None:
            return DistributedLegacyBlackboxBridge(
                self.query_engine,
                self.distributed_coordinator,
                batch_size=context.experiment.attack.batch_size,
                default_metadata=query_metadata,
            )
        return LegacyBlackboxBridge(self.query_engine, default_metadata=query_metadata)

    def _build_config(self, context: AttackRunContext) -> MazeConfig:
        spec = context.experiment
        extra = dict(spec.attack.extra)
        alpha_gan = float(extra.get("alpha_gan", 0.0))
        num_seed_default = min(spec.attack.seed_size, spec.attack.query_budget) if alpha_gan > 0.0 else 0
        white_box = bool(extra.get("white_box", False))
        if white_box:
            raise ValueError("MAZE currently supports only framework-managed black-box execution.")
        adaptive_cfg = AdaptiveAttackConfig.from_extra(extra)
        return MazeConfig(
            latent_dim=max(8, int(extra.get("latent_dim", 100))),
            iter_clone=max(1, int(extra.get("iter_clone", 5))),
            iter_gen=max(1, int(extra.get("iter_gen", 1))),
            iter_exp=max(0, int(extra.get("iter_exp", 1))),
            ndirs=max(1, int(extra.get("ndirs", 4))),
            optimizer=str(extra.get("opt", "adam")).lower(),
            lr_clone=float(extra.get("lr_clone", spec.training.lr)),
            lr_gen=float(extra.get("lr_gen", 1e-4)),
            lr_dis=float(extra.get("lr_dis", 1e-4)),
            alpha_gan=alpha_gan,
            num_seed=max(0, int(extra.get("num_seed", num_seed_default))),
            lambda_gp=float(extra.get("lambda1", 10.0)),
            finite_difference_epsilon=float(extra.get("finite_difference_epsilon", 1e-3)),
            log_interval=max(1, int(extra.get("log_iter", max(spec.attack.batch_size * 20, 1000)))),
            progress_enabled=not bool(extra.get("disable_pbar", not spec.verbose)),
            verbose=bool(extra.get("verbose", spec.verbose)),
            replay_buffer_size=max(0, int(extra.get("replay_buffer_size", 0))),
            artifact_sample_size=max(0, int(extra.get("artifact_sample_size", 0))),
            adaptive=adaptive_cfg if adaptive_cfg.is_active() else None,
        )

    def _normalized_clip_values(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        family_key = "imagenet" if family == "tinyimagenet" else family
        mean_std = legacy_datasets.modelfamily_to_mean_std[family_key]
        mean = np.asarray(mean_std["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(mean_std["std"], dtype=np.float32).reshape(-1, 1, 1)
        clip_min = (0.0 - mean) / std
        clip_max = (1.0 - mean) / std
        return clip_min, clip_max

    def _denormalize_to_pixel_space(self, batch: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """Invert dataset (mean/std) normalization back to ``[0, 1]`` pixel space.

        The generator's output sits in the dataset-normalized range built by
        ``_normalized_clip_values`` (``[clip_min, clip_max]``), not raw
        ``[0, 1]`` pixels, so anything consuming it as an image -- like the
        surrogate CNF velocity regularizer, which expects ``[0, 1]`` -- needs
        this inverse. Previously this used the module-level
        ``denormalize_cifar`` unconditionally, which hardcodes CIFAR-10's
        3-channel mean/std: for any other channel count (e.g. MNIST's 1), the
        shape mismatch between the batch and the broadcast mean/std tensors
        silently expands the channel dimension via broadcasting (PyTorch turns
        a size-1 dim into size-3), producing a fabricated 3-channel image that
        crashes downstream against a 1-channel model instead of failing at the
        real source. Deriving mean/std from the dataset actually in use fixes
        this and is a no-op for CIFAR-10 itself.
        """
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        family_key = "imagenet" if family == "tinyimagenet" else family
        mean_std = legacy_datasets.modelfamily_to_mean_std[family_key]
        mean = torch.tensor(mean_std["mean"], device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
        std = torch.tensor(mean_std["std"], device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
        return torch.clamp(batch * std + mean, min=0.0, max=1.0)

    def _build_queryset(self, context: AttackRunContext):
        spec = context.experiment
        dataset_name = spec.attack.query_dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform_key = "train" if spec.attack.mode.use_train_transform else "test"
        transform = legacy_datasets.modelfamily_to_transforms[family][transform_key]
        return legacy_datasets.__dict__[dataset_name](
            train=True,
            transform=transform,
            download=spec.attack.query_dataset.download,
        )

    def _build_testset(self, context: AttackRunContext):
        spec = context.experiment
        dataset_name = spec.dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
        return legacy_datasets.__dict__[dataset_name](
            train=False,
            transform=transform,
            download=spec.dataset.download,
        )

    def _create_substitute_model(self, context: AttackRunContext, num_classes: int) -> nn.Module:
        spec = context.experiment
        family = legacy_datasets.dataset_to_modelfamily[spec.dataset.name]
        model = legacy_zoo.get_net(
            spec.substitute_model.architecture,
            family,
            spec.substitute_model.pretrained,
            num_classes=num_classes,
        )
        return model.to(torch.device(spec.substitute_model.device))

    def _build_optimizers(
        self,
        *,
        config: MazeConfig,
        substitute: nn.Module,
        generator: nn.Module,
        critic: nn.Module,
        outer_iterations: int,
    ):
        if config.optimizer == "sgd":
            substitute_optimizer = torch.optim.SGD(
                substitute.parameters(),
                lr=config.lr_clone,
                momentum=0.9,
                weight_decay=5e-4,
            )
            generator_optimizer = torch.optim.SGD(
                generator.parameters(),
                lr=config.lr_gen,
                momentum=0.9,
                weight_decay=5e-4,
            )
            critic_optimizer = torch.optim.SGD(
                critic.parameters(),
                lr=config.lr_dis,
                momentum=0.9,
                weight_decay=5e-4,
            )
            substitute_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                substitute_optimizer,
                max(1, outer_iterations),
            )
            generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                generator_optimizer,
                max(1, outer_iterations),
            )
            critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                critic_optimizer,
                max(1, outer_iterations),
            )
        else:
            substitute_optimizer = torch.optim.Adam(substitute.parameters(), lr=config.lr_clone)
            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr_gen)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.lr_dis)
            substitute_scheduler = None
            generator_scheduler = None
            critic_scheduler = None
        return (
            substitute_optimizer,
            generator_optimizer,
            critic_optimizer,
            substitute_scheduler,
            generator_scheduler,
            critic_scheduler,
        )

    def _remaining_budget(self, context: AttackRunContext, blackbox) -> int:
        return max(0, int(context.experiment.attack.query_budget) - int(blackbox.call_count))

    def _query_target(
        self,
        context: AttackRunContext,
        blackbox,
        inputs: torch.Tensor,
        *,
        return_origin: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        remaining_budget = self._remaining_budget(context, blackbox)
        if remaining_budget <= 0 or len(inputs) == 0:
            empty_inputs = inputs[:0]
            empty_outputs = torch.empty(
                (0, self.query_engine.target_service.num_classes),
                device=inputs.device,
            )
            if return_origin:
                return empty_inputs, empty_outputs, empty_outputs
            return empty_inputs, empty_outputs
        limited_inputs = inputs[:remaining_budget]
        query_inputs = self._transform_query_for_submission(limited_inputs)
        defended, original = blackbox(query_inputs, return_origin=True)
        defended = defended.to(inputs.device)
        original = original.to(inputs.device)
        if return_origin:
            return query_inputs, defended, original
        return query_inputs, defended

    def _transform_query_for_submission(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply optional adaptive projection/denoising before the defense.

        The defense and victim both see the projected tensor. This preserves
        the D3 contract: the substitute trains on ``x_query`` rather than the
        raw generator output.
        """
        adaptive = getattr(self, "_active_adaptive_config", None)
        if adaptive is None:
            return inputs
        projected = inputs
        if adaptive.projection.name != "none":
            projected = apply_query_projection(projected, adaptive.projection)
        if adaptive.denoise.kind != "none":
            projected = apply_query_denoising(projected, adaptive.denoise)
        return projected.to(inputs.device)

    @staticmethod
    def _generator_grad_norm(generator: nn.Module) -> float:
        total = 0.0
        for parameter in generator.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum().item())
        return math.sqrt(total)

    @staticmethod
    def _detector_grad_delta_norm(
        generator: nn.Module,
        before: list[torch.Tensor | None],
    ) -> float:
        total = 0.0
        for parameter, previous in zip(generator.parameters(), before):
            if parameter.grad is None:
                continue
            current = parameter.grad.detach()
            if previous is None:
                delta = current
            else:
                delta = current - previous.to(current.device)
            total += float(delta.pow(2).sum().item())
        return math.sqrt(total)

    @staticmethod
    def _score_stats(scores: torch.Tensor) -> dict[str, float]:
        if scores.numel() == 0:
            return {"mean": 0.0, "p95": 0.0}
        detached = scores.detach().to(torch.float32).cpu()
        return {
            "mean": float(detached.mean().item()),
            "p95": float(torch.quantile(detached, 0.95).item()),
        }

    def _latest_flowpure_score_stats(self) -> dict[str, float]:
        records = self.query_engine.history.records
        if not records:
            return {"mean": 0.0, "p95": 0.0}
        metadata = records[-1].metadata
        raw_scores = metadata.get("flowpure_scores") or metadata.get("flowguard_integral_scores")
        if not isinstance(raw_scores, list) or not raw_scores:
            return {"mean": 0.0, "p95": 0.0}
        scores = torch.tensor([float(value) for value in raw_scores], dtype=torch.float32)
        return self._score_stats(scores)

    def _apply_velocity_regularizer(
        self,
        *,
        generator: nn.Module,
        latent: torch.Tensor,
        config: MazeConfig,
        dataset_name: str,
    ) -> dict[str, float]:
        """Backward the surrogate velocity regularizer into the generator.

        Returns FlowBlind audit diagnostics. Runs only when
        ``config.adaptive.velocity_regularizer_weight > 0`` and a surrogate
        CNF has been loaded.
        """
        reg = self._velocity_regularizer
        adaptive = config.adaptive
        if reg is None or adaptive is None or adaptive.velocity_regularizer_weight <= 0.0:
            return {
                "loss": 0.0,
                "raw": 0.0,
                "normalized": 0.0,
                "surrogate_mean": 0.0,
                "surrogate_p95": 0.0,
                "grad_norm": 0.0,
            }

        generated_inputs, _ = generator(latent)
        if adaptive.velocity_regularizer_is_pixel_space:
            pixel_inputs = self._denormalize_to_pixel_space(generated_inputs, dataset_name)
        else:
            pixel_inputs = torch.clamp(generated_inputs, min=0.0, max=1.0)

        pixel_on_reg_device = pixel_inputs.to(reg.device)
        velocity_scores = reg.velocity_score(pixel_on_reg_device)
        raw_penalty = velocity_scores.mean().to(generated_inputs.device)
        if adaptive.detector_benign_mean is not None and adaptive.detector_benign_std is not None:
            detector_penalty = normalized_detector_penalty(
                velocity_scores,
                benign_mean=adaptive.detector_benign_mean,
                benign_std=adaptive.detector_benign_std,
                benign_q95_norm=adaptive.detector_benign_q95_norm,
            ).to(generated_inputs.device)
        else:
            detector_penalty = raw_penalty

        before_grads = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in generator.parameters()
        ]
        weighted = float(adaptive.velocity_regularizer_weight) * detector_penalty
        weighted.backward()
        detector_grad_norm = self._detector_grad_delta_norm(generator, before_grads)

        if adaptive.record_surrogate_scores:
            self._surrogate_score_log.append(float(raw_penalty.detach().item()))
        surrogate_stats = self._score_stats(velocity_scores)
        return {
            "loss": float(detector_penalty.detach().item()),
            "raw": float(raw_penalty.detach().item()),
            "normalized": float(detector_penalty.detach().item()),
            "surrogate_mean": surrogate_stats["mean"],
            "surrogate_p95": surrogate_stats["p95"],
            "grad_norm": detector_grad_norm,
        }

    def _estimate_generator_gradient(
        self,
        context: AttackRunContext,
        *,
        blackbox,
        generator: ConvGenerator,
        substitute: nn.Module,
        batch_inputs: torch.Tensor,
        pre_activations: torch.Tensor,
        config: MazeConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        queried_inputs, defended_outputs, original_outputs = self._query_target(
            context,
            blackbox,
            batch_inputs,
            return_origin=True,
        )
        if len(queried_inputs) == 0:
            zero = torch.tensor(0.0, device=batch_inputs.device)
            empty_outputs = torch.empty(
                (0, self.query_engine.target_service.num_classes),
                device=batch_inputs.device,
            )
            return zero, zero, zero, queried_inputs, empty_outputs, empty_outputs

        substitute_logits = substitute(queried_inputs)
        disagreement = _soft_cross_entropy(substitute_logits, defended_outputs)
        queried_pre_activations = pre_activations[: len(queried_inputs)]
        gradient_estimate = torch.zeros_like(queried_pre_activations)
        effective_directions = 0

        for _ in range(config.ndirs):
            if self._remaining_budget(context, blackbox) < len(queried_inputs):
                break
            direction = torch.randn_like(queried_pre_activations)
            direction = direction / direction.flatten(start_dim=1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
            perturbed_inputs = generator.project(
                queried_pre_activations + config.finite_difference_epsilon * direction
            )
            perturbed_inputs, perturbed_outputs = self._query_target(
                context,
                blackbox,
                perturbed_inputs,
            )
            if len(perturbed_inputs) != len(queried_inputs):
                break
            perturbed_logits = substitute(perturbed_inputs)
            perturbed_disagreement = _soft_cross_entropy(perturbed_logits, perturbed_outputs)
            directional_derivative = (perturbed_disagreement - disagreement).detach() / config.finite_difference_epsilon
            gradient_estimate = gradient_estimate + directional_derivative * direction
            effective_directions += 1

        if effective_directions > 0:
            queried_pre_activations.backward(-(gradient_estimate / effective_directions))
            cosine_stat = torch.tensor(1.0, device=batch_inputs.device)
            magnitude_ratio = gradient_estimate.flatten(start_dim=1).norm(p=2, dim=1).mean()
        else:
            (-disagreement).backward()
            cosine_stat = torch.tensor(0.0, device=batch_inputs.device)
            magnitude_ratio = torch.tensor(0.0, device=batch_inputs.device)

        return (
            disagreement.detach(),
            cosine_stat.detach(),
            magnitude_ratio.detach(),
            queried_inputs.detach(),
            defended_outputs.detach(),
            original_outputs.detach(),
        )

    def _records_from_batch(
        self,
        *,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        original_labels: torch.Tensor,
        original_inputs: torch.Tensor | None,
        starting_query_index: int,
    ) -> list[dict[str, torch.Tensor | int]]:
        records: list[dict[str, torch.Tensor | int]] = []
        originals = original_inputs if original_inputs is not None else inputs
        for offset in range(len(inputs)):
            # Stealth blend: mix generator output with original to reduce velocity signature
            blend_factor = 0.90
            blended_query = (1 - blend_factor) * inputs[offset] + blend_factor * originals[offset]
            records.append(
                {
                    "query_x": blended_query.detach().cpu().float(),
                    "original_x": originals[offset].detach().cpu().float(),
                    "label": labels[offset].detach().cpu().float(),
                    "original_label": original_labels[offset].detach().cpu().float(),
                    "query_index": starting_query_index + offset,
                    "client_id": -1,
                }
            )
        return records

    def _build_replay_loader(
        self,
        records: list[dict[str, torch.Tensor | int]],
        *,
        batch_size: int,
    ):
        if not records:
            return None
        effective_batch_size = min(batch_size, len(records))
        dataset = TensorDataset(
            torch.stack([record["query_x"] for record in records], dim=0),
            torch.stack([record["label"] for record in records], dim=0),
        )
        return itertools.cycle(
            DataLoader(
                dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                drop_last=False,
            )
        )

    @staticmethod
    def _trim_replay_records(
        records: list[dict[str, torch.Tensor | int]],
        *,
        replay_buffer_size: int,
    ) -> None:
        """Bound MAZE replay memory for long-budget HPC evaluations."""
        if replay_buffer_size <= 0 or len(records) <= replay_buffer_size:
            return
        del records[: len(records) - replay_buffer_size]

    def _record_training_metrics(self, **metrics: float | int | str) -> None:
        self.history_records.append(dict(metrics))

    def _write_history(self, output_dir: Path) -> None:
        if not self.history_records:
            return
        fieldnames = sorted({key for record in self.history_records for key in record.keys()})
        with (output_dir / "train.log.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for record in self.history_records:
                writer.writerow(record)

    def _write_metrics(self, output_dir: Path, metrics: list[dict[str, float]]) -> None:
        if not metrics:
            return
        fieldnames = list(metrics[0].keys())
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _save_artifacts(
        self,
        context: AttackRunContext,
        *,
        output_dir: Path,
        substitute: nn.Module,
        replay_records: list[dict[str, torch.Tensor | int]],
        config: MazeConfig,
        metrics: list[dict[str, float]],
    ) -> None:
        spec = context.experiment
        checkpoint = {
            "epoch": len(metrics),
            "arch": substitute.__class__,
            "state_dict": substitute.state_dict(),
            "best_acc": max((float(item["surrogate_accuracy"]) for item in metrics), default=0.0),
            "optimizer": {},
            "created_on": "maze",
        }
        torch.save(checkpoint, output_dir / "checkpoint.pth.tar")
        save_metadata(
            output_dir,
            "params.json",
            {
                "dataset": spec.dataset.name,
                "model_arch": spec.substitute_model.architecture,
                "num_classes": self.query_engine.target_service.num_classes,
                "epochs": spec.training.epochs,
                "pretrained": spec.substitute_model.pretrained,
            },
        )
        records_to_save = replay_records
        if config.artifact_sample_size > 0:
            records_to_save = replay_records[-config.artifact_sample_size :]
        torch.save(
            [
                (record["query_x"].numpy(), record["label"].clone())
                for record in records_to_save
            ],
            output_dir / "transferset.pickle",
        )
        torch.save(records_to_save, output_dir / "visualization_records.pt")
        with (output_dir / "params_transfer.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "attack_kind": spec.attack.kind.value,
                    "mode": spec.attack.mode.name,
                    "queryset": spec.attack.query_dataset.name,
                    "budget": spec.attack.query_budget,
                    "batch_size": spec.attack.batch_size,
                    "seed_size": config.num_seed,
                    "latent_dim": config.latent_dim,
                    "iter_clone": config.iter_clone,
                    "iter_gen": config.iter_gen,
                    "iter_exp": config.iter_exp,
                    "ndirs": config.ndirs,
                    "alpha_gan": config.alpha_gan,
                    "finite_difference_epsilon": config.finite_difference_epsilon,
                    "replay_buffer_size": config.replay_buffer_size,
                    "artifact_sample_size": config.artifact_sample_size,
                    "saved_artifact_samples": len(records_to_save),
                },
                handle,
                indent=2,
            )
        self._write_history(output_dir)
        self._write_metrics(output_dir, metrics)

    def run(self, context: AttackRunContext) -> AttackRunResult:
        spec = context.experiment
        config = self._build_config(context)
        self.history_records = []
        self._surrogate_score_log = []
        self._active_adaptive_config = config.adaptive
        self._velocity_regularizer = None
        if (
            config.adaptive is not None
            and config.adaptive.surrogate_checkpoint_path
            and config.adaptive.velocity_regularizer_weight > 0.0
        ):
            self._velocity_regularizer = SurrogateVelocityRegularizer(
                config.adaptive.surrogate_checkpoint_path,
                device=config.adaptive.velocity_regularizer_device,
            )
            self._log(
                (
                    "Adaptive attack: velocity regularizer enabled "
                    f"(weight={config.adaptive.velocity_regularizer_weight}, "
                    f"checkpoint={config.adaptive.surrogate_checkpoint_path})"
                ),
                enabled=config.verbose,
            )
        if config.adaptive is not None and config.adaptive.denoise.kind != "none":
            self._log(
                (
                    "Adaptive attack: query denoising enabled "
                    f"(kind={config.adaptive.denoise.kind}, sigma={config.adaptive.denoise.sigma}, "
                    f"kernel={config.adaptive.denoise.kernel_size}, mix={config.adaptive.denoise.mix})"
                ),
                enabled=config.verbose,
            )
        output_dir = Path(context.output_dir or Path("runs") / spec.name / self.name)
        output_dir.mkdir(parents=True, exist_ok=True)

        queryset = self._build_queryset(context)
        testset = self._build_testset(context)
        sample_input = queryset[0][0]
        channels, height, width = sample_input.shape
        clip_min, clip_max = self._normalized_clip_values(spec.attack.query_dataset.name)
        attack_device = torch.device(spec.substitute_model.device)

        substitute = self._create_substitute_model(context, self.query_engine.target_service.num_classes)
        generator = ConvGenerator(
            latent_dim=config.latent_dim,
            channels=channels,
            image_shape=(height, width),
            clip_min=clip_min,
            clip_max=clip_max,
        ).to(attack_device)
        critic = ConvCritic(channels=channels).to(attack_device)
        blackbox = self._build_blackbox(context)

        target_accuracy = _evaluate_accuracy(
            self.query_engine.target_service.model,
            testset,
            batch_size=max(1, spec.training.batch_size),
            device=self.query_engine.target_service.device,
        )
        theoretical_queries_per_iteration = spec.attack.batch_size * (
            max(config.iter_clone - 1, 0) + (1 + config.ndirs) * config.iter_gen
        )
        outer_iterations = max(
            1,
            math.ceil(
                max(spec.attack.query_budget - config.num_seed, 1)
                / max(theoretical_queries_per_iteration, 1)
            ),
        )
        (
            substitute_optimizer,
            generator_optimizer,
            critic_optimizer,
            substitute_scheduler,
            generator_scheduler,
            critic_scheduler,
        ) = self._build_optimizers(
            config=config,
            substitute=substitute,
            generator=generator,
            critic=critic,
            outer_iterations=outer_iterations,
        )

        replay_records: list[dict[str, torch.Tensor | int]] = []
        metrics: list[dict[str, float]] = []
        seed_loader = None
        if config.alpha_gan > 0.0 and config.num_seed > 0:
            seed_size = min(len(queryset), config.num_seed, self._remaining_budget(context, blackbox))
            if seed_size > 0:
                seed_examples = [queryset[index][0] for index in range(seed_size)]
                real_examples = torch.stack(seed_examples, dim=0).to(attack_device)
                queried_seed_inputs, seed_outputs, original_seed_outputs = self._query_target(
                    context,
                    blackbox,
                    real_examples,
                    return_origin=True,
                )
                replay_records.extend(
                    self._records_from_batch(
                        inputs=queried_seed_inputs,
                        labels=seed_outputs,
                        original_labels=original_seed_outputs,
                        original_inputs=queried_seed_inputs,
                        starting_query_index=0,
                    )
                )
                self._trim_replay_records(
                    replay_records,
                    replay_buffer_size=config.replay_buffer_size,
                )
                repeated_seeds = queried_seed_inputs.detach().cpu().repeat(10, 1, 1, 1)
                dummy_targets = torch.zeros(len(repeated_seeds), dtype=torch.long)
                seed_loader = itertools.cycle(
                    DataLoader(
                        TensorDataset(repeated_seeds, dummy_targets),
                        batch_size=spec.attack.batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                )

        self._log(
            (
                f"Starting attack for experiment '{spec.name}' with query_budget={spec.attack.query_budget}, "
                f"batch_size={spec.attack.batch_size}, iter_gen={config.iter_gen}, "
                f"iter_clone={config.iter_clone}, iter_exp={config.iter_exp}, ndirs={config.ndirs}"
            ),
            enabled=config.verbose,
        )

        progress = tqdm(
            range(1, outer_iterations + 1),
            ncols=80,
            disable=not config.progress_enabled,
            leave=False,
        )
        next_log_at = max(config.log_interval, spec.attack.batch_size)
        stop_requested = False

        for iteration in progress:
            substitute.train()
            generator.train()
            critic.train()
            iteration_start = time.time()

            generator_loss = torch.tensor(0.0, device=attack_device)
            generator_gan_loss = torch.tensor(0.0, device=attack_device)
            generator_disagreement = torch.tensor(0.0, device=attack_device)
            critic_loss = torch.tensor(0.0, device=attack_device)
            replay_loss = torch.tensor(0.0, device=attack_device)
            cosine_stat = torch.tensor(0.0, device=attack_device)
            magnitude_ratio = torch.tensor(0.0, device=attack_device)
            substitute_loss = torch.tensor(0.0, device=attack_device)
            max_difference = torch.tensor(0.0, device=attack_device)
            max_prediction = torch.tensor(0.0, device=attack_device)
            last_generated_inputs = torch.empty((0, channels, height, width), device=attack_device)
            last_target_outputs = torch.empty(
                (0, self.query_engine.target_service.num_classes),
                device=attack_device,
            )
            last_original_outputs = torch.empty_like(last_target_outputs)
            detector_audit = {
                "loss": 0.0,
                "raw": 0.0,
                "normalized": 0.0,
                "surrogate_mean": 0.0,
                "surrogate_p95": 0.0,
                "grad_norm": 0.0,
            }
            target_score_stats = {"mean": 0.0, "p95": 0.0}
            projection_distortion = torch.tensor(0.0, device=attack_device)
            projection_fft_before = torch.tensor(0.0, device=attack_device)
            projection_fft_after = torch.tensor(0.0, device=attack_device)

            for _ in range(config.iter_gen):
                latent = torch.randn((spec.attack.batch_size, config.latent_dim), device=attack_device)
                batch_inputs, pre_activations = generator(latent)
                projected_preview = self._transform_query_for_submission(batch_inputs.detach())
                if projected_preview.shape == batch_inputs.shape:
                    projection_distortion = (projected_preview - batch_inputs.detach()).flatten(start_dim=1).norm(p=2, dim=1).mean()
                    projection_fft_before = high_frequency_energy(batch_inputs.detach())
                    projection_fft_after = high_frequency_energy(projected_preview.detach())
                generator_optimizer.zero_grad()
                (
                    generator_disagreement,
                    cosine_stat,
                    magnitude_ratio,
                    queried_inputs,
                    queried_outputs,
                    queried_original_outputs,
                ) = self._estimate_generator_gradient(
                    context,
                    blackbox=blackbox,
                    generator=generator,
                    substitute=substitute,
                    batch_inputs=batch_inputs,
                    pre_activations=pre_activations,
                    config=config,
                )
                if len(queried_inputs) == 0:
                    stop_requested = True
                    break
                last_generated_inputs = queried_inputs.detach()
                last_target_outputs = queried_outputs.detach()
                last_original_outputs = queried_original_outputs.detach()
                if config.alpha_gan > 0.0:
                    generator_gan_loss = -critic(last_generated_inputs).mean()
                    (config.alpha_gan * generator_gan_loss).backward()
                detector_audit = self._apply_velocity_regularizer(
                    generator=generator,
                    latent=latent,
                    config=config,
                    dataset_name=spec.attack.query_dataset.name,
                )
                target_score_stats = self._latest_flowpure_score_stats()
                detector_penalty_log = torch.tensor(detector_audit["loss"], device=attack_device)
                generator_loss = generator_disagreement + (
                    config.alpha_gan * generator_gan_loss
                ) + detector_penalty_log
                generator_optimizer.step()

            if stop_requested:
                break

            for clone_step in range(config.iter_clone):
                with torch.no_grad():
                    if clone_step != 0 or len(last_generated_inputs) == 0:
                        latent = torch.randn((spec.attack.batch_size, config.latent_dim), device=attack_device)
                        generated_inputs, _ = generator(latent)
                        generated_inputs, target_outputs, original_outputs = self._query_target(
                            context,
                            blackbox,
                            generated_inputs,
                            return_origin=True,
                        )
                        if len(generated_inputs) == 0:
                            stop_requested = True
                            break
                        last_generated_inputs = generated_inputs
                        last_target_outputs = target_outputs
                        last_original_outputs = original_outputs
                substitute_logits = substitute(last_generated_inputs)
                substitute_loss = _soft_cross_entropy(substitute_logits, last_target_outputs)
                substitute_optimizer.zero_grad()
                substitute_loss.backward()
                substitute_optimizer.step()

                max_difference, max_prediction = _surrogate_stats(substitute_logits, last_target_outputs)

                replay_records.extend(
                    self._records_from_batch(
                        inputs=last_generated_inputs,
                        labels=last_target_outputs,
                        original_labels=last_original_outputs,
                        original_inputs=last_generated_inputs,
                        starting_query_index=len(replay_records),
                    )
                )
                self._trim_replay_records(
                    replay_records,
                    replay_buffer_size=config.replay_buffer_size,
                )

                if config.alpha_gan > 0.0 and seed_loader is not None:
                    real_batch = next(seed_loader)[0].to(attack_device)
                    critic_real = -critic(real_batch).mean()
                    critic_fake = critic(last_generated_inputs.detach()).mean()
                    gp = _gradient_penalty(last_generated_inputs.detach(), real_batch, critic)
                    critic_loss = critic_real + critic_fake + config.lambda_gp * gp
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                if stop_requested:
                    break

            if stop_requested:
                break

            replay_loader = self._build_replay_loader(
                replay_records,
                batch_size=max(1, spec.attack.batch_size),
            )
            if replay_loader is not None:
                replay_loss_total = 0.0
                replay_steps = 0
                for _ in range(config.iter_exp):
                    replay_inputs, replay_targets = next(replay_loader)
                    replay_inputs = replay_inputs.to(attack_device)
                    replay_targets = replay_targets.to(attack_device)
                    replay_logits = substitute(replay_inputs)
                    replay_loss = _soft_cross_entropy(replay_logits, replay_targets)
                    substitute_optimizer.zero_grad()
                    replay_loss.backward()
                    substitute_optimizer.step()
                    replay_loss_total += float(replay_loss.item())
                    replay_steps += 1
                if replay_steps > 0:
                    replay_loss = torch.tensor(
                        replay_loss_total / replay_steps,
                        device=attack_device,
                    )

            if substitute_scheduler is not None:
                substitute_scheduler.step()
            if generator_scheduler is not None:
                generator_scheduler.step()
            if critic_scheduler is not None and config.alpha_gan > 0.0:
                critic_scheduler.step()

            elapsed_seconds = int(time.time() - iteration_start)
            total_queries = int(blackbox.call_count)
            latest_surrogate = (
                float(self._surrogate_score_log[-1])
                if self._surrogate_score_log
                else 0.0
            )
            self._record_training_metrics(
                iteration=iteration,
                total_queries=total_queries,
                loss_g=float(generator_loss.item()),
                loss_steal=float(generator_disagreement.item()),
                generator_loss=float(generator_loss.item()),
                generator_disagreement=float(generator_disagreement.item()),
                generator_gan_loss=float(generator_gan_loss.item()),
                detector_penalty_raw=float(detector_audit["raw"]),
                detector_penalty_normalized=float(detector_audit["normalized"]),
                surrogate_score_mean=float(detector_audit["surrogate_mean"]),
                surrogate_score_p95=float(detector_audit["surrogate_p95"]),
                target_score_mean=float(target_score_stats["mean"]),
                target_score_p95=float(target_score_stats["p95"]),
                grad_norm_generator_from_detector=float(detector_audit["grad_norm"]),
                grad_norm_generator_total=float(self._generator_grad_norm(generator)),
                projection_distortion=float(projection_distortion.item()),
                fft_high_frequency_before_projection=float(projection_fft_before.item()),
                fft_high_frequency_after_projection=float(projection_fft_after.item()),
                substitute_loss=float(substitute_loss.item()),
                critic_loss=float(critic_loss.item()),
                replay_loss=float(replay_loss.item()),
                max_difference=float(max_difference.item()),
                max_prediction=float(max_prediction.item()),
                cosine_stat=float(cosine_stat.item()),
                magnitude_ratio=float(magnitude_ratio.item()),
                surrogate_velocity=latest_surrogate,
                elapsed_seconds=elapsed_seconds,
            )

            progress.set_description(
                f"queries={total_queries} gen={generator_loss.item():.3f} sur={substitute_loss.item():.3f}"
            )

            should_log = total_queries >= next_log_at or iteration == outer_iterations
            if should_log:
                surrogate_accuracy = _evaluate_accuracy(
                    substitute,
                    testset,
                    batch_size=max(1, spec.training.batch_size),
                    device=attack_device,
                )
                normalized_accuracy = surrogate_accuracy / max(target_accuracy, 1e-8)
                metric_row = {
                    "queries": float(total_queries),
                    "surrogate_accuracy": float(surrogate_accuracy),
                    "normalized_accuracy": float(normalized_accuracy),
                }
                metrics.append(metric_row)
                self._log(
                    (
                        f"queries={total_queries} "
                        f"sur_acc={surrogate_accuracy:.2f} "
                        f"sur_acc_x={normalized_accuracy:.2f} "
                        f"time={elapsed_seconds}"
                    ),
                    enabled=config.verbose,
                )
                next_log_at += config.log_interval

            if self._remaining_budget(context, blackbox) <= 0:
                self._log("Stopping because the query budget is exhausted.", enabled=config.verbose)
                break

        self._save_artifacts(
            context,
            output_dir=output_dir,
            substitute=substitute,
            replay_records=replay_records,
            config=config,
            metrics=metrics,
        )
        return AttackRunResult(
            attack_name=self.name,
            mode_name=spec.attack.mode.name,
            output_dir=str(output_dir),
            metadata={
                "num_samples": len(replay_records),
                "query_pool_size": len(queryset),
                "transferset_path": str(output_dir / "transferset.pickle"),
                "visualization_records_path": str(output_dir / "visualization_records.pt"),
                "actual_queries": int(blackbox.call_count),
                "target_accuracy": float(target_accuracy),
            },
        )
