from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F

from flowguard.defenses.query.base import QueryContext, QueryDefense
from flowguard.flow_matching.training import (
    FlowMatchingCheckpoint,
    VelocityFieldWrapper,
    load_flow_matching_checkpoint,
)


_MODELFAMILY_MEAN_STD: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

_CHECKPOINT_CACHE: dict[str, FlowMatchingCheckpoint] = {}
_WARNED_GAUSS_CHECKPOINTS: set[str] = set()


_NOISE_TYPE_METADATA_FILES: tuple[tuple[str, str], ...] = (
    ("flowpure_boundary_config.json", "boundary"),
    ("flowpure_pgd_config.json", "pgd"),
)


def _infer_noise_type(checkpoint: FlowMatchingCheckpoint) -> str:
    """Return the training noise type recorded on the checkpoint.

    FlowPure^PGD / FlowPure^CW / FlowPureBoundary trainers stamp `noise_type`
    onto a sidecar JSON metadata file. Legacy generative Gaussian CNFs do not
    have this field.
    """
    ckpt_dir = checkpoint.checkpoint_path.parent
    for filename, default in _NOISE_TYPE_METADATA_FILES:
        payload_path = ckpt_dir / filename
        if payload_path.exists():
            try:
                import json

                with payload_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return str(data.get("noise_type", default)).lower()
            except (OSError, ValueError):
                pass
    return "gauss"


class FlowPureQueryDefense(QueryDefense):
    """Detect anomalous queries via initial velocity norm at t=0."""

    name = "flowpure"

    def __init__(
        self,
        fm_checkpoint_path: str,
        *,
        dataset_name: str = "CIFAR10",
        device: str = "cpu",
        velocity_threshold: float = 5000.0,
        inputs_normalized: bool = True,
        input_modelfamily: str | None = None,
        audit_only: bool = False,
        **parameters: Any,
    ) -> None:
        super().__init__(
            fm_checkpoint_path=fm_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
            velocity_threshold=velocity_threshold,
            inputs_normalized=inputs_normalized,
            input_modelfamily=input_modelfamily,
            audit_only=audit_only,
            **parameters,
        )
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.velocity_threshold = float(velocity_threshold)
        self.inputs_normalized = bool(inputs_normalized)
        self.input_modelfamily = input_modelfamily or self._infer_modelfamily(dataset_name)
        self.audit_only = bool(audit_only)

        cache_key = f"{fm_checkpoint_path}_{self.device}"
        if cache_key not in _CHECKPOINT_CACHE:
            _CHECKPOINT_CACHE[cache_key] = load_flow_matching_checkpoint(
                fm_checkpoint_path,
                device=self.device,
            )
        self.flow_checkpoint = _CHECKPOINT_CACHE[cache_key]
        self.flow_checkpoint.model.eval()

        self.velocity_model = VelocityFieldWrapper(
            model=self.flow_checkpoint.model,
            class_conditioned=self.flow_checkpoint.dataset_config.class_conditioned,
        )

        self.noise_type = _infer_noise_type(self.flow_checkpoint)
        if self.noise_type == "gauss" and fm_checkpoint_path not in _WARNED_GAUSS_CHECKPOINTS:
            _WARNED_GAUSS_CHECKPOINTS.add(fm_checkpoint_path)
            warnings.warn(
                "FlowPureQueryDefense loaded a Gaussian-source CNF checkpoint "
                f"({fm_checkpoint_path}). The paper's detection signal "
                "||v(t=0, x)||^2 requires a FlowPure^PGD or FlowPure^CW "
                "checkpoint (x_0 = adversarial, x_1 = clean). Scores from a "
                "Gaussian CNF will not meaningfully separate adversarial from "
                "benign inputs.",
                RuntimeWarning,
                stacklevel=2,
            )

    def before_query(
        self,
        batch: torch.Tensor,
        context: QueryContext,
    ) -> tuple[torch.Tensor, QueryContext]:
        scores = self._estimate_velocity_scores(batch)
        context.metadata["flowpure_scores"] = scores.detach().cpu().tolist()
        is_anomalous = scores > self.velocity_threshold
        context.metadata["flowpure_blocked"] = is_anomalous.detach().cpu().tolist()
        context.metadata["flowpure_threshold"] = float(self.velocity_threshold)

        if not self.audit_only and bool(torch.any(is_anomalous).item()):
            max_score = float(scores.max().item())
            raise RuntimeError(
                "Query blocked by FlowPure defense. "
                f"Maximum velocity score {max_score:.4f} exceeds threshold "
                f"{self.velocity_threshold:.4f}."
            )
        return batch, context

    def _estimate_velocity_scores(self, batch: torch.Tensor) -> torch.Tensor:
        fm_inputs = self._prepare_inputs(batch)
        t_zero = torch.zeros((fm_inputs.shape[0],), device=self.device, dtype=fm_inputs.dtype)
        with torch.no_grad():
            velocity = self.velocity_model(fm_inputs, t_zero)
        return velocity.flatten(start_dim=1).pow(2).sum(dim=1)

    def calibrate_threshold(
        self,
        benign_batch: torch.Tensor,
        *,
        target_fpr: float = 0.05,
    ) -> float:
        """Pick a detection threshold at `target_fpr` on a benign reference batch.

        Uses the (1 - target_fpr) empirical quantile of velocity scores on
        benign samples, which is the standard way to set a detector operating
        point given a target false-positive rate.
        """
        if not 0.0 < target_fpr < 1.0:
            raise ValueError(f"target_fpr must be in (0, 1); got {target_fpr}.")
        scores = self._estimate_velocity_scores(benign_batch)
        quantile = torch.quantile(scores.detach().cpu(), 1.0 - float(target_fpr))
        threshold = float(quantile.item())
        self.velocity_threshold = threshold
        return threshold

    def _prepare_inputs(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.detach().to(self.device, dtype=torch.float32)
        if self.inputs_normalized:
            mean, std = _MODELFAMILY_MEAN_STD[self.input_modelfamily]
            mean_tensor = torch.tensor(mean, device=self.device, dtype=x.dtype).view(1, -1, 1, 1)
            std_tensor = torch.tensor(std, device=self.device, dtype=x.dtype).view(1, -1, 1, 1)
            x = x * std_tensor + mean_tensor
        x = torch.clamp(x, min=0.0, max=1.0)

        expected_size = self.flow_checkpoint.dataset_config.image_size
        if x.ndim != 4:
            raise ValueError(f"Expected image batch with shape (B, C, H, W); got {tuple(x.shape)}.")
        if x.shape[-2:] != (expected_size, expected_size):
            x = F.interpolate(
                x,
                size=(expected_size, expected_size),
                mode="bilinear",
                align_corners=False,
            )
        return x * 2.0 - 1.0

    @staticmethod
    def _infer_modelfamily(dataset_name: str) -> str:
        normalized = dataset_name.lower()
        if "mnist" in normalized:
            return "mnist"
        if "cifar" in normalized or normalized in {"svhn", "stl10"}:
            return "cifar"
        return "imagenet"
