from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from defenses import datasets as legacy_datasets
from flowguard.experiments.spec import AttackKind, ExperimentSpec


class VictimEvalAdapter(nn.Module):
    """Map substitute-eval images to the victim's native resolution and normalization."""

    def __init__(
        self,
        victim: nn.Module,
        *,
        dataset_name: str,
        victim_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.victim = victim
        self.victim_h, self.victim_w = victim_shape
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        family_key = "imagenet" if family == "tinyimagenet" else family
        mean_std = legacy_datasets.modelfamily_to_mean_std[family_key]
        mean = torch.tensor(mean_std["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(mean_std["std"], dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.victim_h, self.victim_w):
            x = F.interpolate(
                x,
                size=(self.victim_h, self.victim_w),
                mode="bilinear",
                align_corners=False,
            )
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return self.victim(x)


def wrap_victim_for_substitute_eval(
    victim: nn.Module | None,
    spec: ExperimentSpec,
) -> nn.Module | None:
    if victim is None:
        return None
    if spec.attack.kind != AttackKind.DISGUIDE:
        return victim
    family = legacy_datasets.dataset_to_modelfamily[spec.dataset.name]
    if family != "mnist":
        return victim
    from flowguard.training.eval_datasets import victim_native_spatial_shape

    victim_shape = victim_native_spatial_shape(spec)
    adapter = VictimEvalAdapter(
        victim,
        dataset_name=spec.dataset.name,
        victim_shape=victim_shape,
    )
    device = next(victim.parameters()).device
    return adapter.to(device)
