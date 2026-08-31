from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from flowguard.serving.model_loader import LoadedModel


@dataclass(slots=True)
class PredictionContext:
    loaded_model: LoadedModel
    defense_name: str = "none"
    parameters: dict[str, Any] = field(default_factory=dict)


class PredictionDefense(ABC):
    name = "none"

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters

    def prepare(self, context: PredictionContext) -> None:
        self.context = context

    def auxiliary(self, inputs: torch.Tensor) -> torch.Tensor | None:
        return None

    @abstractmethod
    def transform(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class IdentityPredictionDefense(PredictionDefense):
    name = "none"

    def transform(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return probabilities


class LegacyPredictionDefense(PredictionDefense):
    legacy_class = None

    def __init__(self, checkpoint_dir: str | Path | None = None, **parameters: Any) -> None:
        super().__init__(**parameters)
        self.checkpoint_dir = str(checkpoint_dir) if checkpoint_dir is not None else None
        self.legacy = None

    def prepare(self, context: PredictionContext) -> None:
        super().prepare(context)
        if self.legacy_class is None:
            raise RuntimeError("legacy_class must be set")
        checkpoint_dir = self.checkpoint_dir or str(context.loaded_model.checkpoint_dir)
        self.legacy = self.legacy_class.from_modeldir(
            checkpoint_dir,
            device=context.loaded_model.device,
            output_type="probs",
            out_path=None,
            **self.parameters,
        )
        self.legacy.eval()

    def auxiliary(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if self.legacy is None or not getattr(self.legacy, "require_xinfo", False):
            return None
        return self.legacy.get_xinfo(inputs)

    def transform(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.legacy is None:
            raise RuntimeError("prepare() must be called before transform()")
        return self.legacy.get_yprime(probabilities, x_info=auxiliary)
