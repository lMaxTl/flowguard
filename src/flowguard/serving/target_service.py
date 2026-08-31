from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from flowguard.defenses.prediction.base import (
    IdentityPredictionDefense,
    PredictionDefense,
)
from flowguard.serving.model_loader import LoadedModel
from flowguard.serving.response_formats import OutputFormat, format_output


@dataclass(slots=True)
class PredictionBundle:
    logits: torch.Tensor
    probabilities: torch.Tensor
    payload: torch.Tensor


class TargetService:
    def __init__(
        self,
        loaded_model: LoadedModel,
        prediction_defenses: Iterable[PredictionDefense] | None = None,
    ) -> None:
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.device = loaded_model.device
        self.num_classes = loaded_model.num_classes
        self.prediction_defenses = list(prediction_defenses or [IdentityPredictionDefense()])

    @property
    def top1_preserve(self) -> bool:
        for defense in self.prediction_defenses:
            legacy = getattr(defense, "legacy", None)
            if legacy is not None and hasattr(legacy, "top1_preserve"):
                return bool(legacy.top1_preserve)
        return True

    @property
    def requires_xinfo(self) -> bool:
        for defense in self.prediction_defenses:
            legacy = getattr(defense, "legacy", None)
            if legacy is not None and hasattr(legacy, "require_xinfo") and legacy.require_xinfo:
                return True
        return False

    def health(self) -> dict[str, str]:
        return {"status": "ok", "device": str(self.device), "model_arch": self.loaded_model.model_arch}

    def get_auxiliary_info(self, inputs: torch.Tensor) -> list[torch.Tensor | None]:
        return [defense.auxiliary(inputs.to(self.device)) for defense in self.prediction_defenses]

    def apply_prediction_defenses(
        self, inputs: torch.Tensor, probabilities: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        defended = probabilities
        auxiliaries = self.get_auxiliary_info(inputs)
        for defense, auxiliary in zip(self.prediction_defenses, auxiliaries, strict=True):
            defended = defense.transform(
                defended,
                logits=logits,
                inputs=inputs.to(self.device),
                auxiliary=auxiliary,
            )
        return defended

    def transform_probabilities(
        self,
        probabilities: torch.Tensor,
        *,
        inputs: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        transformed = probabilities
        for defense in self.prediction_defenses:
            aux = auxiliary
            if aux is None and inputs is not None:
                aux = defense.auxiliary(inputs.to(self.device))
            transformed = defense.transform(
                transformed,
                logits=None,
                inputs=inputs.to(self.device) if inputs is not None else None,
                auxiliary=aux,
            )
        return transformed

    def predict_bundle(
        self, inputs: torch.Tensor, output_format: OutputFormat = OutputFormat.SOFT
    ) -> PredictionBundle:
        batch = inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)
        defended_probabilities = self.apply_prediction_defenses(batch, probabilities, logits)
        payload = format_output(
            logits=logits,
            probabilities=defended_probabilities,
            output_format=output_format,
        )
        return PredictionBundle(
            logits=logits.detach(),
            probabilities=defended_probabilities.detach(),
            payload=payload.detach(),
        )

    def predict(self, inputs: torch.Tensor, output_format: OutputFormat = OutputFormat.SOFT) -> torch.Tensor:
        return self.predict_bundle(inputs, output_format=output_format).payload
