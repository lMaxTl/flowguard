from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from defenses.victim.blackbox import Blackbox
from defenses.victim.quantize import incremental_kmeans

from flowguard.defenses.prediction.base import PredictionContext, PredictionDefense


@dataclass(slots=True)
class _IdentityQuantizationBlackbox:
    model: torch.nn.Module
    device: torch.device
    num_classes: int
    out_path: str | None = None
    log_path: str | None = None
    log_prefix: str = "predict"
    require_xinfo: bool = False
    top1_preserve: bool = True

    def __call__(
        self,
        query_input: torch.Tensor,
        stat: bool = True,
        return_origin: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.model(query_input.to(self.device))
            probs = F.softmax(logits, dim=1)

        if return_origin:
            return probs, probs
        return probs

    def get_yprime(self, y: torch.Tensor, x_info: torch.Tensor | None = None) -> torch.Tensor:
        return y

    def get_xinfo(self, x: torch.Tensor) -> None:
        return None

    def calc_query_distances(self, queries):
        return Blackbox.calc_query_distances(self, queries)


class QuantizationDefense(PredictionDefense):
    name = "quantization"

    def prepare(self, context: PredictionContext) -> None:
        super().prepare(context)
        identity = _IdentityQuantizationBlackbox(
            model=context.loaded_model.model,
            device=context.loaded_model.device,
            num_classes=context.loaded_model.num_classes,
            out_path=self.parameters.get("out_path", None),
        )
        self.quantizer = incremental_kmeans(
            identity,
            epsilon=self.parameters.get("epsilon", 1.0),
            ydist=self.parameters.get("ydist", "l1"),
            optim=self.parameters.get("optim", False),
            frozen=self.parameters.get("frozen", True),
            ordered_quantization=self.parameters.get("ordered_quantization", True),
            buffer_size=self.parameters.get("buffer_size"),
            trainingset_name=self.parameters.get("trainingset_name", None),
        )

    def transform(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.quantizer.get_yprime(probabilities)
