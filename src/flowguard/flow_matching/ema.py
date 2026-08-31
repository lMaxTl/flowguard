from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from facebookresearch/flow_matching examples/image/models/ema.py
import torch
from torch.nn import Module, Parameter, ParameterList


class EMA(Module):
    """Simple exponential moving average wrapper for a model."""

    def __init__(self, model: Module, decay: float = 0.999) -> None:
        super().__init__()
        self.model = model
        self.decay = decay
        self.register_buffer("num_updates", torch.tensor(0))
        self.shadow_params: ParameterList = ParameterList(
            [
                Parameter(parameter.clone().detach(), requires_grad=False)
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        )
        self.backup_params: list[torch.Tensor] = []

    def train(self, mode: bool = True):
        if self.training == mode:
            return super().train(mode)
        if not mode:
            self.backup()
            self.copy_to_model()
        else:
            self.restore_to_model()
        return super().train(mode)

    def update_ema(self) -> None:
        self.num_updates += 1
        num_updates = int(self.num_updates.item())
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        with torch.no_grad():
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            for shadow, parameter in zip(self.shadow_params, parameters, strict=True):
                shadow.sub_((1 - decay) * (shadow - parameter))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        for shadow, parameter in zip(self.shadow_params, parameters, strict=True):
            parameter.data.copy_(shadow.data)

    def backup(self) -> None:
        if not self.training:
            raise RuntimeError("EMA backup can only be created while the wrapper is in train mode.")
        if self.backup_params:
            for parameter, backup in zip(self.model.parameters(), self.backup_params, strict=True):
                backup.data.copy_(parameter.data)
        else:
            self.backup_params = [parameter.clone() for parameter in self.model.parameters()]

    def restore_to_model(self) -> None:
        for parameter, backup in zip(self.model.parameters(), self.backup_params, strict=True):
            parameter.data.copy_(backup.data)
