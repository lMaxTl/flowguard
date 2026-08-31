from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from facebookresearch/flow_matching examples/image/models/nn.py
import math

import torch as th
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs) -> nn.Linear:
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate: float = 0.99) -> None:
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().mul_(scale)
    return module


def mean_flat(tensor: th.Tensor) -> th.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels: int) -> GroupNorm32:
    return GroupNorm32(32, channels)


def timestep_embedding(
    timesteps: th.Tensor,
    dim: int,
    max_period: int = 10000,
) -> th.Tensor:
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag: bool):
    del params
    if flag:
        return th.utils.checkpoint.checkpoint(func, *inputs)
    return func(*inputs)
