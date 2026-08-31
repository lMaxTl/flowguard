from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from facebookresearch/flow_matching examples/image/models/unet.py
# which itself is derived from OpenAI guided-diffusion.
import math
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowguard.flow_matching.nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


class ConstantEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.embedding_table = nn.Parameter(torch.empty((1, out_channels)))
        nn.init.uniform_(self.embedding_table, -(in_channels**0.5), in_channels**0.5)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.embedding_table.repeat(emb.shape[0], 1)


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the block to `x` with timestep embeddings `emb`."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        emb_off: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        if emb_off:
            self.emb_layers = ConstantEmbedding(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            )
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            self._forward,
            (x, emb),
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest = self.in_layers[:-1]
            in_conv = self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm = self.out_layers[0]
            out_rest = self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ) -> None:
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            if channels % num_head_channels != 0:
                raise ValueError("channels must be divisible by num_head_channels")
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            self._forward,
            (x,),
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = x.shape
        residual = x.reshape(batch, channels, -1)
        qkv = self.qkv(self.norm(residual))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (residual + h).reshape(batch, channels, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError("qkv width must be divisible by 3 * n_heads")
        channels = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(batch * self.n_heads, channels * 3, length).split(channels, dim=1)
        scale = 1 / math.sqrt(math.sqrt(channels))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        attention = torch.einsum("bts,bcs->bct", weight, v)
        return attention.reshape(batch, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError("qkv width must be divisible by 3 * n_heads")
        channels = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(channels))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(batch * self.n_heads, channels, length),
            (k * scale).view(batch * self.n_heads, channels, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        attention = torch.einsum(
            "bts,bcs->bct",
            weight,
            v.reshape(batch * self.n_heads, channels, length),
        )
        return attention.reshape(batch, -1, length)


@dataclass(eq=False)
class UNetModel(nn.Module):
    """Guided-diffusion style UNet used by the flow matching image examples."""

    in_channels: int
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: tuple[int, ...] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: int | None = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    with_fourier_features: bool = False
    ignore_time: bool = False
    input_projection: bool = True
    image_size: int = -1

    def __post_init__(self) -> None:
        super().__init__()
        if self.with_fourier_features:
            self.in_channels += 12
        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.time_embed_dim = self.model_channels * 4
        if self.ignore_time:
            self.time_embed = lambda x: torch.zeros(
                x.shape[0], self.time_embed_dim, device=x.device, dtype=x.dtype
            )
        else:
            self.time_embed = nn.Sequential(
                linear(self.model_channels, self.time_embed_dim),
                nn.SiLU(),
                linear(self.time_embed_dim, self.time_embed_dim),
            )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(
                self.num_classes + 1,
                self.time_embed_dim,
                padding_idx=self.num_classes,
            )

        channels = int(self.channel_mult[0] * self.model_channels)
        input_channels = channels
        if self.input_projection:
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(conv_nd(self.dims, self.in_channels, channels, 3, padding=1))]
            )
        else:
            self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Identity())])

        input_block_channels = [channels]
        downsample_factor = 1

        for level, multiplier in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        channels,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(multiplier * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                channels = int(multiplier * self.model_channels)
                if downsample_factor in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)

            if level != len(self.channel_mult) - 1:
                out_channels = channels
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            channels,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_channels,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Downsample(channels, self.conv_resample, dims=self.dims, out_channels=out_channels)
                    )
                )
                channels = out_channels
                input_block_channels.append(channels)
                downsample_factor *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
            AttentionBlock(
                channels,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            ResBlock(
                channels,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, multiplier in list(enumerate(self.channel_mult))[::-1]:
            for block_index in range(self.num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ResBlock(
                        channels + skip_channels,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * multiplier),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                channels = int(self.model_channels * multiplier)
                if downsample_factor in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and block_index == self.num_res_blocks:
                    out_channels = channels
                    layers.append(
                        ResBlock(
                            channels,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_channels,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Upsample(channels, self.conv_resample, dims=self.dims, out_channels=out_channels)
                    )
                    downsample_factor //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, input_channels, self.out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        extra: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.with_fourier_features:
            x = torch.cat([x, base2_fourier_features(x, start=6, stop=8, step=1)], dim=1)

        hidden_states: list[torch.Tensor] = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x))
        if self.ignore_time:
            emb = emb * 0.0

        if self.num_classes and "label" not in extra:
            extra["label"] = torch.full(
                (x.size(0),),
                self.num_classes,
                dtype=torch.long,
                device=x.device,
            )
        if self.num_classes is not None and "label" in extra:
            labels = extra["label"]
            emb = emb + self.label_emb(labels)

        hidden = x
        if "concat_conditioning" in extra:
            hidden = torch.cat([x, extra["concat_conditioning"]], dim=1)

        for module in self.input_blocks:
            hidden = module(hidden, emb)
            hidden_states.append(hidden)

        hidden = self.middle_block(hidden, emb)
        for module in self.output_blocks:
            hidden = torch.cat([hidden, hidden_states.pop()], dim=1)
            hidden = module(hidden, emb)

        hidden = hidden.type(x.dtype)
        return self.out(hidden)


def base2_fourier_features(
    inputs: torch.Tensor,
    start: int = 0,
    stop: int = 8,
    step: int = 1,
) -> torch.Tensor:
    freqs = torch.arange(start, stop, step, device=inputs.device, dtype=inputs.dtype)
    weights = 2.0**freqs * 2 * np.pi
    weights = torch.tile(weights[None, :], (1, inputs.size(1)))
    repeated_inputs = torch.repeat_interleave(inputs, len(freqs), dim=1)
    repeated_inputs = weights[:, :, None, None] * repeated_inputs
    return torch.cat([torch.sin(repeated_inputs), torch.cos(repeated_inputs)], dim=1)
