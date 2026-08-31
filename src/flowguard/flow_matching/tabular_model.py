from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Project scalar timesteps into a sinusoidal embedding space."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.float().view(-1, 1)
        half = self.embed_dim // 2
        if half == 0:
            return timesteps
        scale = math.log(10000.0) / max(1, half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=timesteps.dtype) * -scale
        )
        angles = timesteps * frequencies.view(1, -1)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(timesteps)], dim=1)
        return embedding


class TabularFlowModel(nn.Module):
    """Time-conditioned MLP velocity field for tabular flow matching."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        time_embed_dim: int,
        dropout: float,
        num_classes: int | None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = num_classes

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.label_embedding = (
            nn.Embedding(num_classes, time_embed_dim) if isinstance(num_classes, int) else None
        )

        layer_dims = [self.input_dim + time_embed_dim, *hidden_dims, self.input_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_dims[:-2], layer_dims[1:-1]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU(),
                    nn.Dropout(p=dropout),
                ]
            )
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        extra: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected tabular batch shape (B, F), got {tuple(x.shape)}")
        t_embed = self.time_projection(self.time_embedding(timesteps).to(x.dtype))
        if self.label_embedding is not None and "label" in extra:
            labels = extra["label"].to(device=x.device, dtype=torch.long)
            t_embed = t_embed + self.label_embedding(labels)
        features = torch.cat([x, t_embed], dim=1)
        return self.network(features)
