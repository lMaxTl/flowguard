"""Minimal L-infinity PGD for victim models that consume normalized inputs."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _as_channel_tensor(
    values: Sequence[float] | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    return tensor.view(1, -1, 1, 1)


def pgd_linf(
    model: torch.nn.Module,
    x01: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float | torch.Tensor,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
    mean: Sequence[float] | torch.Tensor,
    std: Sequence[float] | torch.Tensor,
    random_start: bool = True,
) -> torch.Tensor:
    """Run L-infinity PGD on inputs in [0, 1].

    The victim `model` is assumed to expect normalized inputs ((x - mean) / std),
    so normalization is applied inside the loss to keep gradients consistent.
    `eps` may be a scalar or a per-sample tensor with shape broadcastable to
    (B, 1, 1, 1), which allows randomized training budgets.
    """
    if x01.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) tensor in [0,1]; got {tuple(x01.shape)}.")

    device = x01.device
    dtype = x01.dtype
    mean_t = _as_channel_tensor(mean, device=device, dtype=dtype)
    std_t = _as_channel_tensor(std, device=device, dtype=dtype)

    if isinstance(eps, torch.Tensor):
        eps_t = eps.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
    else:
        eps_t = torch.full((x01.shape[0], 1, 1, 1), float(eps), device=device, dtype=dtype)

    was_training = model.training
    model.eval()

    x_clean = x01.detach()
    if random_start:
        noise = (torch.rand_like(x_clean) * 2.0 - 1.0) * eps_t
        x_adv = torch.clamp(x_clean + noise, 0.0, 1.0).detach()
    else:
        x_adv = x_clean.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model((x_adv - mean_t) / std_t)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_clean, -eps_t, eps_t)
            x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

    if was_training:
        model.train()
    return x_adv
