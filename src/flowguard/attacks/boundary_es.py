"""BAM-style evolutionary search toward a classifier's decision boundary.

Implements the selection + mutation rule from Biton Dor & Mirsky (2024,
arXiv:2410.15429). Two seeding modes let the same helper drive training
(start from a clean input, so each output has a natural clean counterpart)
and evaluation (start from uniform noise, matching the paper's attacker).

The victim `model` is assumed to consume normalized inputs. Normalization is
applied internally, so both the seeds `x01` and the returned tensor live in
[0, 1].
"""

from __future__ import annotations

from typing import Literal, Sequence

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


def _max_softmax(
    model: torch.nn.Module,
    x01_flat: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    *,
    eval_batch: int,
) -> torch.Tensor:
    """Return max(softmax(logits)) per sample, chunked to bound activation memory."""
    chunks: list[torch.Tensor] = []
    for start in range(0, x01_flat.shape[0], eval_batch):
        chunk = x01_flat[start : start + eval_batch]
        logits = model((chunk - mean_t) / std_t)
        chunks.append(F.softmax(logits, dim=1).max(dim=1).values)
    return torch.cat(chunks, dim=0)


def boundary_es(
    model: torch.nn.Module,
    x01: torch.Tensor,
    *,
    mean: Sequence[float] | torch.Tensor,
    std: Sequence[float] | torch.Tensor,
    iters: int = 8,
    pop: int = 32,
    top_k: int = 8,
    gamma: float = 0.1,
    seed: Literal["data", "noise"] = "data",
    init_eps: float = 0.05,
    eval_batch: int = 256,
) -> torch.Tensor:
    """Return one boundary-proximal sample per input via BAM-style ES.

    Args:
        model: victim classifier producing logits from normalized inputs.
        x01: (B, C, H, W) seeds in [0, 1]. In `seed="data"` mode values are
            used as starting points; in `seed="noise"` mode only the shape
            and device are used.
        mean, std: per-channel normalization constants of the victim.
        iters: number of ES generations (>= 1).
        pop: population size N per seed.
        top_k: selection size k per seed; must satisfy 0 < k <= N.
        gamma: mutation scale in BAM eq. (2).
        seed: "data" starts population around `x01` (training). "noise" draws
            the starting population from U[0, 1] (paper-faithful attacker).
        init_eps: initial symmetric jitter around each seed (data mode only).
        eval_batch: chunk size for victim forward passes.

    Returns:
        Tensor of shape (B, C, H, W) in [0, 1]: for each input, the sample
        in the final generation with the lowest max(softmax(victim(x))).
    """
    if x01.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) tensor; got {tuple(x01.shape)}.")
    if pop <= 0:
        raise ValueError(f"pop must be positive; got {pop}.")
    if not 0 < top_k <= pop:
        raise ValueError(f"top_k must satisfy 0 < top_k <= pop; got top_k={top_k}, pop={pop}.")
    iterations = max(1, int(iters))

    device = x01.device
    dtype = x01.dtype
    B, C, H, W = x01.shape
    N = int(pop)
    k = int(top_k)

    mean_t = _as_channel_tensor(mean, device=device, dtype=dtype)
    std_t = _as_channel_tensor(std, device=device, dtype=dtype)

    was_training = model.training
    model.eval()

    if seed == "data":
        seeds = x01.detach().unsqueeze(1).expand(B, N, C, H, W)
        jitter = (torch.rand(B, N, C, H, W, device=device, dtype=dtype) * 2.0 - 1.0) * init_eps
        population = (seeds + jitter).clamp_(0.0, 1.0)
    elif seed == "noise":
        population = torch.rand(B, N, C, H, W, device=device, dtype=dtype)
    else:
        raise ValueError(f"seed must be 'data' or 'noise'; got {seed!r}.")

    conf: torch.Tensor | None = None
    for iter_idx in range(iterations):
        flat = population.reshape(B * N, C, H, W)
        with torch.no_grad():
            conf_flat = _max_softmax(model, flat, mean_t, std_t, eval_batch=eval_batch)
        conf = conf_flat.view(B, N)

        if iter_idx == iterations - 1:
            break

        # Selection: lowest max-softmax = highest BAM fitness.
        _, top_idx = torch.topk(-conf, k=k, dim=1)
        gather_idx = top_idx.view(B, k, 1, 1, 1).expand(B, k, C, H, W)
        parents = torch.gather(population, dim=1, index=gather_idx)

        # Mutation: per-element span across the current population (BAM eq. 2).
        span = population.max(dim=1).values - population.min(dim=1).values
        span = span.unsqueeze(1)

        repeats = (N + k - 1) // k
        expanded_parents = parents.unsqueeze(2).expand(B, k, repeats, C, H, W)
        z = (torch.rand(B, k, repeats, C, H, W, device=device, dtype=dtype) * 2.0 - 1.0) * span.unsqueeze(2)
        children = (expanded_parents + gamma * z).clamp_(0.0, 1.0)
        population = children.reshape(B, k * repeats, C, H, W)[:, :N].contiguous()

    assert conf is not None
    flat = population.reshape(B * N, C, H, W)
    _, best_idx = conf.min(dim=1)
    flat_best_idx = best_idx + torch.arange(B, device=device) * N
    best = flat.index_select(0, flat_best_idx)

    if was_training:
        model.train()
    return best.detach()
