from __future__ import annotations

from collections.abc import Iterable

import torch


def batch_tensor(inputs: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(inputs), batch_size):
        yield inputs[start : start + batch_size]
