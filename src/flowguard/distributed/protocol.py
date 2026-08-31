from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class QueryTask:
    task_id: str
    batch: torch.Tensor
    output_format: str = "soft"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryTaskResult:
    task_id: str
    outputs: torch.Tensor
    original_probabilities: torch.Tensor | None
    worker_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
