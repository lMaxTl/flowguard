from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class QueryContext:
    total_queries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryDefense(ABC):
    name = "query_defense"

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        return batch, context

    def after_query(
        self, batch: torch.Tensor, outputs: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        return outputs, context
