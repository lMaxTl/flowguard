from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class QueryContext:
    total_queries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def query_identities(context: QueryContext, batch_size: int) -> list[Any]:
    """Identity key of every query in the batch, for stateful detectors.

    With Sybil rotation the engine assigns identities per *query* and publishes
    them as ``metadata["client_ids"]``; a batch then usually spans many
    identities. Without it, every query shares the caller's ``user_id`` /
    ``client_id`` (or ``"default"``), which is the historical behaviour.
    """
    identities = context.metadata.get("client_ids")
    if isinstance(identities, (list, tuple)) and len(identities) == batch_size:
        return list(identities)
    key = context.metadata.get("user_id", context.metadata.get("client_id", "default"))
    return [key] * batch_size


def group_by_identity(identities: list[Any]) -> dict[Any, list[int]]:
    """Map identity -> positions in the batch, in first-seen order."""
    groups: dict[Any, list[int]] = {}
    for index, identity in enumerate(identities):
        groups.setdefault(identity, []).append(index)
    return groups


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
