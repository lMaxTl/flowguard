from __future__ import annotations

import hashlib

import torch

from flowguard.defenses.query.base import QueryContext, QueryDefense


class BudgetingQueryDefense(QueryDefense):
    name = "budgeting"

    def __init__(self, budget: int, deduplicate: bool = False, **parameters):
        super().__init__(budget=budget, deduplicate=deduplicate, **parameters)
        self.budget = int(budget)
        self.deduplicate = bool(deduplicate)
        self.seen_hashes: set[str] = set()

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        if context.total_queries + len(batch) > self.budget:
            raise RuntimeError(
                f"Query budget exceeded: {context.total_queries + len(batch)} > {self.budget}"
            )
        if not self.deduplicate:
            return batch, context
        keep = []
        for row in batch:
            digest = hashlib.sha1(row.detach().cpu().numpy().tobytes()).hexdigest()
            if digest not in self.seen_hashes:
                self.seen_hashes.add(digest)
                keep.append(row)
        if not keep:
            return batch[:0], context
        return torch.stack(keep), context

    def after_query(
        self, batch: torch.Tensor, outputs: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        context.total_queries += len(batch)
        return outputs, context
