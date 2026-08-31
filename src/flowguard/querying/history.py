from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryRecord:
    batch_size: int
    output_format: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryHistory:
    records: list[QueryRecord] = field(default_factory=list)

    def append(self, record: QueryRecord) -> None:
        self.records.append(record)

    @property
    def total_queries(self) -> int:
        return sum(record.batch_size for record in self.records)

    @property
    def total_batches(self) -> int:
        return len(self.records)

    @property
    def average_batch_size(self) -> float:
        if not self.records:
            return 0.0
        return self.total_queries / self.total_batches

    def merge(self, other: QueryHistory) -> None:
        self.records.extend(other.records)
