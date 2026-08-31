from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(slots=True)
class ProgressEvent:
    phase: str
    message: str
    current: int = 0
    total: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.current / self.total)
