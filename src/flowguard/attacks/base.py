from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from flowguard.experiments.spec import ExperimentSpec


@dataclass(slots=True)
class AttackRunContext:
    experiment: ExperimentSpec
    output_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttackRunResult:
    attack_name: str
    mode_name: str
    output_dir: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class AttackRunner(ABC):
    name = "attack"

    @abstractmethod
    def run(self, context: AttackRunContext) -> AttackRunResult:
        raise NotImplementedError
