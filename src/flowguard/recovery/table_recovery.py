from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from defenses.adversary.wb_recover import Table_Recover


@dataclass(slots=True)
class TableRecoveryConfig:
    table_size: int = 1_000_000
    batch_size: int = 1
    epsilon: float | None = None
    perturb_norm: int = 1
    recover_mean: bool = True
    recover_norm: int = 2
    tolerance: float = 1e-4
    concentration_factor: float = 4.0
    shadow_path: str | None = None
    recover_nn: bool = False
    recover_proc: int = 1
    recover_nn_epochs: int = 200


class TableRecovery:
    def __init__(self, blackbox, config: TableRecoveryConfig):
        self.blackbox = blackbox
        self.config = config
        self.legacy = Table_Recover(
            blackbox,
            table_size=config.table_size,
            batch_size=config.batch_size,
            epsilon=config.epsilon,
            perturb_norm=config.perturb_norm,
            recover_mean=config.recover_mean,
            recover_norm=config.recover_norm,
            tolerance=config.tolerance,
            concentration_factor=config.concentration_factor,
            shadow_path=config.shadow_path,
            recover_nn=config.recover_nn,
            recover_proc=config.recover_proc,
            recover_nn_epochs=config.recover_nn_epochs,
        )

    def build(self, estimation_set=None, load_path: str | Path | None = None) -> None:
        self.legacy.generate_lookup_table(
            load_path=str(load_path) if load_path is not None else None,
            estimation_set=estimation_set,
        )

    def recover(self, labels: torch.Tensor, progress=None) -> torch.Tensor:
        return self.legacy(labels, progress)
