from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from defenses.utils import model as legacy_model_utils
from defenses.utils.utils import suppress_stdout


@dataclass(slots=True)
class TrainerConfig:
    batch_size: int = 32
    epochs: int = 30
    lr: float = 0.01
    lr_step: int = 10
    lr_gamma: float = 0.5
    momentum: float = 0.5
    num_workers: int = 4
    weighted_loss: bool = False
    semi_train_weight: float = 0.0
    verbose: bool = True


def train_model(
    model: torch.nn.Module,
    trainset,
    *,
    output_dir: str | Path,
    config: TrainerConfig,
    testset=None,
    semi_dataset=None,
    device: str | torch.device = "cpu",
    gt_model: torch.nn.Module | None = None,
    criterion_train=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_context = nullcontext() if config.verbose else suppress_stdout()
    with stream_context:
        return legacy_model_utils.train_model(
            model,
            trainset,
            out_path=str(output_dir),
            batch_size=config.batch_size,
            testset=testset,
            device=torch.device(device),
            num_workers=config.num_workers,
            lr=config.lr,
            momentum=config.momentum,
            lr_step=config.lr_step,
            lr_gamma=config.lr_gamma,
            epochs=config.epochs,
            weighted_loss=config.weighted_loss,
            semi_train_weight=config.semi_train_weight,
            semi_dataset=semi_dataset,
            gt_model=gt_model,
            criterion_train=criterion_train,
        )
