from __future__ import annotations

from pathlib import Path

from flowguard.experiments.spec import ExperimentSpec
from flowguard.training.target import train_target_model


def train_proxy_model(spec: ExperimentSpec, *, output_dir: str | Path):
    return train_target_model(spec, output_dir=output_dir)
