from __future__ import annotations

from pathlib import Path

from flowguard.experiments.spec import ExperimentSpec
from flowguard.workflows.outlier_exposure import run_outlier_exposure_workflow


def run_misinformation_workflow(
    spec: ExperimentSpec,
    *,
    output_dir: str | Path,
    oe_dataset_name: str,
    oe_lambda: float,
):
    return run_outlier_exposure_workflow(
        spec,
        output_dir=output_dir,
        oe_dataset_name=oe_dataset_name,
        oe_lambda=oe_lambda,
    )
