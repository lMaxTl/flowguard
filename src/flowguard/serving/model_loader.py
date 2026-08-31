from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.training.checkpoints import load_checkpoint


@dataclass(slots=True)
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    checkpoint_dir: Path
    checkpoint_path: Path
    params: dict[str, Any]
    dataset_name: str
    model_arch: str
    num_classes: int
    modelfamily: str


def _resolve_dataset_name(params: dict[str, Any]) -> str:
    for key in ("dataset", "queryset", "testdataset"):
        if key in params:
            return params[key]
    raise KeyError("Could not resolve dataset name from params.json")


def load_legacy_model(
    checkpoint_dir: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedModel:
    checkpoint_dir = Path(checkpoint_dir)
    params_path = checkpoint_dir / "params.json"
    checkpoint_path = checkpoint_dir / "checkpoint.pth.tar"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    with params_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    dataset_name = _resolve_dataset_name(params)
    model_arch = params["model_arch"]
    num_classes = params["num_classes"]
    modelfamily = legacy_datasets.dataset_to_modelfamily[dataset_name]
    torch_device = torch.device(device)
    model = legacy_zoo.get_net(model_arch, modelfamily, num_classes=num_classes)
    checkpoint = load_checkpoint(checkpoint_path, map_location=torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(torch_device)
    model.eval()
    return LoadedModel(
        model=model,
        device=torch_device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        params=params,
        dataset_name=dataset_name,
        model_arch=model_arch,
        num_classes=num_classes,
        modelfamily=modelfamily,
    )
