from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_metadata(output_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / filename
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def load_checkpoint(
    checkpoint_path: str | Path | Any,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    # ModelGuard loads trusted local training artifacts that may contain legacy
    # module objects, so we opt out of PyTorch 2.6's stricter default here.
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
