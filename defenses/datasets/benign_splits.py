"""Held-out benign query streams for detector calibration and evaluation.

The evaluation harness builds its benign stream from ``Dataset(train=True)``.
For the defended dataset that is the *training* split, i.e. the very images the
victim and both CNFs were fitted on. Such queries score as unusually typical
(the flow has seen them), which inflates AUROC and makes the calibrated false-
positive rate optimistic for real users, who send images the model never saw.

The classes generated here serve the defended dataset's *test* split instead,
cut into two fixed, disjoint halves:

- ``<Name>BenignCal``  -- sets thresholds and the benign reference statistics
  (component means/stds, KS reference sample, label/entropy histograms);
- ``<Name>BenignEval`` -- the benign side of every reported metric.

So no benign query both sets an operating point and is scored against it. Both
ignore the ``train`` flag, because callers always ask for ``train=True`` when
they want a query pool.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

SPLIT_SEED = 1234


class BenignSplit(Dataset):
    base_name: str = ""
    half: str = "cal"
    base_registry: dict[str, Any] = {}

    def __init__(
        self,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        base_cls = self.base_registry[self.base_name]
        base = base_cls(train=False, transform=None, download=download)
        data = base.data
        if hasattr(data, "numpy"):  # torchvision MNIST keeps a tensor
            data = data.numpy()
        data = np.asarray(data)
        permutation = np.random.RandomState(SPLIT_SEED).permutation(len(data))
        middle = len(data) // 2
        chosen = permutation[:middle] if self.half == "cal" else permutation[middle:]
        chosen = np.sort(chosen)
        self.data = data[chosen]
        targets = list(base.targets)
        self.targets = [int(targets[index]) for index in chosen]
        self.classes = list(base.classes)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return int(len(self.data))

    def __getitem__(self, index: int):
        array = self.data[index]
        image = Image.fromarray(array, mode="L") if array.ndim == 2 else Image.fromarray(array)
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def register_benign_splits(
    namespace: dict[str, Any],
    dataset_to_modelfamily: dict[str, str],
    base_names: list[str],
) -> list[str]:
    """Create ``<base>BenignCal`` / ``<base>BenignEval`` classes in ``namespace``."""
    created: list[str] = []
    for base_name in base_names:
        for half, suffix in (("cal", "BenignCal"), ("eval", "BenignEval")):
            name = f"{base_name}{suffix}"
            cls = type(
                name,
                (BenignSplit,),
                {"base_name": base_name, "half": half, "base_registry": namespace},
            )
            namespace[name] = cls
            dataset_to_modelfamily[name] = dataset_to_modelfamily[base_name]
            created.append(name)
    return created
