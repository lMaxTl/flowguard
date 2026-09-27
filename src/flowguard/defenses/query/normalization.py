"""Map a dataset name to the normalization its victim model was trained with.

The flow-based detectors receive *victim-normalized* query tensors and must
undo that normalization before scoring (the CNFs work in pixel space). They
used to guess the model family from the dataset *name* -- "cifar" in the name,
or SVHN/STL10, otherwise ImageNet -- which silently diverged from the legacy
dataset registry the victim was actually trained with. GTSRB, for example, is
trained with CIFAR statistics but was de-normalized with ImageNet statistics,
so every CNF score on GTSRB was computed on colour-shifted images.

Resolving through ``defenses.datasets`` keeps the detectors and the victim on
one source of truth.
"""

from __future__ import annotations

from defenses import datasets as legacy_datasets

_FALLBACK_MEAN_STD: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def resolve_modelfamily(dataset_name: str) -> str:
    """Return the legacy model family for ``dataset_name``."""
    family = legacy_datasets.dataset_to_modelfamily.get(str(dataset_name))
    if family is not None:
        return family
    # Unknown names keep the historical heuristic.
    normalized = str(dataset_name).lower()
    if "mnist" in normalized:
        return "mnist"
    if "cifar" in normalized or normalized in {"svhn", "stl10"}:
        return "cifar"
    return "imagenet"


def modelfamily_mean_std(family: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(mean, std)`` used by the victim transform of ``family``."""
    key = "imagenet" if family == "tinyimagenet" else family
    entry = legacy_datasets.modelfamily_to_mean_std.get(key)
    if entry is not None:
        return tuple(entry["mean"]), tuple(entry["std"])
    if key in _FALLBACK_MEAN_STD:
        return _FALLBACK_MEAN_STD[key]
    raise KeyError(f"No normalization statistics known for model family '{family}'.")
