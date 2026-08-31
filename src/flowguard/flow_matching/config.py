from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class FlowMatchingDatasetConfig:
    """Dataset-specific configuration for continuous flow matching."""

    name: str
    dataset_type: str
    image_size: int | None
    architecture: str
    num_classes: int | None
    class_conditioned: bool
    default_data_path: str
    feature_dim: int | None = None
    download_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class FlowMatchingTrainingConfig:
    """Runtime configuration for training a flow matching model."""

    dataset: str
    data_path: str
    output_dir: str
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    class_drop_prob: float = 0.2
    use_ema: bool = False
    ema_decay: float = 0.999
    sample_every: int = 10
    sample_count: int = 16
    checkpoint_every: int = 10
    sample_ode_method: str = "midpoint"
    sample_step_size: float = 0.01
    download: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def output_path(self) -> Path:
        """Return the output directory as a path object."""
        return Path(self.output_dir)


_DATASET_CONFIGS: dict[str, FlowMatchingDatasetConfig] = {
    "cifar10": FlowMatchingDatasetConfig(
        name="cifar10",
        dataset_type="cifar10",
        image_size=32,
        architecture="cifar10",
        num_classes=None,
        class_conditioned=False,
        default_data_path="data/cifar10",
        download_supported=True,
    ),
    "mnist": FlowMatchingDatasetConfig(
        name="mnist",
        dataset_type="mnist",
        image_size=28,
        architecture="mnist",
        num_classes=None,
        class_conditioned=False,
        default_data_path="data/mnist",
        download_supported=True,
    ),
    "imagenet32": FlowMatchingDatasetConfig(
        name="imagenet32",
        dataset_type="imagefolder",
        image_size=32,
        architecture="imagenet",
        num_classes=1000,
        class_conditioned=True,
        default_data_path="data/ILSVRC2012/train",
    ),
    "imagenet64": FlowMatchingDatasetConfig(
        name="imagenet64",
        dataset_type="imagefolder",
        image_size=64,
        architecture="imagenet",
        num_classes=1000,
        class_conditioned=True,
        default_data_path="data/ILSVRC2012/train",
    ),
    "ereno": FlowMatchingDatasetConfig(
        name="ereno",
        dataset_type="tabular",
        image_size=None,
        architecture="tabular",
        num_classes=8,
        class_conditioned=False,
        default_data_path="data/ereno",
    ),
    "cifar_patch8": FlowMatchingDatasetConfig(
        name="cifar_patch8",
        dataset_type="cifar10_patch",
        image_size=8,
        architecture="cifar_patch8",
        num_classes=None,
        class_conditioned=False,
        default_data_path="data/cifar10",
        download_supported=True,
    ),
}


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "imagenet": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    # Single-channel 28x28 variant of the CIFAR-10 UNet. Two downsamples take
    # 28 -> 14 -> 7 exactly, so the skip connections line up without padding.
    # Channel counts stay multiples of 32 because ``normalization`` hard-codes
    # GroupNorm(32, C).
    "mnist": {
        "in_channels": 1,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.1,
        "channel_mult": [1, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "tabular": {
        "input_dim": None,
        "hidden_dims": [512, 256, 256],
        "time_embed_dim": 128,
        "dropout": 0.1,
        "num_classes": None,
    },
    "cifar_patch8": {
        "in_channels": 3,
        "model_channels": 64,
        "out_channels": 3,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.1,
        "channel_mult": [2, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def list_dataset_names() -> list[str]:
    """Return the supported dataset preset names."""
    return sorted(_DATASET_CONFIGS)


def resolve_dataset_config(name: str, data_path: str | None = None) -> FlowMatchingDatasetConfig:
    """Resolve a dataset preset and optionally override its data path."""
    normalized_name = name.lower()
    if normalized_name not in _DATASET_CONFIGS:
        available = ", ".join(list_dataset_names())
        raise ValueError(f"Unsupported flow matching dataset '{name}'. Available: {available}")
    config = _DATASET_CONFIGS[normalized_name]
    if data_path is None:
        return config
    return FlowMatchingDatasetConfig(
        name=config.name,
        dataset_type=config.dataset_type,
        image_size=config.image_size,
        architecture=config.architecture,
        num_classes=config.num_classes,
        class_conditioned=config.class_conditioned,
        default_data_path=data_path,
        feature_dim=config.feature_dim,
        download_supported=config.download_supported,
    )


def get_model_config(
    dataset_config: FlowMatchingDatasetConfig,
    *,
    feature_dim: int | None = None,
) -> dict[str, Any]:
    """Return a copy of the model configuration for the requested dataset."""
    model_config = dict(MODEL_CONFIGS[dataset_config.architecture])
    model_config["num_classes"] = dataset_config.num_classes
    if dataset_config.dataset_type == "tabular":
        inferred_dim = feature_dim if feature_dim is not None else dataset_config.feature_dim
        if inferred_dim is None:
            raise ValueError("Tabular flow matching requires a finite feature dimension.")
        model_config["input_dim"] = int(inferred_dim)
    return model_config
