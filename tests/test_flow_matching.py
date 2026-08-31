from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import TensorDataset

from flowguard.flow_matching.config import (
    FlowMatchingDatasetConfig,
    FlowMatchingTrainingConfig,
    list_dataset_names,
    resolve_dataset_config,
)
from flowguard.flow_matching.data import build_flow_matching_dataset
from flowguard.flow_matching.training import sample_images, train_flow_matching_model


class TinyFlowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        extra: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del timesteps, extra
        return self.scale * x


def test_resolve_dataset_config_supports_expected_presets():
    assert list_dataset_names() == [
        "cifar10",
        "cifar_patch8",
        "ereno",
        "imagenet32",
        "imagenet64",
        "mnist",
    ]
    dataset_config = resolve_dataset_config("imagenet32", "custom/path")
    assert dataset_config.image_size == 32
    assert dataset_config.class_conditioned is True
    assert dataset_config.default_data_path == "custom/path"
    assert resolve_dataset_config("imagenet64").default_data_path == "data/ILSVRC2012/train"

    ereno_config = resolve_dataset_config("ereno")
    assert ereno_config.dataset_type == "tabular"
    assert ereno_config.default_data_path == "data/ereno"
    assert ereno_config.image_size is None


def test_build_flow_matching_imagefolder_dataset(tmp_path: Path):
    class_dir = tmp_path / "class_a"
    class_dir.mkdir(parents=True)
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    Image.fromarray(image).save(class_dir / "sample.png")

    dataset_config = resolve_dataset_config("imagenet32", str(tmp_path))
    dataset = build_flow_matching_dataset(dataset_config)

    sample, label = dataset[0]
    assert sample.shape == (3, 32, 32)
    assert label == 0


def test_build_flow_matching_cifar10_dataset_reuses_framework_wrapper(monkeypatch):
    captured: dict[str, object] = {}

    class DummyCIFAR10:
        def __init__(self, train=True, transform=None, target_transform=None, download=True):
            captured["train"] = train
            captured["transform"] = transform
            captured["target_transform"] = target_transform
            captured["download"] = download

    monkeypatch.setattr("flowguard.flow_matching.data.LegacyCIFAR10", DummyCIFAR10)

    dataset_config = resolve_dataset_config("cifar10")
    dataset = build_flow_matching_dataset(dataset_config, download=True)

    assert isinstance(dataset, DummyCIFAR10)
    assert captured["train"] is True
    assert captured["download"] is True
    assert captured["transform"] is not None


def test_build_flow_matching_ereno_dataset(tmp_path: Path):
    train_data = (
        "@relation ereno\n"
        "@attribute f1 numeric\n"
        "@attribute proto {GOOSE,SV}\n"
        "@attribute @class@ {normal,injection}\n"
        "@data\n"
        "1.0, GOOSE, normal\n"
        "2.0, SV, injection\n"
        "3.0, GOOSE, normal\n"
    )

    (tmp_path / "train.arff").write_text(train_data, encoding="utf-8")
    dataset_config = resolve_dataset_config("ereno", str(tmp_path))
    dataset = build_flow_matching_dataset(dataset_config)

    sample, label = dataset[0]
    assert sample.ndim == 1
    assert sample.shape[0] == 3
    assert int(label.item()) in {0, 1}
    assert len(dataset) == 3
    assert dataset.feature_dim == 3
    assert dataset.num_classes == 2


def test_train_flow_matching_model_smoke(tmp_path: Path, monkeypatch):
    dataset = TensorDataset(torch.rand(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))

    monkeypatch.setattr(
        "flowguard.flow_matching.training.build_flow_matching_dataset",
        lambda dataset_config, download=False: dataset,
    )
    monkeypatch.setattr(
        "flowguard.flow_matching.training.build_model",
        lambda dataset_config, use_ema, ema_decay, feature_dim=None: TinyFlowModel(),
    )

    config = FlowMatchingTrainingConfig(
        dataset="cifar10",
        data_path="unused",
        output_dir=str(tmp_path),
        batch_size=2,
        epochs=1,
        lr=1e-3,
        num_workers=0,
        device="cpu",
        sample_every=1,
        sample_count=4,
        checkpoint_every=1,
        sample_step_size=0.1,
    )
    checkpoint = train_flow_matching_model(config)

    assert checkpoint.epoch == 0
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert (tmp_path / "checkpoint_epoch_0001.pt").exists()
    assert (tmp_path / "samples" / "epoch_0001.png").exists()
    assert (tmp_path / "fm_training_config.json").exists()
    assert (tmp_path / "fm_dataset_config.json").exists()


def test_sample_images_saves_grid(tmp_path: Path):
    dataset_config = resolve_dataset_config("cifar10", "unused")
    output_path = tmp_path / "sample.png"
    saved_path = sample_images(
        TinyFlowModel(),
        dataset_config,
        output_path=output_path,
        device="cpu",
        sample_count=4,
        ode_method="midpoint",
        step_size=0.1,
        seed=0,
    )
    assert saved_path == output_path
    assert output_path.exists()


def test_sample_images_saves_tabular_tensor(tmp_path: Path):
    dataset_config = FlowMatchingDatasetConfig(
        name="ereno",
        dataset_type="tabular",
        image_size=None,
        architecture="tabular",
        num_classes=2,
        class_conditioned=False,
        default_data_path="unused",
        feature_dim=6,
    )
    output_path = tmp_path / "sample.pt"
    saved_path = sample_images(
        TinyFlowModel(),
        dataset_config,
        output_path=output_path,
        device="cpu",
        sample_count=5,
        ode_method="midpoint",
        step_size=0.1,
        seed=0,
    )

    assert saved_path == output_path
    assert output_path.exists()
    sample_tensor = torch.load(output_path, map_location="cpu")
    assert tuple(sample_tensor.shape) == (5, 6)
