from __future__ import annotations

from pathlib import Path

import torch

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.experiments.spec import ExperimentSpec
from flowguard.training.checkpoints import save_metadata
from flowguard.training.core import TrainerConfig, train_model


def train_target_model(spec: ExperimentSpec, output_dir: str | Path):
    if spec.verbose:
        print(
            f"[FlowGuard++] Training target model '{spec.target_model.architecture}' "
            f"on dataset '{spec.dataset.name}' -> '{output_dir}'"
        )
    dataset_name = spec.dataset.name
    dataset_cls = legacy_datasets.__dict__[dataset_name]
    modelfamily = legacy_datasets.dataset_to_modelfamily[dataset_name]
    train_transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["train"]
    test_transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    trainset = dataset_cls(train=True, transform=train_transform, download=spec.dataset.download)
    testset = dataset_cls(train=False, transform=test_transform, download=spec.dataset.download)
    model = legacy_zoo.get_net(
        spec.target_model.architecture,
        modelfamily,
        spec.target_model.pretrained,
        num_classes=len(trainset.classes),
    )
    model = model.to(torch.device(spec.target_model.device))
    config = TrainerConfig(
        batch_size=spec.training.batch_size,
        epochs=spec.training.epochs,
        lr=spec.training.lr,
        lr_step=spec.training.lr_step,
        lr_gamma=spec.training.lr_gamma,
        momentum=spec.training.momentum,
        num_workers=spec.training.num_workers,
        verbose=spec.verbose,
    )
    train_model(
        model,
        trainset,
        output_dir=output_dir,
        config=config,
        testset=testset,
        device=spec.target_model.device,
    )
    save_metadata(output_dir, "params.json", {
        "dataset": dataset_name,
        "model_arch": spec.target_model.architecture,
        "num_classes": len(trainset.classes),
        "epochs": spec.training.epochs,
        "pretrained": spec.target_model.pretrained,
    })
    if spec.verbose:
        print(f"[FlowGuard++] Finished target model training -> '{output_dir}'")
    return Path(output_dir)
