from __future__ import annotations

from pathlib import Path

import torch

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from defenses.utils import admis as legacy_admis
from flowguard.experiments.spec import ExperimentSpec
from flowguard.training.checkpoints import save_metadata


def run_outlier_exposure_workflow(
    spec: ExperimentSpec,
    *,
    output_dir: str | Path,
    oe_dataset_name: str,
    oe_lambda: float,
):
    dataset_cls = legacy_datasets.__dict__[spec.dataset.name]
    modelfamily = legacy_datasets.dataset_to_modelfamily[spec.dataset.name]
    train_transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["train"]
    test_transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    trainset = dataset_cls(train=True, transform=train_transform, download=spec.dataset.download)
    testset = dataset_cls(train=False, transform=test_transform, download=spec.dataset.download)

    oe_cls = legacy_datasets.__dict__[oe_dataset_name]
    oe_family = legacy_datasets.dataset_to_modelfamily[oe_dataset_name]
    train_oe_transform = legacy_datasets.modelfamily_to_transforms[oe_family]["train"]
    test_oe_transform = legacy_datasets.modelfamily_to_transforms[oe_family]["test"]
    trainset_oe = oe_cls(train=True, transform=train_oe_transform, download=spec.dataset.download)
    testset_oe = oe_cls(train=False, transform=test_oe_transform, download=spec.dataset.download)

    model = legacy_zoo.get_net(
        spec.target_model.architecture,
        modelfamily,
        spec.target_model.pretrained,
        num_classes=len(trainset.classes),
    ).to(torch.device(spec.target_model.device))
    model_poison = legacy_zoo.get_net(
        spec.target_model.architecture,
        modelfamily,
        spec.target_model.pretrained,
        num_classes=len(trainset.classes),
    ).to(torch.device(spec.target_model.device))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_admis.train_model(
        model,
        trainset=trainset,
        trainset_OE=trainset_oe,
        testset=testset,
        testset_OE=testset_oe,
        model_poison=model_poison,
        device=torch.device(spec.target_model.device),
        out_path=str(output_dir),
        batch_size=spec.training.batch_size,
        epochs=spec.training.epochs,
        lr=spec.training.lr,
        momentum=spec.training.momentum,
        lr_step=spec.training.lr_step,
        lr_gamma=spec.training.lr_gamma,
        num_workers=spec.training.num_workers,
        oe_lamb=oe_lambda,
    )
    torch.save(model_poison.state_dict(), output_dir / "model_poison.pt")
    save_metadata(output_dir, "params.json", {
        "dataset": spec.dataset.name,
        "model_arch": spec.target_model.architecture,
        "num_classes": len(trainset.classes),
        "epochs": spec.training.epochs,
        "oe_lamb": oe_lambda,
        "dataset_oe": oe_dataset_name,
    })
    return output_dir
