from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Subset

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.experiments.spec import ExperimentSpec
from flowguard.training.checkpoints import save_metadata
from flowguard.training.core import TrainerConfig, train_model


def train_shadow_models(
    spec: ExperimentSpec,
    *,
    dataset_name: str,
    output_dir: str | Path,
    num_shadows: int,
    num_classes: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cls = legacy_datasets.__dict__[dataset_name]
    modelfamily = legacy_datasets.dataset_to_modelfamily[dataset_name]
    train_transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["train"]
    test_transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    trainset = dataset_cls(train=True, transform=train_transform, download=spec.dataset.download)
    testset = dataset_cls(train=False, transform=test_transform, download=spec.dataset.download)
    def get_classes(dataset):
        if hasattr(dataset, 'samples'):
            return np.array([s[1] for s in dataset.samples])
        elif hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        return np.array([dataset[i][1] for i in range(len(dataset))])

    train_classes = get_classes(trainset)
    test_classes = get_classes(testset)
    config = TrainerConfig(
        batch_size=spec.training.batch_size,
        epochs=spec.training.epochs,
        lr=spec.training.lr,
        lr_step=spec.training.lr_step,
        lr_gamma=spec.training.lr_gamma,
        momentum=spec.training.momentum,
        num_workers=spec.training.num_workers,
    )
    outputs = []
    for shadow_idx in range(num_shadows):
        if num_classes is None:
            shadow_trainset = trainset
            shadow_testset = testset
            classes = len(trainset.classes)
        else:
            sub_classes = np.random.choice(np.arange(np.max(train_classes) + 1), num_classes, replace=False)
            train_indices = np.concatenate([np.arange(len(trainset))[train_classes == class_id] for class_id in sub_classes])
            test_indices = np.concatenate([np.arange(len(testset))[test_classes == class_id] for class_id in sub_classes])
            shadow_trainset = Subset(trainset, train_indices)
            shadow_testset = Subset(testset, test_indices)
            classes = num_classes
        model = legacy_zoo.get_net(
            spec.target_model.architecture,
            modelfamily,
            spec.target_model.pretrained,
            num_classes=classes,
        ).to(spec.target_model.device)
        shadow_dir = output_dir / f"shadow_{shadow_idx}"
        train_model(
            model,
            shadow_trainset,
            output_dir=shadow_dir,
            config=config,
            testset=shadow_testset,
            device=spec.target_model.device,
        )
        save_metadata(shadow_dir, "params.json", {
            "dataset": dataset_name,
            "model_arch": spec.target_model.architecture,
            "num_classes": classes,
        })
        outputs.append(shadow_dir)
    return outputs
