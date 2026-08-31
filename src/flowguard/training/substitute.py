from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from defenses.utils import model as legacy_model_utils
from defenses.utils.utils import samples_to_transferset
from flowguard.experiments.spec import AttackKind, ExperimentSpec, LabelType
from flowguard.training.checkpoints import save_metadata
from flowguard.training.core import TrainerConfig, train_model
from flowguard.training.eval_datasets import build_eval_testset
from flowguard.training.victim_eval import wrap_victim_for_substitute_eval


def _coerce_soft_transfer_labels(samples) -> list:
    """Ensure stored labels are valid probability vectors for soft_cross_entropy."""
    coerced: list = []
    for x, y in samples:
        label = torch.as_tensor(y, dtype=torch.float32)
        if float(label.min()) < 0.0 or abs(float(label.sum()) - 1.0) > 0.05:
            label = F.softmax(label, dim=-1)
        coerced.append((x, label))
    return coerced


def train_substitute_model(
    spec: ExperimentSpec,
    transfer_samples,
    *,
    output_dir: str | Path,
    victim_model: torch.nn.Module | None = None,
):
    if spec.verbose:
        print(
            f"[FlowGuard++] Training substitute model '{spec.substitute_model.architecture}' "
            f"with {len(transfer_samples)} transfer samples -> '{output_dir}'"
        )
    dataset_name = spec.dataset.name
    modelfamily = legacy_datasets.dataset_to_modelfamily[dataset_name]
    testset = build_eval_testset(spec)
    transfer_dataset_name = spec.attack.query_dataset.name
    transfer_modelfamily = legacy_datasets.dataset_to_modelfamily[transfer_dataset_name]
    # CEM / DisGUIDE attacks store tensors already in model input space.
    disguide_style = spec.attack.kind in {
        AttackKind.DISGUIDE,
        AttackKind.LATENT_MANIFOLD,
        AttackKind.PROCEDURAL_NATURAL,
    }
    if disguide_style:
        transfer_transform = None
        transfer_samples = _coerce_soft_transfer_labels(transfer_samples)
    else:
        transfer_transform = legacy_datasets.modelfamily_to_transforms[transfer_modelfamily]["test"]
    transferset = samples_to_transferset(
        transfer_samples,
        budget=len(transfer_samples),
        transform=transfer_transform,
    )
    model = legacy_zoo.get_net(
        spec.substitute_model.architecture,
        modelfamily,
        spec.substitute_model.pretrained,
        num_classes=len(testset.classes),
        rot_semi=spec.training.semi_train_weight > 0,
    )
    model = model.to(torch.device(spec.substitute_model.device))
    semi_dataset = None
    if spec.training.semi_train_weight > 0 and spec.training.semi_dataset:
        semi_name = spec.training.semi_dataset
        semi_family = legacy_datasets.dataset_to_modelfamily[semi_name]
        semi_transform = legacy_datasets.modelfamily_to_transforms[semi_family]["test"]
        semi_dataset = legacy_datasets.__dict__[semi_name](
            train=True,
            transform=semi_transform,
            download=spec.dataset.download,
        )
    config = TrainerConfig(
        batch_size=spec.training.batch_size,
        epochs=spec.training.epochs,
        lr=spec.training.lr,
        lr_step=spec.training.lr_step,
        lr_gamma=spec.training.lr_gamma,
        momentum=spec.training.momentum,
        num_workers=spec.training.num_workers,
        semi_train_weight=spec.training.semi_train_weight,
        verbose=spec.verbose,
    )
    criterion_train = None
    if spec.attack.mode.label_type == LabelType.SOFT:
        criterion_train = legacy_model_utils.soft_cross_entropy
    train_model(
        model,
        transferset,
        output_dir=output_dir,
        config=config,
        testset=testset,
        semi_dataset=semi_dataset,
        device=spec.substitute_model.device,
        gt_model=wrap_victim_for_substitute_eval(victim_model, spec),
        criterion_train=criterion_train,
    )
    save_metadata(output_dir, "params_train.json", {
        "testdataset": dataset_name,
        "queryset": transfer_dataset_name,
        "model_arch": spec.substitute_model.architecture,
        "epochs": spec.training.epochs,
        "budgets": spec.training.budgets or [len(transfer_samples)],
    })
    if spec.verbose:
        print(f"[FlowGuard++] Finished substitute training -> '{output_dir}'")
    return Path(output_dir)
