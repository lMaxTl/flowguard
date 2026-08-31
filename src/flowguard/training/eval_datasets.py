from __future__ import annotations

from torchvision import transforms as tv_transforms

from defenses import datasets as legacy_datasets
from flowguard.experiments.spec import AttackKind, ExperimentSpec


def build_eval_testset(spec: ExperimentSpec):
    """Test set aligned with how the substitute is evaluated for this attack."""
    dataset_name = spec.dataset.name
    dataset_cls = legacy_datasets.__dict__[dataset_name]
    modelfamily = legacy_datasets.dataset_to_modelfamily[dataset_name]
    if spec.attack.kind == AttackKind.DISGUIDE and modelfamily == "mnist":
        if "image_size" in spec.attack.extra:
            image_size = int(spec.attack.extra["image_size"])
        else:
            # Must match DisguideAttackRunner's own default (disguide.py),
            # which is the victim's native height -- not a hardcoded 32.
            # disguide-main's canonical setup is CIFAR-10 at 32x32, where a
            # hardcoded 32 happens to be correct, but for a victim with a
            # different native size (MNIST's 28x28) it silently evaluates
            # the substitute against images at the wrong resolution: LeNet's
            # `view(-1, 4*4*50)` then produces more "rows" than the batch
            # actually has (a 128-image batch of 32x32 inputs reshapes to
            # 200 rows, not 128), which crashes cross_entropy with a
            # batch-size mismatch instead of failing at the real source.
            image_size, _ = victim_native_spatial_shape(spec)
        transform = tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size)),
            tv_transforms.ToTensor(),
        ])
    else:
        transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    return dataset_cls(train=False, transform=transform, download=spec.dataset.download)


def victim_native_spatial_shape(spec: ExperimentSpec) -> tuple[int, int]:
    dataset_name = spec.dataset.name
    family = legacy_datasets.dataset_to_modelfamily[dataset_name]
    transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
    sample = legacy_datasets.__dict__[dataset_name](
        train=False,
        transform=transform,
        download=spec.dataset.download,
    )[0][0]
    return int(sample.shape[-2]), int(sample.shape[-1])
