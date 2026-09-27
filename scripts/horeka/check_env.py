"""Quick self-check of a FlowGuard++ installation (run by setup_env.sh).

Checks that the package imports, that every benchmark architecture builds and
runs a 32x32 forward pass, that every dataset cache loads with the expected
number of classes, and whether CUDA is visible (on a login node it usually is
not, which is fine).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

import defenses.config as cfg  # noqa: E402
import defenses.models.zoo as zoo  # noqa: E402
import flowguard  # noqa: E402,F401
from defenses import datasets as legacy_datasets  # noqa: E402

ARCHITECTURES = [
    ("vgg19_bn", "cifar", 10),
    ("mobilenetv2_cifar", "gtsrb", 43),
    ("densenet121_cifar", "celeba", 2),
    ("resnet50_cifar", "skin", 7),
]
DATASETS = {
    "CIFAR10": 10, "CIFAR100": 100, "GTSRB": 43, "BelgiumTS": 62, "CelebA": 2,
    "LFW": None, "LFW10": None, "SkinCancer": 7, "BCN20000": 8,
}


def main() -> int:
    problems: list[str] = []
    print(f"[check] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    print(f"[check] data root: {cfg.DATASET_ROOT}")
    for arch, family, classes in ARCHITECTURES:
        model = zoo.get_net(arch, family, None, num_classes=classes).eval()
        with torch.no_grad():
            shape = tuple(model(torch.zeros(1, 3, 32, 32)).shape)
        status = "ok" if shape == (1, classes) else f"BAD output {shape}"
        print(f"[check] model {arch:20s} {status}")
        if status != "ok":
            problems.append(arch)
    for name, classes in DATASETS.items():
        try:
            for train in (True, False):
                dataset = legacy_datasets.__dict__[name](train=train, download=False)
                if classes is not None and len(dataset.classes) != classes:
                    raise RuntimeError(f"{len(dataset.classes)} classes, expected {classes}")
            print(f"[check] dataset {name:11s} ok ({len(dataset.classes)} classes)")
        except Exception as error:  # noqa: BLE001
            print(f"[check] dataset {name:11s} MISSING/BROKEN: {error}")
            problems.append(name)
    try:
        import diffusers  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as error:
        problems.append(str(error))
    if problems:
        print(f"[check] problems: {problems}")
        return 1
    print("[check] all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
