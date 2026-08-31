"""Train a victim/target classifier on a supported dataset.

This is a thin CLI wrapper around ``flowguard.training.core.train_model`` that
reproduces the same checkpoint layout as the notebook-driven victim training
run (``checkpoint.pth.tar`` + ``params.json``), so downstream tools such as
``flowguard.serving.model_loader.load_legacy_model`` pick it up without any
further config changes.

Defaults target a good CIFAR-10 / VGG16_bn classifier (~93-94 % test acc):
- 100 epochs, batch size 128
- SGD with momentum 0.9, lr 0.01, step LR (step=25, gamma=0.5)
- weight decay 5e-4 (fixed in the legacy trainer)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.training.checkpoints import save_metadata
from flowguard.training.core import TrainerConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a victim/target classifier used as the model-stealing target "
            "and as the feedback oracle for FlowPure^PGD / FlowPureBoundary "
            "training."
        ),
    )
    default_data = os.environ.get("SM_CHANNEL_TRAINING", "./data")
    default_output = os.environ.get(
        "SM_MODEL_DIR",
        str(PROJECT_ROOT / "runs" / "victim" / "cifar10_vgg16_bn"),
    )

    parser.add_argument("--dataset", default="CIFAR10",
                        help="Dataset key from defenses.datasets (e.g. CIFAR10, CIFAR100).")
    parser.add_argument("--architecture", default="vgg16_bn",
                        help="Architecture name understood by defenses.models.zoo.get_net.")
    parser.add_argument("--pretrained", default=None,
                        help="Optional pretrained tag passed to get_net (e.g. 'imagenet').")
    parser.add_argument("--output-dir", default=default_output,
                        help="Directory to write checkpoint.pth.tar and params.json into.")
    parser.add_argument("--data-path", default=default_data,
                        help="Informational only; legacy datasets discover the path internally.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-step", type=int, default=25)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through to the legacy dataset constructor.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cls = legacy_datasets.__dict__[args.dataset]
    modelfamily = legacy_datasets.dataset_to_modelfamily[args.dataset]
    train_tf = legacy_datasets.modelfamily_to_transforms[modelfamily]["train"]
    test_tf = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]

    trainset = dataset_cls(train=True, transform=train_tf, download=args.download)
    testset = dataset_cls(train=False, transform=test_tf, download=args.download)

    model = legacy_zoo.get_net(
        args.architecture,
        modelfamily,
        args.pretrained,
        num_classes=len(trainset.classes),
    )
    device = torch.device(args.device)
    model = model.to(device)

    config = TrainerConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        momentum=args.momentum,
        num_workers=args.num_workers,
        verbose=True,
    )

    print(
        f"[train_victim] dataset={args.dataset} arch={args.architecture} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} "
        f"momentum={args.momentum} device={args.device}"
    )
    print(f"[train_victim] output_dir={output_dir}")

    train_model(
        model,
        trainset,
        output_dir=output_dir,
        config=config,
        testset=testset,
        device=device,
    )

    save_metadata(
        output_dir,
        "params.json",
        {
            "dataset": args.dataset,
            "model_arch": args.architecture,
            "num_classes": len(trainset.classes),
            "epochs": args.epochs,
            "pretrained": args.pretrained,
        },
    )

    print(f"[train_victim] finished. Best checkpoint: {output_dir / 'checkpoint.pth.tar'}")


if __name__ == "__main__":
    main()
