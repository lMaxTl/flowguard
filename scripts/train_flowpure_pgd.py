from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flowguard.flow_matching.flowpure_training import (
    FlowPurePGDConfig,
    train_flowpure_pgd,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for FlowPure^PGD training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a FlowPure^PGD CNF that learns an adversarial->clean velocity "
            "field on a 32x32 dataset (CIFAR-10, GTSRB, CelebA, SkinCancer, or an "
            "attacker pool). Requires a trained classifier for the PGD pairs."
        ),
    )
    default_data = os.environ.get("SM_CHANNEL_TRAINING", "./data")
    default_output = os.environ.get("SM_MODEL_DIR", "./runs/flow_matching/cifar10_flowpure_pgd")

    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-path", default=default_data)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument(
        "--victim-checkpoint-dir",
        default=str(
            PROJECT_ROOT
            / "runs"
            / "notebook"
            / "training-victim-cifar10-vgg16_bn-nodefense"
            / "target_model"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--pgd-eps-max", type=float, default=0.05)
    parser.add_argument("--pgd-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Continue from <output-dir>/checkpoint_latest.pt if present.")
    parser.add_argument(
        "--pgd-label-source", choices=("dataset", "prediction"), default="dataset",
        help="PGD target label: ground truth (defender) or the classifier's own "
             "prediction (attacker surrogate on an unlabelled public pool).",
    )
    return parser


def main() -> None:
    """Parse arguments and launch FlowPure^PGD training."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FlowPurePGDConfig(
        victim_checkpoint_dir=args.victim_checkpoint_dir,
        output_dir=str(output_dir),
        dataset=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        pgd_eps_max=args.pgd_eps_max,
        pgd_alpha=args.pgd_alpha,
        pgd_steps=args.pgd_steps,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        download=args.download,
        resume=args.resume,
        pgd_label_source=args.pgd_label_source,
    )
    print(f"[FlowPurePGD] config: {cfg.to_dict()}")
    checkpoint = train_flowpure_pgd(cfg)
    print(f"[FlowPurePGD] final checkpoint: {checkpoint.checkpoint_path}")


if __name__ == "__main__":
    main()
