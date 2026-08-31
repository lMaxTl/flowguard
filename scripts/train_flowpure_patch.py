from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flowguard.flow_matching.flowpure_patch_training import (
    FlowPurePatchConfig,
    train_flowpure_patch,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for FlowPurePatch training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a FlowPurePatch CNF: a small Gaussian-source flow matching "
            "model over random 8x8 CIFAR-10 patches. No victim classifier is "
            "required: the model is a pure patch-level density model used for "
            "attack-agnostic detection of off-manifold local anomalies (e.g. "
            "PRADA adversarial patches)."
        ),
    )
    default_data = os.environ.get("SM_CHANNEL_TRAINING", "./data")
    default_output = os.environ.get(
        "SM_MODEL_DIR", "./runs/flow_matching/cifar10_flowpure_patch"
    )

    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-path", default=default_data)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-images", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    """Parse arguments and launch FlowPurePatch training."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FlowPurePatchConfig(
        output_dir=str(output_dir),
        dataset=args.dataset,
        data_path=args.data_path,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        batch_images=args.batch_images,
        max_steps=args.max_steps,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        download=args.download,
    )
    print(f"[FlowPurePatch] config: {cfg.to_dict()}")
    checkpoint = train_flowpure_patch(cfg)
    print(f"[FlowPurePatch] final checkpoint: {checkpoint.checkpoint_path}")


if __name__ == "__main__":
    main()
