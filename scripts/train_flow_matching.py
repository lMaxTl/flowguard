from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flowguard.flow_matching import (
    FlowMatchingTrainingConfig,
    list_dataset_names,
    train_flow_matching_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for FM training."""
    parser = argparse.ArgumentParser(
        description="Train a continuous flow matching model for image or tabular datasets (including ERENO).",
    )
    default_data = os.environ.get('SM_CHANNEL_TRAINING', './data')
    default_model_dir = os.environ.get('SM_MODEL_DIR', './outputs')

    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Preset ({', '.join(list_dataset_names())}) or any 32x32 RGB dataset "
        "registered in defenses.datasets (e.g. GTSRB, CelebA, SkinCancer, LFW, BCN20000).",
    )
    parser.add_argument("--data-path", default=default_data)
    parser.add_argument("--output-dir", default=default_model_dir)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--class-drop-prob", type=float, default=0.2)
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sample-ode-method", default="midpoint")
    parser.add_argument("--sample-step-size", type=float, default=0.01)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps (overrides --epochs).")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0,
                        help="Also refresh checkpoint_latest.pt every N steps (0 = per epoch only).")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Continue from <output-dir>/checkpoint_latest.pt if it exists.")
    return parser


def main() -> None:
    """Parse arguments and launch the FM trainer."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = FlowMatchingTrainingConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        class_drop_prob=args.class_drop_prob,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        sample_every=args.sample_every,
        sample_count=args.sample_count,
        checkpoint_every=args.checkpoint_every,
        sample_ode_method=args.sample_ode_method,
        sample_step_size=args.sample_step_size,
        download=args.download,
        max_steps=args.max_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume=args.resume,
    )
    checkpoint = train_flow_matching_model(config)
    print(f"[FlowMatching] final checkpoint: {checkpoint.checkpoint_path}")


if __name__ == "__main__":
    main()
