from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flowguard.flow_matching import load_flow_matching_checkpoint, sample_images


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for sampling from a saved checkpoint."""
    parser = argparse.ArgumentParser(
        description="Sample images from a trained continuous flow matching checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--ode-method", default="midpoint")
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """Load the checkpoint and save a snapshot grid."""
    args = build_parser().parse_args()
    checkpoint = load_flow_matching_checkpoint(args.checkpoint, device=args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = sample_images(
        checkpoint.model,
        checkpoint.dataset_config,
        output_path=output_path,
        device=args.device,
        sample_count=args.sample_count,
        ode_method=args.ode_method,
        step_size=args.step_size,
        seed=args.seed,
    )
    print(f"[FlowMatching] saved samples to {saved_path}")


if __name__ == "__main__":
    main()
