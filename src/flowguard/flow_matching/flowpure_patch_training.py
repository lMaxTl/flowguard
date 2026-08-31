"""Trainer for FlowPurePatch: a Gaussian CNF over local CIFAR image patches.

The goal is an attack-agnostic density model for *local* natural image content.
Compared to the global Gaussian CNF (32x32 whole image) it is much more
sensitive to compositional anomalies like PRADA-style adversarial patches or
cut-and-paste query synthesis, because those produce regions whose local pixel
statistics are off-manifold even though the full-image statistics can still
appear natural.

Training recipe (paper-agnostic):
- Draw a random 32x32 CIFAR image (ToTensor, [0, 1], horizontal flip aug).
- Crop ``patches_per_image`` random ``patch_size x patch_size`` sub-regions.
- Train a small UNet (``cifar_patch8`` preset) via Conditional-OT flow matching
  with ``x_0 ~ N(0, I)`` and ``x_1 = patch * 2 - 1``.

At evaluation (handled elsewhere), a query image is sliced into overlapping
patches and each patch is scored by ``||v_theta(t=0, patch)||^2``. A query is
flagged if *any* patch has a score above the benign-calibrated threshold
(max-pool aggregation).

This file only produces the checkpoint (plus a ``flowpure_patch_config.json``
sidecar). The patch-slicing inference adapter lives elsewhere.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from defenses.datasets.cifarlike import CIFAR10 as LegacyCIFAR10
from flowguard.flow_matching.config import (
    FlowMatchingDatasetConfig,
    FlowMatchingTrainingConfig,
)
from flowguard.flow_matching.training import (
    FlowMatchingCheckpoint,
    build_model,
    load_flow_matching_checkpoint,
    save_flow_matching_checkpoint,
    set_random_seed,
)

PATCH_ARCHITECTURE = "cifar_patch8"
PATCH_DATASET_TYPE = "cifar10_patch"


@dataclass(slots=True)
class FlowPurePatchConfig:
    """Runtime configuration for FlowPurePatch training."""

    output_dir: str
    dataset: str = "cifar10"
    data_path: str | None = None
    patch_size: int = 8
    patches_per_image: int = 8
    batch_images: int = 32
    max_steps: int = 300_000
    lr: float = 2e-4
    device: str = "cuda"
    seed: int = 0
    checkpoint_every: int = 5_000
    log_every: int = 100
    num_workers: int = 0
    subset_size: int | None = None
    download: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "dataset": self.dataset,
            "data_path": self.data_path,
            "patch_size": self.patch_size,
            "patches_per_image": self.patches_per_image,
            "batch_images": self.batch_images,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "device": self.device,
            "seed": self.seed,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
            "num_workers": self.num_workers,
            "subset_size": self.subset_size,
            "download": self.download,
            "noise_type": "gauss",
            "source_type": "patch",
            "architecture": PATCH_ARCHITECTURE,
        }


def _build_patch_dataset_config(cfg: FlowPurePatchConfig) -> FlowMatchingDatasetConfig:
    """Build a dataset config that round-trips through the checkpoint loader.

    The only downstream consumer of this config is ``build_model`` in
    ``FlowPureQueryDefense``/``load_flow_matching_checkpoint`` which only reads
    ``architecture`` and ``num_classes``. We therefore set the image-size field
    to the patch size so that scoring code can branch on it.
    """
    return FlowMatchingDatasetConfig(
        name=f"{cfg.dataset}_patch{cfg.patch_size}",
        dataset_type=PATCH_DATASET_TYPE,
        image_size=cfg.patch_size,
        architecture=PATCH_ARCHITECTURE,
        num_classes=None,
        class_conditioned=False,
        default_data_path=cfg.data_path or "",
        download_supported=True,
    )


def _build_training_config_for_checkpoint(
    cfg: FlowPurePatchConfig,
) -> FlowMatchingTrainingConfig:
    return FlowMatchingTrainingConfig(
        dataset=cfg.dataset,
        data_path=cfg.data_path or "",
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_images,
        epochs=1,
        lr=cfg.lr,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device=cfg.device,
        use_ema=False,
        ema_decay=0.999,
        sample_every=0,
        checkpoint_every=cfg.checkpoint_every,
        download=cfg.download,
    )


def _build_cifar_image_dataset(cfg: FlowPurePatchConfig):
    """Load full CIFAR-10 images at 32x32 in [0, 1]; patch sampling is done per step."""
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = LegacyCIFAR10(train=True, download=cfg.download, transform=transform)
    if cfg.subset_size is not None and cfg.subset_size < len(dataset):
        dataset = Subset(dataset, list(range(int(cfg.subset_size))))
    return dataset


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _crop_random_patches(
    images01: torch.Tensor,
    *,
    patch_size: int,
    patches_per_image: int,
) -> torch.Tensor:
    """Return ``B * K`` random patches of shape ``(B*K, C, P, P)``.

    Uses vectorised index arithmetic so the whole batch stays on-device.
    """
    if images01.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) image batch, got {tuple(images01.shape)}")
    batch_size, channels, height, width = images01.shape
    if patch_size > min(height, width):
        raise ValueError(
            f"patch_size={patch_size} exceeds image dims ({height}x{width})."
        )

    device = images01.device
    total = batch_size * patches_per_image

    top = torch.randint(0, height - patch_size + 1, (total,), device=device)
    left = torch.randint(0, width - patch_size + 1, (total,), device=device)
    image_idx = torch.arange(batch_size, device=device).repeat_interleave(patches_per_image)

    row_offsets = torch.arange(patch_size, device=device).view(1, -1)
    col_offsets = torch.arange(patch_size, device=device).view(1, -1)
    rows = top.view(-1, 1) + row_offsets
    cols = left.view(-1, 1) + col_offsets

    gathered_rows = images01[image_idx]
    gathered_rows = gathered_rows.gather(
        2,
        rows.view(total, 1, patch_size, 1).expand(total, channels, patch_size, width),
    )
    patches = gathered_rows.gather(
        3,
        cols.view(total, 1, 1, patch_size).expand(total, channels, patch_size, patch_size),
    )
    return patches


def train_flowpure_patch(cfg: FlowPurePatchConfig) -> FlowMatchingCheckpoint:
    """Train a patch-level Gaussian CNF and persist a checkpoint."""
    set_random_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    dataset_config = _build_patch_dataset_config(cfg)

    dataset = _build_cifar_image_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_images,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    data_iter = _infinite_loader(loader)

    model = build_model(
        dataset_config,
        use_ema=False,
        ema_decay=0.999,
        feature_dim=None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    path = CondOTProbPath()
    training_cfg_for_ckpt = _build_training_config_for_checkpoint(cfg)

    running_loss = 0.0
    running_count = 0
    latest_checkpoint_path: Path | None = None

    for step in range(1, cfg.max_steps + 1):
        images01, _labels = next(data_iter)
        images01 = images01.to(device, non_blocking=True)

        patches01 = _crop_random_patches(
            images01,
            patch_size=cfg.patch_size,
            patches_per_image=cfg.patches_per_image,
        )
        x1 = patches01 * 2.0 - 1.0
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=device)

        path_sample = path.sample(x_0=x0, x_1=x1, t=t)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(path_sample.x_t, t, {})
        loss = F.mse_loss(prediction, path_sample.dx_t)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        running_count += 1

        if step % cfg.log_every == 0 or step == 1:
            avg_loss = running_loss / max(1, running_count)
            print(
                f"[FlowPurePatch] step={step}/{cfg.max_steps} "
                f"patches/step={x1.shape[0]} loss={avg_loss:.6f}"
            )
            running_loss = 0.0
            running_count = 0

        if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
            latest_checkpoint_path = save_flow_matching_checkpoint(
                output_dir,
                "checkpoint_latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=step,
                average_loss=float(loss.item()),
                training_config=training_cfg_for_ckpt,
                dataset_config=dataset_config,
            )
            meta_path = output_dir / "flowpure_patch_config.json"
            meta_path.write_text(_json_dump(cfg.to_dict()), encoding="utf-8")

    if latest_checkpoint_path is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    return load_flow_matching_checkpoint(latest_checkpoint_path, device=device)


def _json_dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a FlowPurePatch CNF: a small Gaussian-source flow matching "
            "model over random 8x8 CIFAR patches. Used for attack-agnostic local "
            "off-manifold detection (e.g. PRADA adversarial patches)."
        )
    )
    parser.add_argument("--output-dir", type=str, required=False)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-images", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short CPU/MPS-friendly smoke test (tiny subset, ~200 steps).",
    )
    return parser


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    args.batch_images = min(args.batch_images, 8)
    args.patches_per_image = min(args.patches_per_image, 4)
    args.max_steps = min(args.max_steps, 200)
    args.checkpoint_every = min(args.checkpoint_every, 100)
    args.log_every = min(args.log_every, 20)
    args.subset_size = args.subset_size or 512
    if args.device is None:
        args.device = "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        _apply_smoke_defaults(args)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output_dir is None:
        suffix = "_smoke" if args.smoke else ""
        args.output_dir = str(
            Path("runs") / "flow_matching" / f"cifar10_flowpure_patch{suffix}"
        )

    cfg = FlowPurePatchConfig(
        output_dir=args.output_dir,
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
    train_flowpure_patch(cfg)


if __name__ == "__main__":
    main()
