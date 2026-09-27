"""Trainer for FlowPure^PGD: CNF mapping adversarial -> clean samples.

This mirrors `FlowPure/FlowPure/trainer_flowpure.py` but reuses the local
UNet/checkpoint plumbing so that the resulting checkpoint can be loaded by
`load_flow_matching_checkpoint` and consumed by `FlowPureQueryDefense`.
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

from defenses import datasets as legacy_datasets
from flowguard.attacks.pgd import pgd_linf
from flowguard.flow_matching.config import (
    FlowMatchingTrainingConfig,
    resolve_dataset_config,
)
from flowguard.flow_matching.data import build_flow_matching_dataset
from flowguard.flow_matching.training import (
    FlowMatchingCheckpoint,
    build_model,
    load_flow_matching_checkpoint,
    save_flow_matching_checkpoint,
    set_random_seed,
    write_done_marker,
)
from flowguard.serving.model_loader import load_legacy_model


@dataclass(slots=True)
class FlowPurePGDConfig:
    """Runtime configuration for FlowPure^PGD training."""

    victim_checkpoint_dir: str
    output_dir: str
    dataset: str = "cifar10"
    data_path: str | None = None
    batch_size: int = 64
    max_steps: int = 300_000
    lr: float = 2e-4
    device: str = "cuda"
    seed: int = 0
    checkpoint_every: int = 5_000
    log_every: int = 100
    pgd_eps_max: float = 0.05
    pgd_alpha: float = 2.0 / 255.0
    pgd_steps: int = 10
    num_workers: int = 0
    subset_size: int | None = None
    download: bool = False
    # Continue from <output_dir>/checkpoint_latest.pt (model, optimizer, step).
    resume: bool = True
    # Which label PGD ascends the loss of. "dataset": the ground-truth label
    # (defender, who owns labelled training data). "prediction": the classifier's
    # own top-1 prediction, which lets an attacker build PGD pairs on an
    # unlabelled public pool with a proxy classifier it trained elsewhere.
    pgd_label_source: str = "dataset"

    def to_dict(self) -> dict[str, Any]:
        return {
            "victim_checkpoint_dir": self.victim_checkpoint_dir,
            "output_dir": self.output_dir,
            "dataset": self.dataset,
            "data_path": self.data_path,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "device": self.device,
            "seed": self.seed,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
            "pgd_eps_max": self.pgd_eps_max,
            "pgd_alpha": self.pgd_alpha,
            "pgd_steps": self.pgd_steps,
            "num_workers": self.num_workers,
            "subset_size": self.subset_size,
            "download": self.download,
            "resume": self.resume,
            "pgd_label_source": self.pgd_label_source,
            "noise_type": "pgd",
        }


def _build_training_config_for_checkpoint(
    cfg: FlowPurePGDConfig,
) -> FlowMatchingTrainingConfig:
    """Build a FlowMatchingTrainingConfig compatible with the standard checkpoint loader.

    We piggy-back on the existing `FlowMatchingTrainingConfig` schema so the
    saved checkpoint can be read by `load_flow_matching_checkpoint` unchanged.
    """
    return FlowMatchingTrainingConfig(
        dataset=cfg.dataset,
        data_path=cfg.data_path or "",
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_size,
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


def _get_mean_std(dataset_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    family = legacy_datasets.dataset_to_modelfamily[dataset_name]
    mean_std = legacy_datasets.modelfamily_to_mean_std[family]
    return tuple(mean_std["mean"]), tuple(mean_std["std"])


def _as_dataset_name(dataset: str) -> str:
    """Map a (case-insensitive) dataset name to its ``defenses.datasets`` key."""
    lookup = {key.lower(): key for key in legacy_datasets.dataset_to_modelfamily}
    return lookup.get(dataset.lower(), dataset)


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train_flowpure_pgd(cfg: FlowPurePGDConfig) -> FlowMatchingCheckpoint:
    """Train a FlowPure^PGD CNF (x_0 = PGD(x), x_1 = x) and persist a checkpoint.

    Re-running the same command resumes from ``checkpoint_latest.pt``; a
    finished run (``DONE.json`` present) returns immediately.
    """
    set_random_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "checkpoint_latest.pt"

    device = torch.device(cfg.device)
    if cfg.resume and (output_dir / "DONE.json").exists() and latest_path.exists():
        print(f"[FlowPurePGD] {output_dir} is already complete; nothing to do.")
        return load_flow_matching_checkpoint(latest_path, device=device)
    if cfg.pgd_label_source not in {"dataset", "prediction"}:
        raise ValueError(f"pgd_label_source must be 'dataset' or 'prediction', got {cfg.pgd_label_source!r}")

    # The sidecar marks the checkpoint as FlowPure-family (noise_type='pgd') for
    # the detectors' checkpoint-kind checks; write it before the first save.
    (output_dir / "flowpure_pgd_config.json").write_text(_json_dump(cfg.to_dict()), encoding="utf-8")

    dataset_config = resolve_dataset_config(cfg.dataset, cfg.data_path)

    dataset = build_flow_matching_dataset(dataset_config, download=cfg.download)
    if cfg.subset_size is not None and cfg.subset_size < len(dataset):
        indices = list(range(int(cfg.subset_size)))
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    data_iter = _infinite_loader(loader)

    loaded_victim = load_legacy_model(cfg.victim_checkpoint_dir, device=device)
    victim = loaded_victim.model
    for parameter in victim.parameters():
        parameter.requires_grad_(False)
    victim.eval()

    dataset_name = _as_dataset_name(dataset_config.legacy_dataset or cfg.dataset)
    mean, std = _get_mean_std(dataset_name)
    if tuple(mean) != tuple(_get_mean_std(loaded_victim.dataset_name)[0]):
        raise ValueError(
            f"Classifier {cfg.victim_checkpoint_dir} was trained on {loaded_victim.dataset_name}, "
            f"whose normalization differs from {dataset_name}'s; PGD would attack mis-scaled inputs."
        )
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)

    model = build_model(
        dataset_config,
        use_ema=False,
        ema_decay=0.999,
        feature_dim=None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    path = CondOTProbPath()
    training_cfg_for_ckpt = _build_training_config_for_checkpoint(cfg)

    start_step = 1
    latest_checkpoint_path: Path | None = None
    if cfg.resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["epoch"]) + 1  # "epoch" stores the step count here
        latest_checkpoint_path = latest_path
        print(f"[FlowPurePGD] resuming from {latest_path} at step {start_step}")

    running_loss = 0.0
    running_count = 0
    checked_labels = False

    for step in range(start_step, cfg.max_steps + 1):
        samples, labels = next(data_iter)
        x_clean01 = samples.to(device, non_blocking=True)
        if cfg.pgd_label_source == "prediction":
            with torch.no_grad():
                y = victim((x_clean01 - mean_t) / std_t).argmax(dim=1)
        else:
            y = labels.to(device, non_blocking=True).long()
            if not checked_labels:
                if int(y.max().item()) >= int(loaded_victim.num_classes):
                    raise ValueError(
                        f"Dataset labels reach {int(y.max().item())} but the classifier has "
                        f"{loaded_victim.num_classes} classes; use --pgd-label-source prediction."
                    )
                checked_labels = True

        eps_batch = torch.rand(x_clean01.shape[0], device=device) * cfg.pgd_eps_max
        x_adv01 = pgd_linf(
            victim,
            x_clean01,
            y,
            eps=eps_batch,
            alpha=cfg.pgd_alpha,
            steps=cfg.pgd_steps,
            mean=mean,
            std=std,
            random_start=True,
        )

        x1 = x_clean01 * 2.0 - 1.0
        x0 = x_adv01 * 2.0 - 1.0
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

        if step % cfg.log_every == 0 or step == start_step:
            avg_loss = running_loss / max(1, running_count)
            print(f"[FlowPurePGD] step={step}/{cfg.max_steps} loss={avg_loss:.6f}", flush=True)
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

    if latest_checkpoint_path is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    write_done_marker(output_dir, global_step=cfg.max_steps, kind="flowpure_pgd")
    return load_flow_matching_checkpoint(latest_checkpoint_path, device=device)


def _json_dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a FlowPure^PGD CNF.")
    parser.add_argument("--victim-checkpoint-dir", type=str, required=False)
    parser.add_argument("--output-dir", type=str, required=False)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--pgd-eps-max", type=float, default=0.05)
    parser.add_argument("--pgd-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short CPU-friendly smoke test (tiny subset, ~200 steps).",
    )
    return parser


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    args.batch_size = min(args.batch_size, 16)
    args.max_steps = min(args.max_steps, 200)
    args.checkpoint_every = min(args.checkpoint_every, 100)
    args.log_every = min(args.log_every, 20)
    args.pgd_steps = min(args.pgd_steps, 3)
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

    if args.victim_checkpoint_dir is None:
        args.victim_checkpoint_dir = str(
            Path("runs") / "notebook" / "training-victim-cifar10-vgg16_bn-nodefense" / "target_model"
        )
    if args.output_dir is None:
        suffix = "_smoke" if args.smoke else ""
        args.output_dir = str(
            Path("runs") / "flow_matching" / f"cifar10_flowpure_pgd{suffix}"
        )

    cfg = FlowPurePGDConfig(
        victim_checkpoint_dir=args.victim_checkpoint_dir,
        output_dir=args.output_dir,
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
    )
    print(f"[FlowPurePGD] config: {cfg.to_dict()}")
    train_flowpure_pgd(cfg)


if __name__ == "__main__":
    main()
