"""Trainer for FlowPureBoundary: CNF mapping boundary-proximal -> clean samples.

Structurally mirrors ``flowpure_training.py`` but replaces online PGD with a
BAM-style evolutionary search on the victim's softmax confidence. Training
pairs are therefore attack-agnostic: the only signal used to construct
``x_0`` is the victim's own decision geometry.
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
from flowguard.attacks.boundary_es import boundary_es
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
)
from flowguard.serving.model_loader import load_legacy_model


@dataclass(slots=True)
class FlowPureBoundaryConfig:
    """Runtime configuration for FlowPureBoundary training."""

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
    es_pop: int = 8
    es_top_k: int = 2
    es_iters: int = 4
    es_gamma: float = 0.1
    es_init_eps: float = 0.05
    es_eval_batch: int = 256
    num_workers: int = 0
    subset_size: int | None = None
    download: bool = False

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
            "es_pop": self.es_pop,
            "es_top_k": self.es_top_k,
            "es_iters": self.es_iters,
            "es_gamma": self.es_gamma,
            "es_init_eps": self.es_init_eps,
            "es_eval_batch": self.es_eval_batch,
            "num_workers": self.num_workers,
            "subset_size": self.subset_size,
            "download": self.download,
            "noise_type": "boundary",
        }


def _build_training_config_for_checkpoint(
    cfg: FlowPureBoundaryConfig,
) -> FlowMatchingTrainingConfig:
    """Build a FlowMatchingTrainingConfig compatible with the checkpoint loader.

    We piggy-back on the existing schema so the saved checkpoint is readable
    by ``load_flow_matching_checkpoint`` without modification.
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
    return {"cifar10": "CIFAR10", "cifar100": "CIFAR100"}.get(dataset.lower(), dataset)


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train_flowpure_boundary(cfg: FlowPureBoundaryConfig) -> FlowMatchingCheckpoint:
    """Train a FlowPureBoundary CNF and persist a checkpoint."""
    set_random_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
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

    mean, std = _get_mean_std(_as_dataset_name(cfg.dataset))

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
        samples, _labels = next(data_iter)
        x_clean01 = samples.to(device, non_blocking=True)

        x_boundary01 = boundary_es(
            victim,
            x_clean01,
            mean=mean,
            std=std,
            iters=cfg.es_iters,
            pop=cfg.es_pop,
            top_k=cfg.es_top_k,
            gamma=cfg.es_gamma,
            seed="data",
            init_eps=cfg.es_init_eps,
            eval_batch=cfg.es_eval_batch,
        )

        x1 = x_clean01 * 2.0 - 1.0
        x0 = x_boundary01 * 2.0 - 1.0
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
            print(f"[FlowPureBoundary] step={step}/{cfg.max_steps} loss={avg_loss:.6f}")
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
            meta_path = output_dir / "flowpure_boundary_config.json"
            meta_path.write_text(_json_dump(cfg.to_dict()), encoding="utf-8")

    if latest_checkpoint_path is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    return load_flow_matching_checkpoint(latest_checkpoint_path, device=device)


def _json_dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a FlowPureBoundary CNF.")
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
    parser.add_argument("--es-pop", type=int, default=8)
    parser.add_argument("--es-top-k", type=int, default=2)
    parser.add_argument("--es-iters", type=int, default=4)
    parser.add_argument("--es-gamma", type=float, default=0.1)
    parser.add_argument("--es-init-eps", type=float, default=0.05)
    parser.add_argument("--es-eval-batch", type=int, default=256)
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
    args.es_pop = min(args.es_pop, 4)
    args.es_top_k = min(args.es_top_k, 2)
    args.es_iters = min(args.es_iters, 2)
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
            Path("runs") / "flow_matching" / f"cifar10_flowpure_boundary{suffix}"
        )

    cfg = FlowPureBoundaryConfig(
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
        es_pop=args.es_pop,
        es_top_k=args.es_top_k,
        es_iters=args.es_iters,
        es_gamma=args.es_gamma,
        es_init_eps=args.es_init_eps,
        es_eval_batch=args.es_eval_batch,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        download=args.download,
    )
    print(f"[FlowPureBoundary] config: {cfg.to_dict()}")
    train_flowpure_boundary(cfg)


if __name__ == "__main__":
    main()
