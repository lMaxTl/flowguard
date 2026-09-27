"""Train the attacker's 32x32 DDPM prior for Guided DisGUIDE on a public pool.

Guided DisGUIDE decodes its queries with a frozen, pretrained diffusion model
"pretrained on public data from the victim's coarse input domain". For the
multi-dataset benchmark no suitable public checkpoint exists (and the CIFAR-10
default, ``google/ddpm-cifar10-32``, was trained on the CIFAR-10 *training set*,
i.e. on the victim's own data). This script trains that prior on the attacker's
public pool instead (CIFAR100, BelgiumTS, LFW, BCN20000), so the attacker never
touches victim data.

The architecture matches ``google/ddpm-cifar10-32`` (35.7M-parameter UNet, 1000
linear-beta DDPM steps, epsilon prediction). The output directory uses the same
flat diffusers layout (``config.json`` + weights + ``scheduler_config.json``), so
``--diffusion-model-id <output-dir>`` works wherever the model id was used.

Resumable: re-running the same command continues from ``training_state.pt``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defenses import datasets as legacy_datasets  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="defenses.datasets key, e.g. LFW.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--sample-every", type=int, default=25_000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_unet(sample_size: int = 32):
    from diffusers import UNet2DModel

    # Same block layout as google/ddpm-cifar10-32.
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        dropout=0.1,
    )


def build_dataset(name: str, download: bool):
    registry = {key.lower(): key for key in legacy_datasets.dataset_to_modelfamily}
    key = registry.get(name.lower())
    if key is None:
        raise KeyError(f"Unknown dataset '{name}'.")
    family = legacy_datasets.dataset_to_modelfamily[key]
    steps = [transforms.Resize((32, 32))]
    if family not in {"gtsrb", "mnist"}:  # mirrored traffic signs change meaning
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return legacy_datasets.__dict__[key](train=True, download=download, transform=transforms.Compose(steps))


def infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def update_ema(ema_model, model, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters(), strict=True):
        ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)


@torch.no_grad()
def save_samples(unet, scheduler_config, path: Path, device, seed: int) -> None:
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(50, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(64, 3, 32, 32, generator=generator).to(device)
    unet.eval()
    for timestep in scheduler.timesteps:
        sample = scheduler.step(unet(sample, timestep).sample, timestep, sample).prev_sample
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image((sample.clamp(-1, 1) + 1) * 0.5, path, nrow=8)


def main() -> None:
    args = build_parser().parse_args()
    from diffusers import DDPMScheduler

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_dir / "DONE.json").exists():
        print(f"[prior] {output_dir} is already complete; nothing to do.")
        return

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = build_dataset(args.dataset, args.download)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    batches = infinite(loader)

    unet = build_unet().to(device)
    ema_unet = copy.deepcopy(unet).eval()
    for parameter in ema_unet.parameters():
        parameter.requires_grad_(False)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        clip_sample=True,
        prediction_type="epsilon",
    )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.0)
    lr_lambda = lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))  # noqa: E731
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    state_path = output_dir / "training_state.pt"
    step = 0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        unet.load_state_dict(state["unet"])
        ema_unet.load_state_dict(state["ema_unet"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        step = int(state["step"])
        print(f"[prior] resuming at step {step}")

    (output_dir / "prior_config.json").write_text(
        json.dumps({**vars(args), "num_images": len(dataset)}, indent=2), encoding="utf-8"
    )
    print(f"[prior] dataset={args.dataset} images={len(dataset)} params="
          f"{sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M steps={args.max_steps}")

    use_amp = device.type == "cuda"
    started = time.time()
    running = 0.0
    while step < args.max_steps:
        images, _ = next(batches)
        images = images.to(device, non_blocking=True) * 2.0 - 1.0
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, args.num_train_timesteps, (images.shape[0],), device=device)
        noisy = scheduler.add_noise(images, noise, timesteps)
        unet.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            prediction = unet(noisy, timesteps).sample
        loss = F.mse_loss(prediction.float(), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        update_ema(ema_unet, unet, args.ema_decay if step >= args.warmup_steps else 0.0)
        step += 1
        running += float(loss.item())

        if step % args.log_every == 0:
            rate = step / max(1e-6, time.time() - started)
            print(f"[prior] step={step}/{args.max_steps} loss={running / args.log_every:.5f} "
                  f"({rate:.1f} it/s this session)", flush=True)
            running = 0.0
        if step % args.checkpoint_every == 0 or step == args.max_steps:
            temporary = state_path.with_name(state_path.name + ".tmp")
            torch.save(
                {
                    "unet": unet.state_dict(),
                    "ema_unet": ema_unet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "step": step,
                },
                temporary,
            )
            temporary.replace(state_path)
        if args.sample_every and (step % args.sample_every == 0 or step == args.max_steps):
            save_samples(ema_unet, scheduler.config,
                         output_dir / "samples" / f"step_{step:07d}.png", device, args.seed)

    # Flat diffusers layout, like google/ddpm-cifar10-32:
    # UNet2DModel.from_pretrained(dir) and <Scheduler>.from_pretrained(dir).
    ema_unet.save_pretrained(output_dir, safe_serialization=True)
    scheduler.save_pretrained(output_dir)
    (output_dir / "DONE.json").write_text(
        json.dumps({"kind": "ddpm_prior", "global_step": step, "dataset": args.dataset}, indent=2),
        encoding="utf-8",
    )
    print(f"[prior] saved EMA UNet + scheduler to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
