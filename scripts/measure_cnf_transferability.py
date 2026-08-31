"""Measure surrogate-CNF score transferability for FlowBlind (plan A3).

Loads a target CNF (the ``real`` defense checkpoint) and one or more
surrogate CNFs (trained with different seeds / datasets / victims) and
computes the correlation of their ``||v(t=0, x)||^2`` scores on a shared
probe batch. A high Pearson r (``>= 0.7`` is the plan's threshold) shows
that the adaptive FlowBlind attack can rely on transferability to evade
the real detector without query access.

Usage:

.. code-block:: powershell

    .\\.venv\\Scripts\\python.exe scripts\\measure_cnf_transferability.py \\
        --target-checkpoint runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt \\
        --surrogate-checkpoints \\
            runs/flow_matching/surrogate_cifar100_seed0/checkpoint_latest.pt \\
            runs/flow_matching/surrogate_cifar100_seed1/checkpoint_latest.pt \\
        --probe-dataset CIFAR10 --num-probes 256

The script also evaluates a mix of probe types: clean CIFAR-10,
PGD-adversarial CIFAR-10 (eps=8/255), and uniform noise, so transferability
can be reported per probe family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flowguard.attacks.adaptive import SurrogateVelocityRegularizer
from flowguard.attacks.pgd import pgd_linf
from flowguard.flow_matching.config import resolve_dataset_config
from flowguard.flow_matching.data import build_flow_matching_dataset
from flowguard.serving.model_loader import load_legacy_model


def _gather_clean_probes(
    *,
    dataset_name: str,
    num_probes: int,
    download: bool,
) -> torch.Tensor:
    dataset_config = resolve_dataset_config(dataset_name.lower(), None)
    dataset = build_flow_matching_dataset(dataset_config, download=download)
    indices = list(range(min(num_probes, len(dataset))))
    samples = torch.stack([dataset[i][0] for i in indices], dim=0)
    return samples


def _make_pgd_probes(
    victim_checkpoint_dir: Path,
    clean_probes_01: torch.Tensor,
    *,
    device: torch.device,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """Generate PGD-adversarial probes on top of clean [0,1] tensors."""
    loaded = load_legacy_model(str(victim_checkpoint_dir), device=device)
    victim = loaded.model
    for parameter in victim.parameters():
        parameter.requires_grad_(False)
    victim.eval()

    with torch.no_grad():
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
        normalized = (clean_probes_01.to(device) - mean) / std
        logits = victim(normalized)
        labels = logits.argmax(dim=1).detach()

    adversarial = pgd_linf(
        victim,
        clean_probes_01.to(device),
        labels,
        eps=eps,
        alpha=alpha,
        steps=steps,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        random_start=True,
    )
    return adversarial.detach().cpu()


def _score_probes(
    regularizer: SurrogateVelocityRegularizer,
    probes_01: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    scores: list[float] = []
    for start in range(0, probes_01.shape[0], batch_size):
        chunk = probes_01[start : start + batch_size]
        with torch.no_grad():
            chunk_scores = regularizer.velocity_score(chunk)
        scores.extend(chunk_scores.detach().cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((a_centered * b_centered).sum() / denom)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--surrogate-checkpoints", type=Path, nargs="+", required=True,
        help="One or more surrogate CNF checkpoints to compare against target.",
    )
    parser.add_argument("--probe-dataset", default="CIFAR10")
    parser.add_argument("--num-probes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--victim-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "notebook"
        / "training-victim-cifar10-vgg16_bn-nodefense"
        / "target_model",
        help="Used for generating PGD probes.",
    )
    parser.add_argument("--pgd-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--pgd-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--skip-pgd", action="store_true",
                        help="Skip PGD probe generation (e.g. when no victim is available).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    clean_probes_01 = _gather_clean_probes(
        dataset_name=args.probe_dataset,
        num_probes=int(args.num_probes),
        download=bool(args.download),
    )

    probe_sets: dict[str, torch.Tensor] = {"clean": clean_probes_01}
    if not args.skip_pgd and args.victim_checkpoint_dir.exists():
        print(f"Generating PGD probes (eps={args.pgd_eps:.4f}, steps={args.pgd_steps})...")
        probe_sets["pgd"] = _make_pgd_probes(
            args.victim_checkpoint_dir,
            clean_probes_01,
            device=device,
            eps=float(args.pgd_eps),
            alpha=float(args.pgd_alpha),
            steps=int(args.pgd_steps),
        )
    probe_sets["uniform_noise"] = torch.rand_like(clean_probes_01)

    print(f"Loading target CNF: {args.target_checkpoint}")
    target = SurrogateVelocityRegularizer(str(args.target_checkpoint), device=device)

    results: dict[str, dict] = {}
    target_scores_by_probe = {
        probe_name: _score_probes(target, tensor, batch_size=int(args.batch_size))
        for probe_name, tensor in probe_sets.items()
    }

    for surrogate_path in args.surrogate_checkpoints:
        print(f"\nLoading surrogate CNF: {surrogate_path}")
        try:
            surrogate = SurrogateVelocityRegularizer(str(surrogate_path), device=device)
        except Exception as exc:
            print(f"  failed to load surrogate {surrogate_path!s}: {exc!r}")
            results[str(surrogate_path)] = {"error": repr(exc)}
            continue

        surrogate_scores_by_probe = {
            probe_name: _score_probes(surrogate, tensor, batch_size=int(args.batch_size))
            for probe_name, tensor in probe_sets.items()
        }
        per_probe: dict[str, dict] = {}
        for probe_name, target_scores in target_scores_by_probe.items():
            surrogate_scores = surrogate_scores_by_probe[probe_name]
            r = _pearson(target_scores, surrogate_scores)
            per_probe[probe_name] = {
                "pearson_r": r,
                "target_mean": float(target_scores.mean()),
                "target_std": float(target_scores.std()),
                "surrogate_mean": float(surrogate_scores.mean()),
                "surrogate_std": float(surrogate_scores.std()),
            }
            print(f"  {probe_name:15s}: r = {r:+.4f}")

        results[str(surrogate_path)] = per_probe

    payload = {
        "target_checkpoint": str(args.target_checkpoint),
        "probe_dataset": args.probe_dataset,
        "num_probes": int(args.num_probes),
        "pgd_eps": float(args.pgd_eps),
        "pgd_steps": int(args.pgd_steps),
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved transferability report to: {args.output_json}")


if __name__ == "__main__":
    main()
