"""Solver diagnostics and density-model audit for the CIFAR-10 flow detectors.

Addresses reviewer Major Comment 7, items 5 and 6 (divergence estimator, ODE
solver / tolerance / NFE / numerical error) on the *deployed* CIFAR-10 models,
and quantifies the difference between the two candidate density models:

``conditional`` -- the FlowPure-style CNF (``x_0`` = PGD-perturbed image,
    ``x_1`` = clean image) that supplies the velocity signals C1 and C2 and that
    the reported experiments also used for the likelihood component C3.

``generative``  -- a CNF trained with conditional flow matching from
    ``p_0 = N(0, I)`` to the CIFAR-10 image distribution, i.e. the model for
    which Eq. 9 returns the benign data log-density by construction.

For each model the script reports:

* the log-likelihood and its decomposition into the base term ``log p_0(z_0)``
  and the divergence integral;
* round-trip invertibility of the learned transport (``1 -> 0 -> 1``);
* the Hutchinson divergence estimator's spread across independent probes;
* a solver sweep (method, step size, tolerance) against a high-accuracy
  reference, with the number of function evaluations;
* how much of the score is explained by the trivial statistic
  ``-||x||^2 / 2`` (the base density read off at the query itself), which is
  what the change-of-variables integral degenerates to when the learned
  transport is close to the identity;
* detection AUROC of the two-sided typicality statistic on the saved D1/D2
  bypass queries.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\validate_likelihood_cifar.py \\
        --benign-samples 1024 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader, Subset

from defenses import datasets as legacy_datasets
from flowguard.defenses.query.flow_matching import _standard_gaussian_log_prob
from flowguard.defenses.query.flowguard import _prepare_inputs
from flowguard.flow_matching.training import (
    VelocityFieldWrapper,
    load_flow_matching_checkpoint,
)

DEFAULT_MODELS = {
    "conditional": PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd" / "checkpoint_latest.pt",
    "generative": PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_notebook" / "checkpoint_latest.pt",
}

DEFAULT_ATTACKS = {
    "D1_latent_manifold": PROJECT_ROOT / "runs" / "notebook" / "data_free_bypass_viz" / "d1_latent_manifold" / "transferset.pickle",
    "D2_procedural_natural": PROJECT_ROOT / "runs" / "notebook" / "data_free_bypass_viz" / "d2_procedural_natural" / "transferset.pickle",
}


class CountingVelocity:
    """Wrap a velocity field and count network evaluations."""

    def __init__(self, velocity: Callable[..., Tensor]) -> None:
        self.velocity = velocity
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(self, x: Tensor, t: Tensor, **extras: Any) -> Tensor:
        self.calls += 1
        return self.velocity(x=x, t=t, **extras)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_benign_images(*, dataset_name: str, count: int, download: bool) -> Tensor:
    """Benign CIFAR-10 test images with the same transform the evaluation used."""
    modelfamily = legacy_datasets.dataset_to_modelfamily[dataset_name]
    transform = legacy_datasets.modelfamily_to_transforms[modelfamily]["test"]
    dataset = legacy_datasets.__dict__[dataset_name](
        train=False,
        transform=transform,
        download=download,
    )
    subset = Subset(dataset, list(range(min(count, len(dataset)))))
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
    return torch.cat([inputs for inputs, _ in loader], dim=0)


def load_attack_queries(path: Path) -> Tensor | None:
    """Load saved bypass queries (already in the victim's normalized space)."""
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = [torch.as_tensor(np.asarray(item[0]), dtype=torch.float32) for item in payload]
    return torch.stack(samples, dim=0)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def compute_likelihood(
    velocity: Callable[..., Tensor],
    inputs: Tensor,
    *,
    method: str,
    step_size: float | None,
    atol: float,
    rtol: float,
    batch_size: int,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Return (log_likelihood, base_term, divergence_integral, NFE per solve).

    The divergence integral is recovered as ``log p_1(x) - log p_0(z_0)``, which
    by Eq. 9 equals ``-int_0^1 div v_theta dt``.
    """
    counter = CountingVelocity(velocity)
    solver = ODESolver(velocity_model=counter)
    likelihoods: list[Tensor] = []
    base_terms: list[Tensor] = []
    num_batches = 0
    for start in range(0, inputs.shape[0], batch_size):
        chunk = inputs[start : start + batch_size]
        if seed is not None:
            torch.manual_seed(seed + start)
        latent, log_likelihood = solver.compute_likelihood(
            x_1=chunk,
            log_p0=_standard_gaussian_log_prob,
            step_size=step_size,
            time_grid=torch.tensor([1.0, 0.0], device=chunk.device),
            method=method,
            atol=atol,
            rtol=rtol,
            exact_divergence=False,
            enable_grad=False,
        )
        likelihoods.append(log_likelihood.detach().cpu())
        base_terms.append(_standard_gaussian_log_prob(latent).detach().cpu())
        num_batches += 1
    log_likelihood = torch.cat(likelihoods, dim=0)
    base_term = torch.cat(base_terms, dim=0)
    return (
        log_likelihood,
        base_term,
        log_likelihood - base_term,
        counter.calls // max(1, num_batches),
    )


def round_trip_error(
    velocity: Callable[..., Tensor],
    inputs: Tensor,
    *,
    method: str,
    step_size: float | None,
    batch_size: int,
) -> dict[str, float]:
    """Integrate ``1 -> 0`` and back, and measure the reconstruction error."""
    solver = ODESolver(velocity_model=velocity)
    relative_errors: list[Tensor] = []
    latent_norms: list[Tensor] = []
    for start in range(0, inputs.shape[0], batch_size):
        chunk = inputs[start : start + batch_size]
        latent = solver.sample(
            x_init=chunk,
            step_size=step_size,
            method=method,
            time_grid=torch.tensor([1.0, 0.0], device=chunk.device),
        )
        recovered = solver.sample(
            x_init=latent,
            step_size=step_size,
            method=method,
            time_grid=torch.tensor([0.0, 1.0], device=chunk.device),
        )
        numerator = (recovered - chunk).flatten(start_dim=1).norm(dim=1)
        denominator = chunk.flatten(start_dim=1).norm(dim=1).clamp_min(1e-12)
        relative_errors.append((numerator / denominator).detach().cpu())
        latent_norms.append(latent.flatten(start_dim=1).norm(dim=1).detach().cpu())
    errors = torch.cat(relative_errors, dim=0)
    norms = torch.cat(latent_norms, dim=0)
    dimension = int(np.prod(inputs.shape[1:]))
    return {
        "mean_relative_l2_error": float(errors.mean().item()),
        "max_relative_l2_error": float(errors.max().item()),
        "mean_latent_norm": float(norms.mean().item()),
        "expected_gaussian_norm": float(math.sqrt(dimension)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUROC via the rank-sum identity (positive = attack, higher = flagged)."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = np.argsort(np.argsort(combined)) + 1.0
    # Average ranks for ties.
    order = np.argsort(combined)
    sorted_values = combined[order]
    index = 0
    while index < sorted_values.size:
        stop = index
        while stop + 1 < sorted_values.size and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        if stop > index:
            ranks[order[index : stop + 1]] = np.mean(ranks[order[index : stop + 1]])
        index = stop + 1
    positive_rank_sum = ranks[: positive.size].sum()
    return float(
        (positive_rank_sum - positive.size * (positive.size + 1) / 2.0)
        / (positive.size * negative.size)
    )


def correlation(a: Tensor, b: Tensor) -> float:
    return float(np.corrcoef(a.double().numpy(), b.double().numpy())[0, 1])


def typicality(scores: Tensor, mean: float, std: float) -> np.ndarray:
    return (scores.double().numpy() - mean).__abs__() / max(std, 1e-12)


# ---------------------------------------------------------------------------
# Per-model audit
# ---------------------------------------------------------------------------


def audit_model(
    name: str,
    checkpoint_path: Path,
    *,
    benign: Tensor,
    attacks: dict[str, Tensor],
    device: torch.device,
    batch_size: int,
    sweep_samples: int,
    probe_repeats: int,
) -> dict[str, Any]:
    print(f"\n=== {name}: {checkpoint_path} ===")
    checkpoint = load_flow_matching_checkpoint(str(checkpoint_path), device=device)
    checkpoint.model.eval()
    velocity = VelocityFieldWrapper(
        model=checkpoint.model,
        class_conditioned=checkpoint.dataset_config.class_conditioned,
    )
    image_size = int(checkpoint.dataset_config.image_size)

    def prepare(batch: Tensor) -> Tensor:
        return _prepare_inputs(
            batch,
            device=device,
            inputs_normalized=True,
            modelfamily="cifar",
            expected_size=image_size,
        )

    benign_inputs = prepare(benign)
    attack_inputs = {key: prepare(value) for key, value in attacks.items()}

    results: dict[str, Any] = {"checkpoint": str(checkpoint_path)}

    # --- Reference likelihood (the deployed operating point) ----------------
    print("  [1/5] likelihood at the deployed setting (midpoint, h=0.05)")
    benign_ll, benign_base, benign_divergence, nfe = compute_likelihood(
        velocity,
        benign_inputs,
        method="midpoint",
        step_size=0.05,
        atol=1e-5,
        rtol=1e-5,
        batch_size=batch_size,
        seed=0,
    )
    benign_mean = float(benign_ll.mean().item())
    benign_std = float(benign_ll.std().item())
    dimension = 3 * image_size * image_size
    results["deployed_setting"] = {
        "method": "midpoint",
        "step_size": 0.05,
        "num_function_evaluations": nfe,
        "benign_log_likelihood_mean": benign_mean,
        "benign_log_likelihood_std": benign_std,
        "benign_log_likelihood_mean_per_dim": benign_mean / dimension,
        "benign_base_term_mean": float(benign_base.mean().item()),
        "benign_divergence_integral_mean": float(benign_divergence.mean().item()),
        "base_term_share_of_variance": float(
            benign_base.var().item() / max(benign_ll.var().item(), 1e-12)
        ),
        "dimension": dimension,
        "num_benign_samples": int(benign_inputs.shape[0]),
    }
    print(
        f"        log p = {benign_mean:.1f} +/- {benign_std:.1f} nats "
        f"({benign_mean / dimension:.3f} nats/dim), NFE={nfe}"
    )

    # --- How much is just the base density read at the query? ---------------
    naive = _standard_gaussian_log_prob(benign_inputs).detach().cpu()
    results["naive_norm_statistic"] = {
        "pearson_r_with_log_likelihood": correlation(benign_ll, naive),
        "description": "-||x||^2/2 - (d/2) log 2pi evaluated at the query itself",
    }
    print(
        "  [2/5] correlation with the trivial -||x||^2/2 statistic: "
        f"r={results['naive_norm_statistic']['pearson_r_with_log_likelihood']:+.4f}"
    )

    # --- Round-trip invertibility ------------------------------------------
    print("  [3/5] round-trip invertibility (1 -> 0 -> 1)")
    subset = benign_inputs[: min(sweep_samples, benign_inputs.shape[0])]
    results["round_trip"] = round_trip_error(
        velocity, subset, method="midpoint", step_size=0.05, batch_size=batch_size
    )
    print(
        f"        mean relative L2 error = "
        f"{results['round_trip']['mean_relative_l2_error']:.3e}; "
        f"||z_0|| = {results['round_trip']['mean_latent_norm']:.1f} "
        f"(sqrt(d) = {results['round_trip']['expected_gaussian_norm']:.1f})"
    )

    # --- Hutchinson probe spread -------------------------------------------
    print(f"  [4/5] Hutchinson estimator spread over {probe_repeats} probes")
    repeats = []
    for repeat in range(probe_repeats):
        value, _, _, _ = compute_likelihood(
            velocity,
            subset,
            method="midpoint",
            step_size=0.05,
            atol=1e-5,
            rtol=1e-5,
            batch_size=batch_size,
            seed=1000 * (repeat + 1),
        )
        repeats.append(value)
    stacked = torch.stack(repeats, dim=0)
    per_sample_std = stacked.std(dim=0)
    results["hutchinson"] = {
        "num_probes_per_estimate": 1,
        "num_independent_repeats": int(probe_repeats),
        "per_sample_std_nats": float(per_sample_std.mean().item()),
        "per_sample_std_nats_per_dim": float(per_sample_std.mean().item() / dimension),
        "std_relative_to_benign_spread": float(
            per_sample_std.mean().item() / max(benign_std, 1e-12)
        ),
        "note": (
            "Exact divergence needs one backward pass per input dimension "
            f"({dimension} per solver step) and is not tractable here; the "
            "spread across independent Rademacher probes is reported instead."
        ),
    }
    print(
        f"        std = {results['hutchinson']['per_sample_std_nats']:.2f} nats "
        f"= {results['hutchinson']['std_relative_to_benign_spread']:.3f} x the benign spread"
    )

    # --- Solver sweep -------------------------------------------------------
    print("  [5/5] solver sweep against a high-accuracy reference")
    reference, _, _, reference_nfe = compute_likelihood(
        velocity,
        subset,
        method="dopri5",
        step_size=None,
        atol=1e-6,
        rtol=1e-6,
        batch_size=batch_size,
        seed=17,
    )
    sweep: dict[str, Any] = {
        "reference": {
            "method": "dopri5",
            "tolerance": 1e-6,
            "num_function_evaluations": reference_nfe,
        }
    }
    for method, step_size, tolerance in (
        ("euler", 0.1, None),
        ("euler", 0.05, None),
        ("midpoint", 0.1, None),
        ("midpoint", 0.05, None),
        ("midpoint", 0.02, None),
        ("dopri5", None, 1e-4),
    ):
        estimate, _, _, nfe = compute_likelihood(
            velocity,
            subset,
            method=method,
            step_size=step_size,
            atol=tolerance if tolerance is not None else 1e-5,
            rtol=tolerance if tolerance is not None else 1e-5,
            batch_size=batch_size,
            seed=17,
        )
        difference = (estimate - reference).abs()
        key = f"{method}/step={step_size}/tol={tolerance}"
        sweep[key] = {
            "mean_absolute_error_nats": float(difference.mean().item()),
            "mean_absolute_error_nats_per_dim": float(difference.mean().item() / dimension),
            "max_absolute_error_nats": float(difference.max().item()),
            "error_relative_to_benign_spread": float(
                difference.mean().item() / max(benign_std, 1e-12)
            ),
            "num_function_evaluations": nfe,
        }
        print(
            f"        {key:30s} MAE={sweep[key]['mean_absolute_error_nats']:8.2f} nats  "
            f"NFE={nfe}"
        )
    results["solver_sweep"] = sweep

    # --- Detection ----------------------------------------------------------
    detection: dict[str, Any] = {}
    benign_typicality = typicality(benign_ll, benign_mean, benign_std)
    benign_naive_mean = float(naive.mean().item())
    benign_naive_std = float(naive.std().item())
    benign_naive_typicality = typicality(naive, benign_naive_mean, benign_naive_std)
    for attack_name, attack_batch in attack_inputs.items():
        attack_ll, _, _, _ = compute_likelihood(
            velocity,
            attack_batch,
            method="midpoint",
            step_size=0.05,
            atol=1e-5,
            rtol=1e-5,
            batch_size=batch_size,
            seed=0,
        )
        attack_naive = _standard_gaussian_log_prob(attack_batch).detach().cpu()
        detection[attack_name] = {
            "num_attack_samples": int(attack_batch.shape[0]),
            "attack_log_likelihood_mean": float(attack_ll.mean().item()),
            "benign_log_likelihood_mean": benign_mean,
            "shift_in_benign_std": float((attack_ll.mean().item() - benign_mean) / benign_std),
            "auroc_two_sided_typicality": auroc(
                typicality(attack_ll, benign_mean, benign_std), benign_typicality
            ),
            "auroc_lower_tail_only": auroc(
                -attack_ll.double().numpy(), -benign_ll.double().numpy()
            ),
            "auroc_naive_norm_statistic": auroc(
                typicality(attack_naive, benign_naive_mean, benign_naive_std),
                benign_naive_typicality,
            ),
        }
        entry = detection[attack_name]
        print(
            f"        {attack_name:24s} AUROC(typicality)={entry['auroc_two_sided_typicality']:.3f}  "
            f"AUROC(-||x||^2/2)={entry['auroc_naive_norm_statistic']:.3f}  "
            f"shift={entry['shift_in_benign_std']:+.2f} sigma"
        )
    results["detection"] = detection
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "likelihood_validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--benign-samples", type=int, default=1024)
    parser.add_argument("--sweep-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-repeats", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benign = load_benign_images(
        dataset_name=args.dataset, count=args.benign_samples, download=args.download
    )
    print(f"Benign queries: {tuple(benign.shape)}")

    attacks: dict[str, Tensor] = {}
    for attack_name, path in DEFAULT_ATTACKS.items():
        queries = load_attack_queries(path)
        if queries is None:
            print(f"  (skipping {attack_name}: {path} not found)")
            continue
        attacks[attack_name] = queries
        print(f"  {attack_name}: {tuple(queries.shape)}")

    payload: dict[str, Any] = {
        "configuration": {
            "device": str(device),
            "dataset": args.dataset,
            "benign_samples": int(benign.shape[0]),
            "sweep_samples": args.sweep_samples,
            "probe_repeats": args.probe_repeats,
            "torch_version": torch.__version__,
        },
        "models": {},
    }

    for name, checkpoint_path in DEFAULT_MODELS.items():
        if not checkpoint_path.exists():
            print(f"(skipping {name}: {checkpoint_path} not found)")
            continue
        payload["models"][name] = audit_model(
            name,
            checkpoint_path,
            benign=benign,
            attacks=attacks,
            device=device,
            batch_size=args.batch_size,
            sweep_samples=args.sweep_samples,
            probe_repeats=args.probe_repeats,
        )

    output_path = args.output_dir / "likelihood_cifar_diagnostics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
