"""Ground-truth validation of the FlowGuard++ likelihood estimator (paper Eq. 9).

Addresses reviewer Major Comment 7. Every experiment below runs through the
*same* ``flow_matching.solver.ODESolver.compute_likelihood`` code path that the
deployed detector uses (see
``flowguard.defenses.query.flow_matching.FlowMatchingQueryDefense``), so what is
validated is the shipped estimator and not a re-derivation of it.

Experiments
-----------
A. **Analytic flows.** For ``v(t,x) = g(t) A x`` the flow map is
   ``exp(G(t) A)`` with ``G(t) = int_0^t g``, the divergence is ``g(t) tr(A)``,
   and the push-forward of ``N(0,I)`` is a Gaussian with closed-form density.
   Isolates *solver* error from *model* error: any deviation here is numerics.

B. **Generative CNF (noise -> data).** A CNF trained with conditional flow
   matching from ``p_0 = N(0,I)`` to a 2-D Gaussian mixture whose density is
   analytic. Verifies that Eq. 9 recovers the data log-density, and grid
   quadrature verifies that the estimate integrates to one.

C. **FlowPure-style CNF (perturbed -> clean).** The reviewer's concern in a
   controlled setting where every density is analytic. The *same* integral,
   evaluated with a standard-Gaussian base, does not return the data density;
   evaluated with the true source density it does. Both are grid-integrated to
   show that the Gaussian-base quantity is still a normalized density, namely
   the push-forward ``[phi_{0->1}]_# N(0,I)``, just not the data density.

D. **Divergence estimator.** Exact trace vs. the Hutchinson estimator with
   ``k`` Rademacher probes: bias and standard deviation in nats.

E. **Solver sweep.** Method / step size / tolerance against a high-accuracy
   reference, reported jointly with the number of function evaluations.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\validate_likelihood_synthetic.py \\
        --output-dir runs/likelihood_validation --device cuda
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
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

from flowguard.defenses.query.flow_matching import _standard_gaussian_log_prob

# ---------------------------------------------------------------------------
# Analytic reference densities
# ---------------------------------------------------------------------------


def gaussian_log_prob(x: Tensor, covariance: Tensor) -> Tensor:
    """Exact log-density of a zero-mean Gaussian with a full covariance."""
    dimension = covariance.shape[0]
    cholesky = torch.linalg.cholesky(covariance)
    log_det = 2.0 * torch.log(torch.diagonal(cholesky)).sum()
    solved = torch.cholesky_solve(x.unsqueeze(-1).double(), cholesky.double()).squeeze(-1)
    quadratic = (x.double() * solved).sum(dim=-1)
    return (-0.5 * (dimension * math.log(2.0 * math.pi) + log_det.double() + quadratic)).to(x.dtype)


class GaussianMixture:
    """Isotropic 2-D Gaussian mixture on a ring, with an analytic log-density."""

    def __init__(
        self,
        *,
        num_components: int = 8,
        radius: float = 2.5,
        component_std: float = 0.30,
        device: torch.device,
    ) -> None:
        angles = torch.arange(num_components, dtype=torch.float32) * (
            2.0 * math.pi / num_components
        )
        self.means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
        self.means = self.means.to(device)
        self.component_std = float(component_std)
        self.num_components = int(num_components)
        self.device = device

    def with_extra_variance(self, extra_std: float) -> GaussianMixture:
        """Return the mixture convolved with ``N(0, extra_std^2 I)``.

        Convolving a Gaussian mixture with isotropic Gaussian noise yields
        another mixture with the same means and inflated variances. This makes
        the *source* distribution of the FlowPure-style path (experiment C)
        analytic as well.
        """
        convolved = GaussianMixture(
            num_components=self.num_components,
            radius=1.0,
            component_std=math.sqrt(self.component_std**2 + extra_std**2),
            device=self.device,
        )
        convolved.means = self.means.clone()
        return convolved

    def sample(self, count: int, generator: torch.Generator | None = None) -> Tensor:
        indices = torch.randint(
            0, self.num_components, (count,), device=self.device, generator=generator
        )
        noise = torch.randn(count, 2, device=self.device, generator=generator)
        return self.means[indices] + self.component_std * noise

    def log_prob(self, x: Tensor) -> Tensor:
        variance = self.component_std**2
        # (B, K) squared distances to every component mean.
        squared = torch.cdist(x.unsqueeze(0), self.means.unsqueeze(0)).squeeze(0).pow(2)
        component_log_prob = -0.5 * (
            2.0 * math.log(2.0 * math.pi * variance) + squared / variance
        )
        return torch.logsumexp(component_log_prob, dim=1) - math.log(self.num_components)


# ---------------------------------------------------------------------------
# Velocity fields
# ---------------------------------------------------------------------------


class CountingVelocity:
    """Wrap a velocity field and count network evaluations (NFE)."""

    def __init__(self, velocity: Callable[..., Tensor]) -> None:
        self.velocity = velocity
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(self, x: Tensor, t: Tensor, **extras: Any) -> Tensor:
        self.calls += 1
        return self.velocity(x=x, t=t, **extras)


class LinearVelocity:
    """``v(t, x) = g(t) * x A^T`` with a closed-form flow map and divergence."""

    def __init__(self, matrix: Tensor, *, time_profile: str = "constant") -> None:
        self.matrix = matrix
        self.time_profile = time_profile

    def gain(self, t: Tensor | float) -> Tensor:
        value = torch.as_tensor(t, dtype=torch.float32)
        if self.time_profile == "constant":
            return torch.ones_like(value)
        # cos(2 pi t) + 1/2 integrates to 1/2 over [0, 1] and is genuinely
        # time-dependent, so it also exercises the time argument of the solver.
        return torch.cos(2.0 * math.pi * value) + 0.5

    def integrated_gain(self) -> float:
        return 1.0 if self.time_profile == "constant" else 0.5

    def __call__(self, x: Tensor, t: Tensor, **_: Any) -> Tensor:
        gain = self.gain(t).to(x.device)
        if gain.ndim == 0:
            gain = gain.reshape(1, 1)
        else:
            gain = gain.reshape(-1, 1)
        return gain * (x @ self.matrix.T)

    def pushforward_covariance(self) -> Tensor:
        """Covariance of ``[phi_{0->1}]_# N(0, I)``, computed in closed form."""
        transport = torch.matrix_exp(self.integrated_gain() * self.matrix.double())
        return (transport @ transport.T).float()


class TimeConditionedMLP(nn.Module):
    """Small velocity network for the 2-D synthetic problems."""

    def __init__(self, *, dimension: int = 2, hidden: int = 256, num_frequencies: int = 8) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32) * math.pi
        )
        input_dim = dimension + 2 * num_frequencies + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, x: Tensor, t: Tensor, **_: Any) -> Tensor:
        time = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if time.ndim == 0:
            time = time.expand(x.shape[0])
        time = time.reshape(-1, 1)
        scaled = time * self.frequencies.reshape(1, -1)
        features = torch.cat([x, time, torch.sin(scaled), torch.cos(scaled)], dim=1)
        return self.network(features)


# ---------------------------------------------------------------------------
# Likelihood helpers (thin wrappers over the shipped estimator)
# ---------------------------------------------------------------------------


def compute_log_likelihood(
    velocity: Callable[..., Tensor],
    x: Tensor,
    *,
    log_p0: Callable[[Tensor], Tensor],
    method: str = "midpoint",
    step_size: float | None = 0.05,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    exact_divergence: bool = True,
    batch_size: int = 8192,
) -> tuple[Tensor, int]:
    """Evaluate Eq. 9 through the deployed solver; also return the NFE."""
    counter = CountingVelocity(velocity)
    solver = ODESolver(velocity_model=counter)
    outputs: list[Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        chunk = x[start : start + batch_size]
        _, log_likelihood = solver.compute_likelihood(
            x_1=chunk,
            log_p0=log_p0,
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            exact_divergence=exact_divergence,
            enable_grad=False,
        )
        outputs.append(log_likelihood.detach())
    # NFE is per solve; report the cost of a single batch, not the sum over chunks.
    num_chunks = max(1, math.ceil(x.shape[0] / batch_size))
    return torch.cat(outputs, dim=0), counter.calls // num_chunks


def hutchinson_log_likelihood(
    velocity: Callable[..., Tensor],
    x: Tensor,
    *,
    log_p0: Callable[[Tensor], Tensor],
    num_probes: int,
    method: str = "midpoint",
    step_size: float | None = 0.05,
    seed: int = 0,
) -> Tensor:
    """Average ``num_probes`` independent single-probe Hutchinson estimates.

    ``ODESolver.compute_likelihood`` draws one Rademacher vector and holds it
    fixed for the whole solve, which is the correct way to keep the augmented
    ODE consistent. Averaging therefore has to happen over repeated solves.
    """
    estimates: list[Tensor] = []
    for probe in range(num_probes):
        torch.manual_seed(seed + probe)
        value, _ = compute_log_likelihood(
            velocity,
            x,
            log_p0=log_p0,
            method=method,
            step_size=step_size,
            exact_divergence=False,
        )
        estimates.append(value)
    return torch.stack(estimates, dim=0).mean(dim=0)


def grid_integral(
    velocity: Callable[..., Tensor],
    *,
    log_p0: Callable[[Tensor], Tensor],
    device: torch.device,
    extent: float,
    resolution: int,
    method: str = "midpoint",
    step_size: float | None = 0.02,
) -> float:
    """Numerically integrate ``exp(log p_hat)`` over a square grid.

    A density model that is correctly normalized integrates to one. This is the
    check that separates "a valid density that is not the data density" from
    "not a density at all".
    """
    axis = torch.linspace(-extent, extent, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    log_density, _ = compute_log_likelihood(
        velocity,
        points,
        log_p0=log_p0,
        method=method,
        step_size=step_size,
        exact_divergence=True,
    )
    cell_area = (2.0 * extent / (resolution - 1)) ** 2
    return float(torch.exp(log_density.double()).sum().item() * cell_area)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def agreement_metrics(estimate: Tensor, reference: Tensor) -> dict[str, float]:
    estimate_np = estimate.detach().cpu().double().numpy()
    reference_np = reference.detach().cpu().double().numpy()
    difference = estimate_np - reference_np
    correlation = float(np.corrcoef(estimate_np, reference_np)[0, 1])
    ranks_estimate = np.argsort(np.argsort(estimate_np))
    ranks_reference = np.argsort(np.argsort(reference_np))
    spearman = float(np.corrcoef(ranks_estimate, ranks_reference)[0, 1])
    return {
        "mean_absolute_error_nats": float(np.mean(np.abs(difference))),
        "rmse_nats": float(np.sqrt(np.mean(difference**2))),
        "bias_nats": float(np.mean(difference)),
        "max_absolute_error_nats": float(np.max(np.abs(difference))),
        "pearson_r": correlation,
        "spearman_r": spearman,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_velocity_field(
    *,
    sample_pair: Callable[[int], tuple[Tensor, Tensor]],
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TimeConditionedMLP:
    """Train a 2-D CNF with conditional flow matching (same objective as Eq. 5)."""
    torch.manual_seed(seed)
    model = TimeConditionedMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    path = CondOTProbPath()

    model.train()
    for step in range(steps):
        x_0, x_1 = sample_pair(batch_size)
        t = torch.rand(batch_size, device=device)
        path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)
        prediction = model(path_sample.x_t, t)
        loss = torch.nn.functional.mse_loss(prediction, path_sample.dx_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    step {step + 1}/{steps} loss={float(loss.item()):.5f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Experiment A: analytic flows
# ---------------------------------------------------------------------------


def experiment_analytic(device: torch.device, num_points: int) -> dict[str, Any]:
    print("[A] Analytic linear flows (closed-form push-forward density)")
    torch.manual_seed(0)
    results: dict[str, Any] = {}

    matrices = {
        "isotropic_scaling": torch.tensor([[0.4, 0.0], [0.0, 0.4]], device=device),
        "anisotropic_shear": torch.tensor([[0.30, 0.50], [-0.20, 0.15]], device=device),
    }
    for profile in ("constant", "cosine"):
        for name, matrix in matrices.items():
            velocity = LinearVelocity(matrix, time_profile=profile)
            covariance = velocity.pushforward_covariance()
            # Sample from the exact push-forward so the evaluation points sit
            # where the density has mass.
            cholesky = torch.linalg.cholesky(covariance.double()).float()
            points = torch.randn(num_points, 2, device=device) @ cholesky.T
            reference = gaussian_log_prob(points, covariance)

            for method, step_size, atol in (
                ("midpoint", 0.05, 1e-5),
                ("midpoint", 0.01, 1e-5),
                ("dopri5", None, 1e-7),
            ):
                estimate, nfe = compute_log_likelihood(
                    velocity,
                    points,
                    log_p0=_standard_gaussian_log_prob,
                    method=method,
                    step_size=step_size,
                    atol=atol,
                    rtol=atol,
                    exact_divergence=True,
                )
                key = f"{profile}/{name}/{method}/step={step_size}"
                results[key] = {
                    **agreement_metrics(estimate, reference),
                    "num_function_evaluations": nfe,
                    "method": method,
                    "step_size": step_size,
                    "tolerance": atol,
                }
                print(
                    f"    {key:52s} MAE={results[key]['mean_absolute_error_nats']:.3e} "
                    f"nats  NFE={nfe}"
                )
    return results


# ---------------------------------------------------------------------------
# Experiment B: generative CNF on a known density
# ---------------------------------------------------------------------------


def experiment_generative(
    device: torch.device,
    *,
    steps: int,
    num_points: int,
    grid_resolution: int,
) -> tuple[dict[str, Any], TimeConditionedMLP, GaussianMixture]:
    print("[B] Generative CNF: N(0,I) -> Gaussian mixture")
    mixture = GaussianMixture(device=device)

    def sample_pair(count: int) -> tuple[Tensor, Tensor]:
        return torch.randn(count, 2, device=device), mixture.sample(count)

    model = train_velocity_field(
        sample_pair=sample_pair,
        device=device,
        steps=steps,
        batch_size=1024,
        learning_rate=1e-3,
        seed=0,
    )

    torch.manual_seed(1)
    points = mixture.sample(num_points)
    reference = mixture.log_prob(points)
    estimate, nfe = compute_log_likelihood(
        model, points, log_p0=_standard_gaussian_log_prob, method="midpoint", step_size=0.02
    )
    normalization = grid_integral(
        model,
        log_p0=_standard_gaussian_log_prob,
        device=device,
        extent=8.0,
        resolution=grid_resolution,
        step_size=0.02,
    )
    metrics = {
        **agreement_metrics(estimate, reference),
        "num_function_evaluations": nfe,
        "grid_integral_of_density": normalization,
        "num_evaluation_points": int(num_points),
    }
    print(
        f"    MAE={metrics['mean_absolute_error_nats']:.4f} nats  "
        f"r={metrics['pearson_r']:.4f}  integral={normalization:.4f}"
    )
    return metrics, model, mixture


# ---------------------------------------------------------------------------
# Experiment C: FlowPure-style conditional transport
# ---------------------------------------------------------------------------


def experiment_flowpure_style(
    device: torch.device,
    *,
    steps: int,
    num_points: int,
    grid_resolution: int,
    perturbation_std: float,
) -> dict[str, Any]:
    print(
        "[C] FlowPure-style CNF: perturbed mixture -> clean mixture "
        f"(sigma={perturbation_std})"
    )
    mixture = GaussianMixture(device=device)
    source = mixture.with_extra_variance(perturbation_std)

    def sample_pair(count: int) -> tuple[Tensor, Tensor]:
        clean = mixture.sample(count)
        perturbed = clean + perturbation_std * torch.randn_like(clean)
        return perturbed, clean

    model = train_velocity_field(
        sample_pair=sample_pair,
        device=device,
        steps=steps,
        batch_size=1024,
        learning_rate=1e-3,
        seed=0,
    )

    torch.manual_seed(1)
    points = mixture.sample(num_points)
    reference = mixture.log_prob(points)

    # (i) What the paper's Eq. 9 currently computes: standard-Gaussian base.
    gaussian_base, _ = compute_log_likelihood(
        model, points, log_p0=_standard_gaussian_log_prob, method="midpoint", step_size=0.02
    )
    # (ii) The same integral with the *true* source density of this transport.
    correct_base, _ = compute_log_likelihood(
        model, points, log_p0=source.log_prob, method="midpoint", step_size=0.02
    )

    metrics = {
        "gaussian_base_vs_data_density": agreement_metrics(gaussian_base, reference),
        "true_source_base_vs_data_density": agreement_metrics(correct_base, reference),
        "gaussian_base_grid_integral": grid_integral(
            model,
            log_p0=_standard_gaussian_log_prob,
            device=device,
            extent=8.0,
            resolution=grid_resolution,
            step_size=0.02,
        ),
        "true_source_base_grid_integral": grid_integral(
            model,
            log_p0=source.log_prob,
            device=device,
            extent=8.0,
            resolution=grid_resolution,
            step_size=0.02,
        ),
        "perturbation_std": perturbation_std,
        "num_evaluation_points": int(num_points),
    }
    # How close is the Gaussian-base score to simply reading off the base
    # density at the query itself? For a near-identity transport the flow map
    # barely moves the point, so the statistic degenerates to -||x||^2/2.
    base_at_query = _standard_gaussian_log_prob(points)
    metrics["gaussian_base_vs_naive_norm_statistic"] = agreement_metrics(
        gaussian_base, base_at_query
    )
    print(
        "    Gaussian base    : MAE={:.3f} nats  r={:+.4f}  integral={:.4f}".format(
            metrics["gaussian_base_vs_data_density"]["mean_absolute_error_nats"],
            metrics["gaussian_base_vs_data_density"]["pearson_r"],
            metrics["gaussian_base_grid_integral"],
        )
    )
    print(
        "    True source base : MAE={:.3f} nats  r={:+.4f}  integral={:.4f}".format(
            metrics["true_source_base_vs_data_density"]["mean_absolute_error_nats"],
            metrics["true_source_base_vs_data_density"]["pearson_r"],
            metrics["true_source_base_grid_integral"],
        )
    )
    return metrics


# ---------------------------------------------------------------------------
# Experiment D: divergence estimator
# ---------------------------------------------------------------------------


def experiment_divergence(
    model: TimeConditionedMLP,
    mixture: GaussianMixture,
    device: torch.device,
    *,
    num_points: int,
    probe_counts: tuple[int, ...],
    num_repeats: int,
) -> dict[str, Any]:
    print("[D] Divergence estimator: exact trace vs Hutchinson probes")
    torch.manual_seed(2)
    points = mixture.sample(num_points)
    exact, _ = compute_log_likelihood(
        model, points, log_p0=_standard_gaussian_log_prob, method="midpoint", step_size=0.02
    )

    results: dict[str, Any] = {}
    for num_probes in probe_counts:
        repeats = []
        for repeat in range(num_repeats):
            estimate = hutchinson_log_likelihood(
                model,
                points,
                log_p0=_standard_gaussian_log_prob,
                num_probes=num_probes,
                step_size=0.02,
                seed=1000 * repeat + 7,
            )
            repeats.append(estimate)
        stacked = torch.stack(repeats, dim=0)
        difference = stacked - exact.unsqueeze(0)
        results[f"probes={num_probes}"] = {
            "bias_nats": float(difference.mean().item()),
            "std_nats": float(stacked.std(dim=0).mean().item()),
            "rmse_vs_exact_nats": float(difference.pow(2).mean().sqrt().item()),
            "relative_std_of_log_density": float(
                (stacked.std(dim=0).mean() / exact.abs().mean()).item()
            ),
            "num_probes": int(num_probes),
        }
        entry = results[f"probes={num_probes}"]
        print(
            f"    k={num_probes:2d}  bias={entry['bias_nats']:+.4f}  "
            f"std={entry['std_nats']:.4f}  rmse={entry['rmse_vs_exact_nats']:.4f} nats"
        )
    return results


# ---------------------------------------------------------------------------
# Experiment E: solver sweep
# ---------------------------------------------------------------------------


def experiment_solver(
    model: TimeConditionedMLP,
    mixture: GaussianMixture,
    device: torch.device,
    *,
    num_points: int,
) -> dict[str, Any]:
    print("[E] ODE solver sweep against a high-accuracy reference")
    torch.manual_seed(3)
    points = mixture.sample(num_points)
    reference, reference_nfe = compute_log_likelihood(
        model,
        points,
        log_p0=_standard_gaussian_log_prob,
        method="dopri5",
        step_size=None,
        atol=1e-7,
        rtol=1e-7,
        exact_divergence=True,
    )

    settings = [
        ("euler", 0.1, None),
        ("euler", 0.05, None),
        ("euler", 0.02, None),
        ("midpoint", 0.1, None),
        ("midpoint", 0.05, None),
        ("midpoint", 0.02, None),
        ("midpoint", 0.01, None),
        ("rk4", 0.05, None),
        ("dopri5", None, 1e-3),
        ("dopri5", None, 1e-5),
    ]
    results: dict[str, Any] = {
        "reference": {
            "method": "dopri5",
            "tolerance": 1e-7,
            "num_function_evaluations": reference_nfe,
        }
    }
    for method, step_size, tolerance in settings:
        estimate, nfe = compute_log_likelihood(
            model,
            points,
            log_p0=_standard_gaussian_log_prob,
            method=method,
            step_size=step_size,
            atol=tolerance if tolerance is not None else 1e-5,
            rtol=tolerance if tolerance is not None else 1e-5,
            exact_divergence=True,
        )
        key = f"{method}/step={step_size}/tol={tolerance}"
        results[key] = {
            **agreement_metrics(estimate, reference),
            "num_function_evaluations": nfe,
            "method": method,
            "step_size": step_size,
            "tolerance": tolerance,
        }
        print(
            f"    {key:32s} MAE={results[key]['mean_absolute_error_nats']:.3e} nats  NFE={nfe}"
        )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "likelihood_validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--grid-resolution", type=int, default=513)
    parser.add_argument("--perturbation-std", type=float, default=0.30)
    parser.add_argument("--divergence-points", type=int, default=1024)
    parser.add_argument("--divergence-repeats", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "configuration": {
            "device": str(device),
            "train_steps": args.train_steps,
            "num_points": args.num_points,
            "grid_resolution": args.grid_resolution,
            "perturbation_std": args.perturbation_std,
            "torch_version": torch.__version__,
        }
    }

    payload["A_analytic_flows"] = experiment_analytic(device, args.num_points)
    generative_metrics, generative_model, mixture = experiment_generative(
        device,
        steps=args.train_steps,
        num_points=args.num_points,
        grid_resolution=args.grid_resolution,
    )
    payload["B_generative_cnf"] = generative_metrics
    payload["C_flowpure_style_cnf"] = experiment_flowpure_style(
        device,
        steps=args.train_steps,
        num_points=args.num_points,
        grid_resolution=args.grid_resolution,
        perturbation_std=args.perturbation_std,
    )
    payload["D_divergence_estimator"] = experiment_divergence(
        generative_model,
        mixture,
        device,
        num_points=args.divergence_points,
        probe_counts=(1, 2, 4, 8, 16),
        num_repeats=args.divergence_repeats,
    )
    payload["E_solver_sweep"] = experiment_solver(
        generative_model, mixture, device, num_points=args.divergence_points
    )

    output_path = args.output_dir / "likelihood_validation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
