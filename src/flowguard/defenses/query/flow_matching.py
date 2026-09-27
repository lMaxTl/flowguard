from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from flow_matching.solver import ODESolver

from flowguard.defenses.query.base import QueryContext, QueryDefense
from flowguard.defenses.query.normalization import modelfamily_mean_std, resolve_modelfamily
from flowguard.defenses.query.flowpure import _infer_noise_type
from flowguard.flow_matching.training import (
    FlowMatchingCheckpoint,
    VelocityFieldWrapper,
    load_flow_matching_checkpoint,
)




def _standard_gaussian_log_prob(x: torch.Tensor) -> torch.Tensor:
    """Compute the log density of a standard Gaussian for a batch of tensors."""
    flat = x.flatten(start_dim=1)
    dimension = flat.shape[1]
    norm_term = dimension * math.log(2.0 * math.pi)
    return -0.5 * (norm_term + flat.pow(2).sum(dim=1))


_CHECKPOINT_CACHE = {}

_WARNED_NON_GAUSS_LIKELIHOOD: set[str] = set()


def _warn_if_not_gaussian_source(
    checkpoint: FlowMatchingCheckpoint,
    fm_checkpoint_path: str,
    *,
    defense_name: str,
) -> None:
    """Warn when likelihood scoring is asked of a non-generative CNF.

    ``compute_likelihood`` integrates to ``t=0`` and evaluates ``log p_0`` under
    a standard Gaussian, which is only the correct base density for a CNF trained
    with ``x_0 ~ N(0, I)``. FlowPure-family checkpoints are trained with
    ``x_0 = perturbed image`` and ``x_1 = clean image``, so their ``p_0`` is the
    perturbed-image distribution and the resulting "log-likelihood" is not a
    density under any reference measure -- which also makes the two-sided
    typicality z-score built on top of it meaningless.

    This is the mirror of the warning ``FlowPureQueryDefense`` raises for the
    opposite mismatch (a Gaussian CNF used for ``||v(0, x)||^2`` detection).
    """
    noise_type = _infer_noise_type(checkpoint)
    if noise_type == "gauss":
        return
    cache_key = f"{fm_checkpoint_path}_{defense_name}"
    if cache_key in _WARNED_NON_GAUSS_LIKELIHOOD:
        return
    _WARNED_NON_GAUSS_LIKELIHOOD.add(cache_key)
    warnings.warn(
        f"{defense_name} is scoring log-likelihoods with a '{noise_type}' "
        f"FlowPure-family checkpoint ({fm_checkpoint_path}). Likelihoods are "
        "computed against a standard-Gaussian p_0, but this checkpoint's source "
        "distribution is perturbed images, so log p(x) -- and any typicality "
        "score standardized from it -- does not measure density under the benign "
        "data distribution. Use a Gaussian-source CNF for likelihood-based "
        "components.",
        RuntimeWarning,
        stacklevel=3,
    )


class FlowMatchingQueryDefense(QueryDefense):
    """Block queries whose FM reverse-flow likelihood looks OOD."""

    name = "flow_matching"

    def __init__(
        self,
        fm_checkpoint_path: str,
        *,
        dataset_name: str = "CIFAR10",
        device: str = "cpu",
        likelihood_threshold: float = 5000.0,
        step_size: float | None = 0.05,
        method: str = "midpoint",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        exact_divergence: bool = False,
        inputs_normalized: bool = True,
        input_modelfamily: str | None = None,
        audit_only: bool = False,
        scoring: str = "typicality",
        benign_likelihood_mean: float | None = None,
        benign_likelihood_std: float | None = None,
        typicality_threshold: float = float("inf"),
        **parameters: Any,
    ) -> None:
        super().__init__(
            fm_checkpoint_path=fm_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
            likelihood_threshold=likelihood_threshold,
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            exact_divergence=exact_divergence,
            inputs_normalized=inputs_normalized,
            input_modelfamily=input_modelfamily,
            audit_only=audit_only,
            scoring=scoring,
            benign_likelihood_mean=benign_likelihood_mean,
            benign_likelihood_std=benign_likelihood_std,
            typicality_threshold=typicality_threshold,
            **parameters,
        )
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.likelihood_threshold = float(likelihood_threshold)
        self.step_size = step_size
        self.method = method
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.exact_divergence = bool(exact_divergence)
        self.inputs_normalized = bool(inputs_normalized)
        self.input_modelfamily = input_modelfamily or self._infer_modelfamily(dataset_name)
        self.audit_only = bool(audit_only)
        self.scoring = str(scoring).lower()
        self.benign_likelihood_mean = (
            float(benign_likelihood_mean) if benign_likelihood_mean is not None else None
        )
        self.benign_likelihood_std = (
            float(benign_likelihood_std) if benign_likelihood_std is not None else None
        )
        self.typicality_threshold = float(typicality_threshold)
        
        cache_key = f"{fm_checkpoint_path}_{self.device}"
        if cache_key not in _CHECKPOINT_CACHE:
            _CHECKPOINT_CACHE[cache_key] = load_flow_matching_checkpoint(
                fm_checkpoint_path,
                device=self.device,
            )
        self.flow_checkpoint: FlowMatchingCheckpoint = _CHECKPOINT_CACHE[cache_key]
        _warn_if_not_gaussian_source(
            self.flow_checkpoint,
            fm_checkpoint_path,
            defense_name=type(self).name,
        )

        self.flow_checkpoint.model.eval()
        self.velocity_model = VelocityFieldWrapper(
            model=self.flow_checkpoint.model,
            class_conditioned=self.flow_checkpoint.dataset_config.class_conditioned,
        )
        self.solver = ODESolver(velocity_model=self.velocity_model)

    def _has_typicality_reference(self) -> bool:
        return self.scoring == "typicality" and self.benign_likelihood_mean is not None

    def _likelihood_anomaly(self, log_likelihood: torch.Tensor) -> torch.Tensor:
        """Map per-query log-likelihood to a non-negative anomaly score.

        Higher values mean *more anomalous*. Likelihood-based OOD detection is
        fundamentally two-sided: a query can be anomalous because the flow model
        finds it unusually *unlikely* (low ``log p(x)``, the textbook OOD case)
        **or** unusually *likely* (high ``log p(x)``). The latter is the
        well-documented typicality failure of deep generative models (Nalisnick
        et al., 2019) and is exactly why attacks whose queries score a *higher*
        likelihood than benign data (e.g. PRADA, latent-manifold bypass) slip
        past a one-sided lower-tail threshold. The typicality statistic
        ``|log p(x) - mu| / sigma`` flags deviation from the benign typical set
        in either direction; the legacy ``-log p(x)`` lower-tail score is kept
        for backward compatibility when no benign reference is available.
        """
        if self._has_typicality_reference():
            std = self.benign_likelihood_std
            if std is None or not math.isfinite(std) or std <= 0.0:
                std = 1.0
            return (log_likelihood - self.benign_likelihood_mean).abs() / std
        return -log_likelihood

    def before_query(
        self,
        batch: torch.Tensor,
        context: QueryContext,
    ) -> tuple[torch.Tensor, QueryContext]:
        scores = self._estimate_log_likelihood(batch)
        anomaly = self._likelihood_anomaly(scores)
        # Raw log-likelihood is preserved for benign calibration / backward compat;
        # ``flow_matching_scores`` is the (two-sided) anomaly used for ranking.
        context.metadata["queries_blocked"] = scores.detach().cpu().tolist()
        context.metadata["flow_matching_scores"] = anomaly.detach().cpu().tolist()
        context.metadata["flow_matching_scoring"] = self.scoring

        if self._has_typicality_reference():
            is_anomalous = anomaly > self.typicality_threshold
            context.metadata["flow_matching_threshold"] = float(self.typicality_threshold)
        else:
            is_anomalous = scores < self.likelihood_threshold
            context.metadata["flow_matching_threshold"] = float(self.likelihood_threshold)

        context.metadata["flow_matching_blocked"] = is_anomalous.detach().cpu().tolist()

        if not getattr(self, "audit_only", False) and bool(torch.any(is_anomalous).item()):
            raise RuntimeError(
                "Query blocked by FlowMatching defense. "
                "Query log-likelihood is atypical for the benign distribution "
                f"(scoring='{self.scoring}')."
            )
        return batch, context

    def _estimate_log_likelihood(self, batch: torch.Tensor) -> torch.Tensor:
        fm_inputs = self._prepare_inputs(batch)
        return self.solver.compute_likelihood(
            x_1=fm_inputs,
            log_p0=_standard_gaussian_log_prob,
            step_size=self.step_size,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            exact_divergence=self.exact_divergence,
            enable_grad=False,
        )[1]

    def _hutchinson_divergence(
        self,
        x: torch.Tensor,
        velocity: torch.Tensor,
        *,
        hutchinson_samples: int,
    ) -> torch.Tensor:
        sample_count = max(1, int(hutchinson_samples))
        divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for _ in range(sample_count):
            noise = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            probe = (velocity * noise).flatten(start_dim=1).sum(dim=1).sum()
            grad = torch.autograd.grad(
                outputs=probe,
                inputs=x,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                continue
            divergence = divergence + (grad * noise).flatten(start_dim=1).sum(dim=1)
        return divergence / float(sample_count)

    def _exact_divergence(self, x: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        flat_velocity = velocity.flatten(start_dim=1)
        feature_dim = flat_velocity.shape[1]
        per_sample: list[torch.Tensor] = []
        for sample_index in range(x.shape[0]):
            sample_divergence = torch.zeros((), device=x.device, dtype=x.dtype)
            for feature_index in range(feature_dim):
                grad = torch.autograd.grad(
                    outputs=flat_velocity[sample_index, feature_index],
                    inputs=x,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is None:
                    continue
                sample_divergence = sample_divergence + grad[sample_index].flatten()[feature_index]
            per_sample.append(sample_divergence)
        return torch.stack(per_sample, dim=0)

    @torch.no_grad()
    def trace_likelihood_path(
        self,
        batch: torch.Tensor,
        *,
        num_steps: int = 20,
        divergence_mode: str = "hutchinson",
        hutchinson_samples: int = 1,
        include_solver_reference: bool = True,
    ) -> dict[str, Any]:
        """Trace reverse-flow states and Jacobian correction terms for visualization."""
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        mode = divergence_mode.lower()
        if mode not in {"hutchinson", "exact", "none"}:
            raise ValueError("divergence_mode must be one of: 'hutchinson', 'exact', 'none'.")

        prepared_inputs = self._prepare_inputs(batch)
        time_grid = torch.linspace(
            1.0,
            0.0,
            num_steps + 1,
            device=self.device,
            dtype=prepared_inputs.dtype,
        )

        x_current = prepared_inputs.detach()
        trajectory: list[torch.Tensor] = [x_current.detach()]
        velocities: list[torch.Tensor] = []
        divergences: list[torch.Tensor] = []
        correction_steps: list[torch.Tensor] = []

        for step_index in range(num_steps):
            t_current = time_grid[step_index]
            t_next = time_grid[step_index + 1]
            dt = t_next - t_current

            needs_grad = mode in {"hutchinson", "exact"}
            x_state = x_current.detach().requires_grad_(needs_grad)
            t_batch = torch.full(
                (x_state.shape[0],),
                float(t_current.item()),
                device=self.device,
                dtype=x_state.dtype,
            )
            with torch.enable_grad():
                velocity = self.velocity_model(x_state, t_batch)
                if mode == "hutchinson":
                    divergence = self._hutchinson_divergence(
                        x_state,
                        velocity,
                        hutchinson_samples=hutchinson_samples,
                    )
                elif mode == "exact":
                    divergence = self._exact_divergence(x_state, velocity)
                else:
                    divergence = torch.zeros(x_state.shape[0], device=self.device, dtype=x_state.dtype)

            correction_step = divergence * dt
            x_current = (x_state.detach() + velocity.detach() * dt).detach()

            velocities.append(velocity.detach())
            divergences.append(divergence.detach())
            correction_steps.append(correction_step.detach())
            trajectory.append(x_current.detach())

        trajectory_tensor = torch.stack(trajectory, dim=1)
        velocity_tensor = torch.stack(velocities, dim=1)
        divergence_tensor = torch.stack(divergences, dim=1)
        correction_steps_tensor = torch.stack(correction_steps, dim=1)
        correction_cumulative = torch.cumsum(correction_steps_tensor, dim=1)
        correction_total = correction_steps_tensor.sum(dim=1)

        x_0_trace = x_current
        log_p0_trace = _standard_gaussian_log_prob(x_0_trace)
        trace_log_likelihood = log_p0_trace + correction_total
        trace_blocked = trace_log_likelihood > self.likelihood_threshold

        solver_x_0 = None
        solver_log_likelihood = None
        solver_blocked = None
        if include_solver_reference:
            solver_x_0, solver_log_likelihood = self.solver.compute_likelihood(
                x_1=prepared_inputs,
                log_p0=_standard_gaussian_log_prob,
                step_size=self.step_size,
                method=self.method,
                atol=self.atol,
                rtol=self.rtol,
                exact_divergence=self.exact_divergence,
                enable_grad=False,
            )
            solver_blocked = solver_log_likelihood > self.likelihood_threshold

        return {
            "input_batch": batch.detach().cpu(),
            "prepared_inputs": prepared_inputs.detach().cpu(),
            "time_grid": time_grid.detach().cpu(),
            "trajectory": trajectory_tensor.detach().cpu(),
            "velocity": velocity_tensor.detach().cpu(),
            "divergence": divergence_tensor.detach().cpu(),
            "jacobian_correction_steps": correction_steps_tensor.detach().cpu(),
            "jacobian_correction_cumulative": correction_cumulative.detach().cpu(),
            "jacobian_correction_total": correction_total.detach().cpu(),
            "trace_x_0": x_0_trace.detach().cpu(),
            "trace_log_p0": log_p0_trace.detach().cpu(),
            "trace_log_likelihood": trace_log_likelihood.detach().cpu(),
            "trace_blocked": trace_blocked.detach().cpu(),
            "threshold": torch.full_like(trace_log_likelihood, self.likelihood_threshold).detach().cpu(),
            "divergence_mode": mode,
            "hutchinson_samples": max(1, int(hutchinson_samples)),
            "solver_x_0": None if solver_x_0 is None else solver_x_0.detach().cpu(),
            "solver_log_likelihood": None
            if solver_log_likelihood is None
            else solver_log_likelihood.detach().cpu(),
            "solver_blocked": None if solver_blocked is None else solver_blocked.detach().cpu(),
        }

    def _prepare_inputs(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.detach().to(self.device, dtype=torch.float32)
        if self.inputs_normalized:
            mean, std = modelfamily_mean_std(self.input_modelfamily)
            mean_tensor = torch.tensor(mean, device=self.device, dtype=x.dtype).view(1, -1, 1, 1)
            std_tensor = torch.tensor(std, device=self.device, dtype=x.dtype).view(1, -1, 1, 1)
            x = x * std_tensor + mean_tensor
        x = torch.clamp(x, min=0.0, max=1.0)

        expected_size = self.flow_checkpoint.dataset_config.image_size
        if x.ndim != 4:
            raise ValueError(f"Expected image batch with shape (B, C, H, W); got {tuple(x.shape)}.")
        if x.shape[-2:] != (expected_size, expected_size):
            x = F.interpolate(
                x,
                size=(expected_size, expected_size),
                mode="bilinear",
                align_corners=False,
            )
        return x * 2.0 - 1.0

    @staticmethod
    def _infer_modelfamily(dataset_name: str) -> str:
        return resolve_modelfamily(dataset_name)
