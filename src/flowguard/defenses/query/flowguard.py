"""FlowGuard++ defenses (plan Teil 5).

Three modular defenses that address different weaknesses of the paper-
faithful ``FlowPure^PGD`` detector:

- :class:`FlowGuardIntegralDefense` (C1): Replace the single-time-point
  ``||v(t=0, x)||^2`` score with a trajectory integral
  ``sum_{k} ||v(t_k, x_{t_k})||^2 dt`` along a reverse ODE solve. A
  generator that only minimizes the surrogate score at ``t=0``
  (FlowBlind) no longer hits the full integral.

- :class:`FlowGuardUserLevelDefense` (C3): Wrap a per-query FlowPure score
  with a PRADA-style Kolmogorov-Smirnov test over a sliding window of
  queries from the same user. If a user's score distribution differs
  significantly from a stored benign reference, the whole submission is
  flagged. Addresses W6 (Sparse-Probe attack).

- :class:`FlowGuardLabelHistogramDefense` (C4): Monitor the label
  histogram of the victim outputs. Attackers that sweep the label space
  (stealing) produce a significantly flatter histogram than benign users.
  The final score is an MMD between the running-user histogram and a
  benign reference. Addresses W7, W8.

Each class conforms to :class:`~flowguard.defenses.query.base.QueryDefense`
and is opt-in: set ``audit_only=True`` during evaluation to harvest
detection scores without actually rejecting queries.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from flow_matching.solver import ODESolver
from sklearn.linear_model import LogisticRegression

from flowguard.defenses.query.base import QueryContext, QueryDefense
from flowguard.defenses.query.flow_matching import (
    _standard_gaussian_log_prob,
    _warn_if_not_gaussian_source,
)
from flowguard.flow_matching.training import (
    FlowMatchingCheckpoint,
    VelocityFieldWrapper,
    load_flow_matching_checkpoint,
)


_MODELFAMILY_MEAN_STD: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


_CHECKPOINT_CACHE: dict[str, FlowMatchingCheckpoint] = {}


def _load_cached_checkpoint(path: str, device: torch.device) -> FlowMatchingCheckpoint:
    key = f"{path}_{device}"
    if key not in _CHECKPOINT_CACHE:
        _CHECKPOINT_CACHE[key] = load_flow_matching_checkpoint(path, device=device)
    return _CHECKPOINT_CACHE[key]


def _prepare_inputs(
    batch: torch.Tensor,
    *,
    device: torch.device,
    inputs_normalized: bool,
    modelfamily: str,
    expected_size: int,
) -> torch.Tensor:
    x = batch.detach().to(device, dtype=torch.float32)
    if inputs_normalized:
        mean, std = _MODELFAMILY_MEAN_STD[modelfamily]
        mean_t = torch.tensor(mean, device=device, dtype=x.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=device, dtype=x.dtype).view(1, -1, 1, 1)
        x = x * std_t + mean_t
    x = torch.clamp(x, min=0.0, max=1.0)
    if x.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W); got {tuple(x.shape)}.")
    if x.shape[-2:] != (expected_size, expected_size):
        x = F.interpolate(x, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


def _infer_modelfamily(dataset_name: str) -> str:
    normalized = dataset_name.lower()
    if "mnist" in normalized:
        return "mnist"
    if "cifar" in normalized or normalized in {"svhn", "stl10"}:
        return "cifar"
    return "imagenet"


def _as_float_or_none(value: Any) -> float | None:
    """Coerce an optional numeric to a finite float, mapping bad values to ``None``."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_std(std: float | None) -> float:
    """Return a strictly-positive scale for standardization."""
    if std is None or not math.isfinite(std) or std <= 0.0:
        return 1.0
    return float(std)


# ---------------------------------------------------------------------------
# C1 - Trajectory-integral velocity defense
# ---------------------------------------------------------------------------


class FlowGuardIntegralDefense(QueryDefense):
    """Detect anomalous queries via the trajectory-integral velocity norm.

    Integrates the purification ODE *forward* from ``t=0`` to ``t=1`` in
    ``num_steps`` Euler steps and accumulates ``||v(t, x_t)||^2 dt`` along the
    trajectory, approximating the energy functional
    ``E(x) = int_0^1 ||v(t, x_t)||^2 dt``. This is harder to spoof than the
    single-time-point score used by ``FlowPure^PGD``.

    The direction matters. A FlowPure checkpoint is trained with
    ``x_0 = perturbed`` and ``x_1 = clean``, so a query enters the path at
    ``t=0`` and the first term of the sum is exactly FlowPure's
    ``||v(0, x)||^2``: this score strictly contains the baseline's, which is
    what makes it a generalization rather than a different statistic. Solving
    from ``t=1`` down to ``t=0`` instead places the query at the *clean*
    endpoint and runs the flow in reverse, so the score mostly measures where
    the reverse trajectory drifted rather than the query itself -- and can then
    rank attack queries far *below* benign ones even when their ``t=0`` score
    sits above the benign mean (observed as AUROC ~= 0.007 against the D2-family
    generators).
    """

    name = "flowguard_integral"

    def __init__(
        self,
        fm_checkpoint_path: str,
        *,
        dataset_name: str = "CIFAR10",
        device: str = "cpu",
        integral_threshold: float = 5000.0,
        num_steps: int = 8,
        inputs_normalized: bool = True,
        input_modelfamily: str | None = None,
        audit_only: bool = False,
        **parameters: Any,
    ) -> None:
        super().__init__(
            fm_checkpoint_path=fm_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
            integral_threshold=integral_threshold,
            num_steps=num_steps,
            inputs_normalized=inputs_normalized,
            input_modelfamily=input_modelfamily,
            audit_only=audit_only,
            **parameters,
        )
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.integral_threshold = float(integral_threshold)
        self.num_steps = max(1, int(num_steps))
        self.inputs_normalized = bool(inputs_normalized)
        self.input_modelfamily = input_modelfamily or _infer_modelfamily(dataset_name)
        self.audit_only = bool(audit_only)

        self.flow_checkpoint = _load_cached_checkpoint(fm_checkpoint_path, self.device)
        self.flow_checkpoint.model.eval()
        self.velocity_model = VelocityFieldWrapper(
            model=self.flow_checkpoint.model,
            class_conditioned=self.flow_checkpoint.dataset_config.class_conditioned,
        )

    @torch.no_grad()
    def _trajectory_integral(self, batch: torch.Tensor) -> torch.Tensor:
        expected = int(self.flow_checkpoint.dataset_config.image_size)
        fm_inputs = _prepare_inputs(
            batch,
            device=self.device,
            inputs_normalized=self.inputs_normalized,
            modelfamily=self.input_modelfamily,
            expected_size=expected,
        )
        time_grid = torch.linspace(
            0.0, 1.0, self.num_steps + 1,
            device=self.device, dtype=fm_inputs.dtype,
        )
        x_current = fm_inputs
        integral = torch.zeros(fm_inputs.shape[0], device=self.device, dtype=fm_inputs.dtype)
        for step_index in range(self.num_steps):
            t_current = time_grid[step_index]
            dt = float((time_grid[step_index + 1] - t_current).item())
            t_batch = torch.full((x_current.shape[0],), float(t_current.item()),
                                 device=self.device, dtype=x_current.dtype)
            velocity = self.velocity_model(x_current, t_batch)
            velocity_energy = velocity.flatten(start_dim=1).pow(2).sum(dim=1)
            integral = integral + velocity_energy * dt
            x_current = x_current + velocity * dt
        return integral

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        scores = self._trajectory_integral(batch)
        context.metadata["flowguard_integral_scores"] = scores.detach().cpu().tolist()
        context.metadata["flowguard_integral_threshold"] = float(self.integral_threshold)
        is_anomalous = scores > self.integral_threshold
        # Mirror the flowpure_* keys so existing evaluation/extraction helpers work.
        context.metadata["flowpure_scores"] = scores.detach().cpu().tolist()
        context.metadata["flowpure_blocked"] = is_anomalous.detach().cpu().tolist()
        context.metadata["flowpure_threshold"] = float(self.integral_threshold)

        if not self.audit_only and bool(torch.any(is_anomalous).item()):
            max_score = float(scores.max().item())
            raise RuntimeError(
                "Query blocked by FlowGuardIntegralDefense. "
                f"Maximum integral score {max_score:.4f} exceeds threshold "
                f"{self.integral_threshold:.4f}."
            )
        return batch, context

    def calibrate_threshold(
        self,
        benign_batch: torch.Tensor,
        *,
        target_fpr: float = 0.05,
    ) -> float:
        if not 0.0 < target_fpr < 1.0:
            raise ValueError(f"target_fpr must be in (0, 1); got {target_fpr}.")
        scores = self._trajectory_integral(benign_batch)
        quantile = torch.quantile(scores.detach().cpu(), 1.0 - float(target_fpr))
        threshold = float(quantile.item())
        self.integral_threshold = threshold
        return threshold


# ---------------------------------------------------------------------------
# C3 - User-level distribution test (KS-test over FlowPure scores)
# ---------------------------------------------------------------------------


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no p-value). O((n+m) log(n+m))."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    merged = np.concatenate([a_sorted, b_sorted])
    all_values = np.sort(np.unique(merged))
    cdf_a = np.searchsorted(a_sorted, all_values, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, all_values, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


class FlowGuardUserLevelDefense(QueryDefense):
    """Per-user KS-test over a sliding window of FlowPure velocity scores.

    Each ``before_query`` call computes the per-query score (same as
    :class:`~flowguard.defenses.query.flowpure.FlowPureQueryDefense`) and
    pushes it onto a per-user score window. Once the window contains at
    least ``min_window`` samples, the KS-statistic between the user window
    and the benign reference sample is computed; if it exceeds
    ``ks_threshold``, the user is flagged.

    This targets the plan's sparse-probe attack (A4) which hides a small
    fraction of aggressive queries among many clean queries. A per-query
    threshold cannot see the distributional shift; KS can.
    """

    name = "flowguard_userlevel"

    def __init__(
        self,
        fm_checkpoint_path: str,
        *,
        dataset_name: str = "CIFAR10",
        device: str = "cpu",
        benign_reference_scores: list[float] | None = None,
        ks_threshold: float = 0.25,
        min_window: int = 64,
        max_window: int = 2048,
        inputs_normalized: bool = True,
        input_modelfamily: str | None = None,
        audit_only: bool = False,
        **parameters: Any,
    ) -> None:
        super().__init__(
            fm_checkpoint_path=fm_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
            benign_reference_scores=benign_reference_scores,
            ks_threshold=ks_threshold,
            min_window=min_window,
            max_window=max_window,
            inputs_normalized=inputs_normalized,
            input_modelfamily=input_modelfamily,
            audit_only=audit_only,
            **parameters,
        )
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.ks_threshold = float(ks_threshold)
        self.min_window = max(2, int(min_window))
        self.max_window = max(self.min_window, int(max_window))
        self.inputs_normalized = bool(inputs_normalized)
        self.input_modelfamily = input_modelfamily or _infer_modelfamily(dataset_name)
        self.audit_only = bool(audit_only)

        self.flow_checkpoint = _load_cached_checkpoint(fm_checkpoint_path, self.device)
        self.flow_checkpoint.model.eval()
        self.velocity_model = VelocityFieldWrapper(
            model=self.flow_checkpoint.model,
            class_conditioned=self.flow_checkpoint.dataset_config.class_conditioned,
        )
        self.benign_reference = np.asarray(
            benign_reference_scores if benign_reference_scores else [],
            dtype=np.float64,
        )
        self._user_windows: dict[Any, deque[float]] = {}

    def set_benign_reference(self, scores: list[float]) -> None:
        self.benign_reference = np.asarray(scores, dtype=np.float64)

    @torch.no_grad()
    def _per_query_scores(self, batch: torch.Tensor) -> torch.Tensor:
        expected = int(self.flow_checkpoint.dataset_config.image_size)
        fm_inputs = _prepare_inputs(
            batch,
            device=self.device,
            inputs_normalized=self.inputs_normalized,
            modelfamily=self.input_modelfamily,
            expected_size=expected,
        )
        t_zero = torch.zeros((fm_inputs.shape[0],), device=self.device, dtype=fm_inputs.dtype)
        velocity = self.velocity_model(fm_inputs, t_zero)
        return velocity.flatten(start_dim=1).pow(2).sum(dim=1)

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        if self.benign_reference.size == 0:
            raise RuntimeError(
                "FlowGuardUserLevelDefense requires a benign reference. "
                "Call set_benign_reference(scores) before routing traffic."
            )

        per_query_scores = self._per_query_scores(batch)
        scores_list = per_query_scores.detach().cpu().tolist()
        context.metadata["flowpure_scores"] = scores_list

        user_id = context.metadata.get("user_id", context.metadata.get("client_id", "default"))
        window = self._user_windows.setdefault(user_id, deque(maxlen=self.max_window))
        window.extend(scores_list)

        ks_stat = 0.0
        if len(window) >= self.min_window:
            ks_stat = _ks_statistic(np.asarray(window, dtype=np.float64), self.benign_reference)

        flagged = ks_stat > self.ks_threshold
        context.metadata["flowguard_userlevel_ks"] = float(ks_stat)
        context.metadata["flowguard_userlevel_window_size"] = int(len(window))
        context.metadata["flowguard_userlevel_flag"] = bool(flagged)
        # Per-batch blocked flags (broadcasted from user-level decision).
        context.metadata["flowpure_blocked"] = [bool(flagged)] * len(scores_list)

        if not self.audit_only and flagged:
            raise RuntimeError(
                "User blocked by FlowGuardUserLevelDefense. "
                f"KS-statistic {ks_stat:.4f} exceeds threshold {self.ks_threshold:.4f}."
            )
        return batch, context


# ---------------------------------------------------------------------------
# C4 - Label-histogram side channel (MMD vs. benign reference)
# ---------------------------------------------------------------------------


def _histogram_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two probability histograms.

    Both inputs are treated as discrete distributions over the same support
    and renormalized defensively. ``TV = 0.5 * sum |p_i - q_i|`` lies in
    ``[0, 1]``.

    This replaces the previous RBF-MMD that was computed *over the raw
    histogram-bin vectors* (10 scalar "samples" per side). With bin heights in
    ``[0, 1]`` and ``sigma=1`` every kernel evaluated to ~1, so the statistic
    saturated near zero and never crossed any usable threshold (AUROC ~0.002,
    TPR 0 across the board). TV is sensitive to differences between
    near-uniform label distributions, so a label-sweeping attacker histogram
    and a skewed benign histogram now produce a large value while two similar
    distributions produce a small one. NOTE: on a balanced task whose benign
    label histogram is itself ~uniform, both attacker and benign sit near
    uniform and TV stays small by construction -- that is a property of the
    signal, not a bug.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.size == 0 or q.size == 0 or p.shape != q.shape:
        return 0.0
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0.0 or q_sum <= 0.0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    return float(0.5 * np.abs(p - q).sum())


class FlowGuardLabelHistogramDefense(QueryDefense):
    """Post-victim defense: flag users whose victim-output label histogram
    is significantly flatter than a benign reference.

    This defense runs in ``after_query`` (it needs the victim outputs).
    For each query it records the top-1 predicted label; once a user's
    window exceeds ``min_window`` queries it computes the total-variation
    distance between the running histogram and a benign reference
    histogram. A large distance flags the user. (The ``mmd_*`` metadata
    keys are kept for backward compatibility but now carry a TV distance.)

    Addresses W7 (input-only defense ignores outputs) and W8 (label-
    distribution signal).
    """

    name = "flowguard_labelhist"

    def __init__(
        self,
        *,
        num_classes: int,
        benign_label_histogram: list[float] | None = None,
        mmd_threshold: float = 0.05,
        min_window: int = 128,
        max_window: int = 4096,
        audit_only: bool = False,
        rbf_sigma: float = 1.0,
        **parameters: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            benign_label_histogram=benign_label_histogram,
            mmd_threshold=mmd_threshold,
            min_window=min_window,
            max_window=max_window,
            audit_only=audit_only,
            rbf_sigma=rbf_sigma,
            **parameters,
        )
        self.num_classes = int(num_classes)
        self.mmd_threshold = float(mmd_threshold)
        self.min_window = max(2, int(min_window))
        self.max_window = max(self.min_window, int(max_window))
        self.audit_only = bool(audit_only)
        self.rbf_sigma = float(rbf_sigma)
        reference = (
            np.asarray(benign_label_histogram, dtype=np.float64)
            if benign_label_histogram
            else None
        )
        if reference is not None and reference.shape[0] != self.num_classes:
            raise ValueError(
                f"benign_label_histogram length {reference.shape[0]} "
                f"!= num_classes {self.num_classes}."
            )
        if reference is not None:
            total = reference.sum()
            if total <= 0:
                raise ValueError("benign_label_histogram must sum to > 0.")
            reference = reference / total
        self.benign_reference = reference
        self._user_windows: dict[Any, deque[int]] = {}

    def set_benign_reference_from_labels(self, labels: list[int]) -> None:
        histogram = np.bincount(np.asarray(labels, dtype=np.int64), minlength=self.num_classes)
        histogram = histogram.astype(np.float64)
        self.benign_reference = histogram / max(histogram.sum(), 1.0)

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        return batch, context

    def after_query(
        self, batch: torch.Tensor, outputs: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        if self.benign_reference is None:
            raise RuntimeError(
                "FlowGuardLabelHistogramDefense requires a benign reference histogram."
            )
        probabilities = torch.softmax(outputs.detach(), dim=-1) if outputs.ndim > 1 else outputs.detach()
        top1 = probabilities.argmax(dim=-1).cpu().tolist()

        user_id = context.metadata.get("user_id", context.metadata.get("client_id", "default"))
        window = self._user_windows.setdefault(user_id, deque(maxlen=self.max_window))
        window.extend(int(label) for label in top1)

        mmd_score = 0.0
        flagged = False
        if len(window) >= self.min_window:
            histogram = np.bincount(np.asarray(window, dtype=np.int64), minlength=self.num_classes)
            histogram = histogram.astype(np.float64)
            histogram = histogram / max(histogram.sum(), 1.0)
            mmd_score = _histogram_distance(histogram, self.benign_reference)
            flagged = mmd_score > self.mmd_threshold

        context.metadata["flowguard_labelhist_mmd"] = float(mmd_score)
        context.metadata["flowguard_labelhist_window_size"] = int(len(window))
        context.metadata["flowguard_labelhist_flag"] = bool(flagged)

        if not self.audit_only and flagged:
            raise RuntimeError(
                "User blocked by FlowGuardLabelHistogramDefense. "
                f"MMD={mmd_score:.4f} exceeds threshold {self.mmd_threshold:.4f}."
            )
        return outputs, context


# ---------------------------------------------------------------------------
# FlowGuard++ - composite monitor
# ---------------------------------------------------------------------------


class FlowGuardCompositeDefense(QueryDefense):
    """Composite FlowGuard++ monitor combining per-query and stateful signals.

    This class intentionally wraps FlowPure-style scoring rather than replacing
    it. Per-query bypasses can pass the ``flowpure_t0_score`` threshold while
    still being surfaced by user/global distributional signals.
    """

    name = "flowguard++"

    def __init__(
        self,
        fm_checkpoint_path: str,
        *,
        likelihood_fm_checkpoint_path: str | None = None,
        dataset_name: str = "CIFAR10",
        device: str = "cpu",
        flowpure_threshold: float = 5000.0,
        integral_threshold: float = 5000.0,
        likelihood_threshold: float = -float("inf"),
        fused_threshold: float = float("inf"),
        num_steps: int = 8,
        likelihood_step_size: float | None = 0.05,
        likelihood_method: str = "midpoint",
        likelihood_atol: float = 1e-5,
        likelihood_rtol: float = 1e-5,
        likelihood_exact_divergence: bool = False,
        benign_reference_scores: list[float] | None = None,
        benign_label_histogram: list[float] | None = None,
        benign_entropy_histogram: list[float] | None = None,
        benign_likelihood_mean: float | None = None,
        benign_likelihood_std: float | None = None,
        benign_t0_mean: float | None = None,
        benign_t0_std: float | None = None,
        benign_integral_mean: float | None = None,
        benign_integral_std: float | None = None,
        component_weights: dict[str, float] | None = None,
        num_classes: int | None = None,
        score_ks_threshold: float = 0.25,
        label_mmd_threshold: float = 0.05,
        entropy_mmd_threshold: float = 0.05,
        min_window: int = 64,
        max_window: int = 2048,
        decision_mode: str = "hybrid",
        response_policy: str = "reject",
        inputs_normalized: bool = True,
        input_modelfamily: str | None = None,
        audit_only: bool = False,
        rbf_sigma: float = 1.0,
        **parameters: Any,
    ) -> None:
        super().__init__(
            fm_checkpoint_path=fm_checkpoint_path,
            likelihood_fm_checkpoint_path=likelihood_fm_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
            flowpure_threshold=flowpure_threshold,
            integral_threshold=integral_threshold,
            likelihood_threshold=likelihood_threshold,
            fused_threshold=fused_threshold,
            num_steps=num_steps,
            likelihood_step_size=likelihood_step_size,
            likelihood_method=likelihood_method,
            likelihood_atol=likelihood_atol,
            likelihood_rtol=likelihood_rtol,
            likelihood_exact_divergence=likelihood_exact_divergence,
            benign_reference_scores=benign_reference_scores,
            benign_label_histogram=benign_label_histogram,
            benign_entropy_histogram=benign_entropy_histogram,
            benign_likelihood_mean=benign_likelihood_mean,
            benign_likelihood_std=benign_likelihood_std,
            benign_t0_mean=benign_t0_mean,
            benign_t0_std=benign_t0_std,
            benign_integral_mean=benign_integral_mean,
            benign_integral_std=benign_integral_std,
            component_weights=component_weights,
            num_classes=num_classes,
            score_ks_threshold=score_ks_threshold,
            label_mmd_threshold=label_mmd_threshold,
            entropy_mmd_threshold=entropy_mmd_threshold,
            min_window=min_window,
            max_window=max_window,
            decision_mode=decision_mode,
            response_policy=response_policy,
            inputs_normalized=inputs_normalized,
            input_modelfamily=input_modelfamily,
            audit_only=audit_only,
            rbf_sigma=rbf_sigma,
            **parameters,
        )
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.flowpure_threshold = float(flowpure_threshold)
        self.integral_threshold = float(integral_threshold)
        self.likelihood_threshold = float(likelihood_threshold)
        # Threshold on the fused per-query risk score, i.e. the decision rule the
        # paper deploys. Without it the composite can only enforce the individual
        # component thresholds, which the evaluation sets to +-inf -- so the fused
        # rule existed offline, in the metric computation, but was never
        # enforceable at serving time.
        self.fused_threshold = float(fused_threshold)
        self.num_steps = max(1, int(num_steps))
        self.likelihood_step_size = likelihood_step_size
        self.likelihood_method = str(likelihood_method)
        self.likelihood_atol = float(likelihood_atol)
        self.likelihood_rtol = float(likelihood_rtol)
        self.likelihood_exact_divergence = bool(likelihood_exact_divergence)
        self.inputs_normalized = bool(inputs_normalized)
        self.input_modelfamily = input_modelfamily or _infer_modelfamily(dataset_name)
        self.audit_only = bool(audit_only)
        self.score_ks_threshold = float(score_ks_threshold)
        self.label_mmd_threshold = float(label_mmd_threshold)
        self.entropy_mmd_threshold = float(entropy_mmd_threshold)
        self.min_window = max(2, int(min_window))
        self.max_window = max(self.min_window, int(max_window))
        self.decision_mode = str(decision_mode).lower()
        self.response_policy = str(response_policy).lower()
        self.rbf_sigma = float(rbf_sigma)
        self.num_classes = int(num_classes) if num_classes is not None else None

        # Per-component benign reference statistics. When available, each
        # per-query component is converted to a comparable standardized score
        # before being combined; this is what makes a *weighted* combination
        # meaningful (raw t0/integral/likelihood live on wildly different
        # scales, so a plain max() just tracks whichever has the largest
        # magnitude rather than the most informative signal).
        self.benign_t0_mean = _as_float_or_none(benign_t0_mean)
        self.benign_t0_std = _as_float_or_none(benign_t0_std)
        self.benign_integral_mean = _as_float_or_none(benign_integral_mean)
        self.benign_integral_std = _as_float_or_none(benign_integral_std)
        self.benign_likelihood_mean = _as_float_or_none(benign_likelihood_mean)
        self.benign_likelihood_std = _as_float_or_none(benign_likelihood_std)
        self.component_weights = {
            str(name): float(weight)
            for name, weight in (component_weights or {}).items()
        }

        # The composite's components have mutually exclusive checkpoint
        # requirements, so it takes two CNFs rather than forcing one to be wrong:
        #
        #   C1/C2 (t=0 velocity, trajectory integral) need a FlowPure-family
        #     checkpoint (x_0 = adversarial, x_1 = clean). That flow's velocity
        #     is what encodes "how far off the clean manifold is this input";
        #     a Gaussian CNF's velocity does not separate adversarial from
        #     benign at all.
        #   C3 (log-likelihood / typicality) needs a Gaussian-source CNF, since
        #     the likelihood is computed against a standard-Gaussian p_0. Under
        #     a purification flow whose source is perturbed images, log p(x)
        #     does not measure density under the benign data distribution.
        #
        # Passing one checkpoint for both silently invalidates whichever
        # component it does not suit, which is not visible in the output.
        self.flow_checkpoint = _load_cached_checkpoint(fm_checkpoint_path, self.device)
        self.flow_checkpoint.model.eval()
        self.velocity_model = VelocityFieldWrapper(
            model=self.flow_checkpoint.model,
            class_conditioned=self.flow_checkpoint.dataset_config.class_conditioned,
        )

        likelihood_path = str(likelihood_fm_checkpoint_path or fm_checkpoint_path)
        if likelihood_path == str(fm_checkpoint_path):
            self.likelihood_checkpoint = self.flow_checkpoint
            self.likelihood_velocity_model = self.velocity_model
        else:
            self.likelihood_checkpoint = _load_cached_checkpoint(
                likelihood_path, self.device
            )
            self.likelihood_checkpoint.model.eval()
            self.likelihood_velocity_model = VelocityFieldWrapper(
                model=self.likelihood_checkpoint.model,
                class_conditioned=self.likelihood_checkpoint.dataset_config.class_conditioned,
            )
        self.likelihood_fm_checkpoint_path = likelihood_path
        _warn_if_not_gaussian_source(
            self.likelihood_checkpoint,
            likelihood_path,
            defense_name=f"{type(self).name} (C3 likelihood)",
        )
        self.solver = ODESolver(velocity_model=self.likelihood_velocity_model)
        self.benign_scores = np.asarray(
            benign_reference_scores if benign_reference_scores else [],
            dtype=np.float64,
        )
        self.benign_label_histogram = (
            np.asarray(benign_label_histogram, dtype=np.float64)
            if benign_label_histogram
            else None
        )
        if self.benign_label_histogram is not None:
            total = self.benign_label_histogram.sum()
            if total <= 0.0:
                raise ValueError("benign_label_histogram must sum to > 0.")
            self.benign_label_histogram = self.benign_label_histogram / total
            self.num_classes = int(self.benign_label_histogram.shape[0])
        self.benign_entropy_histogram = (
            np.asarray(benign_entropy_histogram, dtype=np.float64)
            if benign_entropy_histogram
            else None
        )
        if self.benign_entropy_histogram is not None:
            total = self.benign_entropy_histogram.sum()
            if total <= 0.0:
                raise ValueError("benign_entropy_histogram must sum to > 0.")
            self.benign_entropy_histogram = self.benign_entropy_histogram / total
        self._score_windows: dict[Any, deque[float]] = {}
        self._label_windows: dict[Any, deque[int]] = {}
        self._entropy_windows: dict[Any, deque[float]] = {}

    def _window_key(self, context: QueryContext) -> Any:
        if self.decision_mode == "stateful_global":
            return "global"
        return context.metadata.get("user_id", context.metadata.get("client_id", "default"))

    def _uses_per_query(self) -> bool:
        return self.decision_mode in {"per_query_only", "hybrid"}

    def _uses_stateful(self) -> bool:
        return self.decision_mode in {"stateful_user", "stateful_global", "hybrid"}

    @torch.no_grad()
    def _flow_scores(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = int(self.flow_checkpoint.dataset_config.image_size)
        fm_inputs = _prepare_inputs(
            batch,
            device=self.device,
            inputs_normalized=self.inputs_normalized,
            modelfamily=self.input_modelfamily,
            expected_size=expected,
        )
        # Forward solve along the purification path (t=0 -> t=1); see
        # FlowGuardIntegralDefense for why the direction is not interchangeable.
        time_grid = torch.linspace(0.0, 1.0, self.num_steps + 1, device=self.device, dtype=fm_inputs.dtype)
        x_current = fm_inputs
        integral = torch.zeros(fm_inputs.shape[0], device=self.device, dtype=fm_inputs.dtype)
        energies: list[torch.Tensor] = []
        for step_index in range(self.num_steps):
            t_current = time_grid[step_index]
            dt = float((time_grid[step_index + 1] - t_current).item())
            t_batch = torch.full((x_current.shape[0],), float(t_current.item()), device=self.device, dtype=x_current.dtype)
            step_velocity = self.velocity_model(x_current, t_batch)
            energies.append(step_velocity.flatten(start_dim=1).pow(2).sum(dim=1))
            integral = integral + energies[-1] * dt
            x_current = x_current + step_velocity * dt
        # energies[0] is the velocity at t=0 on the unmodified query, i.e. exactly
        # C1's score, so the composite shares that forward pass instead of
        # recomputing it -- which also guarantees the two components agree.
        return energies[0], integral

    @torch.no_grad()
    def _estimate_log_likelihood(self, batch: torch.Tensor) -> torch.Tensor:
        expected = int(self.likelihood_checkpoint.dataset_config.image_size)
        fm_inputs = _prepare_inputs(
            batch,
            device=self.device,
            inputs_normalized=self.inputs_normalized,
            modelfamily=self.input_modelfamily,
            expected_size=expected,
        )
        return self.solver.compute_likelihood(
            x_1=fm_inputs,
            log_p0=_standard_gaussian_log_prob,
            step_size=self.likelihood_step_size,
            method=self.likelihood_method,
            atol=self.likelihood_atol,
            rtol=self.likelihood_rtol,
            exact_divergence=self.likelihood_exact_divergence,
            enable_grad=False,
        )[1]

    def _has_component_reference(self) -> bool:
        """True when at least one per-component benign reference is available."""
        return (
            self.benign_t0_mean is not None
            or self.benign_integral_mean is not None
            or self.benign_likelihood_mean is not None
        )

    def _standardized_components(
        self,
        t0_value: float,
        integral_value: float,
        likelihood_value: float,
    ) -> dict[str, float]:
        """Map raw per-query signals to comparable, non-negative anomaly z-scores.

        - ``t0`` / ``integral`` are velocity-energy scores; benign queries score
          low, so a one-sided upper z-score ``(x - mu) / sigma`` is the anomaly.
        - ``likelihood`` is two-sided: an attack can be atypical with *either*
          lower or higher log-likelihood than benign (the generative-model
          typicality failure), so the anomaly is ``|x - mu| / sigma``.
        """
        components: dict[str, float] = {}
        if self.benign_t0_mean is not None:
            components["t0"] = (float(t0_value) - self.benign_t0_mean) / _safe_std(self.benign_t0_std)
        if self.benign_integral_mean is not None:
            components["integral"] = (
                float(integral_value) - self.benign_integral_mean
            ) / _safe_std(self.benign_integral_std)
        if self.benign_likelihood_mean is not None:
            components["likelihood"] = abs(
                float(likelihood_value) - self.benign_likelihood_mean
            ) / _safe_std(self.benign_likelihood_std)
        return components

    def _combine_components(self, components: dict[str, float]) -> float:
        """Weighted sum of standardized component anomalies (weights default to 1)."""
        return sum(
            self.component_weights.get(name, 1.0) * value
            for name, value in components.items()
        )

    def standardized_components_for(
        self,
        t0_list: list[float],
        integral_list: list[float],
        likelihood_list: list[float],
    ) -> dict[str, list[float]]:
        """Return per-component standardized anomaly columns for a set of queries.

        Convenience for assembling a labeled calibration matrix to pass to
        :func:`learn_component_weights`. Requires the benign per-component
        references to be configured.
        """
        columns: dict[str, list[float]] = {}
        for t0_value, integral_value, likelihood_value in zip(
            t0_list,
            integral_list,
            likelihood_list,
            strict=True,
        ):
            for name, value in self._standardized_components(
                t0_value, integral_value, likelihood_value
            ).items():
                columns.setdefault(name, []).append(value)
        return columns

    def _per_query_anomaly_scores(
        self,
        t0_list: list[float],
        integral_list: list[float],
        likelihood_list: list[float],
    ) -> list[float]:
        """Higher values indicate stronger per-query anomaly (for AUROC ranking).

        With benign per-component references the score is a weighted sum of
        standardized component anomalies (two-sided for likelihood). Without any
        reference it falls back to the legacy scale-dominated ``max`` so callers
        that do not calibrate keep their previous behavior.
        """
        use_standardized = self._has_component_reference()
        scores: list[float] = []
        for t0_value, integral_value, likelihood_value in zip(
            t0_list,
            integral_list,
            likelihood_list,
            strict=True,
        ):
            if use_standardized:
                components = self._standardized_components(
                    t0_value, integral_value, likelihood_value
                )
                scores.append(self._combine_components(components))
            else:
                scores.append(
                    max(
                        float(t0_value),
                        float(integral_value),
                        float(-likelihood_value),
                    )
                )
        return scores

    def _apply_before_policy(self, flagged: bool, context: QueryContext) -> None:
        context.metadata["flowguard++_flag"] = bool(flagged)
        context.metadata["flowguard++_response_policy"] = self.response_policy
        if self.audit_only or not flagged:
            return
        if self.response_policy in {"reject", "rate_limit"}:
            raise RuntimeError("Query blocked by FlowGuard++ composite monitor.")

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        t0_scores, integral_scores = self._flow_scores(batch)
        likelihood_scores = self._estimate_log_likelihood(batch)
        t0_list = t0_scores.detach().cpu().tolist()
        integral_list = integral_scores.detach().cpu().tolist()
        likelihood_list = likelihood_scores.detach().cpu().tolist()
        context.metadata["flowpure_t0_score"] = t0_list
        context.metadata["trajectory_integral_score"] = integral_list
        context.metadata["likelihood_score"] = likelihood_list
        context.metadata["feature_trajectory_drift"] = float(np.mean(integral_list) - np.mean(t0_list))
        context.metadata["composite_anomaly_score"] = self._per_query_anomaly_scores(
            t0_list,
            integral_list,
            likelihood_list,
        )
        # Per-component standardized anomalies enable offline / cross-fitted
        # weight learning to compare against the equal-weight default.
        if self._has_component_reference():
            context.metadata["composite_component_scores"] = self.standardized_components_for(
                t0_list,
                integral_list,
                likelihood_list,
            )

        fused_scores = list(context.metadata["composite_anomaly_score"])
        per_query_flags = []
        for t0_value, integral_value, likelihood_value, fused_value in zip(
            t0_list,
            integral_list,
            likelihood_list,
            fused_scores,
            strict=True,
        ):
            query_flags: list[bool] = []
            if self.flowpure_threshold < float("inf"):
                query_flags.append(float(t0_value) > self.flowpure_threshold)
            if self.integral_threshold < float("inf"):
                query_flags.append(float(integral_value) > self.integral_threshold)
            if self.likelihood_threshold > -float("inf"):
                query_flags.append(float(likelihood_value) < self.likelihood_threshold)
            # The fused rule: a query is flagged when its combined standardized
            # anomaly exceeds the calibrated risk threshold, independently of
            # whether any single component crosses its own threshold.
            if self.fused_threshold < float("inf"):
                query_flags.append(float(fused_value) > self.fused_threshold)
            per_query_flags.append(any(query_flags))
        key = self._window_key(context)
        window = self._score_windows.setdefault(key, deque(maxlen=self.max_window))
        if self._uses_stateful():
            window.extend(float(score) for score in t0_list)
        ks_score = 0.0
        if self.benign_scores.size > 0 and len(window) >= self.min_window:
            ks_score = _ks_statistic(np.asarray(window, dtype=np.float64), self.benign_scores)
        stateful_flag = self._uses_stateful() and ks_score > self.score_ks_threshold
        flagged = (self._uses_per_query() and any(per_query_flags)) or stateful_flag

        context.metadata["flowguard++_query_flags"] = [bool(flag) for flag in per_query_flags]
        context.metadata["flowguard++_fused_threshold"] = float(self.fused_threshold)
        context.metadata["score_window_ks"] = float(ks_score)
        context.metadata["score_window_size"] = int(len(window))
        context.metadata["flowpure_scores"] = t0_list
        context.metadata["flowpure_blocked"] = [bool(flagged)] * len(t0_list)
        context.metadata["flowpure_threshold"] = float(self.flowpure_threshold)
        self._apply_before_policy(flagged, context)
        return batch, context

    def after_query(
        self, batch: torch.Tensor, outputs: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        probabilities = torch.softmax(outputs.detach(), dim=-1) if outputs.ndim > 1 else outputs.detach()
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        if self.num_classes is None:
            self.num_classes = int(probabilities.shape[-1])
        key = self._window_key(context)
        label_window = self._label_windows.setdefault(key, deque(maxlen=self.max_window))
        entropy_window = self._entropy_windows.setdefault(key, deque(maxlen=self.max_window))
        top1 = probabilities.argmax(dim=-1).cpu().tolist()
        label_window.extend(int(label) for label in top1)
        entropy = (-(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=1)).cpu().tolist()
        entropy_window.extend(float(value) for value in entropy)

        label_mmd = 0.0
        if self.benign_label_histogram is not None and len(label_window) >= self.min_window:
            histogram = np.bincount(np.asarray(label_window, dtype=np.int64), minlength=self.num_classes)
            histogram = histogram.astype(np.float64) / max(float(histogram.sum()), 1.0)
            label_mmd = _histogram_distance(histogram, self.benign_label_histogram)

        entropy_mmd = 0.0
        if self.benign_entropy_histogram is not None and len(entropy_window) >= self.min_window:
            hist, _ = np.histogram(np.asarray(entropy_window, dtype=np.float64), bins=len(self.benign_entropy_histogram), range=(0.0, math.log(max(self.num_classes, 2))))
            hist = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
            entropy_mmd = _histogram_distance(hist, self.benign_entropy_histogram)

        stateful_flag = self._uses_stateful() and (
            label_mmd > self.label_mmd_threshold
            or entropy_mmd > self.entropy_mmd_threshold
        )
        prior_flag = bool(context.metadata.get("flowguard++_flag", False))
        flagged = prior_flag or stateful_flag
        context.metadata["label_histogram_mmd"] = float(label_mmd)
        context.metadata["entropy_histogram_mmd"] = float(entropy_mmd)
        context.metadata["flowguard++_flag"] = bool(flagged)

        if self.audit_only:
            return outputs, context

        # Per-query suppression: the flagged rows get an uninformative response
        # and the rest are answered normally. A batch-level rule cannot express
        # the deployed policy, because a serving API decides per request; it also
        # makes an end-to-end extraction measurement impossible, since one
        # flagged query in a batch would either kill the run or blank the whole
        # batch and misalign the attacker's inputs from its labels.
        if self.response_policy == "per_query_suppress":
            query_flags = context.metadata.get("flowguard++_query_flags") or []
            flags = [bool(flag) for flag in query_flags[: outputs.shape[0]]]
            flags += [False] * (outputs.shape[0] - len(flags))
            mask = torch.tensor(flags, dtype=torch.bool, device=outputs.device)
            if bool(stateful_flag):
                # A window-level detection condemns the whole batch: the
                # stateful components decide about a client, not a query.
                mask = torch.ones_like(mask)
            released = outputs.clone()
            if mask.any():
                uniform = torch.full_like(outputs[0], 1.0 / max(outputs.shape[-1], 1))
                released[mask] = uniform
            accepted = int((~mask).sum().item())
            context.metadata["flowguard++_released"] = [bool(value) for value in (~mask).tolist()]
            context.metadata["flowguard++_accepted"] = accepted
            context.metadata["flowguard++_suppressed"] = int(outputs.shape[0] - accepted)
            return released, context

        if not flagged:
            return outputs, context
        if self.response_policy in {"reject", "rate_limit"}:
            raise RuntimeError("Query blocked by FlowGuard++ composite monitor.")
        if self.response_policy == "label_coarsening":
            coarsened = torch.zeros_like(outputs)
            coarsened.scatter_(1, probabilities.argmax(dim=1, keepdim=True).to(outputs.device), 1.0)
            return coarsened, context
        if self.response_policy == "silent_low_information_response":
            return torch.full_like(outputs, 1.0 / max(outputs.shape[-1], 1)), context
        return outputs, context


# ---------------------------------------------------------------------------
# Helpers to build benign references used by C3/C4 callers
# ---------------------------------------------------------------------------


def build_benign_label_histogram(
    labels: list[int],
    *,
    num_classes: int,
) -> list[float]:
    histogram = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    total = float(histogram.sum())
    if total <= 0.0:
        raise ValueError("Cannot build histogram from empty labels list.")
    return (histogram.astype(np.float64) / total).tolist()


def uniform_label_histogram(num_classes: int) -> list[float]:
    value = 1.0 / max(1, int(num_classes))
    return [value] * int(num_classes)


def _component_matrix(components: dict[str, list[float]], names: list[str]) -> np.ndarray:
    columns: list[np.ndarray] = []
    length: int | None = None
    for name in names:
        values = np.asarray(components[name], dtype=np.float64)
        if length is None:
            length = int(values.shape[0])
        elif int(values.shape[0]) != length:
            raise ValueError(
                f"Component '{name}' has {values.shape[0]} samples; expected {length}."
            )
        columns.append(values)
    if not length:
        raise ValueError("Empty component samples.")
    return np.column_stack(columns)


def learn_component_weights(
    benign_components: dict[str, list[float]],
    attack_components: dict[str, list[float]],
    *,
    component_order: list[str] | None = None,
    nonnegative: bool = True,
    max_iter: int = 1000,
) -> dict[str, float]:
    """Learn per-component weights for :class:`FlowGuardCompositeDefense`.

    The composite combines per-query component anomalies as a weighted sum. To
    *prioritize* the more discriminative signals you need supervision: a labeled
    set of benign and attack (or proxy-OOD) queries. This helper fits a logistic
    regression on the per-component **standardized** anomaly scores (e.g. the
    output of :meth:`FlowGuardCompositeDefense.standardized_components_for`) and
    returns the coefficients as weights ready to pass back as
    ``component_weights``.

    Args:
        benign_components: Mapping ``component_name -> list of standardized
            anomaly scores`` for benign queries.
        attack_components: Same mapping for attack queries.
        component_order: Optional explicit component ordering. Defaults to the
            sorted intersection of keys present in both groups.
        nonnegative: Clip negative coefficients to ``0`` so weights remain
            interpretable as "how much this component contributes to anomaly".
        max_iter: Maximum solver iterations for the logistic regression.

    Returns:
        Mapping ``component_name -> weight``.

    Note:
        Learned weights are only as general as the attack samples used to fit
        them. In model-stealing defense you usually lack attack queries at
        deploy time, so fitting on the very attacks you evaluate risks
        overfitting. Prefer learning weights on a *held-out* mix of benign data
        and diverse proxy-OOD/attack queries, and keep the equal-weight
        standardized combination as a robust unsupervised default.
    """
    names = component_order or sorted(set(benign_components) & set(attack_components))
    if not names:
        raise ValueError("No shared components between benign and attack samples.")
    benign_matrix = _component_matrix(benign_components, names)
    attack_matrix = _component_matrix(attack_components, names)
    features = np.vstack([benign_matrix, attack_matrix])
    labels = np.concatenate(
        [
            np.zeros(benign_matrix.shape[0], dtype=np.int64),
            np.ones(attack_matrix.shape[0], dtype=np.int64),
        ]
    )
    model = LogisticRegression(max_iter=int(max_iter))
    model.fit(features, labels)
    coefficients = np.asarray(model.coef_, dtype=np.float64).ravel()
    if nonnegative:
        coefficients = np.clip(coefficients, a_min=0.0, a_max=None)
    return {name: float(weight) for name, weight in zip(names, coefficients, strict=True)}
