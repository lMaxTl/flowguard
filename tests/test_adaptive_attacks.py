"""Tests for the FlowGuard++-adaptive attacker machinery.

Covers the pieces that make an attacker adaptive to the composite rather than
only to the ``t=0`` velocity score: the differentiable trajectory integral, the
differentiable change-of-variables log-likelihood, the fused composite risk, the
distributional and prediction-side penalties, and the distilled score surrogate.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from flowguard.attacks.adaptive import (
    AdaptiveAttackConfig,
    CompositeWeights,
    SurrogateBenignStats,
    SurrogateVelocityRegularizer,
    calibrate_surrogate_stats,
    distribution_matching_penalty,
    label_distribution_penalty,
    two_sided_typicality_penalty,
)
from flowguard.attacks.score_distillation import (
    COMPONENT_NAMES,
    DistilledCompositeScore,
    build_distillation_targets,
    train_score_surrogate,
)
from flowguard.flow_matching.config import (
    FlowMatchingDatasetConfig,
    FlowMatchingTrainingConfig,
)
from flowguard.flow_matching.training import FlowMatchingCheckpoint


class _LinearVelocityField(torch.nn.Module):
    """Velocity field with an analytically known divergence.

    ``v(t, x) = a * x`` has ``div v = a * d`` at every point, independent of
    ``t``, so the exact log-likelihood of the reverse solve can be written down
    and compared against the Hutchinson estimate.
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, t: torch.Tensor, extras: dict) -> torch.Tensor:
        return self.scale * x


def _checkpoint(model: torch.nn.Module) -> FlowMatchingCheckpoint:
    return FlowMatchingCheckpoint(
        model=model,
        training_config=FlowMatchingTrainingConfig(
            dataset="cifar10",
            data_path="data/cifar10",
            output_dir="runs/fm-test",
            device="cpu",
        ),
        dataset_config=FlowMatchingDatasetConfig(
            name="cifar10",
            dataset_type="cifar10",
            image_size=8,
            architecture="cifar10",
            num_classes=None,
            class_conditioned=False,
            default_data_path="data/cifar10",
        ),
        checkpoint_path=Path("flow_checkpoint.pt"),
        epoch=0,
    )


@pytest.fixture()
def regularizer(monkeypatch: pytest.MonkeyPatch) -> SurrogateVelocityRegularizer:
    checkpoint = _checkpoint(_LinearVelocityField(scale=0.25))
    monkeypatch.setattr(
        "flowguard.attacks.adaptive.load_flow_matching_checkpoint",
        lambda path, device: checkpoint,
    )
    return SurrogateVelocityRegularizer("unused.pt", device="cpu")


def test_trajectory_integral_is_differentiable(regularizer) -> None:
    images = torch.rand(3, 3, 8, 8, requires_grad=True)

    integral = regularizer.trajectory_integral_score(images, num_steps=4)

    assert integral.shape == (3,)
    assert torch.all(integral >= 0.0)
    integral.sum().backward()
    assert images.grad is not None
    assert float(images.grad.abs().sum()) > 0.0


def test_log_likelihood_matches_analytic_value(regularizer) -> None:
    """For ``v = a*x`` the reverse Euler solve has a closed form.

    One step of ``x <- x + a*x*dt`` scales ``x`` by ``(1 + a*dt)``, so after
    ``n`` steps of ``dt = -1/n`` the source point is ``x * (1 - a/n)^n`` and the
    accumulated log-determinant is ``-a*d`` (the divergence ``a*d`` integrated
    from 1 to 0). Both are exact, so the estimator must reproduce them.
    """
    scale = 0.25
    steps = 4
    images = torch.rand(2, 3, 8, 8)

    log_likelihood = regularizer.log_likelihood(
        images, num_steps=steps, exact_divergence=True, create_graph=False
    )

    prepared = regularizer.prepare(images)
    dimension = prepared[0].numel()
    source = prepared * (1.0 + scale * (-1.0 / steps)) ** steps
    expected_source_log_prob = -0.5 * (
        dimension * math.log(2.0 * math.pi) + source.flatten(1).pow(2).sum(dim=1)
    )
    expected = expected_source_log_prob - scale * dimension

    assert torch.allclose(log_likelihood, expected, rtol=1e-4, atol=1e-3)


def test_hutchinson_estimate_tracks_exact_divergence(regularizer) -> None:
    images = torch.rand(2, 3, 8, 8)

    exact = regularizer.log_likelihood(
        images, num_steps=2, exact_divergence=True, create_graph=False
    )
    estimated = regularizer.log_likelihood(
        images, num_steps=2, hutchinson_samples=8, create_graph=False
    )

    # The estimator is unbiased; with a constant Jacobian the Rademacher probe
    # is exact up to floating point, so the tolerance can be tight.
    assert torch.allclose(exact, estimated, rtol=1e-3, atol=1e-2)


def test_log_likelihood_gradient_flows_to_input(regularizer) -> None:
    images = torch.rand(2, 3, 8, 8, requires_grad=True)

    regularizer.log_likelihood(images, num_steps=2, hutchinson_samples=1).sum().backward()

    assert images.grad is not None
    assert float(images.grad.abs().sum()) > 0.0


def test_composite_score_combines_all_components(regularizer) -> None:
    images = torch.rand(2, 3, 8, 8, requires_grad=True)
    stats = SurrogateBenignStats(
        velocity_mean=1.0, velocity_std=2.0,
        integral_mean=1.0, integral_std=2.0,
        likelihood_mean=-100.0, likelihood_std=10.0,
    )

    risk, components = regularizer.composite_score(
        images,
        stats=stats,
        weights=CompositeWeights(),
        integral_num_steps=2,
        likelihood_num_steps=2,
    )

    assert set(components) >= {"t0", "integral", "likelihood"}
    # The typicality component is two-sided and therefore non-negative.
    assert torch.all(components["likelihood"] >= 0.0)
    risk.sum().backward()
    assert float(images.grad.abs().sum()) > 0.0


def test_typicality_penalty_is_two_sided() -> None:
    """Deviating above or below the benign mean must cost the attacker."""
    below = two_sided_typicality_penalty(
        torch.tensor([-10.0]), benign_mean=0.0, benign_std=1.0, benign_band=1.0
    )
    above = two_sided_typicality_penalty(
        torch.tensor([10.0]), benign_mean=0.0, benign_std=1.0, benign_band=1.0
    )
    inside = two_sided_typicality_penalty(
        torch.tensor([0.5]), benign_mean=0.0, benign_std=1.0, benign_band=1.0
    )

    assert float(below) > 0.0
    assert float(above) > 0.0
    assert float(below) == pytest.approx(float(above))
    assert float(inside) == pytest.approx(0.0)


def test_distribution_penalty_vanishes_on_matching_samples() -> None:
    reference = torch.linspace(-2.0, 2.0, 64)
    matching = torch.linspace(-2.0, 2.0, 64)
    shifted = matching + 5.0

    assert float(distribution_matching_penalty(matching, reference)) == pytest.approx(
        0.0, abs=1e-5
    )
    assert float(distribution_matching_penalty(shifted, reference)) > 1.0


def test_distribution_penalty_is_differentiable() -> None:
    scores = torch.randn(32, requires_grad=True)
    reference = torch.linspace(-1.0, 1.0, 128)

    distribution_matching_penalty(scores, reference).backward()

    assert scores.grad is not None
    assert float(scores.grad.abs().sum()) > 0.0


def test_label_penalty_rewards_matching_histogram() -> None:
    benign = torch.tensor([0.7, 0.2, 0.1])
    matching = benign.unsqueeze(0).repeat(16, 1)
    uniform = torch.full((16, 3), 1.0 / 3.0)

    matched = label_distribution_penalty(matching, benign_label_histogram=benign)
    swept = label_distribution_penalty(uniform, benign_label_histogram=benign)

    assert float(matched) == pytest.approx(0.0, abs=1e-5)
    assert float(swept) > float(matched)


def test_calibration_produces_usable_bands(regularizer) -> None:
    batches = [torch.rand(8, 3, 8, 8) for _ in range(2)]

    stats, columns = calibrate_surrogate_stats(
        regularizer, batches, integral_num_steps=2, likelihood_num_steps=2
    )

    assert stats.velocity_std > 0.0
    assert stats.integral_std > 0.0
    assert stats.likelihood_std > 0.0
    assert set(columns) == {"t0", "integral", "log_likelihood", "composite"}
    assert all(len(values) == 16 for values in columns.values())


def test_adaptive_config_reports_targeted_components() -> None:
    base = {
        "surrogate_checkpoint_path": "surrogate.pt",
        "velocity_regularizer_weight": 1.0,
    }

    velocity_only = AdaptiveAttackConfig.from_extra(base)
    assert velocity_only.adaptivity_label() == "C1"
    assert not velocity_only.uses_composite_objective()

    full = AdaptiveAttackConfig.from_extra(
        {
            **base,
            "composite_regularizer_weight": 1.0,
            "distribution_regularizer_weight": 1.0,
            "label_regularizer_weight": 1.0,
        }
    )
    assert full.adaptivity_label() == "C1+C2+C3+C4+C5"
    assert full.uses_composite_objective()
    assert full.is_active()


def test_adaptive_config_reads_benign_stats_from_extra() -> None:
    config = AdaptiveAttackConfig.from_extra(
        {
            "surrogate_checkpoint_path": "surrogate.pt",
            "likelihood_regularizer_weight": 2.0,
            "benign_stats": {"likelihood_mean": -3000.0, "likelihood_std": 25.0},
            "composite_weights": {"typicality": 3.0},
        }
    )

    assert config.benign_stats.likelihood_mean == pytest.approx(-3000.0)
    assert config.benign_stats.likelihood_std == pytest.approx(25.0)
    assert config.composite_weights.typicality == pytest.approx(3.0)
    assert config.adaptivity_label() == "C3"
    assert config.has_benign_stats

    # The neutral default must not be mistaken for a real calibration.
    assert not AdaptiveAttackConfig.from_extra({}).has_benign_stats


class _StubRunner:
    """Minimal stand-in exposing the runner's detector-penalty composition.

    Binds the real methods to a bare object so the penalty logic can be tested
    without constructing a query engine, victim model, and dataset.
    """

    def __init__(self, **attributes) -> None:
        from flowguard.attacks.disguide_bypass import DisguideBypassAttackRunner

        self._velocity_regularizer = None
        self._likelihood_regularizer = None
        self._distilled_score = None
        self._benign_score_reference = None
        self._benign_label_histogram = None
        for name, value in attributes.items():
            setattr(self, name, value)
        self._composite_detector_loss = (
            DisguideBypassAttackRunner._composite_detector_loss.__get__(self)
        )
        self._denormalize_to_pixel_space = lambda batch, _name: batch.clamp(0.0, 1.0)


def _bypass_config(**extra):
    """Stand in for ``DisguideBypassConfig``; only ``.adaptive`` is read here."""

    class _Config:
        adaptive = AdaptiveAttackConfig.from_extra(
            {
                "surrogate_checkpoint_path": "surrogate.pt",
                "integral_num_steps": 2,
                "likelihood_num_steps": 2,
                **extra,
            }
        )

    return _Config()


@pytest.mark.parametrize(
    ("extra", "expect_nonzero"),
    [
        ({}, False),  # no adaptive term active
        ({"integral_regularizer_weight": 1.0}, True),  # D3
        ({"likelihood_regularizer_weight": 1.0}, True),  # D4
        (
            {
                "composite_regularizer_weight": 1.0,
                "integral_regularizer_weight": 1.0,
                "likelihood_regularizer_weight": 1.0,
            },
            True,
        ),  # D5
    ],
)
def test_detector_penalty_activates_per_preset(regularizer, extra, expect_nonzero) -> None:
    runner = _StubRunner(_velocity_regularizer=regularizer)
    fake = torch.rand(4, 3, 8, 8, requires_grad=True)
    # Bands of zero guarantee the hinges are outside their benign range, so a
    # zero loss means the term never fired rather than that it was satisfied.
    config = _bypass_config(**extra, benign_stats={"likelihood_std": 1.0})

    loss = runner._composite_detector_loss(fake, config=config, dataset_name="CIFAR10")

    value = float(loss.detach())
    assert value > 0.0 if expect_nonzero else value == 0.0
    if expect_nonzero:
        loss.backward()
        assert float(fake.grad.abs().sum()) > 0.0


def test_stateful_penalty_uses_benign_reference(regularizer) -> None:
    """D6's distributional term must fire only when a reference is present."""
    fake = torch.rand(8, 3, 8, 8, requires_grad=True)
    config = _bypass_config(
        distribution_regularizer_weight=1.0,
        benign_stats={"likelihood_std": 1.0},
    )

    without = _StubRunner(_velocity_regularizer=regularizer)._composite_detector_loss(
        fake, config=config, dataset_name="CIFAR10"
    )
    with_reference = _StubRunner(
        _velocity_regularizer=regularizer,
        _benign_score_reference=torch.full((64,), 50.0),
    )._composite_detector_loss(fake, config=config, dataset_name="CIFAR10")

    assert float(without.detach()) == 0.0
    assert float(with_reference.detach()) > 0.0


def test_label_penalty_reaches_the_generator_loss(regularizer) -> None:
    runner = _StubRunner(
        _velocity_regularizer=regularizer,
        _benign_label_histogram=torch.tensor([0.8, 0.1, 0.1]),
    )
    fake = torch.rand(4, 3, 8, 8)
    probabilities = torch.full((4, 3), 1.0 / 3.0, requires_grad=True)
    config = _bypass_config(
        label_regularizer_weight=1.0, benign_stats={"likelihood_std": 1.0}
    )

    loss = runner._composite_detector_loss(
        fake, config=config, dataset_name="CIFAR10", probabilities=probabilities
    )

    assert float(loss.detach()) > 0.0
    loss.backward()
    assert float(probabilities.grad.abs().sum()) > 0.0


def test_distilled_path_avoids_the_ode_solve(regularizer, tmp_path) -> None:
    """With a distilled surrogate present, no CNF solve should be needed."""
    batches = [torch.rand(8, 3, 8, 8)]
    stats, _ = calibrate_surrogate_stats(
        regularizer, batches, integral_num_steps=2, likelihood_num_steps=2
    )
    images, targets = build_distillation_targets(
        regularizer, batches, stats=stats, integral_num_steps=2, likelihood_num_steps=2
    )
    surrogate, _ = train_score_surrogate(
        images, targets, device="cpu", epochs=1, batch_size=4, width=8
    )

    runner = _StubRunner(_velocity_regularizer=None, _distilled_score=surrogate)
    fake = torch.rand(4, 3, 8, 8, requires_grad=True)
    config = _bypass_config(
        composite_regularizer_weight=1.0,
        integral_regularizer_weight=1.0,
        likelihood_regularizer_weight=1.0,
        benign_stats={"likelihood_std": 1.0},
    )

    loss = runner._composite_detector_loss(fake, config=config, dataset_name="CIFAR10")

    loss.backward()
    assert float(fake.grad.abs().sum()) > 0.0


def test_distilled_surrogate_regresses_component_scores(regularizer, tmp_path) -> None:
    batches = [torch.rand(16, 3, 8, 8) for _ in range(2)]
    stats, _ = calibrate_surrogate_stats(
        regularizer, batches, integral_num_steps=2, likelihood_num_steps=2
    )

    images, targets = build_distillation_targets(
        regularizer,
        batches,
        stats=stats,
        integral_num_steps=2,
        likelihood_num_steps=2,
        hutchinson_samples=2,
    )
    assert images.shape[0] == targets.shape[0] == 32
    assert targets.shape[1] == len(COMPONENT_NAMES)

    surrogate, report = train_score_surrogate(
        images, targets, device="cpu", epochs=2, batch_size=8, width=8
    )
    assert report.num_samples == 32
    assert set(report.component_rmse) == set(COMPONENT_NAMES)

    query = torch.rand(4, 3, 8, 8, requires_grad=True)
    risk = surrogate.risk(query)
    assert risk.shape == (4,)
    risk.sum().backward()
    assert float(query.grad.abs().sum()) > 0.0

    path = tmp_path / "distilled.pt"
    surrogate.save(path, stats=stats)
    reloaded = DistilledCompositeScore.load(path, device="cpu")
    assert torch.allclose(
        reloaded.risk(query.detach()), risk.detach(), rtol=1e-5, atol=1e-5
    )


class _TinyGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 12)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z).view(-1, 3, 2, 2)


def _checkpoint_fixtures():
    from flowguard.attacks.disguide import _ReplayBuffer

    generator = _TinyGenerator()
    ensemble = torch.nn.Linear(3, 4)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_student = torch.optim.SGD(ensemble.parameters(), lr=0.1)
    buffer = _ReplayBuffer(max_size=8, batch_size=2, device=torch.device("cpu"))
    buffer.update(torch.rand(4, 3, 2, 2), torch.rand(4, 4))
    return generator, ensemble, optimizer_generator, optimizer_student, buffer


def test_attack_checkpoint_round_trips_training_state(tmp_path) -> None:
    from flowguard.attacks.disguide_bypass import DisguideBypassAttackRunner as R

    generator, ensemble, opt_g, opt_s, buffer = _checkpoint_fixtures()
    # Take a step so the optimizer carries non-trivial state.
    generator(torch.rand(2, 4)).sum().backward()
    opt_g.step()

    path = tmp_path / "attack_checkpoint.pt"
    R._save_attack_checkpoint(
        R, path,
        generator=generator, ensemble=ensemble,
        optimizer_generator=opt_g, optimizer_student=opt_s,
        generator_scheduler=None, student_scheduler=None,
        replay_buffer=buffer,
        replay_records=[{"query_index": 0}, {"query_index": 1}],
        metrics=[{"epoch": 1.0, "queries": 500.0}],
        epoch=3, global_step=42, queries_used=5_000_000, query_index=5_000_000,
    )
    assert path.exists()
    assert not path.with_suffix(".tmp").exists(), "temp file must be renamed, not left behind"

    fresh_gen, fresh_ens, fresh_og, fresh_os, fresh_buf = _checkpoint_fixtures()
    payload = R._load_attack_checkpoint(
        R, path,
        generator=fresh_gen, ensemble=fresh_ens,
        optimizer_generator=fresh_og, optimizer_student=fresh_os,
        generator_scheduler=None, student_scheduler=None,
        replay_buffer=fresh_buf, device=torch.device("cpu"),
    )

    # Weights restored exactly.
    assert torch.allclose(fresh_gen.linear.weight, generator.linear.weight)
    # Budget accounting continues rather than restarting -- this is the bit
    # that makes a resumed run stop at the right total.
    assert payload["queries_used"] == 5_000_000
    assert payload["query_index"] == 5_000_000
    assert payload["epoch"] == 3
    assert payload["global_step"] == 42
    assert len(payload["replay_records"]) == 2
    assert payload["metrics"][0]["queries"] == 500.0
    assert fresh_buf.size == buffer.size


def test_bounded_record_store_keeps_indices_monotonic() -> None:
    """The cap must not corrupt query indices once the deque saturates."""
    from collections import deque

    from flowguard.attacks.disguide_bypass import DisguideBypassAttackRunner as R

    records: deque = deque(maxlen=4)
    query_index = 0
    for _ in range(5):
        batch = R._records_from_batch(
            R,
            inputs=torch.rand(2, 3, 2, 2),
            labels=torch.rand(2, 4),
            original_labels=torch.rand(2, 4),
            starting_query_index=query_index,
        )
        records.extend(batch)
        query_index += 2

    assert len(records) == 4, "store must stay capped"
    # Had the old len(replay_records) been used, indices would have frozen at 4.
    assert [r["query_index"] for r in records] == [6, 7, 8, 9]


def test_disguide_config_parses_checkpoint_knobs() -> None:
    from flowguard.attacks.disguide import DisguideAttackRunner, DisguideConfig

    assert "artifact_sample_size" in DisguideConfig.__dataclass_fields__
    assert "checkpoint_every_queries" in DisguideConfig.__dataclass_fields__
    assert "resume_from_checkpoint" in DisguideConfig.__dataclass_fields__
    assert DisguideAttackRunner  # the config is consumed by this runner
