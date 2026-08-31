from pathlib import Path

import pytest
import torch

from flowguard.defenses.query.base import QueryContext
from flowguard.defenses.query.fdinet import FDINetQueryDefense
from flowguard.defenses.query.flow_matching import FlowMatchingQueryDefense
from flowguard.defenses.query.flowguard import FlowGuardCompositeDefense
from flowguard.defenses.query.flowpure import FlowPureQueryDefense
from flowguard.defenses.query.prada import PradaQueryDefense
from flowguard.experiments.factory import build_experiment_spec
from flowguard.flow_matching.config import FlowMatchingDatasetConfig, FlowMatchingTrainingConfig
from flowguard.flow_matching.training import FlowMatchingCheckpoint
from flowguard.orchestration.runner import _build_query_defenses
from flowguard.querying.engine import QueryEngine
from flowguard.serving.model_loader import LoadedModel
from flowguard.serving.target_service import TargetService


def _constant_loaded_model() -> LoadedModel:
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
    return LoadedModel(
        model=model,
        device=torch.device("cpu"),
        checkpoint_dir=Path("."),
        checkpoint_path=Path("checkpoint.pth.tar"),
        params={},
        dataset_name="dummy",
        model_arch="linear",
        num_classes=3,
        modelfamily="cifar",
    )


def _dummy_fm_checkpoint() -> FlowMatchingCheckpoint:
    flow_model = torch.nn.Conv2d(3, 3, kernel_size=1)
    return FlowMatchingCheckpoint(
        model=flow_model,
        training_config=FlowMatchingTrainingConfig(
            dataset="cifar10",
            data_path="data/cifar10",
            output_dir="runs/fm-test",
            device="cpu",
        ),
        dataset_config=FlowMatchingDatasetConfig(
            name="cifar10",
            dataset_type="cifar10",
            image_size=32,
            architecture="cifar10",
            num_classes=None,
            class_conditioned=False,
            default_data_path="data/cifar10",
        ),
        checkpoint_path=Path("flow_checkpoint.pt"),
        epoch=0,
    )


def test_prada_query_defense_blocks_suspicious_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.prada.stats.shapiro",
        lambda values: (0.1, 0.0),
    )
    engine = QueryEngine(
        TargetService(_constant_loaded_model()),
        query_defenses=[
            PradaQueryDefense(
                check_interval=1,
                min_class_distances=1,
                min_distribution_samples=1,
            )
        ],
    )

    first_batch = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    second_batch = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    third_batch = torch.tensor([[20.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    first_outputs = engine.query_tensor(first_batch, batch_size=1)
    assert first_outputs.shape == (1, 3)
    assert engine.history.total_queries == 1

    with pytest.raises(RuntimeError, match="PRADA detected"):
        engine.query_tensor(second_batch, batch_size=1)
    assert engine.history.total_queries == 2

    with pytest.raises(RuntimeError, match="Query blocked by PRADA defense"):
        engine.query_tensor(third_batch, batch_size=1)


def test_prada_query_defense_audit_mode_records_blocked_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.prada.stats.shapiro",
        lambda values: (0.1, 0.0),
    )
    engine = QueryEngine(
        TargetService(_constant_loaded_model()),
        query_defenses=[
            PradaQueryDefense(
                check_interval=1,
                min_class_distances=1,
                min_distribution_samples=1,
                audit_only=True,
            )
        ],
    )

    first_batch = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    second_batch = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    third_batch = torch.tensor([[20.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    engine.query_tensor(first_batch, batch_size=1)
    assert "prada_blocked_indices" not in engine.history.records[0].metadata

    engine.query_tensor(second_batch, batch_size=1)
    assert engine.history.records[1].metadata["prada_blocked_indices"] == [0]

    engine.query_tensor(third_batch, batch_size=1)
    assert engine.history.records[2].metadata["prada_blocked_indices"] == [0]


def test_flow_matching_query_defense_blocks_low_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flow_matching.load_flow_matching_checkpoint",
        lambda checkpoint_path, device="cpu": _dummy_fm_checkpoint(),
    )
    monkeypatch.setattr(
        FlowMatchingQueryDefense,
        "_estimate_log_likelihood",
        lambda self, batch: torch.tensor([-3.0], dtype=torch.float32),
    )

    engine = QueryEngine(
        TargetService(_constant_loaded_model()),
        query_defenses=[
            FlowMatchingQueryDefense(
                fm_checkpoint_path="runs/fm-test/checkpoint_latest.pt",
                dataset_name="CIFAR10",
                device="cpu",
                likelihood_threshold=-2.0,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="FlowMatching defense"):
        engine.query_tensor(torch.zeros((1, 4), dtype=torch.float32), batch_size=1)


def test_flow_matching_trace_likelihood_path_returns_stepwise_debug_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flow_matching.load_flow_matching_checkpoint",
        lambda checkpoint_path, device="cpu": _dummy_fm_checkpoint(),
    )

    defense = FlowMatchingQueryDefense(
        fm_checkpoint_path="runs/fm-test/checkpoint_latest.pt",
        dataset_name="CIFAR10",
        device="cpu",
        likelihood_threshold=-10.0,
        inputs_normalized=False,
        audit_only=True,
    )

    class _ZeroVelocity:
        def __call__(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
            return x * 0.0

    defense.velocity_model = _ZeroVelocity()

    batch = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    trace = defense.trace_likelihood_path(
        batch,
        num_steps=4,
        divergence_mode="hutchinson",
        hutchinson_samples=1,
        include_solver_reference=False,
    )

    assert trace["trajectory"].shape == (1, 5, 3, 32, 32)
    assert trace["velocity"].shape == (1, 4, 3, 32, 32)
    assert trace["divergence"].shape == (1, 4)
    assert torch.allclose(trace["trace_x_0"], trace["prepared_inputs"], atol=1e-6)
    assert torch.allclose(
        trace["jacobian_correction_total"],
        torch.zeros_like(trace["jacobian_correction_total"]),
        atol=1e-6,
    )
    assert torch.allclose(trace["trace_log_likelihood"], trace["trace_log_p0"], atol=1e-5)
    assert trace["solver_log_likelihood"] is None
    assert trace["solver_x_0"] is None
    assert trace["solver_blocked"] is None


def test_build_query_defenses_supports_prada() -> None:
    spec = build_experiment_spec(
        name="query-defense-prada",
        dataset="CIFAR10",
        target_architecture="resnet18",
        query_defense="prada",
        query_defense_parameters={"shapiro_threshold": 0.9},
        query_budget=100,
        verbose=False,
    )

    defenses = _build_query_defenses(spec)

    assert len(defenses) == 1
    assert isinstance(defenses[0], PradaQueryDefense)
    assert defenses[0].parameters["shapiro_threshold"] == 0.9


def test_build_query_defenses_supports_flow_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flow_matching.load_flow_matching_checkpoint",
        lambda checkpoint_path, device="cpu": _dummy_fm_checkpoint(),
    )
    spec = build_experiment_spec(
        name="query-defense-flow-matching",
        dataset="CIFAR10",
        target_architecture="resnet18",
        query_defense="flow_matching",
        query_defense_parameters={
            "fm_checkpoint_path": "runs/fm-test/checkpoint_latest.pt",
            "likelihood_threshold": -10.0,
        },
        query_budget=100,
        verbose=False,
    )

    defenses = _build_query_defenses(spec)

    assert len(defenses) == 1
    assert isinstance(defenses[0], FlowMatchingQueryDefense)
    assert defenses[0].dataset_name == "CIFAR10"
    assert defenses[0].parameters["likelihood_threshold"] == -10.0


def test_build_query_defenses_supports_fdinet() -> None:
    spec = build_experiment_spec(
        name="query-defense-fdinet",
        dataset="CIFAR10",
        target_architecture="resnet18",
        query_defense="fdinet",
        query_defense_parameters={
            "bootstrap": False,
            "audit_only": True,
            "detection_threshold": 0.7,
        },
        query_budget=100,
        verbose=False,
    )

    defenses = _build_query_defenses(spec, loaded_model=_constant_loaded_model())

    assert len(defenses) == 1
    assert isinstance(defenses[0], FDINetQueryDefense)
    assert defenses[0].parameters["bootstrap"] is False
    assert defenses[0].parameters["detection_threshold"] == 0.7


def test_flowpure_query_defense_blocks_high_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flowpure.load_flow_matching_checkpoint",
        lambda checkpoint_path, device="cpu": _dummy_fm_checkpoint(),
    )
    monkeypatch.setattr(
        FlowPureQueryDefense,
        "_estimate_velocity_scores",
        lambda self, batch: torch.tensor([6.0], dtype=torch.float32),
    )

    engine = QueryEngine(
        TargetService(_constant_loaded_model()),
        query_defenses=[
            FlowPureQueryDefense(
                fm_checkpoint_path="runs/fm-test/checkpoint_latest.pt",
                dataset_name="CIFAR10",
                device="cpu",
                velocity_threshold=5.0,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="FlowPure defense"):
        engine.query_tensor(torch.zeros((1, 4), dtype=torch.float32), batch_size=1)


def test_flowpure_query_defense_audit_mode_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flowpure.load_flow_matching_checkpoint",
        lambda checkpoint_path, device="cpu": _dummy_fm_checkpoint(),
    )
    monkeypatch.setattr(
        FlowPureQueryDefense,
        "_estimate_velocity_scores",
        lambda self, batch: torch.tensor([3.0, 7.0], dtype=torch.float32),
    )

    engine = QueryEngine(
        TargetService(_constant_loaded_model()),
        query_defenses=[
            FlowPureQueryDefense(
                fm_checkpoint_path="runs/fm-test/checkpoint_latest.pt",
                dataset_name="CIFAR10",
                device="cpu",
                velocity_threshold=5.0,
                audit_only=True,
            )
        ],
    )

    engine.query_batch(torch.zeros((2, 4), dtype=torch.float32))
    metadata = engine.history.records[0].metadata
    assert metadata["flowpure_scores"] == pytest.approx([3.0, 7.0])
    assert metadata["flowpure_blocked"] == [False, True]
    assert metadata["flowpure_threshold"] == pytest.approx(5.0)


def test_flowguard_composite_audit_records_composite_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flowguard._load_cached_checkpoint",
        lambda path, device: _dummy_fm_checkpoint(),
    )
    defense = FlowGuardCompositeDefense(
        fm_checkpoint_path="runs/fm-test/checkpoint_latest.pt",
        dataset_name="CIFAR10",
        device="cpu",
        flowpure_threshold=1.0,
        integral_threshold=1.0,
        audit_only=True,
        inputs_normalized=False,
        num_steps=2,
    )

    class _OnesVelocity:
        def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x)

    context = QueryContext()
    defense.velocity_model = _OnesVelocity()
    monkeypatch.setattr(
        FlowGuardCompositeDefense,
        "_estimate_log_likelihood",
        lambda self, batch: torch.full((batch.shape[0],), -2.0, dtype=torch.float32),
    )
    _, updated_context = defense.before_query(
        torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        context,
    )

    assert updated_context.metadata["flowpure_t0_score"][0] > 1.0
    assert updated_context.metadata["trajectory_integral_score"][0] > 1.0
    assert updated_context.metadata["likelihood_score"] == pytest.approx([-2.0])
    assert updated_context.metadata["composite_anomaly_score"][0] > 1.0
    assert updated_context.metadata["flowpure_blocked"] == [True]
    assert updated_context.metadata["flowguard++_flag"] is True


def test_build_query_defenses_supports_flowguard_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flowguard._load_cached_checkpoint",
        lambda path, device: _dummy_fm_checkpoint(),
    )
    spec = build_experiment_spec(
        name="query-defense-flowguard-composite",
        dataset="CIFAR10",
        target_architecture="resnet18",
        query_defense="flowguard++",
        query_defense_parameters={
            "fm_checkpoint_path": "runs/fm-test/checkpoint_latest.pt",
            "flowpure_threshold": 123.0,
        },
        query_budget=100,
        verbose=False,
    )

    defenses = _build_query_defenses(spec)

    assert len(defenses) == 1
    assert isinstance(defenses[0], FlowGuardCompositeDefense)
    assert defenses[0].parameters["flowpure_threshold"] == 123.0


def test_build_query_defenses_supports_flowpure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flowguard.defenses.query.flowpure.load_flow_matching_checkpoint",
        lambda checkpoint_path, device="cpu": _dummy_fm_checkpoint(),
    )
    spec = build_experiment_spec(
        name="query-defense-flowpure",
        dataset="CIFAR10",
        target_architecture="resnet18",
        query_defense="flowpure",
        query_defense_parameters={
            "fm_checkpoint_path": "runs/fm-test/checkpoint_latest.pt",
            "velocity_threshold": 123.0,
        },
        query_budget=100,
        verbose=False,
    )

    defenses = _build_query_defenses(spec)

    assert len(defenses) == 1
    assert isinstance(defenses[0], FlowPureQueryDefense)
    assert defenses[0].dataset_name == "CIFAR10"
    assert defenses[0].parameters["velocity_threshold"] == 123.0


def test_fdinet_audit_mode_records_scores_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defense = FDINetQueryDefense(bootstrap=False, audit_only=True)
    defense._is_initialized = True
    defense._score_threshold = 0.5

    monkeypatch.setattr(
        defense,
        "_compute_fdi_vectors",
        lambda batch, predicted_classes, use_cached_features: torch.tensor(
            [[0.1], [2.0]], dtype=torch.float32
        ),
    )
    monkeypatch.setattr(
        defense,
        "_score_vectors",
        lambda fdi_vectors: torch.tensor([0.2, 0.9], dtype=torch.float32),
    )

    context = QueryContext()
    outputs, updated_context = defense.after_query(
        batch=torch.zeros((2, 4), dtype=torch.float32),
        outputs=torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32),
        context=context,
    )

    assert outputs.shape == (2, 2)
    assert updated_context.metadata["fdinet_initialized"] is True
    assert updated_context.metadata["fdinet_scores"] == pytest.approx([0.2, 0.9])
    assert updated_context.metadata["fdinet_flags"] == [False, True]
    assert updated_context.metadata["fdinet_pred_classes"] == [0, 1]
    assert updated_context.metadata["fdinet_threshold"] == 0.5


def test_fdinet_blocking_mode_raises_when_threshold_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defense = FDINetQueryDefense(bootstrap=False, audit_only=False)
    defense._is_initialized = True
    defense._score_threshold = 0.5

    monkeypatch.setattr(
        defense,
        "_compute_fdi_vectors",
        lambda batch, predicted_classes, use_cached_features: torch.tensor(
            [[0.3]], dtype=torch.float32
        ),
    )
    monkeypatch.setattr(
        defense,
        "_score_vectors",
        lambda fdi_vectors: torch.tensor([0.75], dtype=torch.float32),
    )

    with pytest.raises(RuntimeError, match="FDINet detected"):
        defense.after_query(
            batch=torch.zeros((1, 4), dtype=torch.float32),
            outputs=torch.tensor([[0.7, 0.3]], dtype=torch.float32),
            context=QueryContext(),
        )


def _composite_with_fused_threshold(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fused_threshold: float,
    likelihood: float,
) -> FlowGuardCompositeDefense:
    monkeypatch.setattr(
        "flowguard.defenses.query.flowguard._load_cached_checkpoint",
        lambda path, device: _dummy_fm_checkpoint(),
    )
    defense = FlowGuardCompositeDefense(
        fm_checkpoint_path="runs/fm-test/checkpoint_latest.pt",
        dataset_name="CIFAR10",
        device="cpu",
        fused_threshold=fused_threshold,
        benign_t0_mean=0.0,
        benign_t0_std=1.0,
        benign_integral_mean=0.0,
        benign_integral_std=1.0,
        benign_likelihood_mean=0.0,
        benign_likelihood_std=1.0,
        audit_only=False,
        response_policy="per_query_suppress",
        inputs_normalized=False,
        num_steps=2,
        num_classes=3,
    )

    class _PerRowVelocity:
        """Row 0 stays near zero, row 1 is far off the benign manifold."""

        def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            scale = torch.tensor([0.0, 1.0], dtype=x.dtype).view(-1, 1, 1, 1)
            return torch.ones_like(x) * scale

    defense.velocity_model = _PerRowVelocity()
    monkeypatch.setattr(
        FlowGuardCompositeDefense,
        "_estimate_log_likelihood",
        lambda self, batch: torch.full((batch.shape[0],), likelihood, dtype=torch.float32),
    )
    return defense


def test_flowguard_composite_fused_threshold_flags_per_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defense = _composite_with_fused_threshold(
        monkeypatch, fused_threshold=10.0, likelihood=0.0
    )
    _, context = defense.before_query(
        torch.zeros((2, 3, 32, 32), dtype=torch.float32), QueryContext()
    )

    # The fused rule decides per query: only the high-velocity row crosses it.
    assert context.metadata["flowguard++_query_flags"] == [False, True]
    assert context.metadata["flowguard++_fused_threshold"] == pytest.approx(10.0)


def test_flowguard_composite_per_query_suppression_releases_only_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defense = _composite_with_fused_threshold(
        monkeypatch, fused_threshold=10.0, likelihood=0.0
    )
    _, context = defense.before_query(
        torch.zeros((2, 3, 32, 32), dtype=torch.float32), QueryContext()
    )
    outputs = torch.tensor([[5.0, 1.0, 0.0], [5.0, 1.0, 0.0]])

    released, context = defense.after_query(
        torch.zeros((2, 3, 32, 32), dtype=torch.float32), outputs, context
    )

    # Accepted row keeps the victim's answer; the flagged row is replaced by a
    # uniform response, which is what "no prediction released" means to an
    # attacker that trains on whatever comes back.
    assert torch.allclose(released[0], outputs[0])
    assert torch.allclose(released[1], torch.full((3,), 1.0 / 3.0))
    assert context.metadata["flowguard++_accepted"] == 1
    assert context.metadata["flowguard++_suppressed"] == 1
    assert context.metadata["flowguard++_released"] == [True, False]


def test_flowguard_composite_without_fused_threshold_releases_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default threshold is +inf, so the fused rule must never fire on its own;
    # otherwise every audit run in the suite would silently start blocking.
    defense = _composite_with_fused_threshold(
        monkeypatch, fused_threshold=float("inf"), likelihood=0.0
    )
    _, context = defense.before_query(
        torch.zeros((2, 3, 32, 32), dtype=torch.float32), QueryContext()
    )
    outputs = torch.tensor([[5.0, 1.0, 0.0], [5.0, 1.0, 0.0]])
    released, context = defense.after_query(
        torch.zeros((2, 3, 32, 32), dtype=torch.float32), outputs, context
    )

    assert context.metadata["flowguard++_query_flags"] == [False, False]
    assert torch.allclose(released, outputs)
    assert context.metadata["flowguard++_accepted"] == 2
