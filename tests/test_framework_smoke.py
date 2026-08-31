from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch.utils.data import TensorDataset

from flowguard.attacks.base import AttackRunContext
from flowguard.attacks.bypass import (
    CemIterationSnapshot,
    CemProceduralNaturalAttackRunner,
    DiffusionLatentGenerator,
    QueryCandidate,
    RejectionOracleAttackRunner,
    procedural_natural_images,
    procedural_parameter_dim,
)
from flowguard.attacks.disguide import DisguideAttackRunner, SubstituteEnsemble
from flowguard.attacks.disguide_bypass import (
    LatentManifoldAttackRunner,
    ProceduralNaturalAttackRunner,
)
from flowguard.attacks.maze import MazeAttackRunner
from flowguard.attacks.modes import build_attack_mode
from flowguard.attacks.registry import build_attack_runner
from flowguard.defenses.query.budgeting import BudgetingQueryDefense
from flowguard.distributed.coordinator import DistributedCoordinator
from flowguard.distributed.worker import build_in_process_worker
from flowguard.evaluation.metrics import accuracy, fidelity, joint_accuracy
from flowguard.experiments.catalog import attack_modes_for_kind
from flowguard.experiments.factory import build_experiment_matrix, build_experiment_spec
from flowguard.experiments.spec import AttackKind
from flowguard.querying.engine import QueryEngine
from flowguard.serving.model_loader import LoadedModel
from flowguard.serving.target_service import TargetService


def _loaded_model() -> LoadedModel:
    model = torch.nn.Linear(4, 3)
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


def test_query_engine_records_batches():
    service = TargetService(_loaded_model())
    engine = QueryEngine(
        service,
        query_defenses=[BudgetingQueryDefense(budget=10)],
    )
    inputs = torch.randn(5, 4)
    outputs = engine.query_tensor(inputs, batch_size=2)
    assert outputs.shape == (5, 3)
    assert engine.history.total_queries == 5
    assert engine.history.total_batches == 3


def test_distributed_coordinator_smoke():
    service = TargetService(_loaded_model())
    workers = [
        build_in_process_worker(QueryEngine(service), worker_id="a"),
        build_in_process_worker(QueryEngine(service), worker_id="b"),
    ]
    coordinator = DistributedCoordinator(workers)
    outputs = coordinator.distribute_tensor(torch.randn(6, 4), batch_size=2)
    assert outputs.shape == (6, 3)
    assert len(coordinator.task_results) == 3


def test_attack_modes_and_metrics():
    mode = build_attack_mode("smoothing")
    assert mode.use_train_transform is True
    predictions = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    targets = torch.tensor([0, 1])
    victim_predictions = torch.tensor([[0.7, 0.3], [0.1, 0.9]])
    assert accuracy(predictions, targets) == 50.0
    assert fidelity(predictions, victim_predictions) == 50.0
    assert joint_accuracy(predictions, targets, victim_predictions) == 50.0


def test_experiment_factory_builds_supported_specs():
    spec = build_experiment_spec(
        name="factory-smoke",
        dataset="CIFAR10",
        target_architecture="resnet18",
        attack_kind=AttackKind.TRANSFER_SET,
        attack_mode="naive",
        query_budget=100,
        verbose=False,
        distributed=True,
        num_workers=2,
    )
    assert spec.distributed.enabled is True
    assert spec.distributed.num_workers == 2
    assert spec.attack.mode.name == "naive"
    assert spec.verbose is False
    assert "jbtr" in attack_modes_for_kind(AttackKind.JACOBIAN)
    prada_spec = build_experiment_spec(
        name="factory-prada",
        dataset="CIFAR10",
        target_architecture="resnet18",
        attack_kind=AttackKind.PRADA,
        attack_mode="prada",
        query_budget=100,
    )
    assert prada_spec.attack.kind == AttackKind.PRADA
    assert prada_spec.attack.mode.name == "prada"
    assert "prada" in attack_modes_for_kind(AttackKind.PRADA)
    maze_spec = build_experiment_spec(
        name="factory-maze",
        dataset="CIFAR10",
        target_architecture="resnet18",
        attack_kind=AttackKind.MAZE,
        attack_mode="maze",
        query_budget=100,
        attack_extra={"latent_dim": 32},
    )
    assert maze_spec.attack.kind == AttackKind.MAZE
    assert maze_spec.attack.mode.name == "maze"
    assert maze_spec.attack.extra["latent_dim"] == 32
    assert "maze" in attack_modes_for_kind(AttackKind.MAZE)
    latent_spec = build_experiment_spec(
        name="factory-latent-manifold",
        dataset="CIFAR10",
        target_architecture="resnet18",
        attack_kind=AttackKind.LATENT_MANIFOLD,
        attack_mode="latent_manifold",
        query_budget=100,
    )
    assert latent_spec.attack.kind == AttackKind.LATENT_MANIFOLD
    assert latent_spec.attack.mode.name == "latent_manifold"
    assert "latent_manifold" in attack_modes_for_kind(AttackKind.LATENT_MANIFOLD)


def test_attack_registry_builds_maze_runner():
    service = TargetService(_loaded_model())
    engine = QueryEngine(service)
    runner = build_attack_runner(AttackKind.MAZE, query_engine=engine, distributed_coordinator=None)
    assert isinstance(runner, MazeAttackRunner)


def test_attack_registry_builds_flowpure_bypass_runners():
    service = TargetService(_loaded_model())
    engine = QueryEngine(service)

    latent = build_attack_runner(AttackKind.LATENT_MANIFOLD, query_engine=engine)
    procedural = build_attack_runner(AttackKind.PROCEDURAL_NATURAL, query_engine=engine)
    rejection = build_attack_runner(AttackKind.REJECTION_ORACLE, query_engine=engine)

    assert isinstance(latent, LatentManifoldAttackRunner)
    assert isinstance(procedural, ProceduralNaturalAttackRunner)
    assert isinstance(rejection, RejectionOracleAttackRunner)


def test_procedural_natural_decoder_returns_normalized_images():
    channels = 3
    dim = procedural_parameter_dim(channels=channels, grid_size=2, fourier_modes=1, num_blobs=1)
    params = torch.zeros((2, dim), dtype=torch.float32)
    clip_min = torch.zeros((channels, 1, 1), dtype=torch.float32).numpy()
    clip_max = torch.ones((channels, 1, 1), dtype=torch.float32).numpy()

    images = procedural_natural_images(
        params,
        channels=channels,
        height=8,
        width=8,
        clip_min=clip_min,
        clip_max=clip_max,
        grid_size=2,
        fourier_modes=1,
        num_blobs=1,
    )

    assert images.shape == (2, channels, 8, 8)
    assert torch.all(images >= 0.0)
    assert torch.all(images <= 1.0)


def test_cem_trace_records_iterations(tmp_path, monkeypatch):
    target_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 8 * 8, 3),
    )
    loaded = LoadedModel(
        model=target_model,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pth.tar",
        params={},
        dataset_name="dummy",
        model_arch="linear",
        num_classes=3,
        modelfamily="cifar",
    )
    runner = CemProceduralNaturalAttackRunner(QueryEngine(TargetService(loaded)))
    synthetic_queryset = TensorDataset(torch.randn(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))

    monkeypatch.setattr(runner, "_build_queryset", lambda context: synthetic_queryset)
    monkeypatch.setattr(
        runner,
        "_normalized_clip_values",
        lambda dataset_name: (
            np.zeros((3, 1, 1), dtype=np.float32),
            np.ones((3, 1, 1), dtype=np.float32),
        ),
    )

    iteration_counter = {"value": 0}

    def _mock_query_candidates(
        self,
        candidates: torch.Tensor,
        *,
        config,
        submitted: int,
        query_budget: int,
    ):
        iteration_counter["value"] += 1
        queried: list[QueryCandidate] = []
        for index in range(candidates.shape[0]):
            blocked = iteration_counter["value"] == 1 and index == 0
            queried.append(
                QueryCandidate(
                    index=index,
                    query=candidates[index],
                    probabilities=torch.tensor([0.2, 0.5, 0.3]),
                    detector_score=float(index),
                    accepted=not blocked,
                )
            )
        submitted_count = candidates.shape[0]
        accepted_count = sum(1 for item in queried if item.accepted)
        blocked_count = submitted_count - accepted_count
        return queried, submitted_count, accepted_count, blocked_count

    monkeypatch.setattr(runner, "_query_candidates", MethodType(_mock_query_candidates, runner))

    spec = build_experiment_spec(
        name="cem-trace-smoke",
        dataset="CIFAR10",
        target_architecture="resnet18",
        target_device="cpu",
        substitute_device="cpu",
        attack_kind=AttackKind.PROCEDURAL_NATURAL,
        attack_mode="procedural_natural",
        query_budget=6,
        attack_batch_size=1,
        attack_extra={
            "population_size": 3,
            "elite_fraction": 0.34,
            "procedural_grid_size": 2,
            "procedural_fourier_modes": 1,
            "procedural_blobs": 1,
            "artifact_sample_size": 4,
            "substitute_train_steps": 0,
        },
    )
    cem_trace: list[CemIterationSnapshot] = []
    result = runner.run(
        AttackRunContext(
            experiment=spec,
            output_dir=str(tmp_path / "procedural"),
            metadata={"cem_trace": cem_trace, "cem_trace_max_images": 2},
        )
    )

    assert len(cem_trace) >= 2
    assert all(isinstance(snapshot, CemIterationSnapshot) for snapshot in cem_trace)
    assert cem_trace[0].population_queries
    assert result.metadata.get("cem_trace_path")
    assert Path(result.metadata["cem_trace_path"]).exists()


def test_d1_disguide_extra_sets_pretrained_generator():
    notebook_utils = Path(__file__).resolve().parents[1] / "notebooks" / "flow_matching_defenses"
    import sys

    sys.path.insert(0, str(notebook_utils))
    from bypass_viz_utils import BypassNotebookConfig, d1_disguide_extra

    config = BypassNotebookConfig(
        project_root=Path("."),
        target_checkpoint_dir=Path("target"),
        flow_checkpoint=Path("flow.pt"),
        diffusion_model_id="google/ddpm-cifar10-32",
        diffusion_scheduler="ddim",
        diffusion_steps=12,
    )
    extra = d1_disguide_extra(config, diffusion_train_steps=2)

    assert extra["generator_hf_model_id"] == "google/ddpm-cifar10-32"
    assert extra["diffusion_steps"] == 12
    assert extra["diffusion_train_steps"] == 2
    assert extra["g_iter"] == 2


def test_steered_diffusion_generator_backprops_through_steering(monkeypatch):
    from flowguard.attacks.disguide_bypass import SteeredDiffusionGenerator

    class _MockUNet(torch.nn.Module):
        # Mirrors the diffusers UNet2DModel surface the generator touches:
        # it reads ``unet.config.in_channels`` to size the latent noise.
        config = SimpleNamespace(in_channels=3, sample_size=32)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, sample, timestep):
            return type("Out", (), {"sample": sample * 0.0})()

    class _MockScheduler:
        def __init__(self):
            self.timesteps = []

        @classmethod
        def from_pretrained(cls, model_id_or_path, subfolder=None):
            return cls()

        def set_timesteps(self, steps, device):
            self.timesteps = list(range(steps))

        def scale_model_input(self, sample, timestep):
            return sample

        def step(self, noise_prediction, timestep, sample):
            return type("StepOut", (), {"prev_sample": sample * 0.5})()

    monkeypatch.setattr(
        "flowguard.attacks.bypass.UNet2DModel.from_pretrained",
        lambda *args, **kwargs: _MockUNet(),
    )
    monkeypatch.setattr("flowguard.attacks.bypass.DDIMScheduler", _MockScheduler)

    clip_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    clip_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    generator = SteeredDiffusionGenerator(
        model_id_or_path="google/ddpm-cifar10-32",
        latent_dim=16,
        image_size=32,
        scheduler_name="ddim",
        num_inference_steps=3,
        clip_min=clip_min,
        clip_max=clip_max,
        device=torch.device("cpu"),
        steering_scale=1.0,
    )
    z = torch.randn(2, 16, requires_grad=True)
    images = generator(z)
    loss = images.mean()
    loss.backward()

    assert images.shape == (2, 3, 32, 32)
    assert generator.steering[0].weight.grad is not None
    assert torch.any(generator.steering[0].weight.grad != 0.0)
    assert all(not parameter.requires_grad for parameter in generator.decoder.parameters())


def test_diffusion_latent_generator_runs_with_mock_diffusers(monkeypatch):
    class _MockUNet(torch.nn.Module):
        # Mirrors the diffusers UNet2DModel surface the generator touches:
        # it reads ``unet.config.in_channels`` to size the latent noise.
        config = SimpleNamespace(in_channels=3, sample_size=32)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, sample, timestep):
            return type("Out", (), {"sample": sample * 0.0})()

    class _MockScheduler:
        def __init__(self):
            self.timesteps = []

        @classmethod
        def from_pretrained(cls, model_id_or_path, subfolder=None):
            return cls()

        def set_timesteps(self, steps, device):
            self.timesteps = list(range(steps))

        def scale_model_input(self, sample, timestep):
            return sample

        def step(self, noise_prediction, timestep, sample):
            return type("StepOut", (), {"prev_sample": sample * 0.5})()

    monkeypatch.setattr(
        "flowguard.attacks.bypass.UNet2DModel.from_pretrained",
        lambda *args, **kwargs: _MockUNet(),
    )
    monkeypatch.setattr("flowguard.attacks.bypass.DDIMScheduler", _MockScheduler)

    generator = DiffusionLatentGenerator(
        model_id_or_path="google/ddpm-cifar10-32",
        latent_dim=32,
        image_size=32,
        scheduler_name="ddim",
        num_inference_steps=3,
        device=torch.device("cpu"),
        use_safetensors=True,
    )
    latent = torch.randn(2, 32)
    images = generator(latent)

    assert images.shape == (2, 3, 32, 32)
    assert torch.all(images >= 0.0)
    assert torch.all(images <= 1.0)


def test_maze_runner_writes_expected_artifacts(tmp_path, monkeypatch):
    target_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(64, 3),
    )
    loaded = LoadedModel(
        model=target_model,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pth.tar",
        params={},
        dataset_name="dummy",
        model_arch="linear",
        num_classes=3,
        modelfamily="cifar",
    )
    runner = MazeAttackRunner(QueryEngine(TargetService(loaded)))
    synthetic_queryset = TensorDataset(torch.randn(8, 1, 8, 8), torch.zeros(8, dtype=torch.long))
    synthetic_testset = TensorDataset(torch.randn(6, 1, 8, 8), torch.zeros(6, dtype=torch.long))

    monkeypatch.setattr(runner, "_build_queryset", lambda context: synthetic_queryset)
    monkeypatch.setattr(runner, "_build_testset", lambda context: synthetic_testset)
    monkeypatch.setattr(
        runner,
        "_normalized_clip_values",
        lambda dataset_name: (
            torch.tensor([[[-1.0]]]).numpy(),
            torch.tensor([[[1.0]]]).numpy(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_create_substitute_model",
        lambda context, num_classes: torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, num_classes),
        ),
    )

    spec = build_experiment_spec(
        name="maze-runner-smoke",
        dataset="CIFAR10",
        target_architecture="resnet18",
        target_device="cpu",
        substitute_device="cpu",
        attack_kind=AttackKind.MAZE,
        attack_mode="maze",
        query_budget=12,
        attack_batch_size=2,
        training_batch_size=2,
        epochs=1,
        attack_extra={
            "latent_dim": 16,
            "iter_gen": 1,
            "iter_clone": 2,
            "iter_exp": 1,
            "ndirs": 1,
            "log_iter": 4,
            "disable_pbar": True,
        },
    )
    result = runner.run(
        AttackRunContext(
            experiment=spec,
            output_dir=str(tmp_path / "maze"),
            metadata={},
        )
    )
    artifact_dir = Path(result.output_dir)
    assert result.attack_name == "maze"
    assert result.mode_name == "maze"
    assert (artifact_dir / "checkpoint.pth.tar").exists()
    assert (artifact_dir / "params.json").exists()
    assert (artifact_dir / "params_transfer.json").exists()
    assert (artifact_dir / "visualization_records.pt").exists()
    assert (artifact_dir / "transferset.pickle").exists()


def test_experiment_matrix_filters_invalid_combinations():
    specs = build_experiment_matrix(
        base_name="matrix",
        dataset="CIFAR10",
        target_architecture="resnet18",
        query_dataset="TinyImageNet200",
        query_budget=100,
        attack_modes=["top1"],
        defenses=["none", "reverse_sigmoid", "adaptive_misinformation"],
        target_device="cpu",
        substitute_device="cpu",
        epochs=1,
        training_batch_size=2,
    )
    defense_names = {spec.prediction_defense.name for spec in specs}
    assert defense_names == {"reverse_sigmoid", "adaptive_misinformation"}


def test_attack_registry_builds_disguide_runner():
    service = TargetService(_loaded_model())
    engine = QueryEngine(service)
    runner = build_attack_runner(AttackKind.DISGUIDE, query_engine=engine, distributed_coordinator=None)
    assert isinstance(runner, DisguideAttackRunner)


def test_experiment_factory_builds_disguide_spec():
    spec = build_experiment_spec(
        name="factory-disguide",
        dataset="CIFAR10",
        target_architecture="resnet18",
        attack_kind=AttackKind.DISGUIDE,
        attack_mode="disguide",
        query_budget=100,
        attack_extra={"ensemble_size": 2, "latent_dim": 32},
    )
    assert spec.attack.kind == AttackKind.DISGUIDE
    assert spec.attack.mode.name == "disguide"
    assert spec.attack.extra["ensemble_size"] == 2
    assert "disguide" in attack_modes_for_kind(AttackKind.DISGUIDE)


def test_disguide_runner_writes_expected_artifacts(tmp_path, monkeypatch):
    target_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(64, 3),
    )
    loaded = LoadedModel(
        model=target_model,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pth.tar",
        params={},
        dataset_name="dummy",
        model_arch="linear",
        num_classes=3,
        modelfamily="cifar",
    )
    runner = DisguideAttackRunner(QueryEngine(TargetService(loaded)))
    synthetic_testset = TensorDataset(torch.randn(6, 1, 8, 8), torch.zeros(6, dtype=torch.long))
    clip_dataset_names: list[str] = []

    monkeypatch.setattr(
        runner,
        "_build_testset",
        lambda context, image_size=8: synthetic_testset,
    )
    monkeypatch.setattr(runner, "_build_victim_testset", lambda context: synthetic_testset)

    def _mock_query_target(context, blackbox, inputs, *, return_origin=False):
        outputs = torch.randn(len(inputs), loaded.num_classes)
        if return_origin:
            return inputs, outputs, outputs.clone()
        return inputs, outputs

    monkeypatch.setattr(runner, "_query_target", _mock_query_target)
    monkeypatch.setattr(
        runner,
        "_normalized_clip_values",
        lambda dataset_name: (
            clip_dataset_names.append(dataset_name) or torch.tensor([[[-1.0]]]).numpy(),
            torch.tensor([[[1.0]]]).numpy(),
        ),
    )

    def _mock_create_ensemble(context, num_classes, ensemble_size):
        models = [
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, num_classes))
            for _ in range(ensemble_size)
        ]
        return SubstituteEnsemble(models)

    monkeypatch.setattr(runner, "_create_substitute_ensemble", _mock_create_ensemble)

    spec = build_experiment_spec(
        name="disguide-runner-smoke",
        dataset="CIFAR10",
        target_architecture="resnet18",
        target_device="cpu",
        substitute_device="cpu",
        attack_kind=AttackKind.DISGUIDE,
        attack_mode="disguide",
        query_budget=12,
        attack_batch_size=2,
        training_batch_size=2,
        epochs=1,
        attack_extra={
            "ensemble_size": 2,
            "latent_dim": 16,
            "g_iter": 1,
            "d_iter": 1,
            "rep_iter": 1,
            "replay_size": 20,
            "epoch_itrs": 3,
            "image_size": 8,
            "disable_pbar": True,
        },
    )
    result = runner.run(
        AttackRunContext(
            experiment=spec,
            output_dir=str(tmp_path / "disguide"),
            metadata={},
        )
    )
    artifact_dir = Path(result.output_dir)
    assert result.attack_name == "disguide"
    assert result.mode_name == "disguide"
    assert clip_dataset_names == ["CIFAR10"]
    assert (artifact_dir / "checkpoint.pth.tar").exists()
    assert (artifact_dir / "checkpoint_ensemble.pth.tar").exists()
    assert (artifact_dir / "params.json").exists()
    assert (artifact_dir / "params_transfer.json").exists()
    assert (artifact_dir / "visualization_records.pt").exists()
    assert (artifact_dir / "transferset.pickle").exists()
    checkpoint = torch.load(artifact_dir / "checkpoint.pth.tar", map_location="cpu", weights_only=False)
    assert all(not key.startswith("subnets.") for key in checkpoint["state_dict"])
