from __future__ import annotations

from pathlib import Path

from flowguard.attacks.modes import build_attack_mode
from flowguard.experiments.catalog import (
    SUPPORTED_MODES,
    compatible_prediction_defenses,
    default_prediction_defense_parameters,
)
from flowguard.experiments.spec import (
    AttackKind,
    AttackSpec,
    DatasetSpec,
    DefenseSpec,
    DistributedSpec,
    EvaluationSpec,
    ExperimentSpec,
    ModelSpec,
    PreparationWorkflows,
    QueryDefenseSpec,
    TrainingSpec,
)


def build_experiment_spec(
    *,
    name: str,
    dataset: str,
    target_architecture: str,
    substitute_architecture: str | None = None,
    attack_kind: AttackKind = AttackKind.TRANSFER_SET,
    attack_mode: str = "naive",
    query_dataset: str = "ImageNet1k",
    query_budget: int = 50_000,
    query_transfer_set_size: int | None = None,
    verbose: bool = True,
    prediction_defense: str = "none",
    prediction_defense_parameters: dict | None = None,
    query_defense: str = "noop",
    query_defense_parameters: dict | None = None,
    target_checkpoint_dir: str | None = None,
    target_device: str = "cuda",
    substitute_device: str = "cuda",
    distributed: bool = False,
    num_workers: int = 1,
    num_clients: int = 1,
    attack_batch_size: int = 1,
    attack_extra: dict | None = None,
    training_batch_size: int = 32,
    epochs: int = 30,
    query_download: bool = True,
    dataset_download: bool = True,
    metadata: dict | None = None,
) -> ExperimentSpec:
    mode = build_attack_mode(attack_mode)
    defense_parameters = (
        dict(prediction_defense_parameters)
        if prediction_defense_parameters is not None
        else default_prediction_defense_parameters(prediction_defense)
    )
    substitute_architecture = substitute_architecture or target_architecture
    training = TrainingSpec(
        batch_size=training_batch_size,
        epochs=epochs,
        semi_train_weight=1.0 if mode.semi_supervised else 0.0,
        semi_dataset=query_dataset if mode.semi_supervised else None,
        budgets=[query_budget],
    )
    preparation = PreparationWorkflows(
        outlier_exposure=prediction_defense.lower() in {"adaptive_misinformation", "am"},
        proxy_training=False,
        shadow_training=mode.name == "ddae",
        misinformation_training=prediction_defense.lower() in {"adaptive_misinformation", "am"},
    )
    return ExperimentSpec(
        name=name,
        dataset=DatasetSpec(name=dataset, download=dataset_download),
        target_model=ModelSpec(
            dataset=dataset,
            architecture=target_architecture,
            checkpoint_dir=target_checkpoint_dir,
            device=target_device,
        ),
        substitute_model=ModelSpec(
            dataset=dataset,
            architecture=substitute_architecture,
            device=substitute_device,
        ),
        attack=AttackSpec(
            kind=attack_kind,
            mode=mode,
            query_dataset=DatasetSpec(name=query_dataset, download=query_download),
            query_budget=query_budget,
            query_transfer_set_size=query_transfer_set_size,
            batch_size=attack_batch_size,
            extra=dict(attack_extra or {}),
        ),
        verbose=verbose,
        prediction_defense=DefenseSpec(
            name=prediction_defense,
            parameters=defense_parameters,
        ),
        query_defense=QueryDefenseSpec(
            name=query_defense,
            parameters=dict(query_defense_parameters or {}),
        ),
        distributed=DistributedSpec(
            enabled=distributed,
            num_workers=num_workers,
            num_clients=num_clients,
            max_inflight_batches=max(1, num_workers),
        ),
        training=training,
        preparation=preparation,
        evaluation=EvaluationSpec(budget_sweeps=[query_budget]),
        metadata=dict(metadata or {}),
    )


def clone_with_overrides(spec: ExperimentSpec, *, name: str | None = None, output_root: str | None = None) -> ExperimentSpec:
    cloned = ExperimentSpec.from_dict(spec.to_dict())
    if name is not None:
        cloned.name = name
    if output_root is not None:
        cloned.metadata["output_root"] = str(Path(output_root))
    return cloned


def build_experiment_matrix(
    *,
    base_name: str,
    dataset: str,
    target_architecture: str,
    substitute_architecture: str | None = None,
    query_dataset: str,
    query_budget: int,
    query_transfer_set_size: int | None = None,
    verbose: bool = True,
    target_checkpoint_dir: str | None = None,
    attack_batch_size: int = 1,
    attack_extra: dict | None = None,
    training_batch_size: int = 32,
    epochs: int = 30,
    target_device: str = "cuda",
    substitute_device: str = "cuda",
    distributed: bool = False,
    num_workers: int = 1,
    num_clients: int = 1,
    attack_modes: list[str] | None = None,
    defenses: list[str] | None = None,
) -> list[ExperimentSpec]:
    mode_definitions = SUPPORTED_MODES if attack_modes is None else [entry for entry in SUPPORTED_MODES if entry.mode_name in attack_modes]
    specs: list[ExperimentSpec] = []
    for definition in mode_definitions:
        available_defenses = defenses or compatible_prediction_defenses(definition.mode_name)
        for defense_name in available_defenses:
            if defense_name not in compatible_prediction_defenses(definition.mode_name):
                continue
            specs.append(
                build_experiment_spec(
                    name=f"{base_name}-{definition.attack_kind.value}-{definition.mode_name}-{defense_name}",
                    dataset=dataset,
                    target_architecture=target_architecture,
                    substitute_architecture=substitute_architecture,
                    attack_kind=definition.attack_kind,
                    attack_mode=definition.mode_name,
                    query_dataset=query_dataset,
                    query_budget=query_budget,
                    query_transfer_set_size=query_transfer_set_size,
                    verbose=verbose,
                    prediction_defense=defense_name,
                    target_checkpoint_dir=target_checkpoint_dir,
                    target_device=target_device,
                    substitute_device=substitute_device,
                    distributed=distributed,
                    num_workers=num_workers,
                    num_clients=num_clients,
                    attack_batch_size=attack_batch_size,
                    attack_extra=attack_extra,
                    training_batch_size=training_batch_size,
                    epochs=epochs,
                )
            )
    return specs
