from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class LabelType(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    LOGITS = "logits"


class AttackKind(str, Enum):
    TRANSFER_SET = "transfer_set"
    JACOBIAN = "jacobian"
    PRADA = "prada"
    MAZE = "maze"
    DISGUIDE = "disguide"
    LATENT_MANIFOLD = "latent_manifold"
    PROCEDURAL_NATURAL = "procedural_natural"
    REJECTION_ORACLE = "rejection_oracle"


@dataclass(slots=True)
class DatasetSpec:
    name: str
    split: str = "train"
    use_train_transform: bool = False
    root: str | None = None
    download: bool = True


@dataclass(slots=True)
class ArtifactRef:
    name: str
    path: str
    description: str = ""


@dataclass(slots=True)
class ModelSpec:
    dataset: str
    architecture: str
    pretrained: str | None = None
    checkpoint_dir: str | None = None
    num_classes: int | None = None
    device: str = "cuda"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DefenseSpec:
    name: str = "none"
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)


@dataclass(slots=True)
class QueryDefenseSpec:
    name: str = "noop"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DistributedSpec:
    enabled: bool = False
    num_workers: int = 1
    num_clients: int = 1
    max_inflight_batches: int = 1
    worker_endpoints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrainingSpec:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.01
    lr_step: int = 10
    lr_gamma: float = 0.5
    momentum: float = 0.5
    num_workers: int = 4
    semi_train_weight: float = 0.0
    semi_dataset: str | None = None
    budgets: list[int] = field(default_factory=list)


@dataclass(slots=True)
class PreparationWorkflows:
    outlier_exposure: bool = False
    proxy_training: bool = False
    shadow_training: bool = False
    misinformation_training: bool = False
    artifacts: list[ArtifactRef] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationSpec:
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "fidelity", "joint_accuracy"]
    )
    budget_sweeps: list[int] = field(default_factory=list)


@dataclass(slots=True)
class AttackMode:
    name: str
    label_type: LabelType = LabelType.SOFT
    hard_label: bool = False
    defense_aware: bool = False
    semi_supervised: bool = False
    smoothing_queries_per_image: int = 1
    use_train_transform: bool = False
    recovery_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttackSpec:
    kind: AttackKind
    mode: AttackMode
    query_dataset: DatasetSpec
    query_budget: int
    query_transfer_set_size: int | None = None
    batch_size: int = 1
    seed_size: int = 1000
    epsilon: float = 0.1
    steps: int = 8
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    dataset: DatasetSpec
    target_model: ModelSpec
    substitute_model: ModelSpec
    attack: AttackSpec
    verbose: bool = True
    prediction_defense: DefenseSpec = field(default_factory=DefenseSpec)
    query_defense: QueryDefenseSpec = field(default_factory=QueryDefenseSpec)
    distributed: DistributedSpec = field(default_factory=DistributedSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    preparation: PreparationWorkflows = field(default_factory=PreparationWorkflows)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.attack.query_budget <= 0:
            raise ValueError("query_budget must be positive")
        if self.attack.query_transfer_set_size is not None and self.attack.query_transfer_set_size <= 0:
            raise ValueError("query_transfer_set_size must be positive when provided")
        if self.attack.batch_size <= 0:
            raise ValueError("attack.batch_size must be positive")
        if self.distributed.enabled and self.distributed.num_workers <= 0:
            raise ValueError("distributed.num_workers must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentSpec:
        dataset = DatasetSpec(**payload["dataset"])
        target_model = ModelSpec(**payload["target_model"])
        substitute_model = ModelSpec(**payload["substitute_model"])
        prediction_defense = DefenseSpec(**payload.get("prediction_defense", {}))
        query_defense = QueryDefenseSpec(**payload.get("query_defense", {}))
        distributed = DistributedSpec(**payload.get("distributed", {}))
        training = TrainingSpec(**payload.get("training", {}))
        preparation = PreparationWorkflows(**payload.get("preparation", {}))
        evaluation = EvaluationSpec(**payload.get("evaluation", {}))
        attack_raw = payload["attack"]
        mode_raw = attack_raw["mode"]
        mode = AttackMode(
            name=mode_raw["name"],
            label_type=LabelType(mode_raw.get("label_type", LabelType.SOFT.value)),
            hard_label=mode_raw.get("hard_label", False),
            defense_aware=mode_raw.get("defense_aware", False),
            semi_supervised=mode_raw.get("semi_supervised", False),
            smoothing_queries_per_image=mode_raw.get("smoothing_queries_per_image", 1),
            use_train_transform=mode_raw.get("use_train_transform", False),
            recovery_parameters=dict(mode_raw.get("recovery_parameters", {})),
        )
        attack = AttackSpec(
            kind=AttackKind(attack_raw["kind"]),
            mode=mode,
            query_dataset=DatasetSpec(**attack_raw["query_dataset"]),
            query_budget=attack_raw["query_budget"],
            query_transfer_set_size=attack_raw.get("query_transfer_set_size"),
            batch_size=attack_raw.get("batch_size", 1),
            seed_size=attack_raw.get("seed_size", 1000),
            epsilon=attack_raw.get("epsilon", 0.1),
            steps=attack_raw.get("steps", 8),
            extra=dict(attack_raw.get("extra", {})),
        )
        artifacts = [ArtifactRef(**item) for item in payload.get("artifacts", [])]
        spec = cls(
            name=payload["name"],
            dataset=dataset,
            target_model=target_model,
            substitute_model=substitute_model,
            attack=attack,
            verbose=payload.get("verbose", True),
            prediction_defense=prediction_defense,
            query_defense=query_defense,
            distributed=distributed,
            training=training,
            preparation=preparation,
            evaluation=evaluation,
            artifacts=artifacts,
            metadata=dict(payload.get("metadata", {})),
        )
        spec.validate()
        return spec
