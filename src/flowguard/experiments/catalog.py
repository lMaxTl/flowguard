from __future__ import annotations

from dataclasses import dataclass

from flowguard.attacks.modes import build_attack_mode
from flowguard.experiments.spec import AttackKind


@dataclass(frozen=True, slots=True)
class AttackModeDefinition:
    attack_kind: AttackKind
    mode_name: str
    description: str


SUPPORTED_ATTACKS: tuple[AttackKind, ...] = (
    AttackKind.TRANSFER_SET,
    AttackKind.JACOBIAN,
    AttackKind.PRADA,
    AttackKind.MAZE,
    AttackKind.DISGUIDE,
    AttackKind.LATENT_MANIFOLD,
    AttackKind.PROCEDURAL_NATURAL,
    AttackKind.REJECTION_ORACLE,
)

SUPPORTED_MODES: tuple[AttackModeDefinition, ...] = (
    AttackModeDefinition(AttackKind.TRANSFER_SET, "naive", "Soft-label transfer-set attack."),
    AttackModeDefinition(AttackKind.TRANSFER_SET, "top1", "Hard-label transfer-set attack."),
    AttackModeDefinition(AttackKind.TRANSFER_SET, "s4l", "Semi-supervised transfer-set attack."),
    AttackModeDefinition(AttackKind.TRANSFER_SET, "smoothing", "Transfer-set attack with repeated smoothed queries."),
    AttackModeDefinition(AttackKind.TRANSFER_SET, "ddae", "Defense-aware transfer-set attack with shadow recovery."),
    AttackModeDefinition(AttackKind.TRANSFER_SET, "ddae+", "Defense-aware transfer-set attack without shadow models."),
    AttackModeDefinition(AttackKind.TRANSFER_SET, "bayes", "Defense-aware transfer-set recovery attack."),
    AttackModeDefinition(AttackKind.JACOBIAN, "jbtr", "Jacobian-based augmentation runner."),
    AttackModeDefinition(AttackKind.PRADA, "prada", "PRADA iterative adversarial extraction attack."),
    AttackModeDefinition(AttackKind.MAZE, "maze", "MAZE data-free extraction attack."),
    AttackModeDefinition(AttackKind.DISGUIDE, "disguide", "DisGUIDE ensemble disagreement extraction attack."),
    AttackModeDefinition(AttackKind.LATENT_MANIFOLD, "latent_manifold", "Latent-manifold CEM FlowPure bypass attack."),
    AttackModeDefinition(AttackKind.PROCEDURAL_NATURAL, "procedural_natural", "Strict procedural natural-statistics CEM attack."),
    AttackModeDefinition(AttackKind.REJECTION_ORACLE, "rejection_oracle", "Detector-oracle rejection attack using accepted queries only."),
)

SUPPORTED_PREDICTION_DEFENSES: tuple[str, ...] = (
    "none",
    "reverse_sigmoid",
    "mad",
    "adaptive_misinformation",
    "modelguard",
    "modelguard_s",
)


def attack_modes_for_kind(kind: AttackKind) -> list[str]:
    return [definition.mode_name for definition in SUPPORTED_MODES if definition.attack_kind == kind]


def all_mode_objects() -> dict[str, object]:
    return {definition.mode_name: build_attack_mode(definition.mode_name) for definition in SUPPORTED_MODES}


def default_prediction_defense_parameters(name: str) -> dict[str, float | int | str]:
    normalized = name.lower()
    if normalized == "reverse_sigmoid":
        return {"beta": 0.008, "gamma": 0.2}
    if normalized == "rand_noise":
        return {"epsilon_z": 0.1, "dist_z": "l2"}
    if normalized == "mad":
        return {"epsilon": 1.0, "ydist": "l1", "oracle": "argmax", "objmax": 1, "batch_constraint": 0}
    if normalized == "adaptive_misinformation":
        return {"defense_level": 0.25}
    if normalized in {"modelguard", "mld"}:
        return {"epsilon": 1.0, "ydist": "l1", "batch_constraint": 0}
    if normalized in {"modelguard_s", "quantization"}:
        return {"epsilon": 1.0, "ydist": "l1", "ordered_quantization": True, "frozen": True}
    return {}


def compatible_prediction_defenses(mode_name: str) -> list[str]:
    normalized = mode_name.lower()
    if normalized == "top1":
        return ["reverse_sigmoid", "adaptive_misinformation"]
    return list(SUPPORTED_PREDICTION_DEFENSES)
