from __future__ import annotations

from flowguard.experiments.spec import AttackMode, LabelType


def build_attack_mode(name: str) -> AttackMode:
    normalized = name.lower()
    if normalized == "naive":
        return AttackMode(name="naive", label_type=LabelType.SOFT)
    if normalized == "top1":
        return AttackMode(name="top1", label_type=LabelType.HARD, hard_label=True)
    if normalized == "s4l":
        return AttackMode(
            name="s4l",
            label_type=LabelType.SOFT,
            semi_supervised=True,
        )
    if normalized == "smoothing":
        return AttackMode(
            name="smoothing",
            label_type=LabelType.SOFT,
            smoothing_queries_per_image=3,
            use_train_transform=True,
        )
    if normalized == "ddae":
        return AttackMode(
            name="ddae",
            label_type=LabelType.SOFT,
            defense_aware=True,
            recovery_parameters={"recover_nn": 1, "recover_nn_epochs": 10},
        )
    if normalized == "ddae+":
        return AttackMode(
            name="ddae+",
            label_type=LabelType.SOFT,
            defense_aware=True,
            recovery_parameters={"recover_nn": 1, "recover_nn_epochs": 10},
        )
    if normalized == "bayes":
        return AttackMode(
            name="bayes",
            label_type=LabelType.SOFT,
            defense_aware=True,
        )
    if normalized in {"jbtr", "jbtr3"}:
        return AttackMode(name="jbtr", label_type=LabelType.SOFT)
    if normalized == "transfer_set":
        return AttackMode(name="transfer_set", label_type=LabelType.SOFT)
    if normalized == "prada":
        return AttackMode(name="prada", label_type=LabelType.SOFT)
    if normalized == "maze":
        return AttackMode(name="maze", label_type=LabelType.SOFT)
    if normalized == "disguide":
        return AttackMode(name="disguide", label_type=LabelType.SOFT)
    if normalized == "latent_manifold":
        return AttackMode(name="latent_manifold", label_type=LabelType.SOFT, defense_aware=True)
    if normalized == "procedural_natural":
        return AttackMode(name="procedural_natural", label_type=LabelType.SOFT, defense_aware=True)
    if normalized == "rejection_oracle":
        return AttackMode(name="rejection_oracle", label_type=LabelType.SOFT, defense_aware=True)
    raise ValueError(f"Unsupported attack mode: {name}")
