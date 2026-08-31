from __future__ import annotations

from flowguard.experiments.spec import AttackKind


def build_attack_runner(kind: AttackKind, **kwargs):
    if kind == AttackKind.TRANSFER_SET:
        from flowguard.attacks.transfer_set import TransferSetAttackRunner

        return TransferSetAttackRunner(**kwargs)
    if kind == AttackKind.JACOBIAN:
        from flowguard.attacks.jacobian import JacobianAttackRunner

        return JacobianAttackRunner(**kwargs)
    if kind == AttackKind.PRADA:
        from flowguard.attacks.prada import PradaAttackRunner

        return PradaAttackRunner(**kwargs)
    if kind == AttackKind.MAZE:
        from flowguard.attacks.maze import MazeAttackRunner

        return MazeAttackRunner(**kwargs)
    if kind == AttackKind.DISGUIDE:
        from flowguard.attacks.disguide import DisguideAttackRunner

        return DisguideAttackRunner(**kwargs)
    if kind == AttackKind.LATENT_MANIFOLD:
        from flowguard.attacks.disguide_bypass import LatentManifoldAttackRunner

        return LatentManifoldAttackRunner(**kwargs)
    if kind == AttackKind.PROCEDURAL_NATURAL:
        from flowguard.attacks.disguide_bypass import ProceduralNaturalAttackRunner

        return ProceduralNaturalAttackRunner(**kwargs)
    if kind == AttackKind.REJECTION_ORACLE:
        from flowguard.attacks.bypass import RejectionOracleAttackRunner

        return RejectionOracleAttackRunner(**kwargs)
    raise ValueError(f"Unsupported attack kind: {kind}")
