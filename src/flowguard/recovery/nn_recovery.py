from __future__ import annotations

from flowguard.recovery.table_recovery import TableRecovery, TableRecoveryConfig


def build_nn_recovery(blackbox, **kwargs) -> TableRecovery:
    config = TableRecoveryConfig(recover_nn=True, **kwargs)
    return TableRecovery(blackbox, config)
