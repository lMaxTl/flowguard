from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from defenses import datasets as legacy_datasets
from defenses.adversary.jacobian import JacobianAdversary
from defenses.utils.utils import suppress_stdout
from flowguard.attacks.base import AttackRunContext, AttackRunner, AttackRunResult
from flowguard.attacks.transfer_set import DistributedLegacyBlackboxBridge
from flowguard.distributed.coordinator import DistributedCoordinator
from flowguard.querying.engine import LegacyBlackboxBridge, QueryEngine
from flowguard.recovery.table_recovery import TableRecovery, TableRecoveryConfig


class JacobianAttackRunner(AttackRunner):
    name = "jacobian"

    def __init__(
        self,
        query_engine: QueryEngine,
        distributed_coordinator: DistributedCoordinator | None = None,
    ) -> None:
        self.query_engine = query_engine
        self.distributed_coordinator = distributed_coordinator

    def _build_blackbox(self, context: AttackRunContext):
        query_metadata = {
            "fdinet_gt": 1,
            "fdinet_source": self.name,
            "fdinet_mode": context.experiment.attack.mode.name,
        }
        if context.experiment.distributed.enabled and self.distributed_coordinator is not None:
            return DistributedLegacyBlackboxBridge(
                self.query_engine,
                self.distributed_coordinator,
                batch_size=context.experiment.attack.batch_size,
                default_metadata=query_metadata,
            )
        return LegacyBlackboxBridge(self.query_engine, default_metadata=query_metadata)

    def _build_recovery(self, blackbox, context: AttackRunContext):
        mode = context.experiment.attack.mode
        if not mode.defense_aware:
            return None
        config = TableRecoveryConfig(
            batch_size=context.experiment.attack.batch_size,
            **mode.recovery_parameters,
        )
        recovery = TableRecovery(blackbox, config)
        return recovery.legacy

    def run(self, context: AttackRunContext) -> AttackRunResult:
        spec = context.experiment
        output_dir = Path(context.output_dir or Path("runs") / spec.name / self.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        if spec.verbose:
            print(
                f"[FlowGuard++][jacobian] Starting attack mode='{spec.attack.mode.name}' "
                f"query_budget={spec.attack.query_budget} seed_size={spec.attack.seed_size}"
            )

        query_dataset_name = spec.attack.query_dataset.name
        query_family = legacy_datasets.dataset_to_modelfamily[query_dataset_name]
        query_transform = legacy_datasets.modelfamily_to_transforms[query_family]["test"]
        queryset = legacy_datasets.__dict__[query_dataset_name](
            train=True,
            transform=query_transform,
            download=spec.attack.query_dataset.download,
        )
        subset_indices = np.random.choice(range(len(queryset)), size=spec.attack.seed_size, replace=False)
        seedset = Subset(queryset, subset_indices)

        target_dataset_name = spec.dataset.name
        target_family = legacy_datasets.dataset_to_modelfamily[target_dataset_name]
        target_transform = legacy_datasets.modelfamily_to_transforms[target_family]["test"]
        testset = legacy_datasets.__dict__[target_dataset_name](
            train=False,
            transform=target_transform,
            download=spec.dataset.download,
        )

        blackbox = self._build_blackbox(context)
        recovery = self._build_recovery(blackbox, context)
        strategy = "jbtr3" if spec.attack.mode.name == "jbtr" else spec.attack.mode.name
        adversary = JacobianAdversary(
            blackbox=blackbox,
            budget=spec.attack.query_budget,
            model_adv_name=spec.substitute_model.architecture,
            model_adv_pretrained=spec.substitute_model.pretrained,
            modelfamily=target_family,
            seedset=seedset,
            testset=testset,
            device=blackbox.device,
            out_dir=str(output_dir),
            query_batch_size=spec.attack.batch_size,
            aug_strategy=strategy,
            epsilon=spec.attack.epsilon,
            T=spec.attack.steps,
            useprobs=not spec.attack.mode.hard_label,
            label_recover=recovery,
        )
        stream_context = nullcontext() if spec.verbose else suppress_stdout()
        with stream_context:
            transferset, _ = adversary.get_transferset(
                epochs=spec.training.epochs,
                batch_size=spec.training.batch_size,
                lr=spec.training.lr,
                lr_step=spec.training.lr_step,
                lr_gamma=spec.training.lr_gamma,
                momentum=spec.training.momentum,
                num_workers=spec.training.num_workers,
            )
        transferset_path = output_dir / "transferset.pickle"
        with transferset_path.open("wb") as handle:
            torch.save(transferset, handle)
        with (output_dir / "params_transfer.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "attack_kind": spec.attack.kind.value,
                    "mode": spec.attack.mode.name,
                    "queryset": query_dataset_name,
                    "budget": spec.attack.query_budget,
                    "batch_size": spec.attack.batch_size,
                    "seed_size": spec.attack.seed_size,
                },
                handle,
                indent=2,
            )
        return AttackRunResult(
            attack_name=self.name,
            mode_name=spec.attack.mode.name,
            output_dir=str(output_dir),
            metadata={
                "dataset_size": len(transferset),
                "transferset_path": str(transferset_path),
            },
        )
