from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import torch

from defenses import datasets as legacy_datasets
from defenses.adversary.transfer import RandomAdversaryIters
from defenses.utils.utils import suppress_stdout
from flowguard.attacks.base import AttackRunContext, AttackRunner, AttackRunResult
from flowguard.distributed.coordinator import DistributedCoordinator
from flowguard.querying.engine import LegacyBlackboxBridge, QueryEngine
from flowguard.recovery.table_recovery import TableRecovery, TableRecoveryConfig


class DistributedLegacyBlackboxBridge(LegacyBlackboxBridge):
    def __init__(
        self,
        query_engine: QueryEngine,
        coordinator: DistributedCoordinator,
        batch_size: int,
        out_path: str | None = None,
        default_metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            query_engine,
            out_path=out_path,
            default_metadata=default_metadata,
        )
        self.coordinator = coordinator
        self.batch_size = batch_size

    def __call__(
        self,
        x: torch.Tensor,
        stat: bool = True,
        return_origin: bool = False,
        metadata: dict[str, object] | None = None,
    ):
        query_metadata = dict(self.default_metadata)
        if metadata:
            query_metadata.update(metadata)
        defended, original = self.coordinator.distribute_with_originals(
            x,
            batch_size=self.batch_size,
            output_format="soft",
            metadata=query_metadata if query_metadata else None,
        )
        if stat:
            self.call_count += len(x)
            self.queries.append(
                (
                    original.detach().cpu().numpy(),
                    defended.detach().cpu().numpy(),
                )
            )
        if return_origin:
            return defended.to(self.device), original.to(self.device)
        return defended.to(self.device)


class TransferSetAttackRunner(AttackRunner):
    name = "transfer_set"

    def __init__(
        self,
        query_engine: QueryEngine,
        distributed_coordinator: DistributedCoordinator | None = None,
    ) -> None:
        self.query_engine = query_engine
        self.distributed_coordinator = distributed_coordinator

    def _build_blackbox(self, context: AttackRunContext, output_dir: Path):
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
                out_path=str(output_dir),
                default_metadata=query_metadata,
            )
        return LegacyBlackboxBridge(
            self.query_engine,
            out_path=str(output_dir),
            default_metadata=query_metadata,
        )

    def _build_recovery(self, blackbox, context: AttackRunContext):
        mode = context.experiment.attack.mode
        if not mode.defense_aware:
            return None
        config = TableRecoveryConfig(
            batch_size=context.experiment.attack.batch_size,
            shadow_path=context.metadata.get("shadow_models") if isinstance(context.metadata, dict) else None,
            **mode.recovery_parameters,
        )
        recovery = TableRecovery(blackbox, config)
        return recovery.legacy

    @staticmethod
    def _append_limited_transferset(
        stored: list,
        chunk: list,
        *,
        artifact_sample_size: int,
    ) -> None:
        if artifact_sample_size <= 0:
            stored.extend(chunk)
            return
        remaining = artifact_sample_size - len(stored)
        if remaining <= 0:
            return
        stored.extend(chunk[:remaining])

    def run(self, context: AttackRunContext) -> AttackRunResult:
        spec = context.experiment
        mode = spec.attack.mode
        extra = dict(spec.attack.extra)
        artifact_sample_size = max(
            0,
            int(
                extra.get(
                    "transfer_artifact_sample_size",
                    extra.get("artifact_sample_size", 0),
                )
            ),
        )
        skip_substitute_training = bool(extra.get("skip_substitute_training", False))
        output_dir = Path(context.output_dir or Path("runs") / spec.name / self.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        if spec.verbose:
            print(
                f"[FlowGuard++][transfer_set] Starting attack mode='{mode.name}' "
                f"query_budget={spec.attack.query_budget} batch_size={spec.attack.batch_size}"
            )

        query_dataset_name = spec.attack.query_dataset.name
        modelfamily = legacy_datasets.dataset_to_modelfamily[query_dataset_name]
        transform_key = "train" if mode.use_train_transform else "test"
        transform = legacy_datasets.modelfamily_to_transforms[modelfamily][transform_key]
        queryset = legacy_datasets.__dict__[query_dataset_name](
            train=True,
            transform=transform,
            download=spec.attack.query_dataset.download,
        )
        queryset_size = len(queryset)
        if queryset_size <= 0:
            raise RuntimeError(f"Query dataset '{query_dataset_name}' is empty.")

        blackbox = self._build_blackbox(context, output_dir)
        recovery = self._build_recovery(blackbox, context)
        transferset: list = []
        remaining_budget = int(spec.attack.query_budget)
        chunk_index = 0
        while remaining_budget > 0:
            chunk_budget = min(remaining_budget, queryset_size)
            if spec.verbose and chunk_index > 0:
                print(
                    "[FlowGuard++][transfer_set] Query budget exceeds dataset size; "
                    f"starting repeated dataset pass {chunk_index + 1} "
                    f"with chunk_budget={chunk_budget}."
                )
            adversary = RandomAdversaryIters(
                blackbox=blackbox,
                label_recover=recovery,
                queryset=queryset,
                batch_size=spec.attack.batch_size,
                hard_label=mode.hard_label,
            )
            stream_context = nullcontext() if spec.verbose else suppress_stdout()
            with stream_context:
                chunk_transferset = adversary.get_transferset(
                    chunk_budget,
                    None,
                    queries_per_image=max(1, mode.smoothing_queries_per_image),
                )
            if not skip_substitute_training:
                self._append_limited_transferset(
                    transferset,
                    chunk_transferset,
                    artifact_sample_size=artifact_sample_size,
                )
            remaining_budget -= chunk_budget
            chunk_index += 1

        transferset_path: Path | None = None
        if not skip_substitute_training:
            transferset_path = output_dir / "transferset.pickle"
            with transferset_path.open("wb") as handle:
                torch.save(transferset, handle)
        with (output_dir / "params_transfer.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "attack_kind": spec.attack.kind.value,
                    "mode": mode.name,
                    "queryset": query_dataset_name,
                    "budget": spec.attack.query_budget,
                    "batch_size": spec.attack.batch_size,
                    "queryset_size": queryset_size,
                    "dataset_passes": chunk_index,
                    "stored_samples": len(transferset),
                    "transfer_artifact_sample_size": artifact_sample_size,
                    "skip_substitute_training": skip_substitute_training,
                },
                handle,
                indent=2,
            )
        metadata = {
            "num_samples": len(transferset),
            "num_queries": int(spec.attack.query_budget),
            "skip_substitute_training": skip_substitute_training,
        }
        if transferset_path is not None:
            metadata["transferset_path"] = str(transferset_path)
        return AttackRunResult(
            attack_name=self.name,
            mode_name=mode.name,
            output_dir=str(output_dir),
            metadata=metadata,
        )
