from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from uuid import uuid4

import torch

from flowguard.distributed.protocol import QueryTask, QueryTaskResult
from flowguard.distributed.worker import QueryWorker
from flowguard.querying.batching import batch_tensor


@dataclass(slots=True)
class DistributedCoordinator:
    workers: list[QueryWorker]
    max_inflight_batches: int = 1
    num_threads: int = 1
    task_results: list[QueryTaskResult] = field(default_factory=list)

    def _submit_tasks(
        self,
        batches: list[torch.Tensor],
        output_format: str,
        metadata: dict[str, object] | None = None,
    ) -> list[QueryTaskResult]:
        if not self.workers:
            raise RuntimeError("DistributedCoordinator requires at least one worker")
        shared_metadata = dict(metadata or {})
        tasks = [
            QueryTask(
                task_id=f"task-{uuid4().hex}",
                batch=batch,
                output_format=output_format,
                metadata={**shared_metadata, "batch_size": len(batch)},
            )
            for batch in batches
        ]
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            futures = []
            for index, task in enumerate(tasks):
                worker = self.workers[index % len(self.workers)]
                futures.append(pool.submit(worker.execute, task))
            results = [future.result() for future in futures]
        self.task_results.extend(results)
        return results

    def distribute_tensor(
        self,
        inputs: torch.Tensor,
        *,
        batch_size: int,
        output_format: str = "soft",
        metadata: dict[str, object] | None = None,
    ) -> torch.Tensor:
        batches = list(batch_tensor(inputs, batch_size))
        results = self._submit_tasks(batches, output_format=output_format, metadata=metadata)
        ordered = [result.outputs for result in results]
        if not ordered:
            return torch.empty((0,))
        return torch.cat(ordered, dim=0)

    def distribute_with_originals(
        self,
        inputs: torch.Tensor,
        *,
        batch_size: int,
        output_format: str = "soft",
        metadata: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batches = list(batch_tensor(inputs, batch_size))
        results = self._submit_tasks(batches, output_format=output_format, metadata=metadata)
        defended = [result.outputs for result in results]
        original = [result.original_probabilities for result in results if result.original_probabilities is not None]
        if not defended:
            empty = torch.empty((0,))
            return empty, empty
        return torch.cat(defended, dim=0), torch.cat(original, dim=0)
