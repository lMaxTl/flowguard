from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import torch

from flowguard.distributed.protocol import QueryTask, QueryTaskResult
from flowguard.querying.engine import QueryEngine
from flowguard.serving.response_formats import OutputFormat


@dataclass(slots=True)
class QueryWorker:
    worker_id: str
    query_engine: QueryEngine

    def heartbeat(self) -> dict[str, str]:
        return {"worker_id": self.worker_id, "status": "ok"}

    def execute(self, task: QueryTask) -> QueryTaskResult:
        output_format = OutputFormat(task.output_format)
        result = self.query_engine.query_batch(
            task.batch,
            output_format=output_format,
            record=True,
            metadata=task.metadata,
        )
        return QueryTaskResult(
            task_id=task.task_id,
            outputs=result.payload.detach().cpu(),
            original_probabilities=torch.softmax(result.bundle.logits, dim=1).detach().cpu(),
            worker_id=self.worker_id,
            metadata=dict(task.metadata),
        )


def build_in_process_worker(query_engine: QueryEngine, worker_id: str | None = None) -> QueryWorker:
    return QueryWorker(worker_id=worker_id or f"worker-{uuid4().hex[:8]}", query_engine=query_engine)
