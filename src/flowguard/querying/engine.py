from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from flowguard.defenses.query.base import QueryContext, QueryDefense
from flowguard.querying.batching import batch_tensor
from flowguard.querying.history import QueryHistory, QueryRecord
from flowguard.serving.response_formats import OutputFormat
from flowguard.serving.target_service import PredictionBundle, TargetService


@dataclass(slots=True)
class QueryResult:
    payload: torch.Tensor
    bundle: PredictionBundle


class QueryEngine:
    def __init__(
        self,
        target_service: TargetService,
        query_defenses: Iterable[QueryDefense] | None = None,
        history: QueryHistory | None = None,
        sybil_num_identities: int = 1,
    ) -> None:
        self.target_service = target_service
        self.query_defenses = list(query_defenses or [])
        self.history = history or QueryHistory()
        self.context = QueryContext(total_queries=self.history.total_queries)
        # Sybil sweep: when > 1, attacker queries are spread round-robin across
        # this many identities so per-user/stateful defenses (KS window, label
        # histogram) see only a fraction of the evidence per identity.
        self.sybil_num_identities = max(1, int(sybil_num_identities))
        self._sybil_counter = 0

    def query_batch(
        self,
        inputs: torch.Tensor,
        *,
        output_format: OutputFormat = OutputFormat.SOFT,
        record: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> QueryResult:
        batch = inputs
        context = self.context
        # Make the per-query identity visible to defenses BEFORE before_query.
        # The engine previously only merged caller metadata into the recorded
        # history *after* the defenses ran, so stateful defenses always saw the
        # fallback "default" user. Propagate the routing keys here so user-level
        # detectors actually key on the intended identity, and apply the Sybil
        # rotation when enabled.
        if self.sybil_num_identities > 1:
            context.metadata["client_id"] = self._sybil_counter % self.sybil_num_identities
            self._sybil_counter += 1
        elif metadata:
            if "client_id" in metadata:
                context.metadata["client_id"] = metadata["client_id"]
            if "user_id" in metadata:
                context.metadata["user_id"] = metadata["user_id"]
        for defense in self.query_defenses:
            batch, context = defense.before_query(batch, context)
        if len(batch) == 0:
            empty = torch.empty((0,), device=self.target_service.device)
            bundle = PredictionBundle(logits=empty, probabilities=empty, payload=empty)
            return QueryResult(payload=empty, bundle=bundle)
        bundle = self.target_service.predict_bundle(batch, output_format=output_format)
        outputs = bundle.payload
        after_query_error: Exception | None = None
        try:
            for defense in self.query_defenses:
                outputs, context = defense.after_query(batch, outputs, context)
        except Exception as error:  # pragma: no cover - exercised by defensive smoke tests.
            after_query_error = error
        if record:
            merged_metadata = dict(metadata or {})
            merged_metadata.update(context.metadata)
            self.history.append(
                QueryRecord(
                    batch_size=len(batch),
                    output_format=output_format.value,
                    metadata=merged_metadata,
                )
            )
            self.context = context
        if after_query_error is not None:
            raise after_query_error
        return QueryResult(payload=outputs, bundle=bundle)

    def query_tensor(
        self,
        inputs: torch.Tensor,
        *,
        batch_size: int,
        output_format: OutputFormat = OutputFormat.SOFT,
        record: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        chunks = []
        for batch in batch_tensor(inputs, batch_size):
            result = self.query_batch(
                batch,
                output_format=output_format,
                record=record,
                metadata=metadata,
            )
            chunks.append(result.payload.detach().cpu())
        if not chunks:
            return torch.empty((0,))
        return torch.cat(chunks, dim=0)


class LegacyBlackboxBridge:
    def __init__(
        self,
        query_engine: QueryEngine,
        out_path: str | None = None,
        default_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.query_engine = query_engine
        self.target_service = query_engine.target_service
        self.device = self.target_service.device
        self.model = self.target_service.model
        self.num_classes = self.target_service.num_classes
        self.out_path = out_path
        self.default_metadata = dict(default_metadata or {})
        self.log_path = None
        self.log_prefix = "framework"
        self.require_xinfo = self.target_service.requires_xinfo
        self.top1_preserve = self.target_service.top1_preserve
        self.call_count = 0
        self.queries: list[tuple[np.ndarray, np.ndarray]] = []

    @staticmethod
    def calc_query_distances(queries):
        l1s, l2s, kls = [], [], []
        for query in queries:
            y_v, y_prime, *_ = query
            y_v, y_prime = torch.tensor(y_v), torch.tensor(y_prime)
            l1s.append((y_v - y_prime).norm(p=1, dim=1))
            l2s.append((y_v - y_prime).norm(p=2, dim=1))
            kls.append(torch.sum(F.kl_div((y_v + 1e-6).log(), y_prime, reduction="none"), dim=1))
        l1s = torch.cat(l1s).cpu().numpy()
        l2s = torch.cat(l2s).cpu().numpy()
        kls = torch.cat(kls).cpu().numpy()
        return (
            np.amax(l1s),
            np.mean(l1s),
            np.std(l1s),
            np.mean(l2s),
            np.std(l2s),
            np.mean(kls),
            np.std(kls),
        )

    def eval(self) -> None:
        self.model.eval()

    def get_xinfo(self, x: torch.Tensor):
        infos = self.target_service.get_auxiliary_info(x)
        return infos[0] if infos else None

    def get_yprime(self, y: torch.Tensor, x_info: torch.Tensor | None = None) -> torch.Tensor:
        return self.target_service.transform_probabilities(y, auxiliary=x_info)

    def __call__(
        self,
        x: torch.Tensor,
        stat: bool = True,
        return_origin: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        query_metadata = dict(self.default_metadata)
        if metadata:
            query_metadata.update(metadata)
        result = self.query_engine.query_batch(
            x,
            output_format=OutputFormat.SOFT,
            record=stat,
            metadata=query_metadata if query_metadata else None,
        )
        defended = result.bundle.probabilities
        original = torch.softmax(result.bundle.logits, dim=1)
        if stat:
            self.call_count += len(x)
            self.queries.append(
                (
                    original.detach().cpu().numpy(),
                    defended.detach().cpu().numpy(),
                )
            )
        if return_origin:
            return defended, original
        return defended
