"""Score one query stream with several detectors at once.

The evaluation harness runs one attack per (defense, attack) cell, so an N-way
defense comparison retrains the attack N times. In *audit-only* mode that is
pure waste: an auditing detector observes the queries and records a score, but
returns the batch and the victim's outputs untouched, so the attack trajectory
does not depend on which detector is watching. One run can therefore feed every
detector.

Besides the N-fold saving this makes the comparison *paired*: every detector
scores the exact same queries from the exact same generator state, instead of N
independently seeded runs whose attacks diverge.

Enforcement is deliberately unsupported. Two detectors that both reject would
each change the query stream the other sees, so their scores would no longer
describe the same experiment, and "which one blocked first" would silently
depend on list order.
"""

from __future__ import annotations

from typing import Any

import torch

from flowguard.defenses.query.base import QueryContext, QueryDefense


# Metadata keys that identify the caller rather than a detector's output. They
# are forwarded into every child context so per-user detectors (KS windows,
# label histograms) still see the identity they need to key their state on.
_PASSTHROUGH_KEYS: tuple[str, ...] = ("user_id", "client_id", "batch_size")


class MultiAuditQueryDefense(QueryDefense):
    """Fan a query batch out to several audit-only detectors.

    Each child runs against its own :class:`QueryContext`, so the children
    cannot overwrite one another's metadata -- which they otherwise would, since
    every flow-based detector deliberately mirrors its score into the shared
    ``flowpure_scores``/``flowpure_blocked`` keys. Results are collected under
    ``metadata["by_defense"][key]``.

    Args:
        defenses: ``(key, defense)`` pairs. ``key`` is the name the evaluation
            harness reports the detector under.
        strict: Re-raise a child's exception instead of recording it. Off by
            default so one misconfigured detector cannot abort a run that is
            producing valid results for the others.
    """

    name = "multi_audit"

    def __init__(
        self,
        defenses: list[tuple[str, QueryDefense]],
        *,
        strict: bool = False,
        **parameters: Any,
    ) -> None:
        super().__init__(**parameters)
        if not defenses:
            raise ValueError("MultiAuditQueryDefense requires at least one child defense.")
        self.defenses = list(defenses)
        self.strict = bool(strict)
        for key, defense in self.defenses:
            # A child that blocks would raise, and the surviving children would
            # then see a different (truncated) query stream than the one their
            # scores claim to describe.
            if getattr(defense, "audit_only", True) is False:
                raise ValueError(
                    f"Child defense '{key}' is in enforcing mode. "
                    "MultiAuditQueryDefense only supports audit-only children; "
                    "run enforcing evaluations one defense at a time."
                )

    def _child_context(self, context: QueryContext) -> QueryContext:
        seeded = {
            key: context.metadata[key]
            for key in _PASSTHROUGH_KEYS
            if key in context.metadata
        }
        return QueryContext(total_queries=context.total_queries, metadata=seeded)

    def _record(
        self,
        context: QueryContext,
        key: str,
        child_metadata: dict[str, Any],
    ) -> None:
        bucket = context.metadata.setdefault("by_defense", {})
        entry = bucket.setdefault(key, {})
        entry.update(child_metadata)

    def before_query(
        self, batch: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        for key, defense in self.defenses:
            child_context = self._child_context(context)
            try:
                defense.before_query(batch, child_context)
            except Exception as error:  # noqa: BLE001 - one child must not sink the run.
                if self.strict:
                    raise
                child_context.metadata["error"] = repr(error)
            self._record(context, key, child_context.metadata)
        # The batch is returned unmodified: auditing detectors observe, they do
        # not filter.
        return batch, context

    def after_query(
        self, batch: torch.Tensor, outputs: torch.Tensor, context: QueryContext
    ) -> tuple[torch.Tensor, QueryContext]:
        for key, defense in self.defenses:
            child_context = self._child_context(context)
            # Carry this child's own before_query metadata forward: the
            # composite monitor's after_query reads the flag its before_query
            # set, and would otherwise re-decide from an empty context.
            previous = (context.metadata.get("by_defense") or {}).get(key)
            if isinstance(previous, dict):
                child_context.metadata.update(previous)
            try:
                defense.after_query(batch, outputs, child_context)
            except Exception as error:  # noqa: BLE001
                if self.strict:
                    raise
                child_context.metadata["error"] = repr(error)
            self._record(context, key, child_context.metadata)
        return outputs, context
