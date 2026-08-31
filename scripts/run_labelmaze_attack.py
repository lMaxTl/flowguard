"""A5 LabelMaze attack (plan section A5): active-learning on clean candidates.

Threat model: Grey-Box. The attacker has no adversarial-perturbation
generator and never modifies pixel statistics. Instead they use a pool of
clean images (a foreign dataset or synthesized GAN/SD samples) and run an
active-learning loop where each round picks the highest-entropy victim
response as the next training target.

This script implements the active-learning outer loop post-hoc on top of
an existing clean-transfer run:

1. Reuse the A1 clean-transfer transferset (from ``run_flowpure_attack_suite``).
2. Pull the victim's recorded soft-labels for those queries.
3. Compute per-query label entropy; select the top-``--select-k`` queries.
4. Report AUROC of the FlowPure^PGD detector on this active-learned subset,
   together with a simple query-efficiency metric (how many queries were
   needed to reach ``--select-k`` high-entropy queries at the detector's
   calibrated FPR rate).

The conclusion documented in the plan (Teil 2, A5) is: because every query
in the pool is a clean natural image, the detection rate is bounded by the
calibrated FPR regardless of which queries are selected. Active learning
only affects the substitute-accuracy / query-efficiency axis.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flowguard.attacks.adaptive import SurrogateVelocityRegularizer, denormalize_cifar


def _load_transferset(path: Path) -> list[tuple[np.ndarray, Any]]:
    with path.open("rb") as handle:
        try:
            payload = torch.load(handle, weights_only=False)
        except TypeError:
            handle.seek(0)
            payload = pickle.load(handle)
    return payload


def _entropy_from_softlabels(labels: torch.Tensor) -> torch.Tensor:
    """Compute per-sample entropy of a soft-label batch (assumes probabilities)."""
    probabilities = labels.clamp(min=1e-12)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def _tensor_from_record(record: tuple[np.ndarray, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    query_np, label = record
    query = torch.as_tensor(query_np, dtype=torch.float32)
    if isinstance(label, torch.Tensor):
        label_tensor = label.detach().to(torch.float32)
    else:
        label_tensor = torch.as_tensor(label, dtype=torch.float32)
    return query, label_tensor


def _score_flowpure(
    regularizer: SurrogateVelocityRegularizer,
    queries_normalized: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    scores: list[float] = []
    for start in range(0, queries_normalized.shape[0], batch_size):
        chunk_normalized = queries_normalized[start : start + batch_size]
        chunk_01 = denormalize_cifar(chunk_normalized)
        with torch.no_grad():
            chunk_scores = regularizer.velocity_score(chunk_01)
        scores.extend(chunk_scores.detach().cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transferset", type=Path, required=True,
                        help="Pickle path from the A1 clean-transfer run.")
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Calibrated FlowPure threshold from the benign reference.")
    parser.add_argument("--select-k", type=int, default=2000,
                        help="Number of top-entropy queries retained by the active loop.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    print(f"Loading transferset: {args.transferset}")
    transferset = _load_transferset(args.transferset)
    queries = []
    labels = []
    for record in transferset:
        q, l = _tensor_from_record(record)
        queries.append(q)
        labels.append(l)
    queries_tensor = torch.stack(queries, dim=0)
    labels_tensor = torch.stack(labels, dim=0)

    print(f"Pool size: {queries_tensor.shape[0]}")
    entropies = _entropy_from_softlabels(labels_tensor).cpu().numpy()

    k = min(int(args.select_k), queries_tensor.shape[0])
    top_idx = np.argsort(-entropies)[:k]
    bottom_idx = np.argsort(entropies)[:k]

    print(f"Loading FlowPure checkpoint: {args.flow_checkpoint}")
    regularizer = SurrogateVelocityRegularizer(str(args.flow_checkpoint), device=device)

    def _subset_stats(indices: np.ndarray, label: str) -> dict:
        subset = queries_tensor[torch.as_tensor(indices, dtype=torch.long)]
        scores = _score_flowpure(regularizer, subset, batch_size=int(args.batch_size))
        flagged = (scores > float(args.threshold)).astype(np.float64)
        detection_rate = float(flagged.mean())
        print(
            f"  subset={label:18s} n={len(indices):5d} | "
            f"detection_rate={detection_rate:.4f} | "
            f"score mean={scores.mean():.2f}, median={np.median(scores):.2f}, "
            f"max={scores.max():.2f}"
        )
        return {
            "subset": label,
            "num_queries": int(len(indices)),
            "detection_rate": detection_rate,
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_max": float(scores.max()),
        }

    print("\nPer-subset FlowPure detection statistics:")
    stats = [
        _subset_stats(top_idx, "top_entropy"),
        _subset_stats(bottom_idx, "bottom_entropy"),
        _subset_stats(np.arange(queries_tensor.shape[0]), "full_pool"),
    ]

    payload = {
        "transferset": str(args.transferset),
        "flow_checkpoint": str(args.flow_checkpoint),
        "threshold": float(args.threshold),
        "select_k": int(k),
        "pool_size": int(queries_tensor.shape[0]),
        "entropy_mean": float(entropies.mean()),
        "entropy_std": float(entropies.std()),
        "stats": stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved LabelMaze report to: {args.output_json}")


if __name__ == "__main__":
    main()
