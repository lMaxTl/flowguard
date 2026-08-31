"""A4 Sparse-Probe analysis (plan section A4).

Given a benign detection-sample file and an attack detection-sample file
produced by ``run_flowpure_attack_suite.py`` (or the existing paperlike
pipeline), simulate a sparse-probe adversary that interleaves a small
fraction of attack queries with benign queries and evaluates whether a
per-query detector (the FlowPure^PGD design) can still identify the user
as malicious. Because the paper-faithful defense has no per-user
aggregation, the answer is structurally "no" as long as the attack
fraction stays within the calibrated FPR envelope.

The analysis is post-hoc: no new experiments are launched; it just
re-weights the already-collected detection scores to reflect a mixed
submission stream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_samples_from_metrics_json(path: Path, attack_key: str) -> list[dict]:
    """Best-effort: load per-query scores from an attack-suite run directory.

    Looks for ``visualization_records.pt`` or ``detection_samples.json``
    produced by earlier runs. Falls back to manual loading with
    ``--scores-json`` when those files are absent.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data:
        for row in data["results"]:
            if row.get("attack") == attack_key:
                return row.get("samples", [])
    raise KeyError(
        f"Attack key '{attack_key}' with embedded samples not found in {path}. "
        f"Please pass --scores-json instead."
    )


def _load_scores_json(path: Path) -> list[float]:
    return [float(v) for v in json.loads(path.read_text(encoding="utf-8"))]


def _simulate_sparse_probe(
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    *,
    probe_rate: float,
    threshold: float,
    user_window: int,
    rng: np.random.Generator,
    num_users: int = 500,
) -> dict:
    """Simulate a user submitting ``user_window`` queries with probe-rate fraction attack queries.

    Returns per-user flag statistics and a user-level ROC AUC assuming the
    defender counts the number of flagged queries in the window as the
    user-level score (the simplest aggregation). Compares against a
    purely-benign reference user-population.
    """
    if user_window <= 0:
        raise ValueError("user_window must be positive.")

    benign_flag_rate = float((benign_scores > threshold).mean())
    attack_flag_rate = float((attack_scores > threshold).mean())
    expected_user_flags = probe_rate * user_window * attack_flag_rate + (
        1.0 - probe_rate
    ) * user_window * benign_flag_rate

    user_flags_attack: list[int] = []
    user_flags_benign: list[int] = []
    for _ in range(num_users):
        chosen = rng.choice(
            benign_scores.size + attack_scores.size,
            size=user_window,
            replace=True,
        )
        # Attack user: probe_rate fraction are drawn from attack scores.
        mask = rng.random(user_window) < probe_rate
        attack_idx = rng.integers(attack_scores.size, size=mask.sum())
        benign_idx = rng.integers(benign_scores.size, size=(~mask).sum())
        attack_user_scores = np.concatenate([
            attack_scores[attack_idx],
            benign_scores[benign_idx],
        ])
        user_flags_attack.append(int((attack_user_scores > threshold).sum()))

        # Benign user: draw entirely from benign scores.
        benign_user_scores = benign_scores[rng.integers(benign_scores.size, size=user_window)]
        user_flags_benign.append(int((benign_user_scores > threshold).sum()))

    attack_flags = np.asarray(user_flags_attack)
    benign_flags = np.asarray(user_flags_benign)

    y_true = np.concatenate([np.zeros_like(benign_flags), np.ones_like(attack_flags)])
    y_score = np.concatenate([benign_flags, attack_flags]).astype(np.float64)
    try:
        user_level_auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        user_level_auroc = float("nan")

    return {
        "probe_rate": float(probe_rate),
        "user_window": int(user_window),
        "threshold": float(threshold),
        "per_query_benign_flag_rate": benign_flag_rate,
        "per_query_attack_flag_rate": attack_flag_rate,
        "expected_user_flags": float(expected_user_flags),
        "user_level_auroc": user_level_auroc,
        "attack_user_flag_mean": float(attack_flags.mean()),
        "attack_user_flag_median": float(np.median(attack_flags)),
        "benign_user_flag_mean": float(benign_flags.mean()),
        "benign_user_flag_median": float(np.median(benign_flags)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benign-scores", type=Path, required=True,
                        help="JSON array of FlowPure scores on benign queries.")
    parser.add_argument("--attack-scores", type=Path, required=True,
                        help="JSON array of FlowPure scores on attack queries.")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Calibrated per-query FlowPure threshold.")
    parser.add_argument("--probe-rates", type=float, nargs="+",
                        default=[0.01, 0.025, 0.05, 0.1, 0.25])
    parser.add_argument("--user-windows", type=int, nargs="+",
                        default=[100, 500, 2000])
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    benign_scores = np.asarray(_load_scores_json(args.benign_scores), dtype=np.float64)
    attack_scores = np.asarray(_load_scores_json(args.attack_scores), dtype=np.float64)

    rng = np.random.default_rng(int(args.seed))
    results: list[dict] = []
    for user_window in args.user_windows:
        for probe_rate in args.probe_rates:
            summary = _simulate_sparse_probe(
                benign_scores,
                attack_scores,
                probe_rate=float(probe_rate),
                threshold=float(args.threshold),
                user_window=int(user_window),
                rng=rng,
                num_users=int(args.num_users),
            )
            print(
                f"window={user_window:5d} | rate={probe_rate:.3f} | "
                f"user-AUROC={summary['user_level_auroc']:.4f} | "
                f"attack_flags_mean={summary['attack_user_flag_mean']:.2f} | "
                f"benign_flags_mean={summary['benign_user_flag_mean']:.2f}"
            )
            results.append(summary)

    payload = {
        "threshold": float(args.threshold),
        "num_benign_scores": int(benign_scores.size),
        "num_attack_scores": int(attack_scores.size),
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved sparse-probe report to: {args.output_json}")


if __name__ == "__main__":
    main()
