"""Master evaluation driver for FlowGuard++ (plan Teil 5 E5.1/E5.2).

Iterates over a grid of (defense, attack) pairs, delegating each individual
run to :mod:`scripts.run_flowpure_attack_suite`, and aggregates the
resulting AUROC / FPR / detection-rate numbers into a single consolidated
JSON + Markdown report.

Grid axes:

- Defenses:  ``flowpure`` (paper baseline, Section 3.3),
             ``flowguard_integral`` (C1),
             ``flowguard_userlevel`` (C3).
- Attacks:   ``a1_clean_transfer, a2_projected_maze, a3_flowblind,
             a4_sparse_probe, a5_labelmaze, adaptive_adaptive,
             maze_baseline``.

A4 and A5 are post-hoc analyses and are delegated to
``scripts/analyze_sparse_probe.py`` and ``scripts/run_labelmaze_attack.py``;
they are launched after the in-line A1/A2/A3 runs so their inputs
(transfer-set artifacts, benign reference scores) already exist.

The script is designed to be re-runnable: any (defense, attack) pair whose
``attack_suite_metrics.json`` already exists is skipped unless
``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------


_DEFENSES: tuple[str, ...] = (
    "flowpure",
    "flowguard_integral",
    "flowguard_userlevel",
)


_INLINE_ATTACKS: tuple[str, ...] = (
    "a1_clean_transfer",
    "a2_projected_maze",
    "a3_flowblind",
    "adaptive_adaptive",
    "maze_baseline",
)


@dataclass(slots=True)
class PairResult:
    defense: str
    attack: str
    auroc: float | None
    fpr: float | None
    detection_rate: float | None
    substitute_accuracy: float | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _python_executable() -> str:
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _run_suite(
    defense: str,
    attack: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> PairResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "attack_suite_metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"[skip] {defense} x {attack} already exists at {metrics_path}")
        return _parse_metrics(metrics_path, defense=defense, attack=attack)

    cmd = [
        _python_executable(),
        str(PROJECT_ROOT / "scripts" / "run_flowpure_attack_suite.py"),
        "--name", f"{defense}__{attack}",
        "--output-root", str(out_dir.parent),
        "--attacks", attack,
        "--defense", defense,
        "--dataset", args.dataset,
        "--query-dataset", args.query_dataset,
        "--clean-transfer-dataset", args.clean_transfer_dataset,
        "--target-architecture", args.target_architecture,
        "--substitute-architecture", args.substitute_architecture,
        "--target-checkpoint-dir", str(args.target_checkpoint_dir),
        "--flow-checkpoint", str(args.flow_checkpoint),
        "--device", args.device,
        "--target-fpr", str(args.target_fpr),
        "--attack-query-budget", str(args.attack_query_budget),
        "--benign-query-budget", str(args.benign_query_budget),
        "--attack-batch-size", str(args.attack_batch_size),
        "--extraction-attack-batch-size", str(args.extraction_attack_batch_size),
        "--training-batch-size", str(args.training_batch_size),
        "--epochs", str(args.epochs),
        "--seed-samples", str(args.seed_samples),
    ]
    if args.a3_surrogate_checkpoint is not None:
        cmd += ["--a3-surrogate-checkpoint", str(args.a3_surrogate_checkpoint)]
    print(f"[run]  {defense} x {attack}\n  -> {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as exc:
        return PairResult(
            defense=defense,
            attack=attack,
            auroc=None,
            fpr=None,
            detection_rate=None,
            substitute_accuracy=None,
            error=f"subprocess exit {exc.returncode}",
        )

    if not metrics_path.exists():
        return PairResult(
            defense=defense,
            attack=attack,
            auroc=None,
            fpr=None,
            detection_rate=None,
            substitute_accuracy=None,
            error="attack_suite_metrics.json missing",
        )
    return _parse_metrics(metrics_path, defense=defense, attack=attack)


def _parse_metrics(metrics_path: Path, *, defense: str, attack: str) -> PairResult:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return PairResult(
            defense=defense,
            attack=attack,
            auroc=None,
            fpr=None,
            detection_rate=None,
            substitute_accuracy=None,
            error=f"unreadable metrics: {exc!r}",
        )
    rows = payload.get("results") or []
    for row in rows:
        if row.get("attack") != attack:
            continue
        if row.get("error"):
            return PairResult(
                defense=defense,
                attack=attack,
                auroc=None,
                fpr=None,
                detection_rate=None,
                substitute_accuracy=None,
                error=str(row["error"]),
            )
        return PairResult(
            defense=defense,
            attack=attack,
            auroc=row.get("auroc"),
            fpr=row.get("fpr"),
            detection_rate=row.get("detection_rate_attack"),
            substitute_accuracy=row.get("substitute_accuracy"),
        )
    return PairResult(
        defense=defense,
        attack=attack,
        auroc=None,
        fpr=None,
        detection_rate=None,
        substitute_accuracy=None,
        error="attack row missing from metrics",
    )


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(results: list[PairResult], report_path: Path) -> None:
    defenses = sorted({r.defense for r in results})
    attacks = sorted({r.attack for r in results})
    lookup = {(r.defense, r.attack): r for r in results}

    lines: list[str] = []
    lines.append("# FlowGuard++ evaluation report")
    lines.append("")
    lines.append("Generated by `scripts/evaluate_flowguard_suite.py`.")
    lines.append("")

    lines.append("## AUROC by (defense, attack)")
    lines.append("")
    header = "| attack \\ defense | " + " | ".join(defenses) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(defenses) + 1))
    for attack in attacks:
        cells = [_fmt(lookup.get((d, attack), PairResult(d, attack, None, None, None, None)).auroc) for d in defenses]
        lines.append(f"| {attack} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## FPR by (defense, attack)")
    lines.append("")
    lines.append(header)
    lines.append("|" + "---|" * (len(defenses) + 1))
    for attack in attacks:
        cells = [_fmt(lookup.get((d, attack), PairResult(d, attack, None, None, None, None)).fpr) for d in defenses]
        lines.append(f"| {attack} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Detection rate (attack samples flagged) by (defense, attack)")
    lines.append("")
    lines.append(header)
    lines.append("|" + "---|" * (len(defenses) + 1))
    for attack in attacks:
        cells = [_fmt(lookup.get((d, attack), PairResult(d, attack, None, None, None, None)).detection_rate) for d in defenses]
        lines.append(f"| {attack} | " + " | ".join(cells) + " |")
    lines.append("")

    errors = [r for r in results if r.error]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- `{r.defense} x {r.attack}`: {r.error}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/notebook/flowguard_suite")
    parser.add_argument("--defenses", default=",".join(_DEFENSES),
                        help="Comma-separated defenses to evaluate.")
    parser.add_argument("--attacks", default=",".join(_INLINE_ATTACKS),
                        help="Comma-separated attacks to evaluate.")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--query-dataset", default="CIFAR10")
    parser.add_argument("--clean-transfer-dataset", default="CIFAR100")
    parser.add_argument("--target-architecture", default="vgg16_bn")
    parser.add_argument("--substitute-architecture", default="vgg16_bn")
    parser.add_argument("--target-checkpoint-dir", type=Path,
                        default=PROJECT_ROOT / "runs" / "notebook"
                        / "training-victim-cifar10-vgg16_bn-nodefense"
                        / "target_model")
    parser.add_argument("--flow-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs" / "flow_matching"
                        / "cifar10_flowpure_pgd" / "checkpoint_latest.pt")
    parser.add_argument("--a3-surrogate-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs" / "flow_matching"
                        / "cifar10_flowpure_pgd" / "checkpoint_latest.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--attack-query-budget", type=int, default=144)
    parser.add_argument("--benign-query-budget", type=int, default=144)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--extraction-attack-batch-size", type=int, default=16)
    parser.add_argument("--training-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed-samples", type=int, default=16)
    parser.add_argument("--force", action="store_true",
                        help="Re-run a pair even if its metrics file already exists.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[PairResult] = []
    for defense in defenses:
        for attack in attacks:
            pair_dir = out_root / f"{defense}__{attack}"
            result = _run_suite(defense, attack, args, pair_dir)
            results.append(result)
            print(
                f"[done] {defense:>22s} x {attack:>20s} | "
                f"AUROC={_fmt(result.auroc)} | "
                f"FPR={_fmt(result.fpr)} | "
                f"det_rate={_fmt(result.detection_rate)}"
            )

    payload = {
        "config": {
            "defenses": defenses,
            "attacks": attacks,
            "dataset": args.dataset,
            "target_fpr": args.target_fpr,
            "attack_query_budget": args.attack_query_budget,
            "benign_query_budget": args.benign_query_budget,
        },
        "results": [
            {
                "defense": r.defense,
                "attack": r.attack,
                "auroc": r.auroc,
                "fpr": r.fpr,
                "detection_rate": r.detection_rate,
                "substitute_accuracy": r.substitute_accuracy,
                "error": r.error,
            }
            for r in results
        ],
    }
    summary_json = out_root / "flowguard_suite_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved summary JSON to: {summary_json}")

    report_md = out_root / "flowguard_suite_report.md"
    _write_report(results, report_md)
    print(f"Saved markdown report to: {report_md}")


if __name__ == "__main__":
    main()
