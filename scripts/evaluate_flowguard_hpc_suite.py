"""HPC-oriented FlowGuard++ evaluation launcher.

This script wraps ``scripts/run_flowpure_attack_suite.py`` for long-running
evaluations where MAZE-style attacks need much larger query budgets than
clean-transfer baselines. It supports two execution modes:

1. Sequential/local: run all selected (defense, attack) pairs.
2. Slurm array: pass ``--task-index`` and ``--num-tasks`` so each array task
   runs one shard of the pair grid.

After runs finish, call the script with ``--aggregate-only`` to rebuild the
consolidated JSON/Markdown report from existing pair artifacts.
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

DEFAULT_DEFENSES: tuple[str, ...] = (
    "flowpure",
    "flowguard_integral",
    "flowguard_userlevel",
)

DEFAULT_ATTACKS: tuple[str, ...] = (
    "a1_clean_transfer",
    "a2_projected_maze",
    "a3_flowblind",
    "adaptive_adaptive",
    "maze_baseline",
)

MAZE_FAMILY_ATTACKS: frozenset[str] = frozenset(
    {
        "a2_projected_maze",
        "a3_flowblind",
        "adaptive_adaptive",
        "maze_baseline",
    }
)


@dataclass(slots=True)
class PairTask:
    index: int
    defense: str
    attack: str
    attack_query_budget: int


@dataclass(slots=True)
class PairResult:
    defense: str
    attack: str
    attack_query_budget: int | None
    auroc: float | None
    fpr: float | None
    detection_rate: float | None
    substitute_accuracy: float | None
    error: str | None = None


def _python_executable() -> str:
    """Return the repository-local Python interpreter when available."""
    linux_candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    if linux_candidate.exists():
        return str(linux_candidate)
    windows_candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if windows_candidate.exists():
        return str(windows_candidate)
    return sys.executable


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _budget_for_attack(attack: str, args: argparse.Namespace) -> int:
    if attack == "a1_clean_transfer":
        return int(args.clean_transfer_query_budget)
    if attack in MAZE_FAMILY_ATTACKS:
        return int(args.maze_query_budget)
    return int(args.attack_query_budget)


def _build_tasks(args: argparse.Namespace) -> list[PairTask]:
    defenses = _split_csv(args.defenses)
    attacks = _split_csv(args.attacks)
    tasks: list[PairTask] = []
    for defense in defenses:
        for attack in attacks:
            tasks.append(
                PairTask(
                    index=len(tasks),
                    defense=defense,
                    attack=attack,
                    attack_query_budget=_budget_for_attack(attack, args),
                )
            )
    return tasks


def _selected_tasks(tasks: list[PairTask], args: argparse.Namespace) -> list[PairTask]:
    if args.task_index is None:
        return tasks
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be positive when --task-index is used.")
    return [task for task in tasks if task.index % int(args.num_tasks) == int(args.task_index)]


def _pair_dir(output_root: Path, task: PairTask) -> Path:
    return output_root / f"{task.defense}__{task.attack}"


def _metrics_path(output_root: Path, task: PairTask) -> Path:
    return _pair_dir(output_root, task) / "attack_suite_metrics.json"


def _run_pair(task: PairTask, args: argparse.Namespace, output_root: Path) -> PairResult:
    metrics_path = _metrics_path(output_root, task)
    if metrics_path.exists() and not args.force:
        print(f"[skip] task={task.index} {task.defense} x {task.attack}: {metrics_path}")
        return _parse_pair_metrics(
            metrics_path,
            defense=task.defense,
            attack=task.attack,
            attack_query_budget=task.attack_query_budget,
        )

    command = [
        _python_executable(),
        str(PROJECT_ROOT / "scripts" / "run_flowpure_attack_suite.py"),
        "--name", f"{task.defense}__{task.attack}",
        "--output-root", str(output_root),
        "--attacks", task.attack,
        "--defense", task.defense,
        "--dataset", args.dataset,
        "--query-dataset", args.query_dataset,
        "--clean-transfer-dataset", args.clean_transfer_dataset,
        "--target-architecture", args.target_architecture,
        "--substitute-architecture", args.substitute_architecture,
        "--target-checkpoint-dir", str(args.target_checkpoint_dir),
        "--flow-checkpoint", str(args.flow_checkpoint),
        "--a3-surrogate-checkpoint", str(args.a3_surrogate_checkpoint),
        "--device", args.device,
        "--target-fpr", str(args.target_fpr),
        "--attack-query-budget", str(task.attack_query_budget),
        "--benign-query-budget", str(args.benign_query_budget),
        "--attack-batch-size", str(args.attack_batch_size),
        "--extraction-attack-batch-size", str(args.extraction_attack_batch_size),
        "--training-batch-size", str(args.training_batch_size),
        "--epochs", str(args.epochs),
        "--seed-samples", str(args.seed_samples),
        "--transfer-artifact-sample-size", str(args.transfer_artifact_sample_size),
        "--maze-latent-dim", str(args.maze_latent_dim),
        "--maze-iter-gen", str(args.maze_iter_gen),
        "--maze-iter-clone", str(args.maze_iter_clone),
        "--maze-iter-exp", str(args.maze_iter_exp),
        "--maze-ndirs", str(args.maze_ndirs),
        "--maze-log-iter", str(args.maze_log_iter),
        "--maze-replay-buffer-size", str(args.maze_replay_buffer_size),
        "--maze-artifact-sample-size", str(args.maze_artifact_sample_size),
        "--a2-denoise-kind", args.a2_denoise_kind,
        "--a2-denoise-sigma", str(args.a2_denoise_sigma),
        "--a2-denoise-kernel", str(args.a2_denoise_kernel),
        "--a2-denoise-mix", str(args.a2_denoise_mix),
        "--a3-regularizer-weight", str(args.a3_regularizer_weight),
        "--integral-num-steps", str(args.integral_num_steps),
        "--ks-threshold", str(args.ks_threshold),
        "--ks-min-window", str(args.ks_min_window),
        "--ks-max-window", str(args.ks_max_window),
    ]
    print(f"[run] task={task.index} {task.defense} x {task.attack}")
    print("      " + " ".join(command))
    try:
        subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as exc:
        return PairResult(
            defense=task.defense,
            attack=task.attack,
            attack_query_budget=task.attack_query_budget,
            auroc=None,
            fpr=None,
            detection_rate=None,
            substitute_accuracy=None,
            error=f"subprocess exit {exc.returncode}",
        )

    if not metrics_path.exists():
        return PairResult(
            defense=task.defense,
            attack=task.attack,
            attack_query_budget=task.attack_query_budget,
            auroc=None,
            fpr=None,
            detection_rate=None,
            substitute_accuracy=None,
            error="attack_suite_metrics.json missing",
        )
    return _parse_pair_metrics(
        metrics_path,
        defense=task.defense,
        attack=task.attack,
        attack_query_budget=task.attack_query_budget,
    )


def _parse_pair_metrics(
    metrics_path: Path,
    *,
    defense: str,
    attack: str,
    attack_query_budget: int | None,
) -> PairResult:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return PairResult(
            defense=defense,
            attack=attack,
            attack_query_budget=attack_query_budget,
            auroc=None,
            fpr=None,
            detection_rate=None,
            substitute_accuracy=None,
            error=f"unreadable metrics: {exc!r}",
        )

    config = payload.get("config", {})
    budget = config.get("attack_query_budget", attack_query_budget)
    for row in payload.get("results", []):
        if row.get("attack") != attack:
            continue
        if row.get("error"):
            return PairResult(defense, attack, budget, None, None, None, None, str(row["error"]))
        return PairResult(
            defense=defense,
            attack=attack,
            attack_query_budget=int(budget) if budget is not None else None,
            auroc=row.get("auroc"),
            fpr=row.get("fpr"),
            detection_rate=row.get("detection_rate_attack"),
            substitute_accuracy=row.get("substitute_accuracy"),
        )
    return PairResult(
        defense=defense,
        attack=attack,
        attack_query_budget=int(budget) if budget is not None else None,
        auroc=None,
        fpr=None,
        detection_rate=None,
        substitute_accuracy=None,
        error="attack row missing from metrics",
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(results: list[PairResult], report_path: Path) -> None:
    defenses = sorted({row.defense for row in results})
    attacks = sorted({row.attack for row in results})
    lookup = {(row.defense, row.attack): row for row in results}

    lines: list[str] = [
        "# FlowGuard++ HPC evaluation report",
        "",
        "Generated by `scripts/evaluate_flowguard_hpc_suite.py`.",
        "",
        "## Attack query budgets",
        "",
        "| attack | query budget |",
        "|---|---:|",
    ]
    for attack in attacks:
        budgets = sorted(
            {
                row.attack_query_budget
                for row in results
                if row.attack == attack and row.attack_query_budget is not None
            }
        )
        budget_text = ", ".join(str(value) for value in budgets) if budgets else "n/a"
        lines.append(f"| {attack} | {budget_text} |")
    lines.append("")

    for metric_name, attr, digits in [
        ("AUROC", "auroc", 4),
        ("FPR", "fpr", 4),
        ("Detection rate", "detection_rate", 4),
        ("Substitute accuracy", "substitute_accuracy", 2),
    ]:
        lines.append(f"## {metric_name} by (defense, attack)")
        lines.append("")
        lines.append("| attack \\ defense | " + " | ".join(defenses) + " |")
        lines.append("|" + "---|" * (len(defenses) + 1))
        for attack in attacks:
            cells = [
                _fmt(getattr(lookup.get((defense, attack)), attr, None), digits=digits)
                if lookup.get((defense, attack)) is not None
                else "n/a"
                for defense in defenses
            ]
            lines.append(f"| {attack} | " + " | ".join(cells) + " |")
        lines.append("")

    errors = [row for row in results if row.error]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for row in errors:
            lines.append(f"- `{row.defense} x {row.attack}`: {row.error}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate(output_root: Path, tasks: list[PairTask], args: argparse.Namespace) -> list[PairResult]:
    results: list[PairResult] = []
    for task in tasks:
        metrics_path = _metrics_path(output_root, task)
        if metrics_path.exists():
            results.append(
                _parse_pair_metrics(
                    metrics_path,
                    defense=task.defense,
                    attack=task.attack,
                    attack_query_budget=task.attack_query_budget,
                )
            )
        else:
            results.append(
                PairResult(
                    defense=task.defense,
                    attack=task.attack,
                    attack_query_budget=task.attack_query_budget,
                    auroc=None,
                    fpr=None,
                    detection_rate=None,
                    substitute_accuracy=None,
                    error="metrics missing",
                )
            )

    payload = {
        "config": {
            "defenses": _split_csv(args.defenses),
            "attacks": _split_csv(args.attacks),
            "dataset": args.dataset,
            "query_dataset": args.query_dataset,
            "clean_transfer_dataset": args.clean_transfer_dataset,
            "target_fpr": args.target_fpr,
            "maze_query_budget": args.maze_query_budget,
            "clean_transfer_query_budget": args.clean_transfer_query_budget,
            "attack_query_budget": args.attack_query_budget,
            "benign_query_budget": args.benign_query_budget,
            "transfer_artifact_sample_size": args.transfer_artifact_sample_size,
            "maze_replay_buffer_size": args.maze_replay_buffer_size,
            "maze_artifact_sample_size": args.maze_artifact_sample_size,
        },
        "results": [
            {
                "defense": row.defense,
                "attack": row.attack,
                "attack_query_budget": row.attack_query_budget,
                "auroc": row.auroc,
                "fpr": row.fpr,
                "detection_rate": row.detection_rate,
                "substitute_accuracy": row.substitute_accuracy,
                "error": row.error,
            }
            for row in results
        ],
    }
    summary_path = output_root / "flowguard_hpc_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = output_root / "flowguard_hpc_report.md"
    _write_report(results, report_path)
    print(f"[aggregate] wrote {summary_path}")
    print(f"[aggregate] wrote {report_path}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "flowguard_hpc" / "full")
    parser.add_argument("--defenses", default=",".join(DEFAULT_DEFENSES))
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--task-index", type=int, default=None,
                        help="Slurm-array task index. If omitted, all tasks run sequentially.")
    parser.add_argument("--num-tasks", type=int, default=1,
                        help="Number of Slurm-array tasks used for modulo sharding.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only rebuild the consolidated summary/report from pair artifacts.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run pairs even if attack_suite_metrics.json already exists.")

    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--query-dataset", default="CIFAR10")
    parser.add_argument("--clean-transfer-dataset", default="CIFAR100")
    parser.add_argument("--target-architecture", default="vgg16_bn")
    parser.add_argument("--substitute-architecture", default="vgg16_bn")
    parser.add_argument("--target-checkpoint-dir", type=Path,
                        default=PROJECT_ROOT / "runs" / "notebook"
                        / "training-victim-cifar10-vgg16_bn-nodefense" / "target_model")
    parser.add_argument("--flow-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs" / "flow_matching"
                        / "cifar10_flowpure_pgd" / "checkpoint_latest.pt")
    parser.add_argument("--a3-surrogate-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs" / "flow_matching"
                        / "cifar10_flowpure_pgd" / "checkpoint_latest.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-fpr", type=float, default=0.05)

    parser.add_argument("--maze-query-budget", type=int, default=20_000_000,
                        help="Budget for MAZE-family attacks.")
    parser.add_argument("--clean-transfer-query-budget", type=int, default=1_000_000,
                        help="Budget for A1 clean-transfer.")
    parser.add_argument("--attack-query-budget", type=int, default=1_000_000,
                        help="Fallback attack budget for non-MAZE, non-A1 attacks.")
    parser.add_argument("--benign-query-budget", type=int, default=200_000)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--extraction-attack-batch-size", type=int, default=32)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed-samples", type=int, default=64)
    parser.add_argument("--transfer-artifact-sample-size", type=int, default=50_000,
                        help=(
                            "Bound saved TRANSFER_SET samples for benign/A1 runs. "
                            "Detection still uses the full query budget."
                        ))

    parser.add_argument("--maze-latent-dim", type=int, default=16)
    parser.add_argument("--maze-iter-gen", type=int, default=1)
    parser.add_argument("--maze-iter-clone", type=int, default=2)
    parser.add_argument("--maze-iter-exp", type=int, default=1)
    parser.add_argument("--maze-ndirs", type=int, default=1)
    parser.add_argument("--maze-log-iter", type=int, default=5000)
    parser.add_argument("--maze-replay-buffer-size", type=int, default=32768,
                        help="Bound MAZE replay memory for long HPC runs.")
    parser.add_argument("--maze-artifact-sample-size", type=int, default=4096,
                        help="Bound saved transfer/visualization records.")

    parser.add_argument("--a2-denoise-kind", default="gaussian_blur",
                        choices=["none", "gaussian_blur", "box_blur", "pixel_soft"])
    parser.add_argument("--a2-denoise-sigma", type=float, default=0.8)
    parser.add_argument("--a2-denoise-kernel", type=int, default=5)
    parser.add_argument("--a2-denoise-mix", type=float, default=1.0)
    parser.add_argument("--a3-regularizer-weight", type=float, default=0.01)
    parser.add_argument("--integral-num-steps", type=int, default=8)
    parser.add_argument("--ks-threshold", type=float, default=0.25)
    parser.add_argument("--ks-min-window", type=int, default=64)
    parser.add_argument("--ks-max-window", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(args)

    if args.aggregate_only:
        _aggregate(output_root, tasks, args)
        return

    selected = _selected_tasks(tasks, args)
    if not selected:
        print(
            f"[warn] no tasks selected for task_index={args.task_index}, "
            f"num_tasks={args.num_tasks}; grid has {len(tasks)} tasks."
        )
        return

    print(f"[info] total tasks={len(tasks)} selected={len(selected)} output_root={output_root}")
    results = [_run_pair(task, args, output_root) for task in selected]
    for result in results:
        print(
            f"[done] {result.defense:>22s} x {result.attack:>20s} | "
            f"budget={result.attack_query_budget} | "
            f"AUROC={_fmt(result.auroc)} | "
            f"FPR={_fmt(result.fpr)} | "
            f"det_rate={_fmt(result.detection_rate)} | "
            f"error={result.error or 'none'}"
        )

    if args.task_index is None:
        _aggregate(output_root, tasks, args)
    else:
        print("[info] array task finished; run --aggregate-only after all array tasks complete.")


if __name__ == "__main__":
    main()
