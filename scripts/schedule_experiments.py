"""Submit every missing FlowGuard++ multi-dataset experiment to Slurm (HoreKa).

Reads ``configs/horeka_experiments.json`` and builds, per dataset, this graph
of stages (arrows are "must finish first"):

    victim ───────────────► defender_flowpure ──┐
    defender_gauss ─────────────────────────────┤
    attacker_proxy ─► attacker_flowpure ─┐      ├──► eval (prada, maze, disguide)
    attacker_gauss ──────────────────────┴► attacker_calibration ─┐
    attacker_prior ───────────────────────────────────────────────┴► eval (D1, D3-D6)

A stage is *done* when its ``DONE.json`` marker exists, *queued* when a job
this script submitted earlier is still in ``squeue``, and *missing* otherwise.
Missing stages are submitted with ``--dependency=afterok`` on their unfinished
prerequisites. Stages that may outlive the 48 h wall clock are submitted as a
chain of ``chain`` jobs linked with ``afterany``; every job resumes from the
stage's own checkpoint and exits at once if the marker already exists.

    python scripts/schedule_experiments.py --status          # what is done / queued / missing
    python scripts/schedule_experiments.py --dry-run         # print the sbatch commands
    python scripts/schedule_experiments.py                   # submit everything missing
    python scripts/schedule_experiments.py --datasets GTSRB --stages victim

Re-running is always safe: done and queued stages are never submitted twice.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STAGE = "scripts/horeka/run_stage.sbatch"
TRAINING_STAGES = (
    "victim",
    "attacker_proxy",
    "defender_flowpure",
    "defender_gauss",
    "attacker_flowpure",
    "attacker_gauss",
    "attacker_prior",
    "attacker_calibration",
)
# Attacks that use the attacker's surrogate CNFs, calibration and diffusion prior.
SURROGATE_ATTACK_PREFIX = "flowguard_d"
SHORT_GROUP = {"detection": "det", "adaptive": "ada"}


@dataclass
class Stage:
    key: str
    job_name: str
    done_marker: Path
    command: list[str]
    deps: list[str]
    time: str
    mem: str
    chain: int = 1
    status: str = "missing"
    job_ids: list[str] = field(default_factory=list)


def cli_args(options: dict[str, Any]) -> list[str]:
    """Turn ``{"batch-size": 64, "flag": True}`` into ``--batch-size 64 --flag``."""
    argv: list[str] = []
    for key, value in options.items():
        key = key.replace("_", "-")  # config files may use either spelling
        if value is None or value is False:
            continue
        if value is True:
            argv.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            argv += [f"--{key}", ",".join(str(item) for item in value)]
        else:
            argv += [f"--{key}", str(value)]
    return argv


def merged(*parts: dict[str, Any] | None) -> dict[str, Any]:
    """Later parts win; ``max_steps`` and ``max-steps`` are the same option."""
    result: dict[str, Any] = {}
    for part in parts:
        for key, value in (part or {}).items():
            result[key.replace("_", "-")] = value
    return result


def build_stages(config: dict[str, Any], seeds: list[int] | None) -> list[Stage]:
    runs_root = Path(config["runs_root"])
    stage_cfg = config["stages"]
    evaluation = config["evaluation"]
    stages: list[Stage] = []

    for dataset, ds in config["datasets"].items():
        root = runs_root / dataset
        paths = {name: root / name for name in TRAINING_STAGES}
        arch = ds["architecture"]
        pool = ds["attacker_pool"]

        def add(name: str, command: list[str], deps: list[str], marker: Path | None = None) -> None:
            spec = stage_cfg[name]
            stages.append(
                Stage(
                    key=f"{dataset}/{name}",
                    job_name=f"fg-{dataset}-{name}",
                    done_marker=marker or paths[name] / "DONE.json",
                    command=command,
                    deps=[f"{dataset}/{dep}" for dep in deps],
                    time=spec["time"],
                    mem=spec["mem"],
                    chain=int(spec.get("chain", 1)),
                )
            )

        def stage_args(name: str) -> dict[str, Any]:
            return merged(stage_cfg[name].get("args"), ds.get(name))

        add(
            "victim",
            ["scripts/train_victim.py", "--dataset", dataset, "--architecture", arch,
             "--output-dir", (paths["victim"]).as_posix(), "--device", "cuda", *cli_args(stage_args("victim"))],
            [],
        )
        add(
            "attacker_proxy",
            ["scripts/train_victim.py", "--dataset", ds["attacker_proxy_dataset"], "--architecture", arch,
             "--output-dir", (paths["attacker_proxy"]).as_posix(), "--device", "cuda",
             *cli_args(stage_args("attacker_proxy"))],
            [],
        )
        add(
            "defender_flowpure",
            ["scripts/train_flowpure_pgd.py", "--dataset", dataset,
             "--output-dir", (paths["defender_flowpure"]).as_posix(),
             "--victim-checkpoint-dir", (paths["victim"]).as_posix(), "--pgd-label-source", "dataset",
             "--device", "cuda", *cli_args(stage_args("defender_flowpure"))],
            ["victim"],
        )
        add(
            "defender_gauss",
            ["scripts/train_flow_matching.py", "--dataset", dataset,
             "--output-dir", (paths["defender_gauss"]).as_posix(), "--device", "cuda",
             *cli_args(stage_args("defender_gauss"))],
            [],
        )
        add(
            "attacker_flowpure",
            ["scripts/train_flowpure_pgd.py", "--dataset", pool,
             "--output-dir", (paths["attacker_flowpure"]).as_posix(),
             "--victim-checkpoint-dir", (paths["attacker_proxy"]).as_posix(), "--pgd-label-source", "prediction",
             "--device", "cuda", *cli_args(stage_args("attacker_flowpure"))],
            ["attacker_proxy"],
        )
        add(
            "attacker_gauss",
            ["scripts/train_flow_matching.py", "--dataset", pool,
             "--output-dir", (paths["attacker_gauss"]).as_posix(), "--device", "cuda",
             *cli_args(stage_args("attacker_gauss"))],
            [],
        )
        add(
            "attacker_prior",
            ["scripts/train_diffusion_prior.py", "--dataset", pool,
             "--output-dir", (paths["attacker_prior"]).as_posix(), "--device", "cuda",
             *cli_args(stage_args("attacker_prior"))],
            [],
        )
        calibration_cache = paths["attacker_calibration"] / "attacker_surrogate_calibration.json"
        add(
            "attacker_calibration",
            ["scripts/evaluate_parallel_defenses.py", "--attacker-calibration-only",
             "--dataset", dataset, "--adaptive-attacker-pool", pool,
             "--a3-surrogate-checkpoint", (paths["attacker_flowpure"] / "checkpoint_latest.pt").as_posix(),
             "--a3-likelihood-surrogate-checkpoint", (paths["attacker_gauss"] / "checkpoint_latest.pt").as_posix(),
             "--attacker-calibration-cache", calibration_cache.as_posix(),
             "--output-root", (paths["attacker_calibration"]).as_posix(), "--device", "cuda",
             *cli_args(stage_args("attacker_calibration"))],
            ["attacker_flowpure", "attacker_gauss"],
        )

        for group_name, group in evaluation["groups"].items():
            if not group.get("enabled", True):
                continue
            for seed in seeds if seeds is not None else evaluation["seeds"]:
                for attack in group["attacks"]:
                    output_root = root / "eval" / group_name / f"seed{seed}" / attack
                    uses_surrogates = attack.startswith(SURROGATE_ATTACK_PREFIX)
                    options = merged(
                        evaluation.get("common_args"),
                        group.get("args"),
                        {
                            "run-label": f"{dataset}-{group_name}-seed{seed}-{attack}",
                            "output-root": output_root.as_posix(),
                            "attacks": attack,
                            "defenses": evaluation["defenses"],
                            "sybil-variants": evaluation.get("sybil_variants") or None,
                            "composite-eval-learned": bool(evaluation.get("composite_eval_learned")),
                            "dataset": dataset,
                            "query-dataset": pool,
                            "clean-transfer-dataset": pool,
                            "adaptive-attacker-pool": pool,
                            "benign-calibration-dataset": f"{dataset}BenignCal",
                            "benign-eval-dataset": f"{dataset}BenignEval",
                            "benign-query-budget": ds["benign_query_budget"],
                            "target-architecture": arch,
                            "substitute-architecture": arch,
                            "target-checkpoint-dir": (paths["victim"]).as_posix(),
                            "flow-checkpoint": (paths["defender_flowpure"] / "checkpoint_latest.pt").as_posix(),
                            "likelihood-flow-checkpoint": (paths["defender_gauss"] / "checkpoint_latest.pt").as_posix(),
                            "seed": seed,
                            "device": "cuda",
                        },
                    )
                    deps = ["victim", "defender_flowpure", "defender_gauss"]
                    if uses_surrogates:
                        options.update(
                            {
                                "a3-surrogate-checkpoint": (paths["attacker_flowpure"] / "checkpoint_latest.pt").as_posix(),
                                "a3-likelihood-surrogate-checkpoint": (paths["attacker_gauss"] / "checkpoint_latest.pt").as_posix(),
                                "attacker-calibration-cache": calibration_cache.as_posix(),
                                "diffusion-model-id": (paths["attacker_prior"]).as_posix(),
                            }
                        )
                        deps += ["attacker_calibration", "attacker_prior"]
                    short_attack = attack.replace("flowguard_", "").replace("_adaptive", "")
                    stages.append(
                        Stage(
                            key=f"{dataset}/eval/{group_name}/seed{seed}/{attack}",
                            job_name=f"fg-{dataset}-{SHORT_GROUP.get(group_name, group_name)}-s{seed}-{short_attack}",
                            done_marker=output_root / "DONE.json",
                            command=["scripts/evaluate_parallel_defenses.py", *cli_args(options)],
                            deps=[f"{dataset}/{dep}" for dep in deps],
                            time=group["time"],
                            mem=group["mem"],
                            chain=int(group.get("chain", 1)),
                        )
                    )
    return stages


# --- Slurm helpers ----------------------------------------------------------------------------


def queued_job_ids() -> set[str] | None:
    """IDs of this user's jobs still in the queue, or None if squeue is unavailable."""
    try:
        output = subprocess.run(
            ["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line.strip().split("_")[0] for line in output.splitlines() if line.strip()}


def sbatch_command(stage: Stage, slurm: dict[str, Any], dependency: str | None) -> list[str]:
    log_dir = PROJECT_ROOT / "logs" / "scheduler"
    command = [
        "sbatch", "--parsable",
        f"--job-name={stage.job_name}",
        f"--partition={slurm['partition']}",
        f"--time={stage.time}",
        f"--mem={stage.mem}",
        f"--gres=gpu:{int(slurm.get('gpus', 1))}",
        f"--cpus-per-task={int(slurm.get('cpus_per_task', 16))}",
        f"--chdir={PROJECT_ROOT.as_posix()}",
        f"--output={(log_dir / (stage.job_name + '_%j.out')).as_posix()}",
        "--kill-on-invalid-dep=yes",
    ]
    if slurm.get("account"):
        command.append(f"--account={slurm['account']}")
    command += list(slurm.get("extra_args", []))
    if dependency:
        command.append(f"--dependency={dependency}")
    command += [RUN_STAGE, stage.done_marker.as_posix(), *stage.command]
    return command


def submit(command: list[str], dry_run: bool, counter: list[int]) -> str:
    if dry_run:
        counter[0] += 1
        return f"DRYRUN{counter[0]}"
    result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout.strip().split(";")[0]


# --- main ---------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "horeka_experiments.json")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands, submit nothing.")
    parser.add_argument("--status", action="store_true", help="Only print the status table.")
    parser.add_argument("--datasets", default="", help="Comma-separated subset of datasets.")
    parser.add_argument("--stages", default="",
                        help=f"Comma-separated subset of {','.join(TRAINING_STAGES)},eval. "
                             "Unfinished prerequisites are always included.")
    parser.add_argument("--groups", default="", help="Evaluation groups to include (default: all enabled).")
    parser.add_argument("--attacks", default="", help="Only these evaluation attacks.")
    parser.add_argument("--seeds", default="", help="Override evaluation seeds, e.g. 0,1,2,3,4.")
    parser.add_argument("--max-submit", type=int, default=0,
                        help="Submit at most this many stages (0 = no limit), for a cautious first run.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()] or None
    stages = build_stages(config, seeds)
    by_key = {stage.key: stage for stage in stages}

    def selected(stage: Stage) -> bool:
        dataset, _, rest = stage.key.partition("/")
        if args.datasets and dataset not in args.datasets.split(","):
            return False
        is_eval = rest.startswith("eval/")
        name = "eval" if is_eval else rest
        if args.stages and name not in args.stages.split(","):
            return False
        if is_eval:
            _, group, _, attack = rest.split("/")
            if args.groups and group not in args.groups.split(","):
                return False
            if args.attacks and attack not in args.attacks.split(","):
                return False
        return True

    wanted: set[str] = set()
    frontier = [stage.key for stage in stages if selected(stage)]
    while frontier:  # add prerequisites
        key = frontier.pop()
        if key in wanted:
            continue
        wanted.add(key)
        frontier.extend(by_key[key].deps)

    state_path = PROJECT_ROOT / config["runs_root"] / "_scheduler" / "jobs.json"
    state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    in_queue = queued_job_ids()
    if in_queue is None and not (args.dry_run or args.status):
        raise SystemExit("squeue is not available; run this on a HoreKa login node (or use --dry-run).")

    for stage in stages:
        if (PROJECT_ROOT / stage.done_marker).exists():
            stage.status = "done"
            continue
        previous = state.get(stage.key, {}).get("job_ids", [])
        if in_queue is not None and any(job_id in in_queue for job_id in previous):
            stage.status = "queued"
            stage.job_ids = previous
        elif previous:
            stage.status = "failed?"  # submitted before, no marker, no longer queued

    if args.status:
        width = max(len(stage.key) for stage in stages)
        counts: dict[str, int] = {}
        for stage in stages:
            if stage.key not in wanted:
                continue
            counts[stage.status] = counts.get(stage.status, 0) + 1
            jobs = ",".join(state.get(stage.key, {}).get("job_ids", []))
            print(f"{stage.key:{width}s}  {stage.status:8s}  {jobs}")
        print("\n" + ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())))
        return

    slurm = config["slurm"]
    counter = [0]
    submitted = 0
    if not args.dry_run:
        # Slurm does not create the --output directory; a missing one loses the log.
        (PROJECT_ROOT / "logs" / "scheduler").mkdir(parents=True, exist_ok=True)
    for stage in stages:
        if stage.key not in wanted or stage.status in {"done", "queued"}:
            continue
        if args.max_submit and submitted >= args.max_submit:
            print(f"[schedule] --max-submit reached; {stage.key} and later stages left for the next run")
            break
        blocked = [by_key[dep] for dep in stage.deps if by_key[dep].status not in {"done"}]
        if any(not dep.job_ids for dep in blocked):
            missing = [dep.key for dep in blocked if not dep.job_ids]
            print(f"[schedule] skip {stage.key}: prerequisites not submitted: {missing}")
            continue
        upstream = ":".join(dep.job_ids[-1] for dep in blocked)
        if stage.status == "failed?":
            print(f"[schedule] resubmitting {stage.key} (previous jobs "
                  f"{state[stage.key]['job_ids']} ended without DONE.json; check "
                  f"logs/scheduler/{stage.job_name}_<jobid>.out)")
        job_ids: list[str] = []
        for link in range(max(1, stage.chain)):
            parts = []
            if upstream:
                parts.append(f"afterok:{upstream}")
            if job_ids:
                parts.append(f"afterany:{job_ids[-1]}")
            command = sbatch_command(stage, slurm, ",".join(parts) or None)
            if args.dry_run and link == 0:
                print(" ".join(shlex.quote(part) for part in command))
            job_ids.append(submit(command, args.dry_run, counter))
        stage.job_ids = job_ids
        stage.status = "queued"
        submitted += 1
        print(f"[schedule] {'would submit' if args.dry_run else 'submitted'} {stage.key}: "
              f"{', '.join(job_ids)}")
        if not args.dry_run:
            state[stage.key] = {
                "job_ids": job_ids,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "command": stage.command,
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"[schedule] {submitted} stage(s) {'planned' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    sys.exit(main())
