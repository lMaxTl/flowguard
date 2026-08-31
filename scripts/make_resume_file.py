"""Emit the resume JSON for an interrupted attack/defense matrix run.

``scripts/evaluate_attack_defense_matrix_smoke.py`` flushes its summary after
every completed ``(defense, attack)`` cell, and resumes from a JSON file naming
the cell to start at. Building that file by hand means reading the summary and
re-deriving the iteration order, which is easy to get wrong after a wall-clock
kill.

This script does it: it reads the summary, works out which cells already have
results, and writes a resume file pointing at the first cell that does not.

Usage::

    python scripts/make_resume_file.py \\
        --summary runs/adaptive_ceiling/attack_defense_matrix_adaptive-ceiling_summary.json \\
        --output runs/adaptive_ceiling/resume.json

Then resubmit with ``RESUME_FILE=runs/adaptive_ceiling/resume.json``.

Note that a cell interrupted mid-attack is *not* lost when the attack itself
was checkpointing (``--attack-checkpoint-every-queries``): resuming re-enters
that cell and the runner picks its training state back up from
``_scratch/<defense>__<attack>/attack_checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True,
                        help="attack_defense_matrix_<label>_summary.json from the killed run.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the resume JSON (default: resume.json beside the summary).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if not args.summary.exists():
        raise FileNotFoundError(f"--summary not found: {args.summary}")

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    defense_order = list(config.get("defenses") or [])
    attack_order = list(config.get("attacks") or [])
    if not defense_order or not attack_order:
        raise ValueError(
            f"Summary is missing the defense/attack order in its config block: {args.summary}"
        )

    completed = {
        (str(row.get("defense")), str(row.get("attack")))
        for row in payload.get("results", [])
        # A cell that errored still needs re-running, so it does not count.
        if not row.get("error")
    }

    # The matrix iterates defenses outer, attacks inner.
    pending: tuple[str, str] | None = None
    for defense in defense_order:
        for attack in attack_order:
            if (defense, attack) not in completed:
                pending = (defense, attack)
                break
        if pending is not None:
            break

    if pending is None:
        print(f"[resume] every cell in {args.summary.name} is complete; nothing to resume.")
        return

    defense, attack = pending
    output = args.output or args.summary.parent / "resume.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "start_defense": defense,
                "start_attack": attack,
                "state_path": str(args.summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(defense_order) * len(attack_order)
    print(f"[resume] {len(completed)}/{total} cells complete")
    print(f"[resume] next cell: defense={defense} attack={attack}")
    print(f"[resume] wrote {output}")

    scratch = args.summary.parent / "_scratch" / f"{defense}__{attack}" / "attack_checkpoint.pt"
    if scratch.exists():
        size_gb = scratch.stat().st_size / 1e9
        print(f"[resume] attack checkpoint present ({size_gb:.1f} GB): {scratch}")
        print("[resume] that cell will continue mid-attack rather than retrain from scratch.")
    else:
        print(f"[resume] no attack checkpoint at {scratch}; that cell restarts from scratch.")


if __name__ == "__main__":
    main()
