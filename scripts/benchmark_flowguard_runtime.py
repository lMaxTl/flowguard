"""Measure the serving cost of FlowGuard++ (latency, throughput, NFE, memory).

Why this script exists
----------------------
The paper proposes an online monitor for ML APIs but reports no cost. TDSC's
scope is explicit about achieving security "without compromising performance",
so a detector whose price is unstated cannot be assessed. The likelihood
component C3 integrates the change-of-variables ODE and estimates its
divergence, which is many neural-function evaluations (NFE) per query, while
the victim needs one forward pass. This script measures that ratio instead of
describing it.

What it measures
----------------
For each configuration and batch size:

- wall-clock latency per batch (mean / median / p95), CUDA-synchronised;
- per-query latency and throughput (queries/s);
- neural-function evaluations per query, counted by wrapping every velocity
  field the configuration touches (so the number is observed, not derived from
  the solver settings);
- peak GPU memory attributable to the configuration
  (``max_memory_allocated`` measured around the timed region);
- overhead relative to victim-only inference.

Configurations
--------------
``victim``            victim forward pass only (the baseline every ratio uses)
``c1``                FlowPure velocity at t=0: one velocity evaluation
``c1c2``              shared forward solve producing C1 and C2 (``--num-steps``)
``c3``                likelihood ODE + divergence (the expensive component)
``c1c2c3``            all three per-query components
``composite``         the deployed ``FlowGuardCompositeDefense.before_query`` /
                      ``after_query`` path, i.e. C1-C5 including the stateful
                      windows, with the victim in the loop
``selective``         C1/C2 for every query, C3 only for the fraction of
                      queries whose C1 score lands in the uncertainty band
                      around the threshold (``--selective-fractions``). This is
                      the routing the paper's limitations section proposes but
                      never evaluates.

Usage
-----
    python scripts/benchmark_flowguard_runtime.py \\
        --target-checkpoint-dir runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model \\
        --flow-checkpoint runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt \\
        --likelihood-flow-checkpoint runs/flow_matching/cifar10_notebook/checkpoint_latest.pt \\
        --output runs/runtime/flowguard_runtime_cifar10.json

The LaTeX table is written next to the JSON with ``--latex-out``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flowguard.defenses.query.base import QueryContext  # noqa: E402
from flowguard.defenses.query.flowguard import (  # noqa: E402
    FlowGuardCompositeDefense,
)
from flowguard.serving.model_loader import load_legacy_model  # noqa: E402


class NFECounter:
    """Count calls into a velocity field, weighted by batch size.

    Wraps the callable in place so the count reflects what the solver actually
    did. A solver that adapts its step count, or an implementation that reuses
    a cached evaluation, is then visible in the number rather than hidden by an
    assumption about the discretisation.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.samples = 0
        self._patched: list[tuple[Any, str, Callable[..., Any]]] = []

    def patch(self, owner: Any, attribute: str) -> None:
        original = getattr(owner, attribute)

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if args and isinstance(args[0], torch.Tensor):
                self.samples += int(args[0].shape[0])
            return original(*args, **kwargs)

        setattr(owner, attribute, counted)
        self._patched.append((owner, attribute, original))

    def reset(self) -> None:
        self.calls = 0
        self.samples = 0

    def restore(self) -> None:
        for owner, attribute, original in reversed(self._patched):
            setattr(owner, attribute, original)
        self._patched.clear()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_calls(
    fn: Callable[[torch.Tensor], Any],
    batch: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        fn(batch)
    _sync(device)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(batch)
        _sync(device)
        timings.append(time.perf_counter() - start)
    return timings


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarise(
    name: str,
    timings: list[float],
    *,
    batch_size: int,
    nfe_calls: int,
    nfe_samples: int,
    repeats: int,
    peak_memory_bytes: int | None,
) -> dict[str, Any]:
    per_batch_ms = [value * 1000.0 for value in timings]
    mean_ms = statistics.fmean(per_batch_ms)
    return {
        "configuration": name,
        "batch_size": batch_size,
        "batch_latency_ms_mean": mean_ms,
        "batch_latency_ms_median": statistics.median(per_batch_ms),
        "batch_latency_ms_p95": _percentile(per_batch_ms, 95.0),
        "query_latency_ms_mean": mean_ms / batch_size,
        "throughput_qps": (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else float("nan"),
        # NFE is counted over the whole timed region and normalised by the
        # queries that passed through it, so a batched solver and a per-sample
        # solver are directly comparable.
        "nfe_per_query": (nfe_samples / (batch_size * repeats)) if repeats else 0.0,
        "velocity_calls_per_batch": (nfe_calls / repeats) if repeats else 0.0,
        "peak_memory_mib": (peak_memory_bytes / (1024**2)) if peak_memory_bytes else None,
    }


def _build_composite(
    args: argparse.Namespace,
    *,
    device: torch.device,
    benign_reference_scores: list[float] | None,
    benign_label_histogram: list[float] | None,
    audit_only: bool = True,
) -> FlowGuardCompositeDefense:
    return FlowGuardCompositeDefense(
        fm_checkpoint_path=str(args.flow_checkpoint),
        likelihood_fm_checkpoint_path=str(args.likelihood_flow_checkpoint),
        dataset_name=args.dataset,
        device=str(device),
        num_steps=int(args.num_steps),
        likelihood_step_size=float(args.likelihood_step_size),
        likelihood_method=args.likelihood_method,
        likelihood_exact_divergence=bool(args.exact_divergence),
        benign_reference_scores=benign_reference_scores,
        benign_label_histogram=benign_label_histogram,
        # Standardisation references: any finite value exercises the same code
        # path as a calibrated deployment. Runtime does not depend on their
        # values, only on the branches they enable.
        benign_t0_mean=0.0,
        benign_t0_std=1.0,
        benign_integral_mean=0.0,
        benign_integral_std=1.0,
        benign_likelihood_mean=0.0,
        benign_likelihood_std=1.0,
        num_classes=int(args.num_classes),
        inputs_normalized=False,
        audit_only=audit_only,
    )


def _load_query_batch(
    args: argparse.Namespace, *, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Return a batch of benign-looking inputs in [0, 1]."""
    generator = torch.Generator().manual_seed(int(args.seed))
    if args.data_root and Path(args.data_root).exists():
        try:
            from torchvision import datasets, transforms  # noqa: PLC0415

            dataset = datasets.CIFAR10(
                root=str(args.data_root),
                train=False,
                download=False,
                transform=transforms.ToTensor(),
            )
            indices = torch.randperm(len(dataset), generator=generator)[:batch_size]
            images = torch.stack([dataset[int(index)][0] for index in indices])
            return images.to(device)
        except Exception as error:  # noqa: BLE001 - fall back, but say so
            print(f"[bench] CIFAR-10 unavailable ({error}); using random inputs")
    images = torch.rand(
        (batch_size, 3, args.image_size, args.image_size), generator=generator
    )
    return images.to(device)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    print(f"[bench] device={device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()})")
    victim = load_legacy_model(args.target_checkpoint_dir, device=device).model
    victim.eval()

    # Benign references make C4/C5 do real work in the composite path instead
    # of short-circuiting, so the composite row is the deployed cost.
    benign_reference_scores = np.random.default_rng(int(args.seed)).normal(
        loc=0.0, scale=1.0, size=int(args.benign_reference_size)
    ).tolist()
    benign_label_histogram = [1.0 / int(args.num_classes)] * int(args.num_classes)

    composite = _build_composite(
        args,
        device=device,
        benign_reference_scores=benign_reference_scores,
        benign_label_histogram=benign_label_histogram,
    )

    rows: list[dict[str, Any]] = []
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]
    selective_fractions = [
        float(value) for value in args.selective_fractions.split(",") if value
    ]
    extra_likelihood_steps = [
        float(value) for value in str(args.extra_likelihood_steps).split(",") if value
    ]

    for batch_size in batch_sizes:
        batch = _load_query_batch(args, batch_size=batch_size, device=device)

        def measure(
            name: str,
            fn: Callable[[torch.Tensor], Any],
            *,
            patch_targets: list[tuple[Any, str]],
        ) -> dict[str, Any]:
            counter = NFECounter()
            for owner, attribute in patch_targets:
                counter.patch(owner, attribute)
            try:
                # Warm-up runs are excluded from the NFE count as well, so the
                # reported figure matches the timed repeats exactly.
                for _ in range(int(args.warmup)):
                    fn(batch)
                _sync(device)
                counter.reset()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline = torch.cuda.memory_allocated(device)
                timings = _time_calls(
                    fn, batch, device=device, warmup=0, repeats=int(args.repeats)
                )
                peak = None
                if device.type == "cuda":
                    peak = int(torch.cuda.max_memory_allocated(device) - baseline)
                row = _summarise(
                    name,
                    timings,
                    batch_size=batch_size,
                    nfe_calls=counter.calls,
                    nfe_samples=counter.samples,
                    repeats=int(args.repeats),
                    peak_memory_bytes=peak,
                )
            except torch.cuda.OutOfMemoryError as error:
                counter.restore()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                row = {
                    "configuration": name,
                    "batch_size": batch_size,
                    "error": f"{type(error).__name__}: {error}".splitlines()[0],
                }
                rows.append(row)
                print(f"[bench] bs={batch_size:>4} {name:<12} OOM, skipped")
                return row
            finally:
                counter.restore()
            print(
                f"[bench] bs={batch_size:>4} {name:<12} "
                f"{row['query_latency_ms_mean']:8.3f} ms/query  "
                f"{row['throughput_qps']:10.1f} q/s  "
                f"NFE/query={row['nfe_per_query']:.1f}"
            )
            rows.append(row)
            return row

        @torch.no_grad()
        def victim_only(inputs: torch.Tensor) -> Any:
            return victim(inputs * 2.0 - 1.0)

        @torch.no_grad()
        def c1_only(inputs: torch.Tensor) -> Any:
            # C1 is the t=0 term of the shared solve; measuring it separately
            # shows what a FlowPure-only deployment costs.
            saved = composite.num_steps
            composite.num_steps = 1
            try:
                return composite._flow_scores(inputs)[0]
            finally:
                composite.num_steps = saved

        @torch.no_grad()
        def c1c2(inputs: torch.Tensor) -> Any:
            return composite._flow_scores(inputs)

        @torch.no_grad()
        def c3(inputs: torch.Tensor) -> Any:
            return composite._estimate_log_likelihood(inputs)

        @torch.no_grad()
        def c1c2c3(inputs: torch.Tensor) -> Any:
            t0, integral = composite._flow_scores(inputs)
            likelihood = composite._estimate_log_likelihood(inputs)
            return t0, integral, likelihood

        def composite_full(inputs: torch.Tensor) -> Any:
            context = QueryContext(total_queries=0, metadata={"user_id": "bench"})
            _, context = composite.before_query(inputs, context)
            with torch.no_grad():
                outputs = victim(inputs * 2.0 - 1.0)
            return composite.after_query(inputs, outputs, context)

        # Count NFE on the checkpoint models themselves, not on the composite's
        # wrapper attributes: the likelihood path runs inside an ODESolver that
        # captured its velocity field at construction, so a wrapper swapped on
        # the composite would never be called and C3 would report zero NFE.
        velocity_targets = [(composite.flow_checkpoint.model, "forward")]
        likelihood_targets = [(composite.likelihood_checkpoint.model, "forward")]
        if composite.likelihood_checkpoint is composite.flow_checkpoint:
            likelihood_targets = []
        all_targets = velocity_targets + likelihood_targets

        measure("victim", victim_only, patch_targets=[])
        measure("c1", c1_only, patch_targets=velocity_targets)
        measure("c1c2", c1c2, patch_targets=velocity_targets)
        measure("c3", c3, patch_targets=likelihood_targets or velocity_targets)
        measure("c1c2c3", c1c2c3, patch_targets=all_targets)
        measure("composite", composite_full, patch_targets=all_targets)

        for step_size in extra_likelihood_steps:
            @torch.no_grad()
            def c3_step(inputs: torch.Tensor, step_size: float = step_size) -> Any:
                saved = composite.likelihood_step_size
                composite.likelihood_step_size = step_size
                try:
                    return composite._estimate_log_likelihood(inputs)
                finally:
                    composite.likelihood_step_size = saved

            measure(
                f"c3_step{step_size:g}",
                c3_step,
                patch_targets=likelihood_targets or velocity_targets,
            )

        # Selective routing: C1/C2 for everyone, C3 for a fraction. Measured by
        # running C3 on a sub-batch of that size rather than by scaling the
        # full-batch cost, because the per-query cost of a small batch is
        # higher and a scaled estimate would overstate the saving.
        for fraction in selective_fractions:
            routed = max(1, int(round(batch_size * fraction)))

            @torch.no_grad()
            def selective(inputs: torch.Tensor, routed: int = routed) -> Any:
                t0, integral = composite._flow_scores(inputs)
                likelihood = composite._estimate_log_likelihood(inputs[:routed])
                return t0, integral, likelihood

            measure(f"selective_{fraction:g}", selective, patch_targets=all_targets)


    # Attach the victim-relative overhead once every row exists, so the ratio
    # always divides by the victim measured at the same batch size.
    victim_by_batch = {
        row["batch_size"]: row["query_latency_ms_mean"]
        for row in rows
        if row["configuration"] == "victim"
    }
    for row in rows:
        baseline = victim_by_batch.get(row["batch_size"])
        row["overhead_vs_victim"] = (
            row["query_latency_ms_mean"] / baseline if baseline else None
        )

    payload = {
        "config": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "dataset": args.dataset,
            "target_checkpoint_dir": str(args.target_checkpoint_dir),
            "flow_checkpoint": str(args.flow_checkpoint),
            "likelihood_flow_checkpoint": str(args.likelihood_flow_checkpoint),
            "num_steps": int(args.num_steps),
            "likelihood_step_size": float(args.likelihood_step_size),
            "likelihood_method": args.likelihood_method,
            "likelihood_exact_divergence": bool(args.exact_divergence),
            "batch_sizes": batch_sizes,
            "selective_fractions": selective_fractions,
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
        },
        "results": rows,
    }
    return payload


LATEX_ROWS: tuple[tuple[str, str], ...] = (
    ("victim", r"Victim only (VGG16-BN)"),
    ("c1", r"$+$ C1 (velocity, $t{=}0$)"),
    ("c1c2", r"$+$ C1\,$+$\,C2 (trajectory solve)"),
    ("c3", r"C3 alone (likelihood \gls{ODE})"),
    ("c1c2c3", r"$+$ C1--C3 (all per-query)"),
    ("composite", r"$+$ FlowGuard++ (C1--C5)"),
)


def _throughput(value: float) -> str:
    """Throughput at a precision that stays informative below 100 queries/s."""
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _memory(row: dict[str, Any]) -> str:
    value = row.get("peak_memory_mib")
    if value is None:
        return "--"
    return f"{value / 1024:.2f}" if value >= 1024 else f"{value / 1024:.3f}"


def _latex_row(label: str, row: dict[str, Any]) -> str:
    overhead = row.get("overhead_vs_victim")
    cells = [
        label,
        f"{row['query_latency_ms_mean']:.3f}",
        _throughput(row["throughput_qps"]),
        f"{row['nfe_per_query']:.1f}",
        _memory(row),
        f"{overhead:,.0f}" + r"$\times$" if overhead else "--",
    ]
    return " &\n".join(cells) + r" \\"


def _render_latex(payload: dict[str, Any], batch_size: int) -> str:
    rows = {
        row["configuration"]: row
        for row in payload["results"]
        if row["batch_size"] == batch_size and "error" not in row
    }
    lines = [_latex_row(label, rows[key]) for key, label in LATEX_ROWS if key in rows]

    # Cheaper solvers and selective routing are the two levers the paper
    # proposes for making the cost tolerable, so they belong in the same table
    # as the cost they are meant to reduce.
    steps = sorted(
        (row for name, row in rows.items() if name.startswith("c3_step")),
        key=lambda row: row["nfe_per_query"],
    )
    for row in steps:
        step = row["configuration"].split("step", 1)[1]
        lines.append(
            _latex_row(rf"\quad C3 with step size ${step}$", row)
        )
    selective = sorted(
        (row for name, row in rows.items() if name.startswith("selective_")),
        key=lambda row: float(row["configuration"].split("_", 1)[1]),
    )
    for row in selective:
        fraction = float(row["configuration"].split("_", 1)[1])
        lines.append(
            _latex_row(rf"\quad selective C3 ({fraction * 100:.0f}\% routed)", row)
        )

    config = payload["config"]
    gpu = config.get("gpu") or "CPU"
    body = "\n\n".join(lines)
    return rf"""% GENERATED FILE - do not edit by hand.
%
% Rebuild with:
%   python scripts/benchmark_flowguard_runtime.py --latex-out paper/tables/runtime_overhead.tex
%
\begin{{table}}[!t]
\caption{{
Serving cost of the FlowGuard++ components on {gpu}, batch size {batch_size},
mean over {config['repeats']} timed repetitions after {config['warmup']} warm-up
batches. \gls{{NFE}} counts velocity-field evaluations per query, observed by
instrumenting the velocity fields rather than derived from the solver settings.
Memory is the peak device allocation attributable to the configuration, and
overhead is the per-query latency relative to an undefended victim forward pass
at the same batch size. C3 integrates the change-of-variables \gls{{ODE}} with
{config['likelihood_method']} steps of size {config['likelihood_step_size']} and
the Hutchinson divergence estimator; C1 and C2 share one forward solve of
{config['num_steps']} steps. The indented rows are the two cost levers discussed
in Section~\ref{{sec:discussion}}: a coarser likelihood discretization, and
evaluating C3 for only a fraction of queries.
}}
\label{{tab:runtime_overhead}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}l rrrrr@{{}}}}
\toprule
\textbf{{Configuration}} &
\textbf{{ms/query}} &
\textbf{{queries/s}} &
\textbf{{\gls{{NFE}}/query}} &
\textbf{{Peak mem.\ (GiB)}} &
\textbf{{Overhead}} \\
\midrule

{body}

\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "notebook"
        / "training-victim-cifar10-vgg16_bn-nodefense" / "target_model",
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_flowpure_pgd"
        / "checkpoint_latest.pt",
    )
    parser.add_argument(
        "--likelihood-flow-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "flow_matching" / "cifar10_notebook"
        / "checkpoint_latest.pt",
    )
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "cifar10")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", default="1,8,64")
    parser.add_argument("--selective-fractions", default="0.1,0.25")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--likelihood-step-size", type=float, default=0.05)
    parser.add_argument("--likelihood-method", default="midpoint")
    parser.add_argument(
        "--exact-divergence",
        action="store_true",
        help="Use the exact divergence instead of the Hutchinson estimator.",
    )
    parser.add_argument(
        "--extra-likelihood-steps",
        default="",
        help="Comma-separated extra C3 step sizes to time, e.g. 0.1,0.25.",
    )
    parser.add_argument("--benign-reference-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "runs" / "runtime" / "flowguard_runtime.json",
    )
    parser.add_argument(
        "--latex-out", type=Path, default=None,
        help="Also write a LaTeX table for --latex-batch-size.",
    )
    parser.add_argument("--latex-batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    payload = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] wrote {args.output}")
    if args.latex_out:
        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        args.latex_out.write_text(
            _render_latex(payload, int(args.latex_batch_size)), encoding="utf-8"
        )
        print(f"[bench] wrote {args.latex_out}")


if __name__ == "__main__":
    main()
