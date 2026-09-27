<h1 align="center">FlowGuard++</h1>

<p align="center">
  <strong>Defenses against distributed model stealing beyond the single-client assumption —<br/>
  with a reproducible attack/defense benchmark.</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#reproducing-the-paper">Reproducing</a> ·
  <a href="#whats-in-the-box">What's in the Box</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status: research code" src="https://img.shields.io/badge/status-research%20code-orange">
</p>

---

## What this is

**FlowGuard++** is a query-level defense against black-box model-extraction
attacks on machine-learning-as-a-service (MLaaS) APIs. It scores each incoming
query with a continuous normalizing flow (CNF) trained by flow matching, and
fuses that per-query signal with *stateful* signals computed over a client's
query stream. The intuition: a single extraction query can be crafted to look
in-distribution, but a *stream* of tens of thousands of them is much harder to
disguise.

FlowGuard++ combines four signals:

| Signal | Class | What it catches |
|---|---|---|
| FlowPure $\lVert v(t{=}0,x)\rVert^2$ | `FlowPureQueryDefense` | Per-query off-manifold inputs (the published baseline) |
| Trajectory integral $\sum_k \lVert v(t_k, x_{t_k})\rVert^2 \mathrm{d}t$ | `FlowGuardIntegralDefense` | Attacks that minimize the score only at $t=0$ |
| Per-user KS test over a sliding score window | `FlowGuardUserLevelDefense` | Low-rate / sparse-probe attackers |
| MMD between the client's label histogram and a benign reference | `FlowGuardLabelHistogramDefense` | Label-space sweeps typical of extraction |
| **Fused** | `FlowGuardCompositeDefense` (`flowguard++`) | All of the above, under one calibrated threshold |

This repository is **also a benchmark**. To evaluate the defense honestly we
implement the attacks that target it, including adaptive attackers with white-box
access to the detector. Everything is driven from a single declarative
`ExperimentSpec`, so an attack × defense grid is one function call.

> [!IMPORTANT]
> **Responsible use.** This repository contains working implementations of model
> extraction attacks. They are published so that defenses can be evaluated
> reproducibly, which is standard practice in security research. Use them only
> against models you own or have written authorization to test. See
> [SECURITY.md](SECURITY.md).

### Lineage

FlowGuard++ began as a fork of the official
[ModelGuard](https://github.com/Yoruko-Tang/ModelGuard) release (USENIX Security
2024) and kept its experimental harness so that the earlier results stay
reproducible here. **The ModelGuard *defense* is one of our baselines** and keeps
its original name throughout the code (`modelguard`, `modelguard_w`,
`modelguard_s`). Only the framework/package was renamed. Full attribution is in
[NOTICE.md](NOTICE.md).

---

## Installation

Requires Python ≥ 3.10 and, for anything beyond the smoke tests, a CUDA GPU.

**conda (recommended — pins the BLAS/PyTorch stack):**

```bash
conda env create -f environment.yml
conda activate flowguard
```

**pip:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[viz,serve,notebooks,dev]"
```

Verify:

```bash
pytest -q                          # 76 tests, ~30 s on CPU
python -c "import flowguard; print(flowguard.__file__)"
```

Optional extras: `viz` (plots/projections), `serve` (FastAPI black-box server),
`notebooks` (JupyterLab), `diffusion` (diffusers-based adaptive generators),
`dev` (pytest/ruff/pre-commit), `all`. See [docs/INSTALL.md](docs/INSTALL.md) for
GPU builds, HPC/SLURM setup, and dataset preparation.

### Datasets

CIFAR-10, CIFAR-100, SVHN and MNIST download automatically via `torchvision`.
The rest must be placed in `./data/` yourself — none are redistributed here, and
several are non-commercial-research-only:

| Dataset | Source |
|---|---|
| Caltech-256 | <https://data.caltech.edu/records/nyy15-4j048> |
| CUB-200 | <https://data.caltech.edu/records/65de6-vp158> |
| Tiny ImageNet-200 | <http://cs231n.stanford.edu/tiny-imagenet-200.zip> |
| Indoor-67 | <http://web.mit.edu/torralba/www/indoor.html> |
| ImageNet-1k | <http://image-net.org/download-images> |

`bash dataset.sh` automates the downloadable subset.

---

## Quick Start

### 1. Run one attack against one defense

```python
from flowguard.experiments.factory import build_experiment_spec
from flowguard.experiments.spec import AttackKind
from flowguard.orchestration.runner import run_experiment

spec = build_experiment_spec(
    name="cifar10-knockoff-undefended",
    dataset="CIFAR10",
    target_architecture="resnet18",
    attack_kind=AttackKind.TRANSFER_SET,
    attack_mode="naive",
    query_dataset="TinyImageNet200",
    query_budget=50_000,
    prediction_defense="none",
)

result = run_experiment(spec, output_dir="runs/quickstart")
print(result.evaluation_summary.metrics)   # substitute accuracy, fidelity, ...
```

### 2. Sweep a whole attack × defense grid

```python
from flowguard.experiments.factory import build_experiment_matrix
from flowguard.orchestration.runner import run_experiment_suite

specs = build_experiment_matrix(
    base_name="cifar10-sweep",
    dataset="CIFAR10",
    target_architecture="resnet18",
    query_dataset="TinyImageNet200",
    query_budget=50_000,
    attack_modes=["naive", "top1", "ddae"],
    defenses=["none", "reverse_sigmoid", "modelguard"],
)

results = run_experiment_suite(specs, output_root="runs/sweep")
```

Suites are **idempotent**: a cell is skipped if its metrics JSON already exists,
so an interrupted 48-hour job resumes by relaunching the same command.

### 3. Serve a defended model over HTTP

```bash
uvicorn flowguard.api.app:app --host 0.0.0.0 --port 8000
```

Gives you a genuinely black-box target: the attack code on the other side sees
only the defended API responses.

---

## Reproducing the paper

Full, ordered instructions — including which checkpoints each step needs and the
expected wall-clock cost — are in **[docs/REPRODUCING.md](docs/REPRODUCING.md)**.
The detector-tuning details live in
[docs/flowguard_evaluation_runbook.md](docs/flowguard_evaluation_runbook.md).

The four-dataset benchmark (CIFAR10/VGG19, GTSRB/MobileNetV2, CelebA/DenseNet-121,
Skin Cancer/ResNet-50) runs on a Slurm cluster with one command; see
**[docs/MULTI_DATASET_EXPERIMENTS.md](docs/MULTI_DATASET_EXPERIMENTS.md)**.

The short version, after training a victim and the two CNFs:

```bash
# Detection quality across attacks × query defenses
python scripts/evaluate_detection_suite.py \
  --name detection-suite --dataset CIFAR10 --target-architecture resnet18 \
  --query-budget 120 --benign-queries 160 --epochs 1

# End-to-end extraction against an *enforcing* FlowGuard++
python scripts/evaluate_active_flowguard_extraction.py \
  --run-label active-fgpp --output-root runs/active_flowguard \
  --attacks flowguard_d1_latent_manifold,flowguard_d5_adaptive_composite \
  --query-budget 1000000 --benign-query-budget 50000 \
  --target-checkpoint-dir <victim> \
  --flow-checkpoint <flowpure_pgd.pt> \
  --likelihood-flow-checkpoint <likelihood_cnf.pt>

# What the monitor costs to run (latency / throughput / NFE / peak memory)
python scripts/benchmark_flowguard_runtime.py \
  --batch-sizes 1,8,32 --selective-fractions 0.05,0.1,0.25 \
  --output runs/runtime/flowguard_runtime.json
```

Every script has a SLURM sibling (`scripts/*.sbatch`) for cluster runs.

**Provenance check.** `scripts/audit_paper_table_provenance.py` verifies that
each number in the paper's tables came from a run with the query budget its
caption claims, for the attack/defense pair it is printed under:

```bash
python scripts/audit_paper_table_provenance.py --markdown docs/table_provenance.md
```

---

## What's in the box

### Attacks

| Kind | Modes | Origin |
|---|---|---|
| `transfer_set` | `naive`, `top1`, `s4l`, `smoothing`, `ddae`, `ddae+`, `bayes` | Knockoff Nets (CVPR'19) + defense-aware variants |
| `jacobian` | `jbtr` | JBDA-TR |
| `prada` | `prada` | PRADA (EuroS&P'19) |
| `maze` | `maze` | MAZE (CVPR'21), data-free |
| `disguide` | `disguide` | DisGUIDE (AAAI'23), data-free |
| `latent_manifold` | `latent_manifold` | **Adaptive**: CEM search in CNF latent space |
| `procedural_natural` | `procedural_natural` | **Adaptive**: procedural natural-image statistics |
| `rejection_oracle` | `rejection_oracle` | **Adaptive**: learns only from accepted queries |

The last three are attacks *we* introduce against FlowGuard++, with white-box
access to the detector. A defense paper that only evaluates against pre-existing
attacks is not evaluating anything.

### Prediction-perturbation defenses

`none`, `reverse_sigmoid`, `mad`, `adaptive_misinformation`, `modelguard`
(= `modelguard_w`), `modelguard_s` (= `quantization`), `random_noise`.

### Query-level defenses

`noop`, `budgeting`, `prada`, `fdinet`, `flow_matching`, `flowpure`,
`flowguard_integral`, `flowguard_userlevel`, `flowguard_labelhist`, and the
fused **`flowguard++`**.

Set `audit_only=True` on any query defense to harvest detection scores without
actually rejecting queries — this is what you want for ROC/AUROC evaluation.
`response_policy` controls what a flagged client receives when enforcement is on.

---

## Repository layout

```
flowguard/
├── src/flowguard/          # The framework (importable package)
│   ├── attacks/            #   Attack runners + adaptive bypass attacks
│   ├── defenses/
│   │   ├── prediction/     #   Output-perturbation defenses (incl. ModelGuard baseline)
│   │   └── query/          #   Query-stream defenses (incl. FlowGuard++)
│   ├── flow_matching/      #   CNF training (FlowPure PGD/patch/boundary variants)
│   ├── experiments/        #   ExperimentSpec, catalog, factory
│   ├── orchestration/      #   run_experiment / run_experiment_suite
│   ├── evaluation/         #   Accuracy, fidelity, detection metrics
│   ├── training/           #   Victim & substitute training
│   ├── serving/ querying/  #   Black-box target service + query engine
│   ├── api/                #   FastAPI front-end
│   └── distributed/        #   Multi-worker / multi-client simulation
├── scripts/                # Reproduction entry points (+ .sbatch siblings)
├── notebooks/              # Interactive analyses (outputs stripped — see CONTRIBUTING)
├── tests/                  # pytest suite
├── docs/                   # Install, reproduction, runbook, table provenance
├── defenses/               # Vendored upstream ModelGuard code (USENIX'24 baselines)
└── pretrainedmodels/       # Vendored Cadene/pretrained-models.pytorch (BSD-3)
```

Two directories are **deliberately untouched vendored code**: `defenses/` and
`pretrainedmodels/`. They are excluded from linting so the USENIX'24 baselines
stay byte-comparable to their published form.

Not in the repository, by design: `runs/` (experiment outputs, tens of GB),
`data/` (datasets), `paper/` (manuscript sources), model checkpoints. See
[docs/REPRODUCING.md](docs/REPRODUCING.md) for how to regenerate them.

---

## Citation

If you use FlowGuard++, please cite:

```bibtex
@article{flowguardpp,
  title   = {FlowGuard++: Defenses Against Distributed Model Stealing Beyond the Single-Client Assumption},
  author  = {Schwarzer, Maxime and Holz, Laurin and Lopes, Roberto Rigolin F. and Loevenich, Johannes F. F. and Moehlenhof, Thies and Hagenmeyer, Veit},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  year    = {2026},
  note    = {https://github.com/lMaxTl/flowguard}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

If you use the ModelGuard defense, the extraction attacks, or the FlowPure
baseline as *baselines*, please also cite their original papers — the full list
with references is in [NOTICE.md](NOTICE.md). In particular:

```bibtex
@inproceedings{tang2024modelguard,
  title     = {ModelGuard: Information-Theoretic Defense Against Model Extraction Attacks},
  author    = {Tang, Minxue and Shejwalkar, Virat and Houmansadr, Amir},
  booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
  year      = {2024}
}
```

---

## Contributing

Bug reports, reproduction failures and new attack implementations are welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md). If you break FlowGuard++, we want to know:
open an issue with the attack configuration and we will add it to the benchmark.

## License

MIT — see [LICENSE](LICENSE). Third-party components and their licenses are
listed in [NOTICE.md](NOTICE.md). Datasets are **not** covered by this license
and carry their own terms.
