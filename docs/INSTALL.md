# Installation

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| PyTorch | 2.1 | 2.4+ |
| GPU | none (tests only) | one CUDA GPU, ≥ 24 GB VRAM |
| Disk | 5 GB | 100 GB+ (datasets + run outputs) |

The test suite runs on CPU in about 30 seconds. Everything else — victim
training, CNF training, extraction attacks — needs a GPU. CNF training in
particular is the expensive step: 300 k steps on CIFAR-10 is roughly a day on a
single A100.

## Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate flowguard
```

This installs a CPU-capable PyTorch. For GPU, replace it after creation with the
build matching your driver:

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Option B — pip + venv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel setuptools

# GPU (CUDA 12.1) — pick the index URL matching your driver:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[viz,serve,notebooks,dev]"
```

Installing torch *first* from the right index avoids pip pulling the default
CUDA wheel bundle, which is ~2.5 GB and usually the wrong CUDA version.

### Optional extras

| Extra | Pulls in | Needed for |
|---|---|---|
| `viz` | matplotlib, plotly, umap-learn, scikit-optimize | plots, projections, `scripts/plot_*.py` |
| `serve` | fastapi, uvicorn, pydantic | the black-box HTTP victim |
| `notebooks` | jupyterlab, ipywidgets, pandas | anything under `notebooks/` |
| `diffusion` | diffusers, safetensors | diffusion-based adaptive generators |
| `dev` | pytest, ruff, nbstripout, pre-commit | contributing |
| `all` | all of the above | one-command reproduction environment |

## Verify

```bash
pytest -q                                  # expect: 76 passed
python -c "import flowguard; print(flowguard.__file__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Contributor setup

```bash
pip install -e ".[all]"
pre-commit install
```

`pre-commit install` is required, not optional — it installs the hook that
strips notebook outputs before they reach a commit. See
[../CONTRIBUTING.md](../CONTRIBUTING.md).

## Datasets

Nothing is redistributed in this repository. `dataset.sh` fetches what can be
fetched automatically; the rest you must obtain yourself and place under
`./data/`.

**Automatic** (via `torchvision`, on first use): CIFAR-10, CIFAR-100, SVHN,
MNIST, STL-10.

**Manual:**

| Dataset | Source | Expected location |
|---|---|---|
| Caltech-256 | <https://data.caltech.edu/records/nyy15-4j048> | `data/256_ObjectCategories/` |
| CUB-200-2011 | <https://data.caltech.edu/records/65de6-vp158> | `data/CUB_200_2011/` |
| Tiny ImageNet-200 | <http://cs231n.stanford.edu/tiny-imagenet-200.zip> | `data/tiny-imagenet-200/` |
| Indoor-67 | <http://web.mit.edu/torralba/www/indoor.html> | `data/indoor/` |
| ImageNet-1k | <http://image-net.org/download-images> | `data/ILSVRC2012/` |

Override the root with `--data-path` on any script.

> Several of these datasets are licensed for **non-commercial research only**.
> Their terms are independent of this repository's MIT license and complying
> with them is your responsibility.

## FlowPure reference implementation

FlowPure is a **baseline we compare against**, not a vendored dependency. The
CNF training code needed to reproduce our FlowPure^PGD detector is included here
(`scripts/train_flowpure_pgd.py`, `src/flowguard/flow_matching/`). If you want
the authors' original implementation for comparison, clone it separately —
outside this repository, so it does not end up in your commits:

```bash
git clone https://github.com/deepmancer/FlowPure ../FlowPure
```

(`/FlowPure/` is gitignored for exactly this reason.)

## HPC / SLURM

Every long-running script has a `.sbatch` sibling in `scripts/`. They assume a
project-local `.venv` and bootstrap it via `scripts/hpc_venv_bootstrap.sh`:

```bash
module load devel/python/3.11 devel/cuda/12.1     # adjust to your site
cd $PROJECT_ROOT
source scripts/hpc_venv_bootstrap.sh
hpc_bootstrap_venv "$PROJECT_ROOT"
sbatch scripts/train_flowpure_pgd_cifar10_300k.sbatch
```

Useful overrides:

| Variable | Effect |
|---|---|
| `FORCE_VENV_INSTALL=1` | Reinstall the editable package even if it imports. |
| `SKIP_VENV_INSTALL=1` | Never run pip; fail if `flowguard` is not importable. |
| `RECREATE_VENV_ON_FAILURE=0` | Don't nuke and rebuild `.venv` when pip fails. |

The partition, account, and wall-clock directives in the `.sbatch` headers are
from our cluster. **Edit them for yours** — in particular
`#SBATCH --partition=accelerated` and any `--account` your site requires.

## Troubleshooting

**`ModuleNotFoundError: No module named 'flowguard'`**
The package was renamed from `modelguard` in v1.0.0. Reinstall with
`pip install -e .`, and update imports to `from flowguard... import ...`.

**`pip install -e .` fails on `typing-inspection`**
Known broken-metadata case with some pydantic versions:
`pip install --force-reinstall --no-cache-dir typing-inspection`, then retry.
`hpc_venv_bootstrap.sh` does this automatically.

**CUDA OOM during CNF training**
Lower `--batch-size` (default 64 for FlowPure PGD). Detection quality is not
very batch-size sensitive; wall-clock is.

**`RuntimeWarning: FlowPureQueryDefense loaded a Gaussian-source CNF checkpoint`**
You pointed `--flow-checkpoint` at a plain flow-matching CNF rather than a
FlowPure^PGD/CW one. The `‖v(t=0,x)‖²` signal requires the latter
(x₀ = adversarial, x₁ = clean). Train one with
`scripts/train_flowpure_pgd.py`. Scores from a Gaussian CNF will not separate
anything meaningfully.
