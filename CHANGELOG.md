# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Multi-dataset benchmark at 32x32 (FDINet protocol): CelebA (gender),
  SkinCancer (ISIC 2018 Task 3), attacker pools LFW/LFW10, BelgiumTS, TSRD,
  BCN20000, all cached as uint8 arrays (`defenses/datasets/cached32.py`).
- `resnet50_cifar`, `densenet121_cifar`, `mobilenetv2_cifar` (torchvision
  architectures with a 32x32 stem).
- Held-out benign streams `<Dataset>BenignCal` / `<Dataset>BenignEval` and the
  `--benign-calibration-dataset` / `--benign-eval-dataset` driver options.
- `--sybil-variants`: stateful detectors are also scored as `<detector>@N<k>`
  on the same stream spread over *k* identities; `fdinet_vote` column (FDINet's
  client-level vote over *bs* queries).
- `scripts/train_diffusion_prior.py` (attacker DDPM prior on public data),
  `scripts/prepare_datasets.py`, `scripts/horeka/*` and
  `scripts/schedule_experiments.py` with `configs/horeka_experiments.json`.
- `--seed`, `--attacker-calibration-cache`, `--attacker-calibration-only`,
  learned-weight composite in `evaluate_parallel_defenses.py`.

### Fixed

- `MultiAuditQueryDefense` (used by `evaluate_parallel_defenses.py`) wrote every
  batch's detector outputs into one shared dict, so all history records showed
  the last batch's scores and the driver's AUROC/F1 were computed from a single
  batch per side. Each batch now gets its own namespace.

### Changed

- Sybil identities rotate per query (`--sybil-granularity batch` restores the
  per-batch rotation).
- PRADA keeps one detector per identity; C4/C5 and the composite windows are
  kept and scored per identity.
- The composite's fused score is `[z_t0]_+ + [z_int]_+ + s_typ` (C2 now has a
  benign reference; negative velocity z-scores no longer cancel typicality). The
  attacker's replica of the fused score uses the same rule.
- Flow detectors de-normalize with the dataset registry's statistics instead of
  guessing from the dataset name (GTSRB was treated as ImageNet-normalized).
- GTSRB is cached like CIFAR10 and has its own model family without horizontal
  flips; the transfer-set path bug (`samples` held bare path strings) is gone.
- FlowPure-PGD and Gaussian flow-matching training resume from
  `checkpoint_latest.pt`, accept any registered 32x32 dataset, support step
  budgets, and write `DONE.json`.
- `evaluate_attack_defense_combined.sbatch` passes the Gaussian likelihood CNF.

## [1.0.0] — 2026-08-31

First public release. The repository history was restarted at this commit: the
pre-release history was development-internal, contained cell outputs with local
filesystem paths, and is not part of the published record.

### Added

- **FlowGuard++ composite query defense** (`flowguard++`), fusing a per-query
  FlowPure velocity score with three stateful signals:
  - `flowguard_integral` — trajectory-integral velocity score along a reverse
    ODE solve, defeating attacks that only minimize the score at *t*=0.
  - `flowguard_userlevel` — sliding-window Kolmogorov–Smirnov test per client.
  - `flowguard_labelhist` — MMD between a client's label histogram and a benign
    reference.
- **Adaptive attacks** developed against the defense with white-box detector
  access: `latent_manifold`, `procedural_natural`, `rejection_oracle`.
- **Detection benchmark** (`scripts/evaluate_detection_suite.py`) reporting
  detection rate, TPR, FPR, precision, recall, F1, F1-macro and ROC-AUC across
  attack × defense grids.
- **End-to-end enforcement evaluation**
  (`scripts/evaluate_active_flowguard_extraction.py`): calibrates the fused
  threshold on benign traffic, measures benign rejection, then runs the attack
  with flagged queries receiving uninformative responses.
- **Runtime cost benchmark** (`scripts/benchmark_flowguard_runtime.py`):
  per-query latency, throughput, observed NFE, peak device memory, and overhead
  relative to an undefended forward pass.
- **Table provenance audit** (`scripts/audit_paper_table_provenance.py`),
  verifying every published number came from a run at the budget its caption
  claims.
- FastAPI black-box serving front-end, distributed multi-client query
  simulation, and a resumable experiment-suite runner.
- Community and reproducibility scaffolding: `CONTRIBUTING.md`, `SECURITY.md`
  (responsible-use policy), `CODE_OF_CONDUCT.md`, `CITATION.cff`, `NOTICE.md`,
  GitHub Actions CI, and pre-commit hooks including mandatory `nbstripout`.

### Changed

- **Renamed the framework package `modelguard` → `flowguard`.** All imports are
  now `from flowguard... import ...`.
  - The **ModelGuard defense** keeps its original identifiers (`modelguard`,
    `modelguard_w`, `modelguard_s`). It is a *baseline*, not the framework, and
    renaming it would break the mapping to the USENIX'24 published results.
- Consolidated dependency declaration into `pyproject.toml` with optional
  extras (`viz`, `serve`, `notebooks`, `diffusion`, `dev`, `all`). `Pipfile` and
  `requirements-dev.txt` were removed as redundant and mutually inconsistent.
- `environment.yml` now creates an environment named `flowguard` and defers to
  `pyproject.toml` for pip-only dependencies.

### Fixed

- `tests/test_adaptive_attacks.py`: the `_StubRunner` fixture lacked
  `_likelihood_regularizer`, which the composite detector-loss path reads,
  causing an `AttributeError` in the D6 preset test.
- `tests/test_flow_matching.py`: the expected dataset-preset list had drifted
  behind the implementation (missing `cifar_patch8` and `mnist`).
- `tests/test_framework_smoke.py`: the `_MockUNet` fixtures did not expose
  `.config.in_channels`, which the diffusion generator reads to size latents.

### Removed

- Committed notebook outputs (~6 MB), which included local filesystem paths
  containing a real name.
- Internal planning documents (`docs/tdsc_revision_plan.md`,
  `docs/autoresearch_flowpure_low_detection_plan.md`) — working notes on the
  peer-review process, not part of the artifact.
- A broken `FlowPure/FlowPure` gitlink with no corresponding `.gitmodules`.
  FlowPure is referenced, not vendored; see `docs/INSTALL.md`.
