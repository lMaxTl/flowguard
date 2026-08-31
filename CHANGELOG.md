# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
