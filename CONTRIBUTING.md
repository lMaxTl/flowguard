# Contributing to FlowGuard++

Thanks for taking the time. This is academic research code, so the bar we care
about most is **reproducibility**: a change that makes a number in a table
irreproducible is worse than a change that is merely inelegant.

## Ways to contribute

| I want to… | Do this |
|---|---|
| Report that a result doesn't reproduce | Open a **Reproduction failure** issue. Include the exact command, commit hash, and hardware. |
| Report a bug | Open a **Bug report** issue with a minimal `ExperimentSpec`. |
| Report that you **broke FlowGuard++** | Open an issue — see [SECURITY.md](SECURITY.md#breaking-the-defense). These are the most valuable reports we get. |
| Add a new attack or defense | Read "Adding an attack/defense" below, then send a PR. |
| Report a security issue in the code itself | See [SECURITY.md](SECURITY.md#reporting). |

## Development setup

```bash
git clone https://github.com/lMaxTl/flowguard.git
cd flowguard
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pre-commit install
pytest -q
```

`pre-commit install` is not optional — it installs the hook that strips notebook
outputs (see below).

## Ground rules

### 1. Never commit notebook outputs

Committed cell outputs bloat the repository, produce unreadable diffs, and have
previously leaked local filesystem paths containing real names. The
`nbstripout` pre-commit hook clears them automatically. If you have to bypass
the hooks for some reason, run `nbstripout notebooks/**/*.ipynb` before
committing.

### 2. Never commit experiment outputs, datasets, or checkpoints

`runs/`, `data/`, `paper/`, `*.pt`, `*.ckpt`, `*.pkl` are all gitignored. If you
find yourself using `git add -f` on one of these, stop — publish the artifact
somewhere with a DOI (Zenodo, HuggingFace) and reference it instead.

### 3. Don't rename the ModelGuard baseline

`modelguard`, `modelguard_w`, `modelguard_s` refer to the **ModelGuard defense**
from USENIX Security 2024, which FlowGuard++ is compared against. They are not
leftovers from the old package name. Renaming them silently breaks the mapping
between this code and published results.

### 4. `defenses/` and `pretrainedmodels/` are vendored — leave them alone

These are near-verbatim copies of upstream code (see [NOTICE.md](NOTICE.md)),
kept so the USENIX'24 baselines stay byte-comparable to their published form.
They are excluded from `ruff`. Please don't reformat, modernize, or "clean up"
these directories. Bug fixes that change behavior need a comment explaining why
the deviation from upstream is necessary.

### 5. Determinism

Anything that consumes randomness must accept and thread through a seed. If you
add a code path that can't be made deterministic (non-deterministic CUDA kernels,
for example), say so in the docstring.

## Adding an attack

1. Implement an `AttackRunner` subclass in `src/flowguard/attacks/`.
2. Add the kind to `AttackKind` in `src/flowguard/experiments/spec.py`.
3. Register it in `src/flowguard/attacks/registry.py`.
4. Add a mode builder in `src/flowguard/attacks/modes.py`.
5. List it in `SUPPORTED_ATTACKS` / `SUPPORTED_MODES` in
   `src/flowguard/experiments/catalog.py`.
6. Add a test in `tests/` that runs it end-to-end at a tiny query budget.

## Adding a query defense

1. Subclass `QueryDefense` in `src/flowguard/defenses/query/`, set a unique
   `name`, and support `audit_only=True` — evaluation depends on being able to
   harvest scores without enforcing.
2. Write per-query scores into `context.metadata` under a `<name>_*` key so the
   detection-metrics code can pick them up.
3. Wire it into `_build_query_defenses` in
   `src/flowguard/orchestration/runner.py`.
4. Add a test in `tests/test_query_defenses.py`.

## Pull requests

- Branch from `main`; keep PRs focused on one thing.
- `pytest -q` and `ruff check .` must pass. CI runs both.
- If your change affects a number that appears in the paper, say so explicitly
  in the PR description and re-run
  `scripts/audit_paper_table_provenance.py`.
- New dependencies need a justification. The install is already heavy.

## Code style

`ruff` with the config in `pyproject.toml` (line length 100). Type hints on
public functions. Docstrings should explain *why*, not restate the signature.

## Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
