## What this changes

<!-- One paragraph. What problem does this solve? -->

## Type

- [ ] Bug fix
- [ ] New attack
- [ ] New defense
- [ ] Documentation / reproducibility
- [ ] Refactor or tooling

## Reproducibility impact

- [ ] **This changes a number that appears in the paper or README.**
      If checked, say which table/figure, and paste the before/after values.
- [ ] I re-ran `python scripts/audit_paper_table_provenance.py` (only needed if
      the box above is checked).

<!-- Before / after values, if applicable: -->

## Checklist

- [ ] `pytest -q` passes.
- [ ] `ruff check .` passes.
- [ ] `pre-commit run --all-files` passes (notebook outputs stripped, no
      personal paths, no large files).
- [ ] New randomness is seeded and threaded through.
- [ ] I did not rename `modelguard` / `modelguard_w` / `modelguard_s` — those
      identify the USENIX'24 **baseline defense**, not the old package name.
- [ ] I did not reformat `defenses/` or `pretrainedmodels/` (vendored upstream).
- [ ] No datasets, checkpoints, or `runs/` output added.
- [ ] New dependencies are justified below, or none were added.

<!-- Dependency justification, if any: -->
