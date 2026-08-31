# FlowGuard++ Evaluation Runbook

Describes
how to empirically evaluate the paper-faithful `FlowPure^PGD` detector and
the proposed `FlowGuard++` composite defense against the five attack
candidates (A1-A5) plus an *adaptive-adaptive* attack that targets
`FlowGuard++` directly.

The actual results tables (AUROC / FPR / detection rate) are emitted
automatically by `scripts/evaluate_flowguard_suite.py` as
`runs/notebook/flowguard_suite/flowguard_suite_report.md` and should be
committed alongside this runbook after every run.

## Prerequisites

1. **Trained victim checkpoint** at
   `runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model/`
   (created by `python scripts/train_victim.py ...`).
2. **Deployed-defense CNF**:
   `runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt`
   (from `scripts/train_flowpure_pgd.py`).
3. **Attacker-surrogate CNF** for A3 / adaptive-adaptive; in the strongest
   white-box-vs-defense threat model this is the same CNF, but for
   transferability experiments train a second CNF on CIFAR-100 / STL-10 /
   TinyImageNet and point `--a3-surrogate-checkpoint` at it. See
   `scripts/measure_cnf_transferability.py` for the Pearson-correlation
   sanity check.

## Defense axis

| ID  | Name                       | Repo class                         | Addresses |
| --- | -------------------------- | ---------------------------------- | --------- |
| -   | `flowpure` (paper)         | `FlowPureQueryDefense`             | baseline  |
| C1  | `flowguard_integral`       | `FlowGuardIntegralDefense`         | W1        |
| C3  | `flowguard_userlevel`      | `FlowGuardUserLevelDefense`        | W6        |
| C4  | `flowguard_labelhist`      | `FlowGuardLabelHistogramDefense`   | W7, W8    |

C4 scores require the victim's top-1 labels, which the unified suite
already logs; post-hoc MMD scoring is done in
`scripts/run_labelmaze_attack.py`.

## Attack axis

| ID  | Name                | Key                | Type          |
| --- | ------------------- | ------------------ | ------------- |
| -   | MAZE (baseline)     | `maze_baseline`    | in-line       |
| A1  | Clean-Transfer      | `a1_clean_transfer`| in-line       |
| A2  | ProjectedMAZE       | `a2_projected_maze`| in-line       |
| A3  | FlowBlind           | `a3_flowblind`     | in-line       |
| A4  | Sparse-Probe        | -                  | post-hoc      |
| A5  | LabelMaze           | -                  | post-hoc      |
| -   | Adaptive-Adaptive   | `adaptive_adaptive`| in-line       |

Post-hoc attacks are analyzed on the outputs of existing in-line runs
using `scripts/analyze_sparse_probe.py` (A4) and
`scripts/run_labelmaze_attack.py` (A5).

## Running the full grid

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_flowguard_suite.py `
    --output-root runs\notebook\flowguard_suite `
    --defenses flowpure,flowguard_integral,flowguard_userlevel `
    --attacks  a1_clean_transfer,a2_projected_maze,a3_flowblind,adaptive_adaptive,maze_baseline `
    --dataset CIFAR10 `
    --device cuda `
    --attack-query-budget 2048 `
    --benign-query-budget 2048 `
    --target-fpr 0.05
```

The driver is idempotent: a (defense, attack) pair is skipped if its
`attack_suite_metrics.json` already exists, so interrupted runs can be
resumed by re-launching the same command. Use `--force` to overwrite.

For the pilot-scale smoke grid, lower the query budget (e.g. 144) to
verify wiring without the full CIFAR-10 GPU cost.

## Running A4 / A5 (post-hoc)

After the in-line runs finish, reuse their detection samples:

```powershell
# A4 - Sparse-Probe: simulate 95% benign + 5% A3 probes per user
.\.venv\Scripts\python.exe scripts\analyze_sparse_probe.py `
    --benign-dir  runs\notebook\flowguard_suite\flowpure__maze_baseline\benign_reference__flowpure `
    --attack-dir  runs\notebook\flowguard_suite\flowpure__a3_flowblind\attack-a3_flowblind__flowpure `
    --probe-rate 0.05 --user-window 256 `
    --output runs\notebook\flowguard_suite\a4_sparse_probe_report.json

# A5 - LabelMaze: entropy-ranked active learning over clean candidates
.\.venv\Scripts\python.exe scripts\run_labelmaze_attack.py `
    --transferset-dir  runs\notebook\flowguard_suite\flowpure__a1_clean_transfer\attack-a1_clean_transfer__flowpure `
    --surrogate-checkpoint runs\flow_matching\cifar10_flowpure_pgd\checkpoint_latest.pt `
    --top-k 1024 `
    --output runs\notebook\flowguard_suite\a5_labelmaze_report.json
```

## Expected directional outcomes (from plan)

The qualitative hypotheses the grid should falsify or confirm:

- **A1 / A5** should *not* be detectable by `flowpure` (AUROC ~= FPR) but
  *should* be caught by `flowguard_userlevel` (C3) and
  `flowguard_labelhist` (C4) because the label distribution over the
  entire user submission is visibly stealing-shaped.
- **A2** should drop `flowpure` AUROC from ~1.0 toward 0.6-0.7 while only
  losing 3-5pp substitute accuracy; `flowguard_integral` (C1) should
  partially recover detection because denoising does not remove the
  off-manifold geometry along the full ODE trajectory.
- **A3** (FlowBlind) should push `flowpure` AUROC to 0.5-0.6 once
  `lambda` is large enough; `flowguard_integral` stays higher because
  the surrogate regularizer only controls `t=0`.
- **Adaptive-Adaptive** should be the hardest case - AUROC below 0.7 on
  *every* single-component defense but recovered to >0.85 by the union
  (C1 OR C3 OR C4 flagged). See
  `flowguard_suite_report.md#aur oc-by-defense-attack` for the empirical
  numbers.

## FlowGuard++-adaptive attackers (D3-D6)

D1 and D2 are adaptive to *single-point flow-velocity detection*: their
generator loss penalizes `||v(0, x)||^2` and nothing else. Evaluating
FlowGuard++ only against them measures whether it closes the gap it was
designed to close, not whether it survives an attacker aimed at itself. D3-D6
close that evaluation gap.

| Key | Adaptive to | Generator penalty |
| --- | --- | --- |
| `flowguard_d3_adaptive_c1c2` | C1+C2 | `t=0` velocity + trajectory integral |
| `flowguard_d4_adaptive_c1c3` | C1+C3 | `t=0` velocity + two-sided likelihood typicality |
| `flowguard_d5_adaptive_composite` | fused risk | the attacker's replica of `R(x)` |
| `flowguard_d6_adaptive_stateful` | C1-C5 | D5 + batch score-distribution + label/entropy match |

All four use the D2 procedural generator.

### Attacker-side calibration

Before the first D3-D6 cell, the harness calibrates the attacker's own benign
statistics: `(mu, sigma)` and a 95th-percentile band per component, computed on
`--adaptive-attacker-pool` (default CIFAR-100) with the attacker's surrogate
CNF. The result is cached at
`<output-root>/attacker_surrogate_calibration.json` and reused across cells.
This pool must stay disjoint from the defender's calibration set, otherwise the
attacker is being handed the defender's operating point.

### Cost and the two gradient routes

The likelihood term is the expensive one: its gradient requires differentiating
a divergence, i.e. one double-backward per ODE step. Measured on an RTX 4060
with the CIFAR-10 CNF (peak allocated, model itself is 0.22 GB):

| batch \ Euler steps | 2 | 4 |
| ---: | ---: | ---: |
| 2 | 1.74 GB | 3.23 GB |
| 4 | 3.24 GB | 6.23 GB |
| 8 | 6.21 GB | 12.15 GB |
| 16 | 12.18 GB | 24.05 GB |

Memory is linear in the product: **~0.38 GB per (image x Euler step)**. Size the
generator batch from that. On an A100-40GB, keeping ~10% headroom means

    EXTRACTION_ATTACK_BATCH_SIZE * ADAPTIVE_LIKELIHOOD_STEPS <= ~90

so `batch=16, steps=4` (~24 GB) or `batch=32, steps=2` (~24 GB) both fit, while
the **default `batch=32, steps=4` needs ~48 GB and will OOM**. Set the batch
explicitly for any run including D4, D5, or D6; D3 has no likelihood term and
can keep the default.

For high-budget runs the exact route is not viable at all: 20M queries at
batch 32 is ~625k generator steps, each carrying a double-backward. Set
`ADAPTIVE_DISTILL_SURROGATE=1` there, which distills the component scores into a
small CNN and gives composite gradients at one forward/backward per step.
Check `attacker_distilled_score.report.json` before trusting the result: if the
per-component correlation is low, the attack is optimizing noise rather than
the defender's score, and the exact route should be used instead.

Scoring (not optimizing) the likelihood is far cheaper because it needs no
second-order graph; that is what `calibrate_surrogate_stats` and the
distillation-target builder use.

### Running

**Do not pass the attack list via `--export`.** Slurm splits `--export` on
commas, so `--export=ALL,ATTACKS=a,b,c` is parsed as `ATTACKS=a` plus two
undefined variable names. Set the variables in the submitting shell instead;
sbatch defaults to `--export=ALL` and propagates them.

```bash
ATTACKS="flowguard_d1_latent_manifold,flowguard_d2_procedural_natural,flowguard_d3_adaptive_c1c2,flowguard_d4_adaptive_c1c3,flowguard_d5_adaptive_composite,flowguard_d6_adaptive_stateful" \
DEFENSES="flowpure,flowguard_integral,flow_matching,flowguard_userlevel,flowguard_composite" \
SYBIL_IDENTITIES=1250 \
EXTRACTION_ATTACK_BATCH_SIZE=16 \
ADAPTIVE_LIKELIHOOD_STEPS=4 \
RUN_LABEL=adaptive-dist \
OUTPUT_ROOT=runs/adaptive_dist \
VERBOSE=1 \
  sbatch scripts/evaluate_attack_defense_combined.sbatch
```

`flow_matching` is **not** in the default `DEFENSES` list but is the C3
(likelihood) column of Table V, so it must be named explicitly or that column
comes back empty.

Then rebuild the paper table:

```bash
python scripts/build_fully_adaptive_table.py \
    --summary runs/adaptive_dist/attack_defense_matrix_adaptive-dist_summary.json \
    --output paper/tables/fully_adaptive_results.tex
```

The script prints which `(attack, defense)` cells are still missing and leaves
them as `\pending` (a red `??` in the PDF) rather than dropping them, so a
partially finished sweep is visible in the paper instead of looking complete.

### Do the adaptive attacks still steal anything?

Detection AUROC does not answer this. Every penalty added to Eq. 15 constrains
the generator, and several fight the extraction objective directly, so a D4-D6
attack could show low AUROC simply because it stopped producing informative
queries. Two runs separate the cases; both report `accuracy` (substitute test
accuracy) and `fidelity` (agreement with the victim) per cell.

**1. Extraction ceiling** - `AUDIT_ONLY=1` scores queries but never blocks, so
the substitute quality reflects only the generator constraint. This matches the
protocol behind Table IV, so the numbers are directly comparable to D1's 71.6%
and D2's 74.3% at 20M queries. If an attack lands near the blocked-DisGUIDE
floor (10.2%) here, it has self-destructed and its low AUROC means nothing.

```bash
ATTACKS="flowguard_d3_adaptive_c1c2,flowguard_d4_adaptive_c1c3,flowguard_d5_adaptive_composite,flowguard_d6_adaptive_stateful" \
DEFENSES="flowpure" \
AUDIT_ONLY=1 \
QUERY_BUDGET=20000000 \
BENIGN_QUERY_BUDGET=50000 \
ATTACK_ARTIFACT_SAMPLE_SIZE=50000 \
ATTACK_CHECKPOINT_EVERY_QUERIES=5000000 \
EPOCHS=50 \
ADAPTIVE_DISTILL_SURROGATE=1 \
SYBIL_IDENTITIES=1 \
RUN_LABEL=adaptive-ceiling \
OUTPUT_ROOT=runs/adaptive_ceiling \
  sbatch --time=48:00:00 scripts/evaluate_attack_defense_combined.sbatch
```

**2. Against an enforcing FlowGuard++** - `AUDIT_ONLY=0` makes the composite
actually reject, so substitute quality reflects detection *and* the generator
constraint. The budget is cut to 1M because the composite runs a likelihood ODE
per query, which is not affordable at 20M; note the reduced budget when
reporting, and compare against the ceiling run rather than against Table IV.

```bash
ATTACKS="flowguard_d3_adaptive_c1c2,flowguard_d4_adaptive_c1c3,flowguard_d5_adaptive_composite,flowguard_d6_adaptive_stateful" \
DEFENSES="flowguard_composite" \
AUDIT_ONLY=0 \
FLOWGUARDPP_RESPONSE_POLICY=reject \
QUERY_BUDGET=1000000 \
BENIGN_QUERY_BUDGET=50000 \
ATTACK_ARTIFACT_SAMPLE_SIZE=50000 \
ATTACK_CHECKPOINT_EVERY_QUERIES=5000000 \
EPOCHS=50 \
ADAPTIVE_DISTILL_SURROGATE=1 \
SYBIL_IDENTITIES=1250 \
RUN_LABEL=adaptive-enforcing \
OUTPUT_ROOT=runs/adaptive_enforcing \
  sbatch --time=48:00:00 scripts/evaluate_attack_defense_combined.sbatch
```

Read the pair together: high ceiling + low enforcing accuracy means FlowGuard++
stopped a working attack; low in both means the adaptive penalties broke the
attack on their own; high in both means the attack genuinely evades.

### Surviving the 48h wall clock

The matrix harness resumes at `(defense, attack)` granularity only. That is
useless for a high-budget cell: if one cell cannot finish inside the allocation,
restarting reruns it from zero and it never completes. Two settings fix this.

`ATTACK_ARTIFACT_SAMPLE_SIZE` is mandatory in `full` mode regardless of
wall clock. It defaults to the entire query budget, and each record holds the
query, a copy of it, and two label vectors (~24.6 KB for CIFAR-10), so a
20M-query run would need ~490 GB of host RAM against a 128 GB job. It OOMs
around 5M queries. Detection and substitute training still use the full budget;
only the retained record set is capped.

`ATTACK_CHECKPOINT_EVERY_QUERIES=5000000` snapshots generator, ensemble, both
optimizers, both LR schedulers, the replay buffer, the retained records, the
metrics history, and the spent-query count to
`<output-root>/_scratch/<defense>__<attack>/attack_checkpoint.pt`. The write goes
to a `.tmp` file and is renamed, so a job killed mid-write leaves the previous
checkpoint intact rather than a truncated one. On restart the runner reloads it
and continues: restoring the spent-query count is what makes the budget continue
rather than restart, so a 20M cell resumed at 5M runs the remaining 15M and
stops at 20M total. The checkpoint is deleted once the cell finishes, since it
can be large and would otherwise make a re-submission resume a completed attack.

Slurm's SIGKILL at the wall clock does not run Python's cleanup handlers, so the
scratch directory and its checkpoint survive the kill. The per-cell directory
name is deterministic (`<defense>__<attack>`), so a resubmission with the same
`OUTPUT_ROOT` finds it.

To restart, generate the resume file and resubmit the identical command with
`RESUME_FILE` added:

```bash
python scripts/make_resume_file.py \
    --summary runs/adaptive_ceiling/attack_defense_matrix_adaptive-ceiling_summary.json \
    --output runs/adaptive_ceiling/resume.json
```

It prints how many cells are done, which cell is next, and whether that cell has
an attack checkpoint to continue from. Then resubmit with
`RESUME_FILE=runs/adaptive_ceiling/resume.json` added to the same variable list.
Repeat until it reports every cell complete.

Two caveats. Checkpointing is implemented for the DisGUIDE-loop attacks (D1-D6)
only; PRADA, MAZE, and the transfer attacks still restart their cell from
scratch, which is fine because none of them approach the wall clock at these
budgets. And a checkpoint taken mid-epoch resumes by re-entering that epoch from
its start, so up to one epoch of iterations is repeated; the query budget, not
the epoch counter, decides when the run stops, so this costs time rather than
correctness. Set `ATTACK_NO_RESUME=1` to force a clean retrain.

### Reading the result

AUROC alone does not decide whether an adaptive attacker won. Each extra
penalty constrains the generator, and several terms fight the extraction
objective directly: the typicality term pulls queries toward the benign
density while the extraction loss rewards atypical, boundary-adjacent queries,
and the label term is in explicit opposition to DisGUIDE's diversity term.
Read the substitute-accuracy column jointly with AUROC. An attack that evades
detection but yields a substitute near the blocked-DisGUIDE floor (10.2%) has
not evaded anything worth having; one that holds near D1/D2's 71.6-74.3% has.

## Interpreting the report

Each cell in `flowguard_suite_report.md` is a single
`(defense, attack)` pair at a fixed target FPR of 5%:

- `AUROC` - standard ranking metric on `||v(t=0, x)||^2` (or integral
  score for C1, KS statistic for C3). Values below 0.5 mean the
  distribution is flipped (score is *lower* on attacks than on benign,
  consistent with the FlowPureBoundary anomaly we diagnosed in
  plan Teil 0).
- `FPR` - observed false-positive rate after threshold calibration; by
  construction very close to `--target-fpr`.
- `detection_rate` - fraction of attack queries flagged; relevant for
  W6-style sparse-probe discussion.

Read `AUROC` together with `TPR`, never alone. AUROC near 0.87 with TPR near
0.001 is not weak detection, it is *no* detection: it means the attack scores
sit systematically just above the benign median but well inside the benign
bulk, so nothing crosses the 95th-percentile threshold. Only `TPR` speaks to
the operating point the defense would actually be deployed at.

## Invalidated results (fixed 2026-07-30)

Three corrections that make earlier C1 / D1 / D2 numbers non-comparable. Any
table produced before this date has to be regenerated.

**C1 integrated the wrong way along the path.** `FlowGuardIntegralDefense` and
`FlowGuardCompositeDefense._flow_scores` solved from `t=1` down to `t=0`,
placing the query at the *clean* endpoint of a checkpoint trained with
`x_0 = perturbed`, `x_1 = clean`, and running the flow in reverse. Both now
integrate forward from `t=0`, so the first term of the sum is exactly
FlowPure's `||v(0, x)||^2` and C1 provably contains the baseline's statistic.
Symptom of the old behavior: benign integral ~70x benign `t=0` score while the
D2-family attacks scored ~0.0026 against a 0.1766 benign mean, i.e. AUROC
~0.007 - a near-perfectly *inverted* detector, misreadable as "the attack
evades C1". `SurrogateVelocityRegularizer.trajectory_integral_score` was
flipped to match; the attacker must optimize the same statistic it is scored
on. The likelihood solve is unchanged - `t=1 -> t=0` is correct there.

**D1/D2 were not actually adaptive to FlowPure.** Their C1 generator penalty
fell through to the raw `scores.mean()` branch because the matrix harness never
passed them benign velocity statistics. Benign `||v(0, x)||^2` on the CIFAR-10
FlowPure CNF is ~2.5e-3, so at `--cem-lambda-detector 1.0` the penalty sat ~3
orders of magnitude below the ensemble-disagreement term and the generator was
effectively unconstrained. D3-D6 got the benign-normalized hinge via
`_adaptive_common_extra` and D1/D2 did not, which is why FlowPure "detected" D1
at AUROC 0.968 / TPR 0.85 while suppressing D3/D4 TPR to ~0.001 - the contrast
measured penalty scale, not attack strength. `_disguide_bypass_common_extra`
now forwards `benign_stats`, and the raw fallback warns once per process.

**Likelihood scoring on a FlowPure checkpoint is undefined.** `flow_matching`
and `flowguard_composite` evaluate `log p_0` under a standard Gaussian, which is
only the right base density for a CNF trained with `x_0 ~ N(0, I)`. Pointing
them at a `pgd`/`cw`/`boundary`/`patch` checkpoint makes `log p(x)` - and the
two-sided typicality z-score standardized from it - meaningless. Both now emit
a `RuntimeWarning`, mirroring the one `FlowPureQueryDefense` already raised for
the opposite mismatch. Table V's C3 column needs a Gaussian-source CNF.
