# Reproducing the FlowGuard++ results

This is the ordered path from a fresh clone to the numbers in the paper. Read
[INSTALL.md](INSTALL.md) first.

Nothing here is optional-but-nice: each stage produces checkpoints the next
stage consumes. If you only want to see the machinery work, jump to
[Smoke tests](#0-smoke-tests-30-minutes-1-gpu) and stop there.

## Cost

Wall-clock budgets below are the SLURM `--time` limits we actually request on a
single-GPU node (A100-class, 128 GB host RAM). Treat them as *upper bounds we
provisioned for*, not as measured runtimes — they are the honest number we have.

| Stage | Script | SLURM budget |
|---|---|---|
| 0. Smoke | `*_smoke.sbatch` | 20–45 min each |
| 1. Victim | `train_victim_cifar10.sbatch` | 12 h |
| 2. FlowPure^PGD CNF | `train_flowpure_pgd_cifar10_300k.sbatch` | 48 h |
| 3. Likelihood CNF | `train_flow_matching_cifar10_3000.sbatch` | 48 h |
| 4. Detection suite | `evaluate_detection_suite.sbatch` | 20 min |
| 5. Attack × defense grid | `evaluate_attack_defense_combined.sbatch` | 48 h |
| 6. Active enforcement | `evaluate_active_flowguard_extraction.sbatch` | 24 h × 3 (array) |
| 7. Runtime overhead | `benchmark_flowguard_runtime.sbatch` | 2 h |
| — Full HPC sweep | `evaluate_flowguard_hpc_full.sbatch` | 7 d, 15-task array, 3 concurrent |

Stages 2 and 3 dominate. Budget several GPU-days before anything downstream can
run at paper scale.

All output lands under `runs/`, which is gitignored. Do not commit it.

---

## 0. Smoke tests (30 minutes, 1 GPU)

Verify the wiring end-to-end at toy budgets before spending GPU-days.

```bash
pytest -q                                        # 76 tests, CPU, ~30 s

python scripts/evaluate_attack_defense_matrix_smoke.py \
  --run-label smoke --query-budget 8 --benign-query-budget 8 --epochs 1
```

On a cluster: `sbatch scripts/evaluate_flowguard_hpc_smoke.sbatch`.

The smoke matrix runs every attack against every defense at budget 8. The
resulting *numbers* are meaningless; the point is that no code path raises.

---

## 1. Train the victim

```bash
python scripts/train_victim.py \
  --dataset CIFAR10 \
  --architecture vgg16_bn \
  --epochs 100 \
  --output-dir runs/notebook/training-victim-cifar10-vgg16_bn-nodefense \
  --device cuda --seed 0
```

Produces `.../target_model/`. Every later stage takes this as
`--target-checkpoint-dir`.

Defaults: `--lr 0.01`, `--lr-step 25`, `--lr-gamma 0.5`, `--batch-size 128`.
For MNIST experiments use `--dataset MNIST --architecture lenet`.

## 2. Train the FlowPure^PGD CNF (the detector)

This is the checkpoint the per-query `‖v(t=0,x)‖²` signal needs. A plain
Gaussian-source flow-matching CNF **will not work** here and the code will warn
you at load time.

```bash
python scripts/train_flowpure_pgd.py \
  --dataset cifar10 \
  --max-steps 300000 \
  --batch-size 64 --lr 2e-4 \
  --pgd-eps-max 0.05 --pgd-alpha 0.00784 --pgd-steps 10 \
  --output-dir runs/flow_matching/cifar10_flowpure_pgd \
  --device cuda --seed 0
```

Checkpoints every 5 000 steps to `checkpoint_latest.pt`. It is resumable —
relaunch the same command.

Variants used in ablations: `scripts/train_flowpure_patch.py` (patch
perturbations) and `scripts/train_flowpure_boundary.py` (boundary
perturbations).

## 3. Train the likelihood CNF

The typicality component of the fused score uses a *standard* Gaussian-source
flow-matching CNF — a different model from stage 2.

```bash
python scripts/train_flow_matching.py \
  --dataset cifar10 \
  --epochs 3000 \
  --output-dir runs/flow_matching/cifar10_notebook \
  --device cuda --seed 0
```

Sanity-check it before use:

```bash
python scripts/validate_likelihood_cifar.py
python scripts/measure_cnf_transferability.py     # attacker-surrogate transfer
```

---

## 4. Detection quality (audit mode)

Detection metrics with the defense observing but **not** rejecting — this is what
you want for ROC/AUROC.

```bash
python scripts/evaluate_detection_suite.py \
  --name detection-suite \
  --dataset CIFAR10 --target-architecture resnet18 \
  --defenses fdinet,prada,flow_matching \
  --attacks naive,maze,disguide \
  --query-budget 120 --benign-queries 160 --epochs 1 \
  --flow-checkpoint runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt
```

Outputs per `<defense>-<attack>` cell under `runs/detection_suite/<name>/`:

- `detection_metrics.json` — detection rate, TPR, FPR, precision, recall, F1,
  F1-macro, ROC-AUC
- `detection_samples.csv` — per-query scores and predictions
- `summary.json` / `summary.csv` — aggregate

To simulate a distributed attacker spread over many client identities:

```bash
python scripts/evaluate_detection_suite.py \
  --name detection-suite-100-clients --distributed \
  --num-clients 100 --num-workers 16
```

A dedicated FDINet benchmark lives in `scripts/evaluate_fdinet_detection.py`.
Use `--audit-only` there for stable statistics; `--blocking` terminates attacks
early and distorts them.

## 5. Full attack × defense grid

```bash
python scripts/evaluate_attack_defense_matrix_smoke.py \
  --run-label full \
  --output-root runs/attack_defense_matrix \
  --dataset CIFAR10 --target-architecture vgg16_bn \
  --query-budget 50000 --benign-query-budget 50000 \
  --target-fpr 0.05 \
  --target-checkpoint-dir runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model \
  --flow-checkpoint runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt \
  --likelihood-flow-checkpoint runs/flow_matching/cifar10_notebook/checkpoint_latest.pt \
  --device cuda
```

Despite the `_smoke` filename, this is the general matrix driver — the name
reflects its origin, and `--query-budget` decides whether a run is a smoke test
or a paper run.

Defaults cover attacks `prada, maze, disguide, transfer_{naive,top1,s4l,smoothing},
flowguard_{a1_clean_transfer,a2_projected_maze,a3_flowblind,adaptive_adaptive}`
against defenses `prada, flowpure, fdinet, flowguard_{integral,userlevel,labelhist,composite}`.
Restrict with `--attacks` / `--defenses` (comma-separated).

Cells are skipped when their metrics JSON already exists, so an interrupted
48-hour job resumes by relaunching the identical command. `--force` overrides.

MNIST counterpart: `sbatch scripts/evaluate_attack_defense_mnist.sbatch`,
tabulated by `scripts/build_mnist_attack_defense_tables.py`.

## 6. End-to-end extraction against an *enforcing* FlowGuard++

Stages 4–5 audit. This one enforces, which is the claim that actually matters:
it calibrates the fused risk threshold on benign traffic, measures the benign
rejection rate under enforcement, then runs the attack with flagged queries
receiving an uninformative response — so the substitute only learns from
released answers.

```bash
python scripts/evaluate_active_flowguard_extraction.py \
  --run-label active-fgpp --output-root runs/active_flowguard \
  --attacks flowguard_d1_latent_manifold,flowguard_d5_adaptive_composite \
  --query-budget 1000000 --benign-query-budget 50000 \
  --target-checkpoint-dir runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model \
  --flow-checkpoint runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt \
  --likelihood-flow-checkpoint runs/flow_matching/cifar10_notebook/checkpoint_latest.pt
```

It accepts every flag of the stage-5 matrix driver plus `--skip-benign-holdout`
(skips phase B, the benign-rejection measurement — useful when iterating, never
for a reported number).

Reports: accepted-query rate, benign rejection rate, substitute accuracy and
fidelity, and detection AUROC/F1 on the same stream.

## 7. What the defense costs to run

A detection number without a cost number is not a deployable claim.

```bash
python scripts/benchmark_flowguard_runtime.py \
  --target-checkpoint-dir runs/notebook/training-victim-cifar10-vgg16_bn-nodefense/target_model \
  --flow-checkpoint runs/flow_matching/cifar10_flowpure_pgd/checkpoint_latest.pt \
  --likelihood-flow-checkpoint runs/flow_matching/cifar10_notebook/checkpoint_latest.pt \
  --batch-sizes 1,8,32 --selective-fractions 0.05,0.1,0.25 \
  --repeats 10 --warmup 3 \
  --output runs/runtime/flowguard_runtime.json \
  --latex-out paper/tables/runtime_overhead.tex
```

Reports per-query latency, throughput, observed neural-function evaluations,
peak device memory, and overhead relative to an undefended victim forward pass.

---

## 8. Figures and tables

```bash
python scripts/build_attack_defense_tables.py
python scripts/build_fully_adaptive_table.py
python scripts/build_mnist_guidance_tradeoff.py
python scripts/plot_score_landscapes.py
python scripts/plot_composite_scores.py
python scripts/plot_query_evolution.py
python scripts/plot_operating_point_from_summary.py
python scripts/generate_flow_matching_ood_figure.py
python scripts/visualize_detection_results.py
```

These read from `runs/` and write LaTeX/PNG. The `paper/` directory is not part
of this repository.

## 9. Provenance audit

Before trusting any table, check that every number came from a run of the budget
its caption claims, for the attack/defense pair it is printed under:

```bash
python scripts/audit_paper_table_provenance.py --markdown docs/table_provenance.md
```

See [table_provenance.md](table_provenance.md) for the current mapping.

---

## Determinism and expected variance

Every script takes `--seed` (default 0) and threads it through data order, model
init, and attack randomness. Exact bit-level reproduction still requires the
same GPU architecture, CUDA version, and PyTorch build — non-deterministic cuDNN
kernels and differing reduction orders will move the last digits.

Expect detection AUROC to be stable across seeds and substitute accuracy to move
by roughly a point. If you see a *qualitative* difference — a defense that
detects nothing, an attack that reaches victim accuracy — that is a reproduction
failure and we want to hear about it: open a
[reproduction failure issue](https://github.com/REPLACE_ME/flowguard/issues/new?template=reproduction_failure.yml).

## Getting the artifacts instead

Training the CNFs from scratch is the expensive part. If you only want to
evaluate, we intend to publish the victim and CNF checkpoints in a DOI-backed
archive.

> **TODO(release):** upload victim + FlowPure^PGD + likelihood CNF checkpoints to
> Zenodo, and replace this paragraph with the DOI and `wget` commands.
