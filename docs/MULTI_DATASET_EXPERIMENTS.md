# Multi-dataset experiments on HoreKa

This runbook covers the four-dataset benchmark: CIFAR10, GTSRB, CelebA and Skin
Cancer, each with its own victim architecture. It explains what every job does,
how to run everything on HoreKa with one command, and where the results end up.

| Task | Victim model | Classes | Attacker's public data (pool) |
|---|---|---|---|
| CIFAR10 | VGG19-BN (`vgg19_bn`) | 10 | CIFAR100 |
| GTSRB (traffic signs) | MobileNetV2 (`mobilenetv2_cifar`) | 43 | BelgiumTS (Belgian traffic signs) |
| CelebA (gender) | DenseNet-121 (`densenet121_cifar`) | 2 | LFW (faces) |
| Skin Cancer (ISIC 2018 / HAM10000) | ResNet-50 (`resnet50_cifar`) | 7 | BCN20000 (dermoscopy) |

The dataset/architecture pairs and the 32x32 input size follow the FDINet
evaluation (Yao et al., TDSC), which is one of the baselines.

---

## 1. What runs, in plain words

For **each** of the four datasets the pipeline trains eight models and then runs
the attacks. The defender's models are trained on the defender's own training
data. The attacker's models are trained only on the attacker's public pool,
which never overlaps with the victim's data.

```
defender side                               attacker side
─────────────                               ─────────────
victim classifier  ──► FlowPure CNF (C1,C2)  proxy classifier ──► surrogate FlowPure CNF ─┐
Gaussian CNF (C3)                            surrogate Gaussian CNF ──────────────────────┴► calibration
                                             diffusion prior (for Guided DisGUIDE)
                      └───────────────────────────────┬───────────────────────────────┘
                                                      ▼
                           evaluation: every attack × every detector
```

| Stage | Script | What it produces and why |
|---|---|---|
| `victim` | `train_victim.py` | The model the attacker wants to steal. |
| `defender_flowpure` | `train_flowpure_pgd.py` | A flow that learned to turn PGD-perturbed images back into clean ones (x₀ = attacked, x₁ = clean). How hard it has to "push" a query (the velocity) is the C1 score. The push summed along the whole path is C2. |
| `defender_gauss` | `train_flow_matching.py` | A standard flow from Gaussian noise to images. It gives a likelihood log p(x), which is the C3 typicality score. It has to be a separate model: log p(x) is only defined for a flow that starts from Gaussian noise. |
| `attacker_proxy` | `train_victim.py` (on the pool) | A classifier the attacker trains on its own public data. It is only used to create PGD examples for the attacker's surrogate flow. |
| `attacker_flowpure` | `train_flowpure_pgd.py` (on the pool) | The attacker's copy of the C1/C2 detector. The attacker never sees the defender's flow. |
| `attacker_gauss` | `train_flow_matching.py` (on the pool) | The attacker's copy of the C3 detector. |
| `attacker_prior` | `train_diffusion_prior.py` | A 32x32 diffusion model trained on the pool. Guided DisGUIDE uses it to generate natural-looking queries. |
| `attacker_calibration` | `evaluate_parallel_defenses.py --attacker-calibration-only` | The attacker measures its surrogate scores on its pool ("what does normal look like to me?"). It also distils the scores into a small, fast network for the high-budget adaptive attacks. |
| `eval/<group>/seed<S>/<attack>` | `evaluate_parallel_defenses.py` | Runs one attack once. **Every** detector scores the same query stream, so the comparison is paired. |

**Evaluation groups** (in `configs/horeka_experiments.json`):

- `detection`: PRADA attack, MAZE, DisGUIDE and the velocity-adaptive Guided
  DisGUIDE (`flowguard_d1_latent_manifold`), with 50,000 queries each.
- `adaptive`: the FlowGuard++-adaptive variants D3–D6, with 1,000,000 queries
  each.

**Detectors scored in every run:** PRADA, FlowPure (C1), FDINet (per query),
FDINet client vote (`fdinet_vote`), trajectory integral (C2), likelihood
typicality (C3, `flow_matching`), KS window (C4, `flowguard_userlevel`), label
histogram (C5, `flowguard_labelhist`), the fused FlowGuard++ score
(`flowguard_composite`) and the learned-weight oracle
(`flowguard_composite_learned`).

**Sybil identities.** Every run is scored single-client, and each detector that
keeps per-identity state (PRADA, FDINet vote, C4, C5) is also scored as
`<detector>@N<count>` for N = 5, 25, 125, 625, 1250 identities. All of these
scores come from the same query stream. Query *i* goes to identity *i mod N*,
so with 50,000 queries and N = 1250 each identity sends exactly 40 queries.
Per-query detectors (C1–C3, FDINet per query, the fused score) do not look at
identities, so one number covers every N.

**Benign traffic.** Benign traffic is drawn from the victim dataset's *test*
split, which no model was trained on. The test split is cut into two fixed,
disjoint halves (`<Dataset>BenignCal` / `<Dataset>BenignEval`, see
`defenses/datasets/benign_splits.py`). The first half sets every threshold and
reference statistic. The second half is the benign side of every reported
number.

---

## 2. Running it on HoreKa

HoreKa facts this relies on: the `accelerated` partition has A100 40 GB GPUs
with a 48 h wall-clock limit. Large data belongs on a workspace
(`ws_allocate`). Because it is not documented whether compute nodes can reach
the internet, **every download happens on the login node**, and jobs run with
`HF_HUB_OFFLINE=1`.

```bash
# on a HoreKa login node, inside the repository
ws_allocate flowguard 60                                   # optional: data on a workspace
cp scripts/horeka/site.env.example scripts/horeka/site.env # set FLOWGUARD_DATA_ROOT etc.
bash scripts/horeka/setup_env.sh                           # venv + all datasets + self-check (~1-2 h)

python scripts/schedule_experiments.py --status            # everything "missing" at first
python scripts/schedule_experiments.py --dry-run           # prints the sbatch commands
python scripts/schedule_experiments.py                     # submits all 64 stages with dependencies
```

Afterwards, simply re-run `python scripts/schedule_experiments.py` whenever you
like:

- stages with a `DONE.json` are skipped,
- stages whose jobs are still in `squeue` are skipped,
- stages that ended without `DONE.json` (crash, timeout after the last chain
  job) are resubmitted, and the log file is named in the output.

Useful filters: `--datasets GTSRB`, `--stages victim,defender_gauss`,
`--groups detection`, `--attacks maze`, `--seeds 0,1,2,3,4`, and
`--max-submit 5` for a careful first round. Unfinished prerequisites of a
selected stage are always included.

**Long jobs.** Stages that may take longer than 48 h are submitted as a chain
(`"chain": 2` or `3` in the config). Each link is an ordinary job that starts
when the previous one ends (`afterany`). Every trainer resumes from its own
checkpoint (`checkpoint_latest.pt` / `training_state.pt`), and the DisGUIDE-loop
attacks resume from `attack_checkpoint.pt`. A link that finds `DONE.json` exits
immediately.

Logs go to `logs/scheduler/<job-name>_<jobid>.out`. Job state is recorded in
`runs/fdinet4/_scheduler/jobs.json`.

---

## 3. Data

`scripts/prepare_datasets.py` (run by `setup_env.sh`) downloads everything and
stores each dataset as a single 32x32 uint8 array
(`<data>/<dataset>/cache32/*.npz`). Jobs only ever read those arrays.

| Dataset | Source | Split used | Preprocessing |
|---|---|---|---|
| CIFAR10 / CIFAR100 | torchvision | official train/test | — |
| GTSRB | sid.erda.dk (official archives) | official train (39,209) / test (12,630) | resize to 32 (no ROI crop). No horizontal flips anywhere: a mirrored "keep right" sign is a "keep left" sign. |
| BelgiumTS | btsd.ethz.ch | official train / test | resize to 32 |
| CelebA | `flwrlabs/celeba` on HuggingFace (ungated parquet mirror); the official files also work | official train (162,770) / valid / test (19,962) | centre crop 148, resize to 32. Label = attribute `Male` (change with `FLOWGUARD_CELEBA_ATTR`). |
| LFW / LFW10 | figshare mirror of `lfw-funneled.tgz` | deterministic 90/10 per image | centre crop 150, resize to 32. LFW10 = people with ≥10 images (158 identities), used only for the attacker's proxy classifier. |
| Skin Cancer | ISIC 2018 Task 3 (S3 bucket) | official train (10,015) / test (1,512); different lesions | centre square, resize to 32 |
| BCN20000 | ISIC 2019 images whose lesion id starts with `BCN_` (12,413 images) | 90/10 by lesion | centre square, resize to 32. The `BCN_` filter excludes every HAM10000 image. |

Approximate downloads: CelebA 11.7 GB, ISIC 2019 9.1 GB, ISIC 2018 2.8 GB, the
rest together about 1 GB. `REMOVE_RAW=1 bash scripts/horeka/setup_env.sh`
deletes the raw files once the caches exist.

The attacker pools differ from FDINet in two places:

- **TSRD → BelgiumTS.** TSRD's download server was unreachable when this was
  set up. `TSRD` is implemented for manual placement; set `attacker_pool` and
  `attacker_proxy_dataset` to `TSRD` for GTSRB to use it.
- **CINIC-10 → CIFAR100.** CINIC-10 contains CIFAR-10 images, so it is not
  disjoint from the victim's data. CIFAR100 is what the earlier CIFAR-10
  experiments used.

---

## 4. Configuration

Everything lives in `configs/horeka_experiments.json`:

- `datasets.<name>`: architecture, attacker pool, benign budget (= half the test
  split), plus per-stage overrides (for example `max_steps`, scaled to the
  dataset size so that small datasets are not trained for thousands of epochs).
- `stages.<stage>`: wall-clock time, memory, chain length, CLI arguments.
- `evaluation`: seeds, Sybil variants, detector list, shared driver arguments,
  and the attack groups.

Values that were judgement calls and are worth revisiting:

| Setting | Value | Note |
|---|---|---|
| `diffusion-guidance-scale` | 7.0 | Step-wise detector guidance of Guided DisGUIDE. The older CIFAR-10 sbatch files default to 0.0, which switches it off. 7–7.5 was the best trade-off in the MNIST sweep. |
| `seeds` | `[0]` | Use `--seeds 0,1,2,3,4` for five seeds (5× the evaluation cost). |
| `adaptive` query budget | 1,000,000 | D3–D6 at ~11 queries/s need ~25 h each; chained over up to 3 jobs. |
| Attacker proxy architecture | same as the victim | Only shapes the attacker's PGD pairs. |
| CNF step budgets | per dataset | E.g. SkinCancer 100k/60k steps vs CIFAR10 300k/200k. |

---

## 5. Results

Each evaluation job writes

```
runs/fdinet4/<dataset>/eval/<group>/seed<S>/<attack>/
    attack_defense_matrix_<label>_summary.json   # one row per detector
    DONE.json
```

Each row has `defense`, `attack`, `auroc`, `tpr`, `fpr`, `f1` (at the benign
95th percentile, i.e. a 5 % target FPR), `accuracy`/`fidelity` of the
attacker's substitute, and `query_curve` (fidelity over queries for the
DisGUIDE-loop attacks). Rows named `prada@N1250` etc. are the Sybil variants.

Things to keep in mind when reading them:

- A detector's AUROC under Sybil fan-out is the `@N` row. When no identity
  reaches the detector's window (C4: 64, C5: 128, FDINet vote: 50 queries), the
  score is 0 for every query, and the AUROC comes out as 0.5. That is the
  measured version of "the detector cannot form its statistic".
- For attacks that resumed after a wall-clock kill, detector scores cover the
  queries issued after the last resume. The attack's own checkpoint does not
  contain the detectors' history.

---

## 6. Behaviour changes compared to the earlier CIFAR-10 runs

These change numbers relative to runs made with older code. The last column
says how to get the old behaviour back, where that makes sense.

| Change | Why | Old behaviour |
|---|---|---|
| The paired driver records each batch's detector scores | Every recorded batch used to show the last batch's scores, so AUROC/F1 of `evaluate_parallel_defenses.py` came from one batch per side | — |
| Sybil identities rotate per query, not per batch | With 32-query batches, 312 of 1,250 identities got 64 queries (more than FDINet's 50 and C4's 64-query window) | `--sybil-granularity batch` |
| PRADA keeps one detector per identity | The old detector pooled all identities, so Sybil splitting could not affect it | — |
| Fused score uses [z]₊ for C1/C2 and includes C2 | C2 had no benign reference and was silently dropped; negative velocity z-scores cancelled typicality evidence | — |
| Flow detectors de-normalize with the victim's registry stats | They guessed from the dataset name (GTSRB was treated as ImageNet-normalized) | — |
| Benign traffic from held-out test halves | The training split is what the CNFs were fitted on | omit `--benign-*-dataset` |
| FlowPure-PGD and Gaussian CNF trainers resume | Previously a relaunch started from step 0 | `--no-resume` |

---

## 7. Troubleshooting

- **`No virtualenv at …`** in a job: run `scripts/horeka/setup_env.sh` on the login node.
- **A dataset is missing in `check_env.py`**: re-run
  `python scripts/prepare_datasets.py --datasets <Name>` on the login node. The
  error message says which raw files were expected.
- **`Distilled score surrogate is too inaccurate`** (calibration stage): raise
  `adaptive-calibration-samples` / `adaptive-distill-epochs`, or lower
  `adaptive-distill-min-correlation` in the config.
- **Out of memory in the adaptive group**: lower `extraction-attack-batch-size`
  in `evaluation.groups.adaptive.args`.
- **A stage keeps being resubmitted**: open the log named in the scheduler
  output. A deterministic error (bad path, missing data) will not fix itself by
  resubmitting.
