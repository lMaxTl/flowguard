# Security & Responsible Use

## What this repository contains

FlowGuard++ is defensive security research. To evaluate a defense against model
extraction you must be able to run the attacks it defends against, so this
repository ships working implementations of published extraction attacks
(Knockoff Nets, JBDA-TR, PRADA, MAZE, DisGUIDE) **and** novel adaptive attacks we
developed specifically to stress-test FlowGuard++.

Publishing attack code alongside a defense is standard practice in the security
literature and is what makes a defense evaluation falsifiable. It also means this
code can be misused.

## Acceptable use

Use this code **only** against models that you own, or that you have explicit
written authorization to test.

Do **not** use it to extract, clone, or reverse-engineer third-party commercial
or public ML APIs. Doing so is very likely to violate the provider's terms of
service, and depending on your jurisdiction may violate computer-misuse,
trade-secret, or copyright law. Nothing here constitutes legal advice, and the
authors accept no liability for how the code is used (see `LICENSE`).

Dataset licenses are separate from this repository's MIT license. Several
datasets used in our experiments (Caltech-256, CUB-200, Indoor-67, ImageNet-1k)
are restricted to non-commercial research. Complying with them is your
responsibility.

## Scope of "vulnerability" for this project

This is research code, not a deployed service. The reports we can act on are:

**In scope**

- A dependency in `pyproject.toml` / `environment.yml` with a known CVE.
- Code that unsafely deserializes untrusted input — in particular
  `torch.load` / `pickle` on checkpoints or transfer sets from an untrusted
  source.
- The FastAPI server in `src/flowguard/api/` doing something unsafe when
  exposed. Note that it ships with **no authentication by design** — it exists
  to simulate a black-box victim API in experiments. Binding it to a public
  interface is a misconfiguration, not a vulnerability.
- Accidental disclosure of credentials or personal data in this repository.

**Out of scope**

- "The defense can be bypassed." That is a *research result*, not a
  vulnerability — please open a normal issue or send a PR. We genuinely want
  these; see below.
- Attacks succeeding against an undefended model. That is the premise.

## Reporting

For anything in the in-scope list above, please use GitHub's **private
vulnerability reporting** (Security → Report a vulnerability) rather than a
public issue, or email the maintainers listed in `CITATION.cff`.

Please include a minimal reproduction and the commit hash. We aim to acknowledge
within 7 days. We have no bug-bounty program.

## Breaking the defense

If you find an attack configuration that defeats FlowGuard++, **open a public
issue**. A defense that is only evaluated against attacks its own authors thought
of is not evaluated. Please include:

1. The `ExperimentSpec` (or CLI invocation) that reproduces it.
2. Query budget, victim checkpoint, and CNF checkpoint used.
3. The detection metrics you observed (AUROC / TPR at fixed FPR / accepted-query
   rate) and the resulting substitute accuracy and fidelity.

We will try to reproduce it and, if it holds, add it to the benchmark with
attribution.
