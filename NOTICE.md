# Third-Party Notices

FlowGuard++ is released under the MIT License (see `LICENSE`). It builds on, and
redistributes code from, the projects listed below. Each component remains under
its own license; those licenses are reproduced or linked here as required.

If you redistribute FlowGuard++, you must keep this file intact.

---

## 1. ModelGuard — Yoruko-Tang et al.

- **Upstream:** <https://github.com/Yoruko-Tang/ModelGuard>
- **License:** MIT
- **Paper:** Tang, Shejwalkar, Houmansadr. *ModelGuard: Information-Theoretic
  Defense Against Model Extraction Attacks.* USENIX Security 2024.
- **What we use:** FlowGuard++ began as a fork of this repository. The
  directories `defenses/`, `scripts/run_caltech256.py`, `scripts/run_cifar10.py`,
  `scripts/run_cifar100.py`, `scripts/run_cub200.py` and `dataset.sh` are
  derived from it, largely unmodified, and are retained so that the original
  USENIX experiments remain reproducible inside this repository.
- **Naming note:** the *ModelGuard defense* itself is preserved throughout the
  codebase under its original identifiers (`modelguard`, `modelguard_w`,
  `modelguard_s`) because it is one of the **baselines** FlowGuard++ is compared
  against. Do not rename these — doing so would silently break the mapping
  between the code and the published results.

The upstream MIT copyright notice is reproduced in `LICENSE`.

---

## 2. Prediction-perturbation baselines

The following defenses are reimplementations of, or derived from, published work.
Cite the original papers, not this repository, when you use them as baselines.

| Identifier in this repo | Original work |
|---|---|
| `reverse_sigmoid` | Lee et al., *Defending Against Model Stealing Attacks Using Deceptive Perturbations*, 2018 |
| `mad` | Orekondy, Schiele, Fritz, *Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks*, ICLR 2020 |
| `adaptive_misinformation` | Kariyappa & Qureshi, *Defending Against Model Stealing Attacks With Adaptive Misinformation*, CVPR 2020 |
| `modelguard`, `modelguard_w`, `modelguard_s` | Tang et al., USENIX Security 2024 (above) |

## 3. Extraction-attack baselines

| Identifier in this repo | Original work |
|---|---|
| `transfer_set` | Orekondy, Schiele, Fritz, *Knockoff Nets: Stealing Functionality of Black-Box Models*, CVPR 2019 |
| `jacobian` | Papernot et al., *Practical Black-Box Attacks against Machine Learning*, AsiaCCS 2017; Juuti et al. (JBDA-TR) |
| `prada` | Juuti, Szyller, Marchal, Asokan, *PRADA: Protecting Against DNN Model Stealing Attacks*, EuroS&P 2019 |
| `maze` | Kariyappa, Prakash, Qureshi, *MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation*, CVPR 2021 |
| `disguide` | Rosenthal et al., *DisGUIDE: Disagreement-Guided Data-Free Model Extraction*, AAAI 2023 |

## 4. Query-level detection baselines

| Identifier in this repo | Original work |
|---|---|
| `prada` (query defense) | Juuti et al., EuroS&P 2019 |
| `fdinet` | Yan et al., *FDINet: Protecting against DNN Model Extraction via Feature Distortion Index* |
| `flowpure` | Collin et al., *FlowPure: Continuous Normalizing Flows for Adversarial Purification* — <https://github.com/deepmancer/FlowPure> |

`FlowPure` is **not vendored** in this repository. If you need the reference
implementation, clone it separately; see `docs/INSTALL.md`.

---

## 5. `pretrainedmodels/`

- **Upstream:** <https://github.com/Cadene/pretrained-models.pytorch>
- **Author:** Remi Cadene
- **License:** BSD-3-Clause
- **What we use:** a vendored snapshot, used by the legacy `defenses/` code path
  for ImageNet-family architectures. Unmodified except for import fixes.

```
BSD 3-Clause License

Copyright (c) 2017, Remi Cadene
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 6. Datasets

No dataset is redistributed here. `dataset.sh` and the `torchvision` loaders
fetch data from the original hosts. Each dataset carries its own terms of use —
in particular Caltech-256, CUB-200, Indoor-67 and ImageNet-1k restrict use to
non-commercial research. You are responsible for complying with them.
