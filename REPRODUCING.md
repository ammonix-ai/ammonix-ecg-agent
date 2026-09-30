# Reproducing the paper

Paper: *The Ammonix ECG Agent: A Local Specialized AI Agent to Transparently Read and Discuss
12-Lead Electrocardiograms* (https://doi.org/10.5281/zenodo.22871232).

This repository is the serving application described in the paper. It is not the experiment
code of the 17-diagnosis study, and the universe package and classifier package are not part
of the initial release (see "Data and model" in the README). This file says, result by result,
what can be recomputed from here, what needs the packages, and what cannot be rerun.

Classification used below:
- **exact** — deterministic recomputation from shipped or published artefacts.
- **needs packages** — it runs once the universe and classifier packages are published.
- **not rerunnable from this release** — the code or data that produced it is not here; the
  number is reported in the paper only.

## Environment

Python 3.10 or newer (checked with 3.12), dependencies in `backend/requirements.txt` and `qpsi/pyproject.toml`; Node (checked with 24)
for the viewer (`frontend/package.json`, lockfile committed); the Rust accelerator under
`qpsi/qpsi_native/` is optional; it is a speed-up, see `qpsi/README.md`. The chat agent
talks to any OpenAI-compatible endpoint (`OPENAI_BASE_URL`); the paper's prototype ran a local
model on one 24 GB GPU.

## Results, one row per table and figure

| Paper element | What it shows | Where it comes from | Class |
|---|---|---|---|
| Fig. 1 | Architecture diagram | drawing | — |
| Table 1 | Feature taxonomy, 26,490 features per ECG | the q_psi pipeline in `qpsi/` defines the features; the count is a property of the full configuration | exact (read off the feature schema) |
| Fig. 2 | Tokenizer output on example recordings | `qpsi/` on any 12-lead WFDB recording (for example from PTB-XL) | exact for a given recording |
| Table 2 | 17-diagnosis vocabulary | read off the paper; cohorts are Chapman-Shaoxing, CPSC 2018, PTB-XL | — |
| Table 3 | Proposed and admitted Skills (193 / 165) | RHO-VR Skill induction, not in this repository | not rerunnable from this release |
| Fig. 3 | Knowledge Universe of the 17-diagnosis cohort | experiment code not in this repository | not rerunnable from this release |
| Table 4 | Held-out test performance (n = 1,135) | experiment code and the learned Skill store are not in this repository | not rerunnable from this release |
| Table 5 | F1 difference by baseline cluster F1 | same | not rerunnable from this release |
| Table 6 | Frontier-model comparators (GPT-6 Astra, Claude Fable 5.1, zero-shot on the trace image) | comparator runs are not in this repository and are stochastic | not rerunnable from this release |
| Table 7 | Per-diagnosis AUROC, 32 diagnoses | the private cross-validation that trained the swarm: out-of-fold scores for the training rows, which are not shipped. The published values of all three AUROC definitions are recorded per class in `docs/auroc-metrics.md` (the table reports the `final` column, except the ST-deviation row, which reports the display-label pool, 0.883 on 2,378 cases; the same-source macro average 0.929 is the `primary` column). `packaging/recompute_metrics.py` states why it does not reproduce them: the published probabilities are a five-model mean, partly in-sample for training rows | not rerunnable from this release; values documented |
| Table 7, Brugada row | 0.906 on Brugada-HUCA, 76 cases vs 287 controls | `analysis/brugada_refit/REPORT.md` and the aggregate JSON files next to it; the refit scripts import a private training module | aggregates shipped; rerun not possible from this release |
| Table 8 | Operating points of the seven critical diagnoses | same private cross-validation scores as Table 7 | not rerunnable from this release |
| Table 9 | NPV as a function of prevalence | computed from Table 8's sensitivity and specificity | exact (arithmetic from Table 8) |
| Fig. 4 | Data efficiency vs a CNN | training code not in this repository | not rerunnable from this release |
| Fig. 5 | The clinical interface | the viewer in `frontend/` with the packages loaded | needs packages |

## What runs today

Without the packages: the backend starts and reports `degraded` on `GET /api/status`, the
viewer builds and loads, and the Universe and Analyze pages say which package is missing.
The hosted demo at https://ecg.ammonix.ai runs the same code with the packages installed. The q_psi feature
pipeline runs on any 12-lead WFDB recording you supply.

## Obtaining data yourself

The open sources (PTB-XL, CPSC 2018, Georgia, Chapman-Shaoxing, Brugada-HUCA, MIMIC-IV-ECG)
are available from PhysioNet, and the Pre-/Post-STEMI ECG Database from the University of
Michigan's Deep Blue Data, under the terms listed in `NOTICE.md`. Some training labels of the
paper's 32-diagnosis universe (infarction diagnoses, echocardiographic LVEF) come from
credentialed MIMIC-IV data, which PhysioNet grants to each user individually. No credentialed
data is needed to run the feature pipeline.
