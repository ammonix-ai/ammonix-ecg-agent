# Brugada classifier refit — HUCA cases vs HUCA controls

Analysis behind the `brugada syndrome` class in
`models/source_safe_canonical_v1/`. Run 2026-08-13/14.

**Read [`REPORT.md`](REPORT.md) first** — it has the headline numbers, the comparison against
the previous build, every leakage check, the transportability result and the caveats.

## Result in one line

Confirmed Brugada (n=69) vs controls from the same study (n=287): **out-of-fold AUROC 0.889**
(10-seed mean 0.8955 ± 0.0091). Counting the 7 atypical cases as positive as the shipped
loader does (n=76): **0.906**. The class that shipped before this refit scored **0.546** on
that same comparison, because it was trained on MIMIC with the HUCA cohort excluded and
counted all 363 HUCA records — including 287 healthy controls — as positives.

## Data source and attribution

*HUCA 12-lead ECG recordings for the study of Brugada syndrome*, v1.0.0, PhysioNet,
Hospital Universitario Central de Asturias, licensed **CC BY-SA 4.0**. Labels come from that
dataset's own `metadata.csv` (`brugada`: 0 = healthy, 1 = confirmed, 2 = other/atypical).

Only derived q_psi features are distributed with this project — never the raw recordings.
The raw dataset is publicly downloadable from PhysioNet under its own licence.

MIMIC-IV ECG (a public credentialed release) is used as an evaluation set only for this class;
its 57 Brugada-labelled records are treated as negatives, for the reason in the next section.

## What transportability showed

| train → test | AUROC |
|---|---:|
| HUCA → HUCA (out-of-fold) | 0.900 |
| MIMIC → MIMIC | 0.710 |
| HUCA → MIMIC | 0.519 |
| MIMIC → HUCA | 0.546 |

Both cross-source directions are chance while each cohort is internally learnable, so the two
"Brugada" label sets describe different constructs. MIMIC's is discharge-letter derived and not
challenge- or echo-validated.

## Scope limit

False positives concentrate in conduction disease: 83% of flagged MIMIC controls carry RBBB,
LBBB, fascicular block, IVCD or pacing — the recognised Brugada phenocopies. The model is
near-silent on clean normals (22 of 19,805 sinus-rhythm normals fire at the default threshold).
**It is not a general-population Brugada detector without a conduction-abnormality
differential.**

## Layout

```
REPORT.md            full writeup
scripts/             the analysis, in the order it was run (see below)
results/             JSON/CSV outputs, including per-record out-of-fold scores
```

Run order: `_huca_scout.py` (verify the label join) → `_huca_brugada_refit.py` (the refit, both
label variants, seed stability, permutation control) → `_huca_artefact_check.py`,
`_huca_gain_check.py`, `_huca_final_checks.py` (leakage and operating point) →
`_huca_label_only_fix.py` (what a label-only correction would have given) →
`_huca_to_mimic_transport.py`, `_huca_mimic_fp_count.py`, `_huca_universe_ppv.py`
(transportability and firing profile) → `_huca_build_model_package.py` (build) →
`_huca_install_open_repo.py`, `_huca_restore_attribution.py` (install) →
`_huca_rebuild_universe.py` (rebuild the `_staging/universe/` cache the agent actually reads),
`_huca_foldmean_vs_oof.py` (why that cache must not be regenerated with a plain fold mean),
`_huca_record_operating_point.py` (write the threshold rationale into the shipped notes).

## Two things that will bite whoever touches this next

**Replacing the model does not change what the agent shows.** The backend reads
`_staging/universe/` — `probabilities.npy`, `patients.jsonl` and the thresholds in its own
`metadata.json`. The model package is not consulted at display time. All three must be rebuilt.

**Never regenerate this class's probability column with a plain 5-fold mean.** The cache builder
averages all five fold models over every row, which is 4/5 in-sample for rows the model trained
on. HUCA is now the training set and holds all 76 positives, so a fold mean yields AUROC
**1.0000** against a true out-of-fold **0.9058**. Use the out-of-fold score for HUCA rows; the
fold mean is only valid for genuinely external rows.

**These scripts are archived for provenance and do not run as-is.** They read the 26,490-feature
training matrices (`apr28_X.npy` and companions) and the `ammonix.diagnostics.source_safe_cv`
harness from the private training tree; neither is part of this repository. They are here so the
numbers in `REPORT.md` can be audited against the code that produced them.
