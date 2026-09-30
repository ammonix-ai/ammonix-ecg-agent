# Brugada classifier refit — HUCA cases vs HUCA controls

Run 2026-08-13. Scripts (in `scripts/` here): `_huca_brugada_refit.py` (main), `_huca_artefact_check.py`,
`_huca_gain_check.py`, `_huca_final_checks.py`, `_huca_scout.py`.
Outputs in `huca_brugada_refit/`. Nothing was written to the private training repository.

---

## Headline

**Confirmed Brugada (n=69) vs same-study controls (n=287): out-of-fold AUROC 0.889**
(seed 0; 10-seed mean 0.8955 ± 0.0091; bootstrap 95% CI 0.833–0.936).

Counting the 7 atypical cases as positive as the shipped loader does (n=76 vs 287):
**0.906** (10-seed mean 0.9009 ± 0.0076; 95% CI 0.863–0.945).

`brugada = 2` was **excluded** from the headline. "Other/atypical" is not "confirmed".
Both variants are reported because the difference is only 7 records.

Fold structure: 5 folds, subject-grouped, ~14 cases per held-out fold, 500 screened features
per fold, **subject overlap 0 in every fold of every run**. One subject = one recording in HUCA,
so grouping is a no-op, but it was left on.

---

## The comparison you asked for

| artefact | positives | negatives | trained on | AUROC |
|---|---:|---:|---|---:|
| Shipped model `metadata.json` | **420** | 33,683 | MIMIC (57 pos) | primary **0.710** · final_full **0.843** |
| Canonical sweep table 2026-06-05 | 76 | 287 | Brugada-HUCA | primary **0.9095** · final 0.9053 |
| **This refit — inclusive (76)** | 76 | 287 | Brugada-HUCA | **0.9058** (10-seed 0.9009) |
| **This refit — confirmed (69)** | 69 | 287 | Brugada-HUCA | **0.8888** (10-seed 0.8955) |

### Two corrections to the handoff's premise

**1. The refit does not land near 0.71, and it should not.** `primary_auc` 0.710 is not an
estimate of "HUCA cases vs HUCA controls" — it is the shipped model's MIMIC-internal number,
57 MIMIC positives against 2,841 MIMIC negatives. Different cohort, different label source,
different task. There was never a reason for the HUCA-internal number to be bounded by it.

**2. "Published 0.905" is not the inflated number — it is the already-corrected one.**
It comes from `<private training repo>/domains/ecg-12lead/data/qpsi_jun3/source_safe_canonical_table_20260605.csv`:

```
brugada syndrome,0.9095,0.9053,0.9710,76,76,287,Brugada-HUCA,HUCA metadata source-of-truth
```

`n_total_pos 76`, `n_train_neg 287`, `train_sources Brugada-HUCA` — that is exactly the refit
being requested here, run on 2026-06-05. So this work **reproduces** an existing corrected
result (0.9058 vs 0.9095, inside seed noise) rather than replacing an inflated one. The handoff
placed 0.905 in the same row as `final_full_auc` 0.843 and `primary_auc` 0.710, which attributes
to the broken shipped model a number produced by its corrected replacement.

### The real defect in the shipped model

Confirmed from `ammonix-ecg-agent/models/source_safe_canonical_v1/diagnoses/brugada_syndrome/metadata.json`:

```
train_sources           ['MIMIC']
excluded_train_sources  ['Brugada-HUCA']
n_total_pos             420
pos_by_source           {'MIMIC': 57, 'Brugada-HUCA': 363}
n_train_pos             57      n_external_pos  363
```

`pos_by_source['Brugada-HUCA'] = 363` is the bug, and it is worse than the handoff states:
**all 363 HUCA records are labelled positive, including the 287 healthy controls.** The model
was built without the HUCA metadata fix, so it used the source-level label. 420 = 57 + 363.
It then scored those 363 rows as `external` (fold-mean, never trained on) and computed
`final_full_auc` 0.843 with 287 healthy people sitting in the positive class.

The handoff's "fitted on MIMIC's two positives" is wrong — it trained on **57** MIMIC positives.
The "2" is from the universe viewer's `patients.jsonl`, not the training set.

---

## Leakage checks

The result is well above what the handoff expected, so it was attacked from five directions.

| check | result | reading |
|---|---|---|
| Permutation control, labels shuffled, full protocol refit ×10 | 0.527 (confirmed) / 0.515 (inclusive), max 0.597 | protocol does not leak |
| `patient_id` alone as a score | 0.506 | no cohort-block or recruitment-order leak |
| Subject overlap across folds | 0 in all folds, all runs | no patient straddles train/test |
| Per-record ADC gain alone | 0.634 | a real channel — chased down below |
| `#Dx` header tag | see below | present, but not reaching the features |

### Gain is not driving it

Per-record gain differs between cases and controls (AUROC 0.634), because gain encodes
peak-to-peak amplitude. Two tests clear it:

- **Gain-stratified**: within gain quartiles, AUROC 0.816 / 0.860 / 0.918 / 0.879,
  size-weighted mean **0.868** vs 0.889 overall. Pearson r(score, gain) = 0.198.
- **Refit on gain-invariant features only** (drop all 7,560 millivolt-valued features,
  keep 18,930): **0.9010 ± 0.0076**, versus 0.9004 for all 26,490 features. Amplitude-only
  features give 0.8870.

Removing every amplitude feature does not move the number. The single most important feature
is a *timing* feature, which is gain-invariant by construction.

### The `#Dx` header tag — real, but dormant

The locally derived `files_500Hz_10s/` headers carry `#Dx: 418818005` (SNOMED: Brugada syndrome)
on all 76 cases **and** on 23 `brugada=0` controls — 99 records total, exactly those with
`brugada>0 OR basal_pattern=1`. The original 100 Hz PhysioNet release has no comment lines at
all and uniform gain 1000.0, so both the tag and the variable gains were introduced by the local
500 Hz conversion. That folder is outside the official release and is not in `SHA256SUMS.txt`.

Those 23 tagged controls are a natural probe: they carry the identical header tag as every case.
If the tag reached the features they would score like cases.

| group | median OOF score |
|---|---:|
| cases | 0.494 |
| controls **with** `#Dx` | 0.026 |
| controls without `#Dx` | 0.014 |

They score with the controls, 19× below the cases. Cases vs tagged controls 0.810, cases vs
untagged controls 0.896 — close, where a tag-driven model would show an enormous gap. The mild
elevation is expected: those 23 genuinely have pathological baseline ECGs, which is what the tag
marks. q_psi consumes the signal array and `fs`, not header comments.

**Still worth fixing:** a diagnosis annotation living in a derived header is exactly the substrate
that produced the original source-level-label bug. It should not be there.

### What the model actually keys on

Top screen features, aggregated across the 5 folds (train rows only):

```
ml_direct::V2::T::pos_peak_time_ms::median        <- #1, and gain-invariant
plane_direct::limb::QRS::duration_ms_rms::median
ml_direct::V1::QRS::duration_ms_rms::median
ml_direct::V2::T::pos_peak_amp_mv::median
ml_direct::V2::T::extr_1_amp_mv::median
ml_direct::V1::T::rms_amplitude_mv::median
```

Right-precordial repolarization timing and V1 QRS duration — coved ST-T in V1–V2 plus RBBB-like
conduction delay. That is Brugada physiology, not a file artefact. V1–V3 hold 28.5% of the top
200 features.

---

## Operating point

The main run's F1-optimal threshold was picked on the same OOF scores it was scored against,
which is optimistic. Below, each fold's threshold is chosen on the OOF scores of the **other four
folds only**, so no fold's own labels inform its threshold.

**Confirmed (69 vs 287)** — per-fold thresholds 0.214 / 0.214 / 0.214 / 0.185 / 0.223:

| | value |
|---|---:|
| sensitivity | 0.696 |
| specificity | 0.937 |
| PPV | 0.727 |
| NPV | 0.928 |
| F1 | 0.711 |
| TP / FP / TN / FN | 48 / 18 / 269 / 21 |

In-sample threshold selection would have reported F1 0.731 — about 2 points of optimism.
Inclusive variant, same procedure: sensitivity 0.724, specificity 0.923, PPV 0.714, F1 0.719.

For scale, the shipped model's `default_threshold` is 0.00725. A refit that can actually see
positives puts the F1-optimal threshold at **~0.21**, roughly 30× higher.

---

## Concealed vs manifest cases — the interesting result

`basal_pattern` is documented as whether the baseline ECG is pathological, *independent of* the
Brugada diagnosis. Cases without it are concealed — diagnosed by sodium-channel-blocker
challenge, not visible as a resting type-1 pattern.

| group | n | AUROC vs all 287 controls |
|---|---:|---:|
| cases **with** pathological baseline | 19 | 0.867 |
| cases **without** (concealed) | 50 | 0.897 |

Concealed cases are detected at least as well as manifest ones. Either the mechanistic features
pick up sub-threshold right-precordial repolarization changes that do not meet the visual type-1
criterion, or `basal_pattern` is coding something narrower than "any visible abnormality".
n=19 in the manifest group, so treat the ordering as noise-limited — but the concealed group at
0.897 on n=50 is the solid half, and it is the clinically interesting one.

---

## Addendum — can we fix only the labels and keep the universe classifier?

Short answer: the label fix is right and necessary, but on its own it does **not** give 0.906.
It gives **0.546**.

The live package is `<private training repo>/domains/ecg-12lead/WaveMedix/models/source_safe_canonical_v1/`
(2026-06-04). Its brugada class reads:

```
train_sources           ['MIMIC']          excluded_train_sources  ['Brugada-HUCA']
n_train_pos             57                 n_external_pos          363
```

**The deployed universe classifier was not trained on the validated HUCA cases — it was trained
only on 57 MIMIC-IV positives, with HUCA excluded from training entirely.** All 32 classes in
`manifest.json` carry `excluded_train_sources: ["Brugada-HUCA"]`; it is the global default.

`validation_scores.npy` holds the deployed per-row scores for all 34,103 rows. Recomputing the
shipped `final_full_auc` from them reproduces 0.8433 exactly, which confirms the row alignment.
Keeping those scores and correcting only the labels (HUCA 76 positive / 287 negative):

| quantity | as shipped | label-only fix |
|---|---:|---:|
| n_total_pos | 420 | 133 (76 HUCA + 57 MIMIC) |
| final_full_auc | 0.8433 | **0.7624** |
| HUCA-internal, 76 vs 287 | not reported | **0.5459** |

### What the shipped 0.843 is actually measuring

| group | median deployed score |
|---|---:|
| HUCA cases | 0.01477 |
| HUCA healthy controls | 0.01299 |
| MIMIC brugada positives | 0.00375 |
| all other rows | 0.00253 |

All 363 HUCA rows — cases **and** healthy controls together — separate from everything non-HUCA at
**AUROC 0.885**. HUCA healthy controls outscore MIMIC brugada cases by 3.5×. The deployed model is
a HUCA-study-membership detector, not a Brugada detector. That is the "was in the HUCA study vs
was not" artefact the handoff described — it belongs to `final_full_auc` 0.843, not to 0.905.

### Consequence

Label-only correction makes the metadata honest but replaces a wrong 0.843 with a correct 0.546,
and it will not match the paper. Reaching ~0.906 inside the universe requires **retraining that one
class on HUCA** — which is what this refit and the 2026-06-05 canonical table both did, and what
the deployed package never did.

## Transportability — HUCA model applied to MIMIC (2026-08-14)

`_huca_to_mimic_transport.py`: train on HUCA (76 vs 287), score the 2,898 MIMIC-derived Apr28
rows as external (5-model mean, the deployed convention). Row counts reproduce the deployed
`n_train_eligible 2898 / n_train_pos 57 / n_train_neg 2841` exactly.

| train → test | AUROC |
|---|---:|
| HUCA → HUCA (out-of-fold) | **0.900** ± 0.010 |
| MIMIC → MIMIC (deployed `primary_auc`) | 0.710 |
| **HUCA → MIMIC** (57 brugada vs 2,841 controls) | **0.519** ± 0.002 |
| **MIMIC → HUCA** (deployed model, 76 vs 287) | **0.546** |

Median scores: MIMIC brugada-labelled 0.0308, MIMIC controls 0.0326 — the controls score
*higher*. Both cross-source directions are chance.

**This is not simply label noise.** MIMIC's Brugada label is internally learnable at 0.710, so it
has real structure — but a HUCA-trained model cannot see it, and a MIMIC-trained model cannot see
HUCA's. The two cohorts' "Brugada" are different constructs. MIMIC's label (discharge-letter /
ICD-derived, never validated) is not the HUCA-validated coved-ST phenotype.

### Where the HUCA model's false positives land

Diagnoses enriched among the top-scoring MIMIC controls: RBBB 7.0×, bifascicular block 8.4×,
LAFB 4.9×. Universe-wide at threshold 0.214 the same picture: bifascicular 6.7×, LBBB 5.5×,
pacing 5.0×, RBBB 3.5×. The model learned right-precordial repolarization plus RBBB-like
conduction delay, so in a pathology-rich database it fires on genuine conduction disease — which
is the real clinical differential for type-1 Brugada, not a bug.

### MIMIC false-positive count at the refit operating point (0.2143)

| | count |
|---|---:|
| MIMIC controls flagged (false positives) | **470 of 2,841 = 16.5%** |
| MIMIC brugada-labelled flagged | 13 of 57 = 22.8% |
| MIMIC brugada-labelled missed | 44 of 57 |
| PPV against MIMIC's own labels | 13/483 = **2.7%** |

22.8% vs 16.5% is the whole of the 0.52 AUROC.

Threshold sweep (HUCA sensitivity is out-of-fold on HUCA; FP is MIMIC controls):

| threshold | HUCA sens | MIMIC FP | FP rate |
|---:|---:|---:|---:|
| 0.05 | 0.855 | 1,144 | 40.3% |
| 0.10 | 0.829 | 793 | 27.9% |
| **0.2143** | **0.763** | **470** | **16.5%** |
| 0.50 | 0.566 | 151 | 5.3% |
| 0.75 | 0.316 | 43 | 1.5% |

**The 470 false positives are not random.** 390 of them (83.0%) carry a conduction abnormality —
RBBB, LBBB, fascicular block, IVCD or pacing. Of the remaining 80, most carry ST deviation (52),
MI (46) or STEMI (27). Conduction disease and ischemic ST elevation are the two textbook Brugada
phenocopies. By cohort the flags concentrate in the sick populations — MIMIC-LVEF40 24.6%,
VT 23.9%, STEMI-NSTEMI 18.7% — while the clean MIMIC-Control cohort is 3 of 616 (0.5%).

The model behaves like a right-precordial ST/T-morphology detector, which is what type-1 Brugada
is. What it cannot do is the differential.

### Universe-wide firing at threshold 0.214

| group | n | flagged |
|---|---:|---:|
| HUCA cases (OOF) | 76 | 76.3% |
| HUCA controls (OOF) | 287 | 6.6% |
| **MIMIC SR normals** | 19,805 | **0.1%** |
| MIMIC-Control | 616 | 0.5% |
| PTB-XL | 4,028 | 10.4% |
| MIMIC-LVEF40 | 1,093 | 24.6% |
| VT | 343 | 23.9% |

**Against clean normals the model is near-silent (22 of 19,805).** It is specific; what it cannot
do is separate Brugada from other conduction abnormalities. Treating HUCA cases as the only true
positives in the universe gives an implied PPV of 3.3% — but that number is driven entirely by
the conduction-disease population, not by normals.

## Installed into source_safe_canonical_v1 (2026-08-14)

Built by `_huca_build_model_package.py` (staged to `model_package/`), installed by
`_huca_install_model_package.py`. The rebuild reproduces the audited harness exactly
(max score delta 0.00e+00). Shipped version backed up to `shipped_brugada_backup_20260814/`.

| field | shipped | rebuilt |
|---|---:|---:|
| train_sources | `['MIMIC']` | `['Brugada-HUCA']` |
| excluded_train_sources | `['Brugada-HUCA']` | `[]` |
| n_total_pos | 420 | **76** |
| n_train_pos / n_train_neg | 57 / 2,841 | 76 / 287 |
| n_external_pos | 363 | 0 |
| pos_by_source['Brugada-HUCA'] | 363 | 76 |
| neg_by_source['Brugada-HUCA'] | 0 | 287 |
| primary_auc | 0.7102 | **0.9058** |
| final_full_auc | 0.8433 | 0.9422 |
| default_threshold | 0.00725 | 0.2738 |

Also `final_apr28_not_dx_auc` 0.896, `final_clean_normal_auc` 0.974. Manifest macro AUCs moved
0.922466 → 0.928579 (primary) and 0.947980 → 0.951071 (final). The global `config` block was left
as-is because it still describes the other 31 classes; the brugada exception is recorded in
`manifest.json` notes. Once the labels are corrected the protocol selects `Brugada-HUCA` as the
only train source **on its own** — MIMIC drops below `min_pos_per_source` with 0 positives — so no
exclusion override is needed any more.

**Not committed.** `.gitignore:107` excludes `domains/*/WaveMedix/models/*` with an explicit
carve-out for `.dvc` pointers, so the whole package is untracked; `git status` shows nothing.
Models here are meant to be DVC-versioned, and `source_safe_canonical_v1` has no `.dvc` pointer.
See "Remaining steps".

## Installed into the Ammonix ECG Agent repo (scrubbed)

`_huca_install_open_repo.py` installs the same rebuilt class into
`/path/to/ammonix-ecg-agent/models/source_safe_canonical_v1/`, scrubbed for public
release by importing the repo's **own** `scripts/scrub_model_manifest.py`, so it cannot drift
from that policy. `train_sources` becomes `["clinical-partner"]`, and the by-source count maps
are folded. Backup: `open_repo_brugada_backup_20260814/`.

**One risk this surfaced.** The scrub script folds *exact* source-name strings
(`node in PRIVATE_SOURCES`). Free-text prose naming the hospital passes straight through. My
private-side `notes` say "Brugada-HUCA" and "HUCA metadata.csv" repeatedly, and the shipped
scrub would not have caught them. The open-repo notes are therefore written public-safe at
source, not scrubbed afterwards, and the installed files are asserted free of the string.

**Pre-existing leak, unchanged by this work:** 30 other `diagnoses/*/metadata.json` files in the
open package still contain `Brugada-HUCA`. `scrub_model_manifest.py` only ever reads
`manifest.json`. Anyone can read the partner name out of the per-diagnosis model cards, which is
exactly what that script's own docstring says the policy exists to prevent.

## Universe cache rebuilt (2026-08-14) — what the agent actually shows

Swapping the model changed nothing the ECG Agent displays: it reads
`ammonix-ecg-agent/_staging/universe/`, not the model package. That cache still held the old
classifier's scores and the old labels. `_huca_rebuild_universe.py` fixes the brugada class only.
Backup: `universe_backup_20260814/`.

| | before | after |
|---|---:|---:|
| brugada positives in the universe | 133 | **76** (HUCA only) |
| rows predicted brugada | 7,294 (11.5%) | **3,982 (6.3%)** |
| universe-wide AUROC | 0.9571 | 0.9159 |
| HUCA-internal AUROC | — | 0.9058 |

The 57 MIMIC-derived brugada labels were dropped. `metadata.json.thresholds` was updated too —
**the backend reads thresholds from the universe metadata, not the model manifest**, so changing
only the model package would have left the old behaviour in place.

Row order comes from `metadata.json.matrix_specs`: apr28 0–14,297 · normals 14,298–34,102 ·
LVEF cases 34,103–40,242 · LVEF controls 40,243–63,255. Alignment was verified before writing:
31,208 of 34,103 rows matched the old column exactly, and the 2,895 that differed are precisely
the MIMIC rows the old model trained on.

### The fold-mean trap, measured

The cache builder stores the **mean of all five fold models for every row**. In 5-fold CV each row
is held out by one fold and *trained on by the other four*, so that mean is 4/5 in-sample for any
row the model saw.

Under the old model this was harmless here: HUCA was excluded from training, so all 363 rows were
external. Under the new model **HUCA is the training set and holds all 76 positives**:

| | AUROC | sensitivity | specificity |
|---|---:|---:|---:|
| Out-of-fold (shipped) | **0.9058** | 0.724 | 0.958 |
| Naive 5-fold mean | **1.0000** | 1.000 | 1.000 |

A perfect classifier, purely from memorisation — median case score 0.617 → 0.883 while controls
stay flat. Letting the standard cache builder regenerate this column would have published that.
The fix: use `validation_scores.npy` verbatim for rows 0–34,102 (it already stores out-of-fold for
trained rows and a fold mean for external ones) and compute a fold mean only for the genuinely
external 29,153 LVEF rows.

*Correction to an earlier draft:* the old 0.9571 was **not** in-sample-inflated. It differed from
the model's 0.8433 because of the label set (133 corrected vs the broken 420) and a larger, easier
negative pool (63,256 vs 34,103).

### Operating point — deliberate, do not "fix"

| threshold | rows firing | sensitivity | PPV |
|---|---:|---:|---:|
| **`primary_f1` 0.2738 (kept)** | **3,982 (6.3%)** | **0.724** | **1.4%** |
| `full_f1` 0.8877 | 100 (0.16%) | 0.184 | 14.0% |

Kept at `primary_f1` by decision 2026-08-14. Brugada carries sudden-death risk, suspicion cases are
common, and the false positives are **conduction disease, not normal ECGs** — the model fires on
~0.1% of clean sinus-rhythm normals. The cost is surfacing abnormal ECGs for review, not alarming
healthy people; raising to `full_f1` would miss four of every five real cases. This rationale is
recorded in the shipped `notes` of the model metadata, the manifest and the universe metadata so
the low PPV is not later mistaken for a defect.

## The three AUROCs in the UI, exactly (`_huca_three_aurocs.py`)

The ROC panel shows `area 0.916`, `held-out 0.942 final · 0.906 primary`. All three use the
**same 76 positives and the same scores**. Only the negative pool differs.

| shown as | negatives | n | AUROC |
|---|---|---:|---:|
| `0.906 primary` | the 287 HUCA controls only | 287 | 0.9058 |
| `0.942 final` | every negative in the model's dataset (Apr28 14,298 + MIMIC SR normals 19,805) | 34,027 | 0.9422 |
| `area 0.916` | every negative in the displayed universe (adds the 29,153 LVEF rows) | 63,180 | 0.9159 |

AUROC is the probability that a random positive outranks a random negative. Split the negatives
into groups and it is **exactly** the weighted average of the per-group AUROCs, weighted by each
group's share of the negatives. Measured against the same 76 positives:

| negative group | n | share | AUROC |
|---|---:|---:|---:|
| MIMIC-LVEF | 29,153 | 46% | 0.8853 |
| MIMIC SR normals | 19,805 | 31% | 0.9754 |
| PTB-XL | 4,028 | 6% | 0.9229 |
| CPSC | 3,150 | 5% | 0.8721 |
| Georgia | 1,994 | 3% | 0.9352 |
| Chapman | 1,301 | 2% | 0.8767 |
| other clinical-partner cohorts | 1,207 | 2% | 0.8516 |
| MIMIC | 1,057 | 2% | 0.8137 |
| MIMIC-Control | 608 | 1% | 0.9766 |
| CPSC-Extra | 403 | <1% | 0.9395 |
| **HUCA controls** | **287** | <1% | **0.9058** |
| MIMIC-LVEF40 | 187 | <1% | 0.8167 |

Σ wᵍ·AUROCᵍ = 0.9159, against 0.9159 computed directly — the identity closes to 1e-16.

So the movement is entirely negative-set composition:

- **0.906 → 0.942** adds 19,805 clean sinus-rhythm normals at 0.9754 plus the PhysioNet sources.
  Easy negatives dominate, so the average rises.
- **0.942 → 0.916** adds 29,153 LVEF rows at 0.8853, which become 46% of all negatives — below
  the running average, so it pulls back down. Low-EF hearts carry conduction disease and abnormal
  repolarization, the same features this model keys on, so they are hard negatives. Consistent
  with 2,565 of the 3,982 firing rows being MIMIC-LVEF.

**0.906 is the number to quote for the classifier.** It is the only one whose negatives are
matched to the cases — same study, same hospital, same protocol, all investigated for suspected
Brugada. The other two are higher or lower according to how easy the borrowed negatives happen
to be, which is a property of the universe's composition, not of the classifier.

## Remaining steps

1. **Versioning decision.** Either `dvc add` the package (the repo's intended mechanism, needs a
   reachable remote) or `git add -f` (overrides a deliberate policy and puts ~2 MB of model JSON in
   git history). Until then the change is machine-local, backed up only by
   `shipped_brugada_backup_20260814/`.
2. **Universe cache rebuild.** `probabilities.npy` is regenerated by the cache builder as a 5-fold
   mean over every row, so the universe still displays the old brugada scores until it is rebuilt.
3. **Threshold change is 38×** (0.00725 → 0.2738). Anything downstream that hardcoded the old
   default, or any stored alert/flag computed with it, needs re-checking.

## Caveats

1. `oof_auroc`, `primary_auc` and `final_full_auc` are **the same number by construction** here.
   With every row `Brugada-HUCA` and `excluded_train_sources=()`, `train_eligible` is all-True,
   so `train_pos == pos_mask`. Do not read them as three corroborating results.
2. `final_clean_normal_auc` changes meaning under the HUCA restriction and is not comparable to
   the same field in shipped source-safe runs. Ignore it.
3. ~14 cases per held-out fold is thin. The 10-seed spread (sd 0.008–0.009) is the honest
   uncertainty on the fold structure; the bootstrap CI (±0.05) is the uncertainty on the cohort.
4. 363 patients from one hospital, one protocol, one conversion pipeline. Nothing here speaks to
   transportability, and per the project's own bimodal-transportability finding, a
   threshold-morphology diagnosis like this is exactly the kind that may not transport.
5. Controls are people **investigated for suspected Brugada and ruled out** — a hard, well-matched
   negative set, not healthy volunteers. That makes 0.889 more meaningful than the same number
   against clean normals would be.
