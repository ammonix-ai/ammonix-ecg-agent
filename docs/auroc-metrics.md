# The three AUROC numbers, and why they differ

The Universe page shows three AUROCs for every diagnosis. In the Model Stats panel:

```
area 0.916
held-out 0.942 final · 0.906 primary
n+ 76 · n− 63,180 in universe · op t=0.274
```

and the per-class table lists `final` and `primary` for all 32 classes.

They are not three estimates of the same thing, and one is not "more correct" than the others.
They are **three different questions**. This document defines each exactly, shows the numbers for
every diagnosis, and says which one to quote.

---

## Definitions

Every classifier in this package is fitted with the same protocol: five folds, grouped by subject,
a top-500 feature screen inside each fold, and an XGBoost model per fold. Each record gets a score
from the one fold that did **not** train on it (out-of-fold); records outside the training pool get
the mean of the five fold models. Details in `analysis/brugada_refit/REPORT.md`.

### `primary` — against same-source controls

* **Positives:** the diagnosis's cases inside the sources the model trained on.
* **Negatives:** records from **those same sources** that do not carry the diagnosis.
* **Scores:** out-of-fold.

This is the hardest and most honest comparison: cases and controls come from the same hospitals,
machines and recording protocols, so nothing can be won by recognising *where* a trace came from.

### `final` — against every negative in the model's own dataset

* **Positives:** the same cases.
* **Negatives:** every non-case in the model's 34,103-record training universe — the Apr28
  multi-source cohort plus 19,805 clean MIMIC sinus-rhythm normals.
* **Scores:** out-of-fold for trained rows, five-model mean for the rest.

Usually **higher** than `primary`, because most of what gets added is easy: clean normals are far
simpler to reject than a same-source patient who was investigated for the same condition.

### `area` — on the displayed universe

* **Positives:** every record in the 63,256-row universe whose gold labels include the diagnosis.
* **Negatives:** all remaining 63,256 records.
* **Scores:** the published `probabilities.npy` column — what the viewer actually plots.

This is the curve the ROC panel draws. It describes **this population as displayed**, which is why
it can sit above or below the held-out numbers.

---

## Three things change at once

For most diagnoses the three numbers differ for more than one reason. All three matter:

**1. The negative pool changes.** AUROC is the probability that a random positive outranks a random
negative. Partition the negatives into groups and the AUROC is *exactly* the weighted average of
the per-group AUROCs, weighted by each group's share of the negatives:

```
AUROC = Σ  w_g · AUROC(positives vs group g)
```

So adding easy negatives raises it and adding hard ones lowers it, with no change to the model.
Worked example for `brugada syndrome`, whose 76 positives and scores are identical in all three:

| negative group | n | share | AUROC |
|---|---:|---:|---:|
| MIMIC-LVEF | 29,153 | 46% | 0.885 |
| MIMIC sinus-rhythm normals | 19,805 | 31% | 0.975 |
| PTB-XL | 4,028 | 6% | 0.923 |
| CPSC | 3,150 | 5% | 0.872 |
| Georgia | 1,994 | 3% | 0.935 |
| Chapman | 1,301 | 2% | 0.877 |
| other clinical-partner cohorts | 1,207 | 2% | 0.852 |
| MIMIC | 1,057 | 2% | 0.814 |
| MIMIC-Control | 608 | 1% | 0.977 |
| CPSC-Extra | 403 | <1% | 0.940 |
| **HUCA controls (same study)** | **287** | <1% | **0.906** |
| MIMIC-LVEF40 | 187 | <1% | 0.817 |

`Σ w_g · AUROC_g` = 0.9159, against 0.9159 computed directly — the identity closes to 1e-16.

* `primary` 0.906 uses only the 287 same-study controls.
* `final` 0.942 adds 19,805 clean normals at 0.975 — easy negatives dominate, so it rises.
* `area` 0.916 then adds 29,153 LVEF rows at 0.885, which become 46% of all negatives and pull it
  back down. Low-EF hearts carry conduction disease and abnormal repolarization — the same features
  this model keys on — so they are hard negatives.

**2. The positive set changes.** `primary`/`final` count cases in the model's 34,103-row dataset;
`area` counts cases across all 63,256 universe rows, which include labels recovered for the LVEF and
normals cohorts. Compare the `n+ (model)` and `n+ (universe)` columns below: `myocardial infarction`
goes from 1,059 to 7,768, `left anterior fascicular block` from 189 to 2,070. Those extra positives
were never part of the held-out evaluation, so `area` is partly measuring generalisation to cases
the reported numbers never covered. **`old myocardial infarction` is the clearest warning**:
`primary` 0.733 but `area` 0.626, because the universe holds 676 positives against the model's 48.

**3. The score semantics differ.** `primary`/`final` always use held-out scores. The universe column
is built as a five-model **mean over every row**, which is 4/5 in-sample for rows the model trained
on — each row is held out by one fold and trained on by the other four. For most classes this
inflates `area` slightly. **`brugada syndrome` is the one exception**: its column was rebuilt to
carry genuine out-of-fold scores for its 363 training rows, because all 76 of its positives sit in
the training set and a plain fold mean scored **AUROC 1.000** — pure memorisation. See
`analysis/brugada_refit/`.

---

## All 32 diagnoses

Sorted by `area`. `n+ (model)` is the positive count behind `primary`/`final`; `n+ (universe)` is
the count behind `area`.

| diagnosis | primary | final | area | n+ (model) | n+ (universe) | trained on |
|---|---:|---:|---:|---:|---:|---|
| tachycardia | 0.973 | 0.964 | 0.999 | 107 | 107 | MIMIC |
| sinus tachycardia | 0.995 | 0.994 | 0.993 | 1016 | 3806 | PTB-XL+MIMIC+Georgia+Chapman+CPSC-Extra |
| bifascicular block | 0.978 | 0.995 | 0.992 | 78 | 961 | MIMIC |
| left bundle branch block / variations | 0.992 | 0.996 | 0.992 | 701 | 1842 | PTB-XL+MIMIC+CPSC+Georgia |
| axis right shift | 0.994 | 0.992 | 0.991 | 498 | 1213 | PTB-XL+MIMIC+Chapman |
| nstemi | 0.773 | 0.935 | 0.990 | 480 | 504 | MIMIC |
| atrial fibrillation | 0.984 | 0.988 | 0.990 | 1075 | 4411 | PTB-XL+MIMIC+CPSC+Georgia |
| left anterior fascicular block | 0.968 | 0.994 | 0.989 | 189 | 2070 | MIMIC |
| supraventricular tachycardia | 0.996 | 0.989 | 0.987 | 387 | 422 | Chapman |
| sinus bradycardia | 0.988 | 0.986 | 0.986 | 1348 | 4805 | PTB-XL+MIMIC+Georgia+Chapman |
| right bundle branch block | 0.984 | 0.978 | 0.985 | 1315 | 3778 | PTB-XL+MIMIC+CPSC+Georgia |
| wolff-parkinson-white | 0.964 | 0.957 | 0.984 | 145 | 166 | PTB-XL+Chapman |
| low qrs voltages | 0.960 | 0.983 | 0.980 | 231 | 3140 | MIMIC |
| ventricular premature beats | 0.962 | 0.978 | 0.980 | 912 | 3013 | MIMIC+CPSC |
| bradycardia | 0.959 | 0.941 | 0.976 | 197 | 302 | MIMIC+CPSC-Extra |
| axis left shift | 0.940 | 0.970 | 0.974 | 1440 | 5868 | PTB-XL+MIMIC+Georgia |
| pacing rhythm | 0.979 | 0.989 | 0.972 | 423 | 1438 | PTB-XL+MIMIC |
| lvef <40% expanded MIMIC EF>=55 controls | 0.921 | 0.939 | 0.971 | 6140 | 6140 | MIMIC-LVEF |
| 1st av-block | 0.974 | 0.980 | 0.969 | 753 | 3310 | PTB-XL+MIMIC+CPSC+Georgia |
| atrial premature beats | 0.911 | 0.930 | 0.949 | 645 | 2262 | PTB-XL+MIMIC+CPSC+Georgia |
| atrial flutter | 0.955 | 0.978 | 0.939 | 340 | 859 | MIMIC+Chapman |
| poor r wave progression | 0.860 | 0.907 | 0.933 | 108 | 1632 | MIMIC |
| left ventricular hypertrophy | 0.957 | 0.954 | 0.931 | 985 | 4017 | PTB-XL+MIMIC+Georgia+Chapman |
| qt interval extension | 0.882 | 0.913 | 0.919 | 305 | 1763 | MIMIC+Georgia |
| left atrial enlargement | 0.861 | 0.913 | 0.916 | 115 | 1561 | MIMIC |
| brugada syndrome | 0.906 | 0.942 | 0.916 | 76 | 76 | Brugada-HUCA |
| myocardial infarction | 0.877 | 0.949 | 0.903 | 1059 | 7768 | MIMIC |
| st deviation | 0.903 | 0.889 | 0.902 | 2250 | 10440 | PTB-XL+MIMIC+CPSC+Georgia+CPSC-Extra |
| STEMI | 0.807 | 0.904 | 0.893 | 531 | 976 | MIMIC |
| t wave change | 0.884 | 0.902 | 0.850 | 1564 | 6786 | PTB-XL+MIMIC+Georgia+Chapman |
| sinus irregularity | 0.893 | 0.879 | 0.833 | 612 | 1572 | PTB-XL+MIMIC+Georgia |
| old myocardial infarction | 0.733 | 0.827 | 0.626 | 48 | 676 | MIMIC |

Macro averages: **primary 0.929 · final 0.951 · area 0.944**. `area` exceeds `primary` for 20 of
32 classes.

---

## Which one to quote

**Quote `primary`.** It is the only one whose negatives are matched to its cases — same sources,
same acquisition, same protocol — so it measures the classifier rather than the composition of the
dataset it was pointed at. It is also the most conservative of the three for most classes.

Use `final` when the question is "how does this behave against everything else we hold, including
clean normals". Use `area` only to describe **this universe as displayed**; it is a property of the
current population, not a transferable performance claim. Adding 100,000 healthy volunteers would
push every `area` upward without a single model improving.

Two traps worth stating plainly:

* **A high `area` with a much lower `primary` usually means easy negatives, not a good classifier.**
  `nstemi` reads 0.990 on the universe and 0.773 against same-source controls. The second number is
  the one that predicts behaviour on a real NSTEMI workup.
* **An `area` far below `primary` means the universe holds positives the model never saw.** Check
  `n+ (universe)` against `n+ (model)` before concluding the model regressed —
  `old myocardial infarction` is 48 versus 676.

## Reproducing this

```bash
python analysis/brugada_refit/scripts/_huca_all_dx_aurocs.py    # the 32-class table
python analysis/brugada_refit/scripts/_huca_three_aurocs.py     # the brugada decomposition
```

Both read `_staging/universe/` and the model package; neither needs the private training matrices.
