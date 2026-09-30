# Paper alignment audit — the paper vs this repository

Audited 2026-09-21 against the paper on Overleaf at commit `547540c`, re-checked 2026-09-30
against Overleaf commit `3260b13` (table numbers below follow that version), and the
repository at source commit `3511e59` (`09ac12d` plus a frontend retheme). Method: the paper's claims about the system and the numbers that the
repository documents were compared one by one. Deviations are written down, not silently
fixed.

## Scope, stated once

The repository is the serving application of the paper's second experiment (the 32-diagnosis
scale-up): the feature pipeline, the inference backend, the dialogue agent and the viewer. The
first experiment (17 diagnoses, RHO-VR Skill induction, the comparators) was run with code that
is not part of this repository. `REPRODUCING.md` classifies every table and figure.

## Numbers checked

| Paper element | Repository source | Result |
|---|---|---|
| Table 7, 32 per-diagnosis AUROCs and case counts | `docs/auroc-metrics.md`, per-class table | 31 of 32 match the documented values: 30 rows equal the `final` column, the Brugada row equals `primary`. The ST-deviation row (0.883, 2,378 cases) is the display-label pool; the documented `final` value is 0.889 on 2,250 cases |
| Same-source macro AUROC 0.929 | `docs/auroc-metrics.md`, macro of `primary` | matches (0.929) |
| Mean AUROC 0.950 (Table 7 caption) | `docs/auroc-metrics.md`, macro of `final`: 0.951 | differs by 0.001; the paper's Brugada row uses the same-source value, the documented macro may not. To be confirmed by the authors |
| Brugada row: 0.906, 76 cases vs 287 controls | `analysis/brugada_refit/REPORT.md` and its result files | matches (0.9058); the ten-seed mean 0.9009 is documented next to it |
| Table 8: NPV and never-fire NPV | sensitivity, specificity and prevalence of the same table | internally consistent (checked for myocardial infarction: 98.28% and 96.85% follow from 0.47, 0.987 and 1,059 of 33,664) |
| 439 quarantined infarction labels (Table 8 caption) | `packaging/recompute_metrics.py`, header | same figure; the quarantine marker is not in the published universe, which the script states |
| Universe size 63,256; features 26,490; 32 classes | `README.md`, `docs/auroc-metrics.md` | match |

## Deviations

1. **Which AUROC is "the" number.** The paper's Table 7 reports each diagnosis against all
   non-case records (the repository's `final`), and gives the same-source figure only as a
   macro average (0.929). The repository's `docs/auroc-metrics.md` calls the same-source
   figure (`primary`) the one to quote. Both are defined and disclosed on both sides; the
   per-class gap is large for the infarction classes (NSTEMI 0.773 vs 0.935, STEMI 0.807 vs
   0.904, myocardial infarction 0.877 vs 0.949, old infarction 0.733 vs 0.827). Neither side
   is wrong; a reader should know the paper's per-class values are the more favourable
   definition.
2. **Label provenance.** The paper describes the universe as recordings "with
   physician-documented diagnoses". `packaging/recover_machine_labels.py` documents that
   48,958 records of the displayed universe (the LVEF arms and the sinus-rhythm normals) had
   no diagnosis codes and received their displayed labels from the recording machine's
   automatic statements. Per `docs/auroc-metrics.md` these recovered labels affect the `area`
   figures of the displayed universe, not the `primary` and `final` figures the paper reports.
3. **Data sources.** `docs/auroc-metrics.md` lists 1,570 "clinical-partner" rows, 363 of them
   Brugada-HUCA and 1,207 from other cohorts. Since Overleaf `3260b13` the paper names the
   Pre-/Post-STEMI ECG Database (University of Michigan) as a further public source, and says
   that infarction diagnoses and LVEF values for MIMIC-derived recordings come from credentialed
   MIMIC-IV data (discharge summaries, echocardiography). Whether the 1,207 rows are exactly the
   Pre-/Post-STEMI database is not stated in the repository.
4. **Names.** The paper calls the system "the Ammonix ECG Agent" and its feature stage "the
   tokenizer" or "mechanistic world model"; the code calls the feature stage `q_psi` and, before
   this release, called the application "Ammonix ECG Agent". The release uses the paper's
   system name; `q_psi` stays as the code-level name of the tokenizer.
5. **Skills.** The paper's central result is that admitted Skills edit the classifier's label
   set (193 proposed, 165 admitted; micro-F1 0.72 to 0.80 on the 17-diagnosis task). That Skill store
   is not in the repository. `backend/services/skills.py` states what was carried over: the
   selection rule and the domain groups, run in observation mode. They tell the dialogue agent
   what to check and flag conflicts; they never change a probability or a prediction. So the
   released application reports the frozen classifier's calls as they are, and does not
   reproduce the paper's Skill-edited labels.

## Limitations the paper states, and whether the repository agrees

Retrospective evidence only; no prospective evaluation; thresholds are design parameters; a
concurrent second reader under the responsible clinician. The repository adds the matching
line in three places: "Research demonstration — not a medical device, not for clinical use."
