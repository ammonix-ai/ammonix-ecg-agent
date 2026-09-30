"""What does fixing ONLY the labels give, with the deployed classifier untouched?

The live universe model (source_safe_canonical_v1) stores per-row scores in
validation_scores.npy for all 34,103 rows (14,298 Apr28 + 19,805 MIMIC SR normals).
That model was trained on 57 MIMIC positives with Brugada-HUCA excluded, and it labels
ALL 363 HUCA rows positive -- the 287 healthy controls included.

Here we keep those scores exactly as they are and only correct the labels
(HUCA: 76 positive / 287 negative), then recompute every AUROC.
Read-only: nothing is written to the private training repository.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
from sklearn.metrics import roc_auc_score

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
MODEL = Path("/path/to/private_training_repo/domains/ecg-12lead/WaveMedix/models/"
             "source_safe_canonical_v1/diagnoses/brugada_syndrome")
OUT = Path("/path/to/huca_brugada_refit_workdir")
DX = ss.HUCA_BRUGADA_LABEL


def p(m=""):
    print(m, flush=True)


scores = np.load(MODEL / "validation_scores.npy")
meta = json.load(open(MODEL / "metadata.json", encoding="utf-8"))
p(f"deployed validation_scores.npy: shape={scores.shape} dtype={scores.dtype}")

apr28_ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
normal_ids = ss.load_recording_ids(DATA / "normals_ids.csv")
ids = apr28_ids + normal_ids
p(f"row order: {len(apr28_ids)} apr28 + {len(normal_ids)} normals = {len(ids)}"
  f"   (metadata n_total={meta['n_total']})")
assert len(ids) == len(scores) == meta["n_total"], "row-count mismatch"

man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = {r: ss.source_of_manifest_path(r, man.get(r, "")) for r in apr28_ids}
huca_ids = [r for r in apr28_ids if cohort[r] == "Brugada-HUCA"]

hmeta = {}
for row in csv.DictReader(open(ss.find_huca_metadata_path(man), encoding="utf-8")):
    hmeta[str(row["patient_id"]).strip()] = int(float(row["brugada"]))

# ---- labels as the DEPLOYED model has them (source-level: every HUCA row positive)
apr28_labels = ss.load_apr28_labels(DATA / "apr28_labels.jsonl")
deployed_pos = np.zeros(len(ids), dtype=bool)
for i, r in enumerate(apr28_ids):
    lab = apr28_labels.get(r, set())
    deployed_pos[i] = (DX in lab) or (cohort[r] == "Brugada-HUCA")

# ---- labels CORRECTED from HUCA metadata.csv
corrected_pos = np.zeros(len(ids), dtype=bool)
excluded = np.zeros(len(ids), dtype=bool)   # atypical brugada==2, dropped
for i, r in enumerate(apr28_ids):
    if cohort[r] == "Brugada-HUCA":
        b = hmeta[r]
        corrected_pos[i] = b > 0          # inclusive rule = the shipped loader's, 76 cases
    else:
        corrected_pos[i] = DX in apr28_labels.get(r, set())

huca_mask = np.zeros(len(ids), dtype=bool)
for i, r in enumerate(apr28_ids):
    huca_mask[i] = cohort[r] == "Brugada-HUCA"

p(f"\ndeployed  positives: {int(deployed_pos.sum())}  (metadata n_total_pos={meta['n_total_pos']})")
p(f"corrected positives: {int(corrected_pos.sum())}   "
  f"= {int((corrected_pos & huca_mask).sum())} HUCA + {int((corrected_pos & ~huca_mask).sum())} MIMIC-derived")
p(f"HUCA rows moving positive -> negative: "
  f"{int((deployed_pos & ~corrected_pos & huca_mask).sum())}")

res = {}

# ------------------------------------------------- the numbers the metadata reports
p("\n" + "=" * 78)
p("AS SHIPPED (deployed labels, deployed scores)")
p("=" * 78)
a = roc_auc_score(deployed_pos.astype(int), scores)
p(f"  final_full_auc recomputed : {a:.4f}   (metadata says {meta['final_full_auc']:.4f})")
res["shipped_final_full_auc_recomputed"] = float(a)

# ------------------------------------------------- label-only fix, same scores
p("\n" + "=" * 78)
p("LABEL-ONLY FIX (corrected labels, SAME deployed scores, classifier untouched)")
p("=" * 78)
a_full = roc_auc_score(corrected_pos.astype(int), scores)
p(f"  final_full_auc            : {a_full:.4f}   (was {meta['final_full_auc']:.4f})")
res["labelfix_final_full_auc"] = float(a_full)

# HUCA-internal: 76 cases vs 287 controls, scored by the deployed MIMIC-trained model
hp = corrected_pos & huca_mask
hn = (~corrected_pos) & huca_mask
a_huca = roc_auc_score(np.r_[np.ones(int(hp.sum())), np.zeros(int(hn.sum()))],
                       np.r_[scores[hp], scores[hn]])
p(f"  HUCA-internal (76 vs 287) : {a_huca:.4f}")
p(f"     median score  HUCA cases   : {np.median(scores[hp]):.5f}")
p(f"     median score  HUCA controls: {np.median(scores[hn]):.5f}")
res["labelfix_huca_internal_auc"] = float(a_huca)
res["n_huca_pos"] = int(hp.sum())
res["n_huca_neg"] = int(hn.sum())

# MIMIC-internal primary, unchanged by the HUCA relabel
mimic_pos = corrected_pos & ~huca_mask
p(f"\n  MIMIC-derived positives ({int(mimic_pos.sum())}) median score: "
  f"{np.median(scores[mimic_pos]):.5f}")

p("\n" + "=" * 78)
p("COMPARISON")
p("=" * 78)
p(f"  {'quantity':46s} {'value':>8s}")
p(f"  {'-' * 56}")
p(f"  {'deployed model, HUCA 76 vs 287 (label fix only)':46s} {a_huca:8.4f}")
p(f"  {'HUCA-trained refit, 76 vs 287 (2026-08-13)':46s} {0.9058:8.4f}")
p(f"  {'canonical sweep table 2026-06-05, primary':46s} {0.9095:8.4f}")
p(f"  {'deployed metadata primary_auc (MIMIC-internal)':46s} {meta['primary_auc']:8.4f}")
p(f"  {'deployed metadata final_full_auc (420 pos)':46s} {meta['final_full_auc']:8.4f}")
p(f"  {'label-fixed final_full_auc (133 pos)':46s} {a_full:8.4f}")

with open(OUT / "label_only_fix.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
p(f"\nwrote {OUT / 'label_only_fix.json'}")
