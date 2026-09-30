"""How many MIMIC records does the HUCA-trained Brugada model flag?

Train on HUCA (76 vs 287), score the 2,898 MIMIC-derived rows as external (5-model mean),
then count flags at several thresholds against MIMIC's own (unvalidated) Brugada labels.
Also: what ARE the flagged controls, and how many have no conduction abnormality to explain
the flag. Saves per-row scores. Read-only.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
OUT = Path("/path/to/huca_brugada_refit_workdir")
DX = ss.HUCA_BRUGADA_LABEL
THR = 0.2143

MIMIC_COHORTS = {"MIMIC-LVEF40", "MIMIC-Control", "Brugada", "VT", "pre-AF",
                 "Amyloidosis", "Chagas", "STEMI-NSTEMI"}
CONDUCTION = {
    "right bundle branch block", "left bundle branch block / variation",
    "bifascicular block", "left anterior fascicular block",
    "left posterior fascicular block", "pacing rhythm",
    "non-specific intraventricular conduction delay",
    "incomplete right bundle branch block", "intraventricular conduction delay",
}


def p(m=""):
    print(m, flush=True)


apr28_ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = [ss.source_of_manifest_path(r, man.get(r, "")) for r in apr28_ids]
labels_file = ss.load_apr28_labels(DATA / "apr28_labels.jsonl")
hmeta = {}
for row in csv.DictReader(open(ss.find_huca_metadata_path(man), encoding="utf-8")):
    hmeta[str(row["patient_id"]).strip()] = int(float(row["brugada"]))

huca_rel = np.asarray([i for i, c in enumerate(cohort) if c == "Brugada-HUCA"])
mimic_rel = np.asarray([i for i, c in enumerate(cohort) if c in MIMIC_COHORTS])
y_h = np.asarray([hmeta[apr28_ids[i]] > 0 for i in huca_rel], dtype=int)
y_m = np.asarray([DX in labels_file.get(apr28_ids[i], set()) for i in mimic_rel], dtype=int)
p(f"HUCA {len(huca_rel)} ({int(y_h.sum())} cases) | MIMIC {len(mimic_rel)} "
  f"({int(y_m.sum())} brugada-labelled, {int((1 - y_m).sum())} controls)")

Xmm = np.load(DATA / "apr28_X.npy", mmap_mode="r")
Xh = np.ascontiguousarray(np.asarray(Xmm[huca_rel], dtype=np.float32))
Xm = np.ascontiguousarray(np.asarray(Xmm[mimic_rel], dtype=np.float32))
del Xmm

groups = np.asarray([f"H:{apr28_ids[i]}" for i in huca_rel], dtype=object)
oof = np.full(len(y_h), np.nan)
acc = np.zeros(len(y_m))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xh, y_h, groups):
    screen = ss.default_xgb_screen_factory()
    screen.fit(Xh[tr], y_h[tr])
    top = ss._top_features_from_screen(screen, 500, Xh.shape[1])
    clf = ss.default_xgb_classifier_factory()
    clf.fit(Xh[tr][:, top], y_h[tr])
    oof[te] = clf.predict_proba(Xh[te][:, top])[:, 1]
    acc += clf.predict_proba(Xm[:, top])[:, 1]
s_m = acc / 5
p(f"HUCA OOF AUROC {roc_auc_score(y_h, oof):.4f} | MIMIC AUROC {roc_auc_score(y_m, s_m):.4f}")

p("\n" + "=" * 78)
p("MIMIC CONFUSION AT SEVERAL THRESHOLDS (vs MIMIC's own unvalidated labels)")
p("=" * 78)
p(f"  {'threshold':>10s} {'sens_HUCA':>10s} {'TP':>5s} {'FP':>6s} {'TN':>6s} {'FN':>4s} "
  f"{'FP rate':>8s} {'PPV':>7s}")
rows = []
for thr in (0.05, 0.10, THR, 0.30, 0.50, 0.75):
    tp = int(((s_m >= thr) & (y_m == 1)).sum()); fp = int(((s_m >= thr) & (y_m == 0)).sum())
    tn = int(((s_m < thr) & (y_m == 0)).sum()); fn = int(((s_m < thr) & (y_m == 1)).sum())
    sens_h = float((oof[y_h == 1] >= thr).mean())
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    mark = "  <- refit operating point" if abs(thr - THR) < 1e-9 else ""
    p(f"  {thr:10.4f} {sens_h:10.3f} {tp:5d} {fp:6d} {tn:6d} {fn:4d} "
      f"{fp / max(1, fp + tn) * 100:7.1f}% {ppv:7.3f}{mark}")
    rows.append({"threshold": thr, "huca_sensitivity": sens_h, "tp": tp, "fp": fp,
                 "tn": tn, "fn": fn, "fp_rate": fp / max(1, fp + tn), "ppv": ppv})

p("\n" + "=" * 78)
p(f"THE {int(((s_m >= THR) & (y_m == 0)).sum())} FLAGGED MIMIC CONTROLS AT {THR}")
p("=" * 78)
fp_mask = (s_m >= THR) & (y_m == 0)
fp_idx = mimic_rel[fp_mask]
by_coh = Counter(cohort[i] for i in fp_idx)
tot_coh = Counter(cohort[i] for i in mimic_rel[y_m == 0])
p("  by cohort:")
for c, n in by_coh.most_common():
    p(f"    {c:18s} {n:5d} of {tot_coh[c]:5d}  ({n / tot_coh[c] * 100:5.1f}%)")

has_cond = 0
no_cond_labels = Counter()
for i in fp_idx:
    lab = labels_file.get(apr28_ids[i], set())
    if lab & CONDUCTION:
        has_cond += 1
    else:
        no_cond_labels.update(lab if lab else {"(no diagnosis)"})
nfp = len(fp_idx)
p(f"\n  with a conduction abnormality (RBBB/LBBB/fascicular/pacing/IVCD): "
  f"{has_cond}/{nfp} = {has_cond / nfp * 100:.1f}%")
p(f"  without any conduction label                                    : "
  f"{nfp - has_cond}/{nfp} = {(nfp - has_cond) / nfp * 100:.1f}%")
p("\n  most common diagnoses among the non-conduction flagged controls:")
for dx, c in no_cond_labels.most_common(10):
    p(f"    {dx[:44]:44s} {c:4d}")

with open(OUT / "mimic_flag_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["recording_id", "cohort", "mimic_brugada_label", "score", "flagged_at_0.2143"])
    for j, i in enumerate(mimic_rel):
        w.writerow([apr28_ids[i], cohort[i], int(y_m[j]), f"{s_m[j]:.6f}",
                    int(s_m[j] >= THR)])
json.dump({"mimic_auc": float(roc_auc_score(y_m, s_m)), "thresholds": rows,
           "fp_with_conduction": has_cond, "fp_total": nfp},
          open(OUT / "mimic_fp_counts.json", "w", encoding="utf-8"), indent=2)
p(f"\nwrote {OUT / 'mimic_fp_counts.json'} and mimic_flag_scores.csv")
