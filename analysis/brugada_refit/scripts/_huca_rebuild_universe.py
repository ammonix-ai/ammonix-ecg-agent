"""Rebuild the brugada column of the ECG Agent universe cache.

The model swap alone changed nothing the agent displays: it reads
ammonix-ecg-agent/_staging/universe/, whose probabilities.npy column 30 and patients.jsonl
still come from the old MIMIC-trained classifier at threshold 0.00725 -- which predicts
brugada for 7,294 of 63,256 records (11.5%), including 2,836 MIMIC sinus-rhythm normals.

This rebuilds, for the brugada class only:
  probabilities.npy[:, 30]   new HUCA-trained scores (OOF for HUCA, 5-fold mean elsewhere)
  patients.jsonl             drops the 57 MIMIC-derived brugada labels; recomputes
                             predictions / top_prediction / max_score / primary / xgb_correct
  metadata.json              thresholds['brugada syndrome'] 0.00725 -> 0.2738, macro AUCs
  metrics_recomputed.json    brugada entry

Row order is verified: universe rows 0-34,102 align with the model's validation_scores.npy
(31,208/34,103 identical; the 2,895 that differ are exactly the MIMIC rows the OLD model
trained on, where validation_scores holds OOF and the universe holds a fold mean).
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")
sys.path.insert(0, "/path/to/ammonix-ecg-agent/backend")

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from ammonix.diagnostics import source_safe_cv as ss
from diagnosis_severity import clinical_labels, primary_diagnosis  # backend/ on sys.path

OPEN = Path("/path/to/ammonix-ecg-agent")
U = OPEN / "_staging" / "universe"
PKG = OPEN / "models" / "source_safe_canonical_v1" / "diagnoses" / "brugada_syndrome"
DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
BACKUP = Path("/path/to/huca_brugada_refit_workdir/universe_backup_20260814")
DX = "brugada syndrome"


def log(m=""):
    print(m, flush=True)


BACKUP.mkdir(parents=True, exist_ok=True)
for name in ("patients.jsonl", "probabilities.npy", "metadata.json", "metrics_recomputed.json"):
    if not (BACKUP / name).exists():
        shutil.copy2(U / name, BACKUP / name)
log(f"backed up universe -> {BACKUP}")

meta = json.load(open(U / "metadata.json", encoding="utf-8"))
classes = meta["classes"]
BI = classes.index(DX)
new_model_meta = json.load(open(PKG / "metadata.json", encoding="utf-8"))
NEW_THR = new_model_meta["default_threshold"]
OLD_THR = meta["thresholds"][DX]
log(f"brugada column index {BI}; threshold {OLD_THR:.6f} -> {NEW_THR:.6f}")

# ---------------------------------------------------------------- new scores
val = np.load(PKG / "validation_scores.npy").astype(np.float64)   # 34,103 apr28+normals
assert val.shape == (34103,)

boosters, feats = [], []
for i in range(5):
    b = xgb.Booster(); b.load_model(str(PKG / f"fold_{i}.xgb.json"))
    boosters.append(b)
    feats.append(np.load(PKG / f"fold_{i}.feature_indices.npy"))


def score_matrix(path: Path, n_expected: int) -> np.ndarray:
    X = np.load(path, mmap_mode="r")
    assert X.shape[0] == n_expected, f"{path.name}: {X.shape[0]} != {n_expected}"
    out = np.zeros(X.shape[0], dtype=np.float64)
    for b, top in zip(boosters, feats):
        for s in range(0, X.shape[0], 4000):
            blk = np.asarray(X[s:s + 4000][:, top], dtype=np.float32)
            out[s:s + 4000] += b.predict(xgb.DMatrix(blk))
    return out / 5.0


log("scoring the LVEF matrices with the 5 fold models (external -> fold mean) ...")
cases = score_matrix(DATA / "cases_X.npy", 6140)
log(f"  cases_X    {cases.shape} median {np.median(cases):.4f}")
controls = score_matrix(DATA / "controls_X.npy", 23013)
log(f"  controls_X {controls.shape} median {np.median(controls):.4f}")

new_col = np.concatenate([val, cases, controls])
assert new_col.shape == (63256,)

P = np.load(U / "probabilities.npy")
old_col = P[:, BI].astype(np.float64).copy()
P[:, BI] = new_col.astype(np.float32)
np.save(U / "probabilities.npy", P)
log(f"\nprobabilities.npy[:, {BI}] rewritten")
log(f"  old: median {np.median(old_col):.5f}  >=old_thr {int((old_col >= OLD_THR).sum())}")
log(f"  new: median {np.median(new_col):.5f}  >=new_thr {int((new_col >= NEW_THR).sum())}")

# ------------------------------------------------- which universe rows are HUCA
apr28_ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = [ss.source_of_manifest_path(r, man.get(r, "")) for r in apr28_ids]
is_huca = np.zeros(63256, dtype=bool)
is_huca[:14298] = np.asarray([c == "Brugada-HUCA" for c in cohort])
log(f"\nHUCA rows in universe: {int(is_huca.sum())}")

# ---------------------------------------------------------------- patients.jsonl
meta["thresholds"][DX] = NEW_THR
T = np.array([meta["thresholds"][c] for c in classes])
scored = {k.lower() for k in meta["thresholds"]}

stats = Counter()
out_path = U / "patients.jsonl.tmp"
with open(U / "patients.jsonl", encoding="utf-8") as fin, \
     open(out_path, "w", encoding="utf-8", newline="\n") as fout:
    for i, line in enumerate(fin):
        row = json.loads(line)
        labels = list(row.get("labels") or [])

        # DECISION 2026-08-14: MIMIC-derived brugada labels are not the HUCA-validated
        # phenotype (transfer is chance both ways) -> drop them.
        if DX in labels and not is_huca[i]:
            labels = [l for l in labels if l != DX]
            row["labels"] = labels
            stats["labels_dropped"] += 1
        elif DX in labels:
            stats["labels_kept_huca"] += 1

        p = P[i].astype(np.float64)
        preds = [c for c, v, t in zip(classes, p, T) if v >= t]
        before = set(row.get("predictions") or [])
        row["predictions"] = preds
        row["top_prediction"] = classes[int(np.argmax(p))]
        row["max_score"] = float(p.max())
        row["primary"] = primary_diagnosis(labels)
        gold = {d.lower() for d in clinical_labels(labels) if d.lower() in scored}
        row["xgb_correct"] = gold == {q.lower() for q in preds}
        if set(preds) != before:
            stats["prediction_sets_changed"] += 1
        if DX in preds:
            stats["still_predict_brugada"] += 1
        if row["xgb_correct"]:
            stats["xgb_correct_after"] += 1
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
out_path.replace(U / "patients.jsonl")

log("\npatients.jsonl rewritten:")
log(f"  brugada labels dropped (MIMIC-derived): {stats['labels_dropped']}")
log(f"  brugada labels kept (HUCA)            : {stats['labels_kept_huca']}")
log(f"  rows whose prediction set changed     : {stats['prediction_sets_changed']}")
log(f"  rows still predicting brugada         : {stats['still_predict_brugada']}  (was 7294)")

# ---------------------------------------------------------------- metadata
for d in meta.get("diagnoses", []):
    if isinstance(d, dict) and d.get("diagnosis") == DX:
        for k in list(d.keys()):
            if k in new_model_meta:
                d[k] = new_model_meta[k]
acc = [d for d in meta.get("diagnoses", []) if isinstance(d, dict) and "primary_auc" in d]
if acc:
    meta["macro_primary_auc"] = float(np.mean([d["primary_auc"] for d in acc]))
    meta["macro_final_auc"] = float(np.mean([d["final_full_auc"] for d in acc]))
meta.setdefault("notes", [])
json.dump(meta, open(U / "metadata.json", "w", encoding="utf-8"), indent=2)
log(f"\nmetadata.json: threshold updated, macro_primary={meta.get('macro_primary_auc')}")

# ---------------------------------------------------------------- metrics
y = np.zeros(63256, dtype=int)
lab_pos = 0
for i, line in enumerate(open(U / "patients.jsonl", encoding="utf-8")):
    if DX in (json.loads(line).get("labels") or []):
        y[i] = 1; lab_pos += 1
full_new = roc_auc_score(y, new_col)
mr = json.load(open(U / "metrics_recomputed.json", encoding="utf-8"))
for e in mr.get("classes", []):
    if e.get("dx") == DX:
        e["n_pos_new"] = int(lab_pos)
        e["full_new"] = float(full_new)
        e["cohort_new"] = float(roc_auc_score(y[is_huca], new_col[is_huca]))
        log(f"\nmetrics_recomputed: n_pos {e['n_pos_old']} -> {lab_pos}, "
            f"full {e['full_old']:.4f} -> {full_new:.4f}, "
            f"HUCA-internal cohort AUROC {e['cohort_new']:.4f}")
json.dump(mr, open(U / "metrics_recomputed.json", "w", encoding="utf-8"), indent=2)

log("\n" + "=" * 66)
log("REBUILD COMPLETE")
log("=" * 66)
log(f"  brugada positives in universe : 133 -> {lab_pos}  (HUCA only)")
log(f"  rows predicted brugada        : 7294 -> {stats['still_predict_brugada']}")
log(f"  universe-wide AUROC           : {full_new:.4f} (labels = HUCA-validated only)")
