"""Quantify the fold-mean vs out-of-fold gap for the HUCA rows.

In 5-fold CV each row is held out by exactly ONE fold and used for TRAINING by the other
four. So the 5-model mean that the universe cache stores is 4/5 in-sample for any row the
model trained on.

Under the OLD brugada model this was harmless for HUCA: HUCA was excluded from training,
so all 363 rows were external and their fold-mean was honest.

Under the NEW model HUCA IS the training set and holds ALL 76 positives, so a naive
fold-mean would make every positive 4/5 in-sample. This measures how big that would be.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
PKG = Path("/path/to/ammonix-ecg-agent/models/source_safe_canonical_v1/diagnoses/brugada_syndrome")


def p(m=""):
    print(m, flush=True)


ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = [ss.source_of_manifest_path(r, man.get(r, "")) for r in ids]
huca_rel = np.asarray([i for i, c in enumerate(cohort) if c == "Brugada-HUCA"])
hmeta = {}
for row in csv.DictReader(open(ss.find_huca_metadata_path(man), encoding="utf-8")):
    hmeta[str(row["patient_id"]).strip()] = int(float(row["brugada"]))
y = np.asarray([hmeta[ids[i]] > 0 for i in huca_rel], dtype=int)

X = np.load(DATA / "apr28_X.npy", mmap_mode="r")
Xh = np.ascontiguousarray(np.asarray(X[huca_rel], dtype=np.float32))
del X

# the honest number the package already stores: OOF for these rows
oof = np.load(PKG / "validation_scores.npy")[huca_rel].astype(np.float64)

# what a naive 5-fold mean over every row would have given instead
acc = np.zeros(len(y))
for i in range(5):
    b = xgb.Booster(); b.load_model(str(PKG / f"fold_{i}.xgb.json"))
    top = np.load(PKG / f"fold_{i}.feature_indices.npy")
    acc += b.predict(xgb.DMatrix(Xh[:, top]))
foldmean = acc / 5.0

auc_oof = roc_auc_score(y, oof)
auc_fm = roc_auc_score(y, foldmean)
p("=" * 70)
p("HUCA rows (76 cases vs 287 controls) under the NEW HUCA-trained model")
p("=" * 70)
p(f"  out-of-fold (what the package stores, what we ship) : AUROC {auc_oof:.4f}")
p(f"  naive 5-fold mean (4/5 in-sample)                   : AUROC {auc_fm:.4f}")
p(f"  inflation if we had used the fold mean              : +{auc_fm - auc_oof:.4f}")
p()
p(f"  median score, cases    : OOF {np.median(oof[y == 1]):.4f}  vs fold-mean {np.median(foldmean[y == 1]):.4f}")
p(f"  median score, controls : OOF {np.median(oof[y == 0]):.4f}  vs fold-mean {np.median(foldmean[y == 0]):.4f}")

thr = 0.273754
p()
p(f"  at threshold {thr}: OOF flags {int((oof >= thr).sum())} of 363, "
  f"fold-mean flags {int((foldmean >= thr).sum())}")
p(f"  sensitivity: OOF {(oof[y == 1] >= thr).mean():.3f}  fold-mean {(foldmean[y == 1] >= thr).mean():.3f}")
p(f"  specificity: OOF {(oof[y == 0] < thr).mean():.3f}  fold-mean {(foldmean[y == 0] < thr).mean():.3f}")
