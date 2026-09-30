"""Apply the HUCA-trained Brugada classifier to MIMIC Brugada vs MIMIC controls.

Train on Brugada-HUCA only (76 cases vs 287 controls, the exact refit that scores 0.906
out-of-fold). MIMIC rows are never trained on, so the harness scores them as `external`
= the mean of the five fold models, which is the same convention the deployed universe
uses for never-trained rows.

Then ask: does that classifier separate MIMIC's Brugada-labelled ECGs from MIMIC controls?

MIMIC here = the 2,898 MIMIC-derived Apr28 rows (57 brugada positive, 2,841 negative),
which reproduces the deployed model's n_train_eligible/n_train_pos/n_train_neg exactly.
Read-only; nothing is written to the private training repository.
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

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
OUT = Path("/path/to/huca_brugada_refit_workdir")
DX = ss.HUCA_BRUGADA_LABEL

MIMIC_COHORTS = {"MIMIC-LVEF40", "MIMIC-Control", "Brugada", "VT", "pre-AF",
                 "Amyloidosis", "Chagas", "STEMI-NSTEMI"}


def p(m=""):
    print(m, flush=True)


apr28_ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = [ss.source_of_manifest_path(r, man.get(r, "")) for r in apr28_ids]
labels_file = ss.load_apr28_labels(DATA / "apr28_labels.jsonl")

hmeta = {}
for row in csv.DictReader(open(ss.find_huca_metadata_path(man), encoding="utf-8")):
    hmeta[str(row["patient_id"]).strip()] = int(float(row["brugada"]))

huca_rel = [i for i, c in enumerate(cohort) if c == "Brugada-HUCA"]
mimic_rel = [i for i, c in enumerate(cohort) if c in MIMIC_COHORTS]
p(f"HUCA rows  : {len(huca_rel)}")
p(f"MIMIC rows : {len(mimic_rel)}   (deployed n_train_eligible = 2898)")

sel = np.asarray(huca_rel + mimic_rel)
sel_ids = [apr28_ids[i] for i in sel]
sources = np.asarray(["Brugada-HUCA"] * len(huca_rel) + ["MIMIC"] * len(mimic_rel), dtype=object)

labels = []
for i, rid in zip(sel, sel_ids):
    lab = set(labels_file.get(rid, set()))
    if cohort[i] == "Brugada-HUCA":
        lab.discard(DX)
        if hmeta[rid] > 0:                      # inclusive rule = 76 cases
            lab.add(DX)
    labels.append(lab)

y = np.asarray([DX in l for l in labels], dtype=int)
is_huca = sources == "Brugada-HUCA"
is_mimic = sources == "MIMIC"
p(f"positives: HUCA {int(y[is_huca].sum())}  MIMIC {int(y[is_mimic].sum())}"
  f"   (deployed n_train_pos = 57)")
p(f"negatives: HUCA {int((1 - y)[is_huca].sum())}  MIMIC {int((1 - y)[is_mimic].sum())}"
  f"   (deployed n_train_neg = 2841)")

X_all = np.load(DATA / "apr28_X.npy", mmap_mode="r")
X = np.ascontiguousarray(np.asarray(X_all[sel], dtype=np.float32))
del X_all
p(f"feature matrix: {X.shape}")

subjects = np.asarray(
    [f"Brugada-HUCA:{r}" if s == "Brugada-HUCA" else f"M?:{r}"
     for r, s in zip(sel_ids, sources)], dtype=object)

ds = ss.SourceSafeDataset(
    X=X, recording_ids=sel_ids, sources=sources, subjects=subjects, labels=labels,
    is_clean_normal=np.asarray([ss.is_clean_apr28_normal(l) for l in labels], dtype=bool),
    is_apr28=np.ones(len(sel_ids), dtype=bool),
)

# MIMIC excluded from training -> HUCA is the only train source; MIMIC becomes `external`
# and is scored by the 5-model mean, the deployed convention for never-trained rows.
rows = []
for seed in range(5):
    cfg = ss.SourceSafeConfig(excluded_train_sources=("MIMIC",), random_state=seed)
    res, scores = ss.run_source_safe_diagnosis(ds, DX, cfg)
    if seed == 0:
        p(f"\ntrain_sources={res.train_sources}  n_train_pos={res.n_train_pos} "
          f"n_train_neg={res.n_train_neg}  n_external_pos={res.n_external_pos}")
        s0 = scores.copy()
    huca_auc = roc_auc_score(y[is_huca], scores[is_huca])
    mimic_auc = roc_auc_score(y[is_mimic], scores[is_mimic])
    rows.append((huca_auc, mimic_auc))
    p(f"  seed {seed}: HUCA-internal(OOF) {huca_auc:.4f}   "
      f"MIMIC-internal(external) {mimic_auc:.4f}")

h = np.asarray([r[0] for r in rows]); m = np.asarray([r[1] for r in rows])
p(f"\n  HUCA-internal  mean {h.mean():.4f} sd {h.std():.4f}")
p(f"  MIMIC-internal mean {m.mean():.4f} sd {m.std():.4f}")

p("\n" + "=" * 78)
p("SCORE DISTRIBUTIONS (seed 0)")
p("=" * 78)
for name, mask in (("HUCA cases", is_huca & (y == 1)), ("HUCA controls", is_huca & (y == 0)),
                   ("MIMIC brugada-labelled", is_mimic & (y == 1)),
                   ("MIMIC controls", is_mimic & (y == 0))):
    v = s0[mask]
    p(f"  {name:24s} n={len(v):5d}  median {np.median(v):.4f}  "
      f"q25 {np.percentile(v, 25):.4f}  q75 {np.percentile(v, 75):.4f}")

# what do the highest-scoring MIMIC controls look like?
p("\n" + "=" * 78)
p("WHAT THE HUCA MODEL CALLS POSITIVE INSIDE MIMIC")
p("=" * 78)
mi = np.where(is_mimic & (y == 0))[0]
order = mi[np.argsort(s0[mi])[::-1]]
top = order[:100]
base = Counter()
for i in mi:
    base.update(labels[i])
topc = Counter()
for i in top:
    topc.update(labels[i])
n_base, n_top = len(mi), len(top)
p(f"  top-100 highest-scoring MIMIC controls vs all {n_base} MIMIC controls:")
p(f"  {'diagnosis':34s} {'top100':>7s} {'base%':>7s} {'top%':>7s} {'enrich':>7s}")
enr = []
for dx, c in topc.most_common():
    if c < 5:
        continue
    bp_ = base[dx] / n_base * 100
    tp_ = c / n_top * 100
    if bp_ > 0:
        enr.append((tp_ / bp_, dx, c, bp_, tp_))
for e, dx, c, bp_, tp_ in sorted(enr, reverse=True)[:12]:
    p(f"  {dx[:34]:34s} {c:7d} {bp_:6.1f}% {tp_:6.1f}% {e:6.2f}x")

json.dump({
    "huca_internal_auc_mean": float(h.mean()), "huca_internal_auc_sd": float(h.std()),
    "mimic_internal_auc_mean": float(m.mean()), "mimic_internal_auc_sd": float(m.std()),
    "per_seed": [{"seed": i, "huca": float(a), "mimic": float(b)}
                 for i, (a, b) in enumerate(rows)],
    "n_mimic_pos": int(y[is_mimic].sum()), "n_mimic_neg": int((1 - y)[is_mimic].sum()),
    "median_scores": {
        "huca_cases": float(np.median(s0[is_huca & (y == 1)])),
        "huca_controls": float(np.median(s0[is_huca & (y == 0)])),
        "mimic_cases": float(np.median(s0[is_mimic & (y == 1)])),
        "mimic_controls": float(np.median(s0[is_mimic & (y == 0)])),
    },
}, open(OUT / "huca_to_mimic_transport.json", "w", encoding="utf-8"), indent=2)
p(f"\nwrote {OUT / 'huca_to_mimic_transport.json'}")
