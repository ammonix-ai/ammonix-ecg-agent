"""If the HUCA-trained Brugada model were deployed universe-wide, what fires?

Train on HUCA (76 vs 287), score every other Apr28 row + the 19,805 MIMIC SR normals as
external (5-model mean), then apply the out-of-fold F1 threshold (~0.21) and count.
Read-only.
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
THR = 0.2143  # out-of-fold F1-optimal threshold from the confirmed-variant refit


def p(m=""):
    print(m, flush=True)


apr28_ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
normal_ids = ss.load_recording_ids(DATA / "normals_ids.csv")
man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = [ss.source_of_manifest_path(r, man.get(r, "")) for r in apr28_ids]
labels_file = ss.load_apr28_labels(DATA / "apr28_labels.jsonl")

hmeta = {}
for row in csv.DictReader(open(ss.find_huca_metadata_path(man), encoding="utf-8")):
    hmeta[str(row["patient_id"]).strip()] = int(float(row["brugada"]))

huca_rel = np.asarray([i for i, c in enumerate(cohort) if c == "Brugada-HUCA"])
y_h = np.asarray([hmeta[apr28_ids[i]] > 0 for i in huca_rel], dtype=int)

Xmm = np.load(DATA / "apr28_X.npy", mmap_mode="r")
Xh = np.ascontiguousarray(np.asarray(Xmm[huca_rel], dtype=np.float32))
rest_rel = np.asarray([i for i, c in enumerate(cohort) if c != "Brugada-HUCA"])
Xr = np.ascontiguousarray(np.asarray(Xmm[rest_rel], dtype=np.float32))
del Xmm
Xn = np.load(DATA / "normals_X.npy", mmap_mode="r")
p(f"HUCA {Xh.shape}  other-apr28 {Xr.shape}  normals {Xn.shape}")

groups = np.asarray([f"H:{apr28_ids[i]}" for i in huca_rel], dtype=object)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

oof = np.full(len(y_h), np.nan)
sum_r = np.zeros(len(rest_rel))
sum_n = np.zeros(Xn.shape[0])
for tr, te in splitter.split(Xh, y_h, groups):
    screen = ss.default_xgb_screen_factory()
    screen.fit(Xh[tr], y_h[tr])
    top = ss._top_features_from_screen(screen, 500, Xh.shape[1])
    clf = ss.default_xgb_classifier_factory()
    clf.fit(Xh[tr][:, top], y_h[tr])
    oof[te] = clf.predict_proba(Xh[te][:, top])[:, 1]
    sum_r += clf.predict_proba(Xr[:, top])[:, 1]
    for s in range(0, Xn.shape[0], 4000):       # normals in chunks
        blk = np.asarray(Xn[s:s + 4000][:, top], dtype=np.float32)
        sum_n[s:s + 4000] += clf.predict_proba(blk)[:, 1]
s_r, s_n = sum_r / 5, sum_n / 5
p(f"HUCA OOF AUROC {roc_auc_score(y_h, oof):.4f}   threshold {THR}")

p("\n" + "=" * 78)
p(f"WHO FIRES AT THRESHOLD {THR}")
p("=" * 78)
p(f"  {'group':38s} {'n':>7s} {'>=thr':>7s} {'rate':>7s}")
groups_out = [("HUCA cases (OOF)", oof[y_h == 1]), ("HUCA controls (OOF)", oof[y_h == 0])]
by_src = {}
for j, i in enumerate(rest_rel):
    by_src.setdefault(cohort[i], []).append(s_r[j])
for src in sorted(by_src, key=lambda k: -len(by_src[k])):
    groups_out.append((f"{src} (external)", np.asarray(by_src[src])))
groups_out.append(("MIMIC SR normals (external)", s_n))
rows = []
for name, v in groups_out:
    k = int((v >= THR).sum())
    p(f"  {name:38s} {len(v):7d} {k:7d} {k / len(v) * 100:6.1f}%")
    rows.append({"group": name, "n": len(v), "n_flagged": k, "rate": k / len(v)})

n_true = int((oof[y_h == 1] >= THR).sum())
n_false = sum(r["n_flagged"] for r in rows if r["group"] not in
              ("HUCA cases (OOF)",))
total_neg = sum(r["n"] for r in rows if r["group"] != "HUCA cases (OOF)")
p(f"\n  true positives flagged : {n_true} of {int(y_h.sum())}")
p(f"  everything else flagged: {n_false} of {total_neg}")
p(f"  implied PPV if HUCA cases were the only true positives in this universe: "
  f"{n_true / max(1, n_true + n_false) * 100:.1f}%")

# which diagnoses dominate the firing set outside HUCA
p("\n" + "=" * 78)
p("DIAGNOSES ENRICHED AMONG FLAGGED NON-HUCA APR28 ROWS")
p("=" * 78)
flag = s_r >= THR
base, topc = Counter(), Counter()
for j, i in enumerate(rest_rel):
    lab = labels_file.get(apr28_ids[i], set())
    base.update(lab)
    if flag[j]:
        topc.update(lab)
nb, nt = len(rest_rel), int(flag.sum())
p(f"  {nt} of {nb} non-HUCA Apr28 rows flagged ({nt / nb * 100:.1f}%)")
enr = []
for dx, c in topc.most_common():
    if c < 10:
        continue
    bp_, tp_ = base[dx] / nb * 100, c / nt * 100
    if bp_ > 0.2:
        enr.append((tp_ / bp_, dx, c, bp_, tp_))
p(f"  {'diagnosis':34s} {'flagged':>8s} {'base%':>7s} {'flag%':>7s} {'enrich':>7s}")
for e, dx, c, bp_, tp_ in sorted(enr, reverse=True)[:12]:
    p(f"  {dx[:34]:34s} {c:8d} {bp_:6.1f}% {tp_:6.1f}% {e:6.2f}x")

json.dump({"threshold": THR, "huca_oof_auroc": float(roc_auc_score(y_h, oof)),
           "groups": rows}, open(OUT / "universe_ppv.json", "w", encoding="utf-8"), indent=2)
p(f"\nwrote {OUT / 'universe_ppv.json'}")
