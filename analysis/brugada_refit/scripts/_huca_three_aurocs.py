"""Why 0.9058, 0.9422 and 0.9159 differ: same positives, same scores, different negatives.

AUROC = P(a random positive scores above a random negative). When the negative pool is
split into groups g with weights w_g (share of all negatives), that probability is exactly
the weighted average of the per-group AUROCs:

    AUROC = sum_g  w_g * AUROC(positives vs group g)

So the three numbers are three different weighted averages over the same per-group values.
This script computes every group and checks the identity closes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

U = Path("/path/to/ammonix-ecg-agent/_staging/universe")
DX = "brugada syndrome"


def p(m=""):
    print(m, flush=True)


meta = json.load(open(U / "metadata.json", encoding="utf-8"))
BI = meta["classes"].index(DX)
P = np.load(U / "probabilities.npy", mmap_mode="r")
s = np.asarray(P[:, BI], dtype=np.float64)

src, lab = [], []
for line in open(U / "patients.jsonl", encoding="utf-8"):
    r = json.loads(line)
    src.append(r.get("source"))
    lab.append(DX in (r.get("labels") or []))
src = np.array(src)
y = np.array(lab)

# 'clinical-partner' covers ALL private cohorts (VT, Chagas, amyloidosis, STEMI, ...),
# so it cannot identify HUCA. Recover the true HUCA rows by index from the apr28 manifest.
import sys
sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")
from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
apr28_ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
is_huca = np.zeros(63256, dtype=bool)
is_huca[:14298] = np.asarray(
    [ss.source_of_manifest_path(r, man.get(r, "")) == "Brugada-HUCA" for r in apr28_ids]
)
huca_ctrl = is_huca & ~y

pos = s[y]
p(f"positives: {int(y.sum())} (identical in all three numbers)")
p(f"total negatives in the universe: {int((~y).sum())}")

# ---------------------------------------------------------------- per-group AUROC
p("\nper-group AUROC of the SAME 76 positives against each negative group:")
p(f"  {'negative group':34s} {'n':>7s} {'AUROC':>8s}")
groups = {}
groups["HUCA controls (same study)"] = huca_ctrl
groups["other clinical-partner cohorts"] = (src == "clinical-partner") & ~y & ~huca_ctrl
for name in ("MIMIC-LVEF", "MIMIC-SR", "PTB-XL", "CPSC", "Georgia", "Chapman",
             "MIMIC", "MIMIC-Control", "CPSC-Extra", "MIMIC-LVEF40"):
    groups[name] = (src == name) & ~y
rows = []
for name, m in groups.items():
    if m.sum() == 0:
        continue
    a = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(int(m.sum()))], np.r_[pos, s[m]])
    rows.append((name, int(m.sum()), a))
    p(f"  {name:34s} {int(m.sum()):7d} {a:8.4f}")
assert sum(n for _, n, _ in rows) == int((~y).sum()), "groups do not partition the negatives"

# ---------------------------------------------------------------- the three numbers
p("\n" + "=" * 72)
p("THE THREE NUMBERS = three negative pools over the same positives")
p("=" * 72)


def auc_over(mask, label):
    a = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(int(mask.sum()))], np.r_[pos, s[mask]])
    p(f"  {label:38s} negatives {int(mask.sum()):6d}   AUROC {a:.4f}")
    return a


# primary: HUCA controls only
auc_over(huca_ctrl, "primary  (HUCA controls only)")
# final_full: the model's own dataset = apr28 + MIMIC SR normals = rows 0..34,102
model_pool = np.zeros(len(y), dtype=bool)
model_pool[:34103] = True
auc_over(model_pool & ~y, "final    (model dataset, 34,103 rows)")
# displayed universe: everything
a_uni = auc_over(~y, "figure   (whole displayed universe)")

# ---------------------------------------------------------------- identity check
w = np.array([n for _, n, _ in rows], dtype=float)
w /= w.sum()
recon = float(np.sum(w * np.array([a for _, _, a in rows])))
p(f"\nweighted-average identity: sum(w_g * AUROC_g) = {recon:.4f}  vs computed {a_uni:.4f}")
p(f"  difference {abs(recon - a_uni):.2e}")

p("\nwhat moves the number:")
lvef = (src == "MIMIC-LVEF") & ~y
p(f"  the 29,153 LVEF rows are {lvef.sum() / (~y).sum() * 100:.0f}% of all negatives and score")
p(f"  median {np.median(s[lvef]):.4f} vs {np.median(s[(src == 'MIMIC-SR') & ~y]):.4f} for clean SR normals")
p(f"  -> they are HARD negatives and pull the average back down.")
