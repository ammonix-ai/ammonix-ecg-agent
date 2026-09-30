"""Rule out per-record ADC gain as the driver of the HUCA Brugada AUROC.

Per-record gain alone separates cases from controls at AUROC ~0.63, so it has to be
excluded as an amplitude/scaling artefact. Two tests:
  A. gain-stratified AUROC on the existing OOF scores (within gain quartiles)
  B. full refit restricted to gain-invariant features (no millivolt-valued features)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
from sklearn.metrics import roc_auc_score

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
OUT = Path("/path/to/huca_brugada_refit_workdir")
DX = ss.HUCA_BRUGADA_LABEL


def p(m=""):
    print(m, flush=True)


ids_all = ss.load_recording_ids(DATA / "apr28_ids.csv")
man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
coh = [ss.source_of_manifest_path(r, man.get(r, "")) for r in ids_all]
idx = np.asarray([i for i, c in enumerate(coh) if c == "Brugada-HUCA"])
ids = [ids_all[i] for i in idx]

meta = {}
for row in csv.DictReader(open(ss.find_huca_metadata_path(man), encoding="utf-8")):
    meta[str(row["patient_id"]).strip()] = {
        k: int(float(row[k])) for k in ("basal_pattern", "sudden_death", "brugada")
    }

keep_rel = [i for i, r in enumerate(ids) if meta[r]["brugada"] in (0, 1)]
sub_ids = [ids[i] for i in keep_rel]
y = np.asarray([meta[r]["brugada"] == 1 for r in sub_ids], dtype=int)


def mean_gain(rid: str) -> float:
    g = []
    for line in Path(man[rid]).read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                g.append(float(parts[2].split("(")[0]))
            except ValueError:
                pass
    return float(np.mean(g)) if g else float("nan")


gain = np.asarray([mean_gain(r) for r in sub_ids])

# ------------------------------------------------------------------ A. stratified
p("=" * 78)
p("A. GAIN-STRATIFIED AUROC on existing OOF scores (confirmed variant)")
p("=" * 78)
score_path = OUT / "oof_scores_confirmed.csv"
if not score_path.exists():
    p(f"  {score_path} not written yet - run _huca_brugada_refit.py first")
else:
    rows = {r["recording_id"]: float(r["oof_score"]) for r in csv.DictReader(open(score_path, encoding="utf-8"))}
    s = np.asarray([rows[r] for r in sub_ids])
    p(f"  overall OOF AUROC            : {roc_auc_score(y, s):.4f}")
    p(f"  AUROC of gain alone          : {roc_auc_score(y, gain):.4f}")
    p(f"  corr(OOF score, gain) Pearson: {np.corrcoef(s, gain)[0, 1]:.4f}")
    q = np.quantile(gain, [0.25, 0.5, 0.75])
    strat = np.digitize(gain, q)
    aucs = []
    for k in range(4):
        m = strat == k
        yy = y[m]
        if len(np.unique(yy)) < 2:
            p(f"  gain quartile {k + 1}: n={int(m.sum())} - degenerate, skipped")
            continue
        a = roc_auc_score(yy, s[m])
        aucs.append((a, int(m.sum()), int(yy.sum())))
        p(f"  gain quartile {k + 1}: n={int(m.sum())} cases={int(yy.sum())} AUROC={a:.4f}")
    if aucs:
        w = sum(n for _, n, _ in aucs)
        p(f"  -> size-weighted mean within-quartile AUROC: {sum(a * n for a, n, _ in aucs) / w:.4f}")

# ------------------------------------------------------------------ B. refit
p()
p("=" * 78)
p("B. REFIT ON GAIN-INVARIANT FEATURES ONLY (drop every millivolt-valued feature)")
p("=" * 78)
cols = (DATA / "apr28_cols.txt").read_text(encoding="utf-8").splitlines()
amp = np.asarray(["mv" in c.lower() for c in cols])
p(f"  total features {len(cols)}; amplitude(mV) {int(amp.sum())}; gain-invariant {int((~amp).sum())}")

X_all = np.load(DATA / "apr28_X.npy", mmap_mode="r")
X = np.ascontiguousarray(np.asarray(X_all[idx[np.asarray(keep_rel)]], dtype=np.float32))
del X_all

base_labels = ss.load_apr28_labels(DATA / "apr28_labels.jsonl")
status = {r: bool(meta[r]["brugada"] == 1) for r in sub_ids}
labels = [ss.apply_huca_brugada_metadata(r, "Brugada-HUCA", set(base_labels.get(r, set())), status)
          for r in sub_ids]
subjects = np.asarray([ss.subject_of_apr28_id(r, "Brugada-HUCA", {}, {}) for r in sub_ids], dtype=object)


def run(Xsub, tag):
    ds = ss.SourceSafeDataset(
        X=Xsub, recording_ids=sub_ids,
        sources=np.asarray(["Brugada-HUCA"] * len(sub_ids), dtype=object),
        subjects=subjects, labels=labels,
        is_clean_normal=np.asarray([ss.is_clean_apr28_normal(l) for l in labels], dtype=bool),
        is_apr28=np.ones(len(sub_ids), dtype=bool),
    )
    aucs = []
    for seed in range(5):
        cfg = ss.SourceSafeConfig(excluded_train_sources=(), random_state=seed)
        _, sc = ss.run_source_safe_diagnosis(ds, DX, cfg)
        aucs.append(float(roc_auc_score(y, sc)))
    p(f"  {tag:28s} n_feat={Xsub.shape[1]:6d}  AUROC mean={np.mean(aucs):.4f} "
      f"sd={np.std(aucs):.4f} min={min(aucs):.4f} max={max(aucs):.4f}")
    return aucs


res = {
    "all_features": run(X, "all features"),
    "gain_invariant": run(np.ascontiguousarray(X[:, ~amp]), "gain-invariant only (no mV)"),
    "amplitude_only": run(np.ascontiguousarray(X[:, amp]), "amplitude only (mV features)"),
}

p()
p("  Interpretation: if gain-invariant features alone stay near the full-feature AUROC,")
p("  the result is not an amplitude/scaling artefact.")

import json
with open(OUT / "gain_check.json", "w", encoding="utf-8") as f:
    json.dump({"auroc_gain_alone": float(roc_auc_score(y, gain)), "refits": res}, f, indent=2)
p(f"  wrote {OUT / 'gain_check.json'}")
