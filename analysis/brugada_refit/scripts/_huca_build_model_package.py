"""Build a drop-in replacement for the source_safe_canonical_v1 brugada_syndrome class.

Changes vs the shipped build:
  * labels come from HUCA metadata.csv (76 positives), not the source-level label
    that made all 363 HUCA rows positive (n_total_pos 420, 287 healthy controls included)
  * the 57 MIMIC brugada-labelled rows are treated as NEGATIVES (decision 2026-08-14)
  * Brugada-HUCA is no longer excluded from training; with the labels corrected it is the
    only source with >= 30 positives, so the protocol selects it on its own

Everything else -- 5-fold subject-grouped CV, top-500 in-fold screen, XGBoost swarm,
external rows scored by the 5-model mean -- is the shipped protocol, unchanged.

Writes to a STAGING dir. Installation into the private training repository is a separate, explicit step.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
LIVE = Path("/path/to/private_training_repo/domains/ecg-12lead/WaveMedix/models/source_safe_canonical_v1")
STAGE = Path("/path/to/huca_brugada_refit_workdir/model_package")
RECORD_LIST = Path(json.load(open(LIVE / "manifest.json", encoding="utf-8"))["record_list"])
DX = ss.HUCA_BRUGADA_LABEL
SAFE = "brugada_syndrome"


def log(m=""):
    print(m, flush=True)


# --------------------------------------------------------------------- dataset
log("loading dataset (apr28 + MIMIC SR normals), this takes a minute ...")
ds = ss.load_qpsi_jun3_dataset(DATA, record_list_path=RECORD_LIST, include_clean_normals=True)
log(f"  rows={len(ds.recording_ids)}  features={ds.X.shape[1]}")

n_before = sum(1 for l in ds.labels if DX in l)
huca = ds.sources == "Brugada-HUCA"
# DECISION 2026-08-14: the 57 MIMIC brugada-labelled rows become negatives.
stripped = 0
for i, l in enumerate(ds.labels):
    if DX in l and not huca[i]:
        l.discard(DX)
        stripped += 1
n_after = sum(1 for l in ds.labels if DX in l)
log(f"  '{DX}' positives: {n_before} -> {n_after}  ({stripped} MIMIC rows relabelled negative)")
assert n_after == 76, f"expected 76 HUCA positives, got {n_after}"

cfg = ss.SourceSafeConfig(excluded_train_sources=())
train_sources, _ = ss.train_sources_for_diagnosis(
    ds, DX, min_pos_per_source=cfg.min_pos_per_source,
    min_neg_per_source=cfg.min_neg_per_source, excluded_train_sources=())
log(f"  auto-selected train_sources = {train_sources}")
assert train_sources == ["Brugada-HUCA"], train_sources

# --------------------------------------- reference run through the audited harness
log("\nreference run via run_source_safe_diagnosis ...")
ref, ref_scores = ss.run_source_safe_diagnosis(ds, DX, cfg)
log(f"  primary_auc={ref.primary_auc:.6f}  final_full_auc={ref.final_full_auc:.6f}")

# ------------------------------------- same protocol again, keeping the fold models
y = np.asarray([DX in l for l in ds.labels], dtype=int)
usable = np.ones(len(y), dtype=bool)
pos_mask = y == 1
train_eligible = np.isin(ds.sources, train_sources) & ds.is_apr28
train_idx = np.where(train_eligible)[0]
external_idx = np.where(~train_eligible)[0]
groups = ds.subjects[train_idx]

scores = np.full(len(y), np.nan)
ext_sum = np.zeros(len(external_idx))
folds_meta = []
STAGE.mkdir(parents=True, exist_ok=True)
cols = (DATA / "apr28_cols.txt").read_text(encoding="utf-8").splitlines()

splitter = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
for fold, (rel_tr, rel_te) in enumerate(splitter.split(ds.X[train_idx], y[train_idx], groups)):
    abs_tr, abs_te = train_idx[rel_tr], train_idx[rel_te]
    screen = ss.default_xgb_screen_factory()
    screen.fit(ds.X[abs_tr], y[abs_tr])
    top = ss._top_features_from_screen(screen, cfg.top_k, ds.X.shape[1])
    clf = ss.default_xgb_classifier_factory()
    clf.fit(ds.X[abs_tr][:, top], y[abs_tr])
    scores[abs_te] = clf.predict_proba(ds.X[abs_te][:, top])[:, 1]
    ext_sum += clf.predict_proba(ds.X[external_idx][:, top])[:, 1]

    clf.get_booster().save_model(str(STAGE / f"fold_{fold}.xgb.json"))
    np.save(STAGE / f"fold_{fold}.feature_indices.npy", top.astype(np.int32))
    folds_meta.append({
        "fold": fold,
        "model_path": f"diagnoses/{SAFE}/fold_{fold}.xgb.json",
        "feature_index_path": f"diagnoses/{SAFE}/fold_{fold}.feature_indices.npy",
        "n_train": len(abs_tr), "n_test": len(abs_te),
        "n_train_pos": int(y[abs_tr].sum()), "n_test_pos": int(y[abs_te].sum()),
        "n_subject_overlap": len(set(ds.subjects[abs_tr]) & set(ds.subjects[abs_te])),
        "top_feature_count": len(top),
        "top_features_preview": [cols[j] for j in top[:20]],
    })
    log(f"  fold {fold}: train={len(abs_tr)} (pos {int(y[abs_tr].sum())}) "
        f"test={len(abs_te)} (pos {int(y[abs_te].sum())}) overlap={folds_meta[-1]['n_subject_overlap']}")
scores[external_idx] = ext_sum / cfg.n_splits

assert all(f["n_subject_overlap"] == 0 for f in folds_meta), "subject overlap!"
delta = float(np.nanmax(np.abs(scores - ref_scores)))
log(f"\n  max |score - harness score| = {delta:.2e}  (must be ~0)")
assert delta < 1e-6, "rebuilt scores diverge from the harness"

# ------------------------------------------------------------------- metrics
train_pos = train_eligible & pos_mask
train_neg = train_eligible & ~pos_mask
primary_auc = ss.auc_from_scores(scores[train_pos], scores[train_neg])
final_full = ss.auc_from_scores(scores[pos_mask], scores[~pos_mask])
final_apr28 = ss.auc_from_scores(scores[pos_mask], scores[~pos_mask & ds.is_apr28])
final_clean = ss.auc_from_scores(scores[pos_mask], scores[~pos_mask & ds.is_clean_normal])
log(f"  primary_auc            {primary_auc:.6f}")
log(f"  final_full_auc         {final_full:.6f}")
log(f"  final_apr28_not_dx_auc {final_apr28:.6f}")
log(f"  final_clean_normal_auc {final_clean:.6f}")


def f1_block(pm, nm):
    ps, ns = scores[pm], scores[nm]
    y_ = np.r_[np.ones(len(ps), int), np.zeros(len(ns), int)]
    s_ = np.r_[ps, ns]
    best = None
    for t in np.unique(s_):
        pred = (s_ >= t).astype(int)
        tp = int(((pred == 1) & (y_ == 1)).sum()); fp = int(((pred == 1) & (y_ == 0)).sum())
        fn = int(((pred == 0) & (y_ == 1)).sum()); tn = int(((pred == 0) & (y_ == 0)).sum())
        f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "specificity": tn / (tn + fp) if tn + fp else 0.0,
                    "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
                    "true_positive_rate": tp / (tp + fn) if tp + fn else 0.0,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return best


thresholds = {"full_f1": f1_block(pos_mask, ~pos_mask),
              "primary_f1": f1_block(train_pos, train_neg)}
log(f"  primary_f1 threshold {thresholds['primary_f1']['threshold']:.6f} "
    f"(shipped was 0.00725)")

srcs = ss.ordered_sources(ds.sources)
meta = {
    "diagnosis": DX, "safe_name": SAFE, "status": "accepted",
    "model_kind": "source_safe_apr28_plus_sr_controls",
    "train_sources": train_sources, "excluded_train_sources": [],
    "n_total": int(usable.sum()), "n_total_pos": int(pos_mask.sum()),
    "n_total_neg": int((~pos_mask).sum()),
    "n_train_eligible": int(train_eligible.sum()),
    "n_train_pos": int(train_pos.sum()), "n_train_neg": int(train_neg.sum()),
    "n_external_pos": int((pos_mask & ~train_eligible).sum()),
    "primary_auc": primary_auc, "final_full_auc": final_full,
    "final_apr28_not_dx_auc": final_apr28, "final_clean_normal_auc": final_clean,
    "thresholds": thresholds, "default_threshold_kind": "primary_f1",
    "default_threshold": thresholds["primary_f1"]["threshold"],
    "folds": folds_meta,
    "pos_by_source": {s: int((pos_mask & (ds.sources == s)).sum()) for s in srcs},
    "neg_by_source": {s: int((~pos_mask & (ds.sources == s)).sum()) for s in srcs},
    "notes": [
        "Refit 2026-08-14 on Brugada-HUCA only: 76 cases vs 287 same-study controls, "
        "labels from the HUCA metadata.csv 'brugada' column (source-of-truth).",
        "Supersedes the build that trained on MIMIC (57 positives) with Brugada-HUCA "
        "excluded and counted all 363 HUCA rows as positive (n_total_pos 420), which "
        "placed 287 healthy HUCA controls in the positive class.",
        "brugada=2 (7 atypical) counted positive, matching load_huca_brugada_metadata. "
        "Confirmed-only variant (69 cases) gives OOF AUROC 0.889.",
        "The 57 MIMIC brugada-labelled records are treated as NEGATIVES by decision "
        "2026-08-14. HUCA<->MIMIC transfer is chance in both directions "
        "(HUCA->MIMIC 0.519, MIMIC->HUCA 0.546) although each cohort is internally "
        "learnable, so the two label sets are different constructs.",
        "SCOPE: false positives concentrate in conduction disease (83% of flagged MIMIC "
        "controls carry RBBB/LBBB/fascicular block/IVCD/pacing); the model is near-silent "
        "on clean normals (22 of 19,805 MIMIC SR normals fire). Not a general-population "
        "Brugada detector without a conduction-abnormality differential.",
    ],
}
np.save(STAGE / "validation_scores.npy", scores.astype(np.float32))
json.dump(meta, open(STAGE / "metadata.json", "w", encoding="utf-8"), indent=2)

log("\n" + "=" * 70)
log("SHIPPED -> REBUILT")
log("=" * 70)
old = json.load(open(LIVE / "diagnoses" / SAFE / "metadata.json", encoding="utf-8"))
for k in ("train_sources", "excluded_train_sources", "n_total_pos", "n_total_neg",
          "n_train_pos", "n_train_neg", "n_external_pos", "primary_auc",
          "final_full_auc", "default_threshold"):
    o, n = old.get(k), meta.get(k)
    fo = f"{o:.4f}" if isinstance(o, float) else str(o)
    fn = f"{n:.4f}" if isinstance(n, float) else str(n)
    log(f"  {k:24s} {fo:>28s}  ->  {fn}")
log(f"  {'pos_by_source(HUCA)':24s} {old['pos_by_source']['Brugada-HUCA']:>28d}  ->  "
    f"{meta['pos_by_source']['Brugada-HUCA']}")
log(f"  {'neg_by_source(HUCA)':24s} {old['neg_by_source']['Brugada-HUCA']:>28d}  ->  "
    f"{meta['neg_by_source']['Brugada-HUCA']}")
log(f"\nstaged in {STAGE}")
