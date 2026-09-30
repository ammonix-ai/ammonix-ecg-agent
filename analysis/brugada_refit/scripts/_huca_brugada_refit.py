"""Refit the Brugada classifier on HUCA cases vs HUCA controls only.

Reuses the shipped 5-fold + XGBoost-swarm protocol (ammonix.diagnostics.source_safe_cv,
read-only) but:
  * restricts the dataset to the 363 Brugada-HUCA rows,
  * takes labels from the HUCA metadata.csv, not apr28_labels.jsonl,
  * sets excluded_train_sources=() so HUCA is trainable at all.

Two label variants:
  confirmed  brugada == 1  -> 69 cases vs 287 controls, the 7 atypical rows dropped
  inclusive  brugada  > 0  -> 76 cases vs 287 controls (matches load_huca_brugada_metadata)

Writes to huca_brugada_refit/. Never writes to the private training repository.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
from sklearn.metrics import roc_auc_score

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
OUT = Path("/path/to/huca_brugada_refit_workdir")
OUT.mkdir(parents=True, exist_ok=True)

DX = ss.HUCA_BRUGADA_LABEL  # "brugada syndrome"
N_SEEDS = 10
N_PERM = 10
N_BOOT = 5000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- load

def load_huca():
    """Return (X, ids, meta_by_id, base_labels) for the 363 HUCA rows."""
    ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
    manifest = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
    cohort = [ss.source_of_manifest_path(r, manifest.get(r, "")) for r in ids]
    idx = np.asarray([i for i, c in enumerate(cohort) if c == "Brugada-HUCA"])
    huca_ids = [ids[i] for i in idx]

    meta_path = ss.find_huca_metadata_path(manifest)
    if meta_path is None or not meta_path.exists():
        raise SystemExit("HUCA metadata.csv not found")
    meta = {}
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[str(row["patient_id"]).strip()] = {
                k: int(float(str(row[k]).strip()))
                for k in ("basal_pattern", "sudden_death", "brugada")
            }
    missing = [r for r in huca_ids if r not in meta]
    if missing:
        raise SystemExit(f"{len(missing)} HUCA matrix rows missing from metadata.csv")

    apr28_labels = ss.load_apr28_labels(DATA / "apr28_labels.jsonl")
    base_labels = [set(apr28_labels.get(r, set())) for r in huca_ids]

    X_all = np.load(DATA / "apr28_X.npy", mmap_mode="r")
    X = np.ascontiguousarray(np.asarray(X_all[idx], dtype=np.float32))
    del X_all
    log(f"loaded HUCA slice {X.shape} from apr28_X.npy; metadata source = {meta_path}")
    return X, huca_ids, meta, base_labels


def build_dataset(X, ids, meta, base_labels, variant: str):
    """Restrict to HUCA rows and set the brugada label from metadata.csv."""
    if variant == "confirmed":       # 69 cases, 7 atypical dropped
        keep = [i for i, r in enumerate(ids) if meta[r]["brugada"] in (0, 1)]
        is_pos = lambda r: meta[r]["brugada"] == 1
    elif variant == "inclusive":     # 76 cases (brugada > 0), matches shipped loader
        keep = list(range(len(ids)))
        is_pos = lambda r: meta[r]["brugada"] > 0
    else:
        raise ValueError(variant)

    keep = np.asarray(keep)
    sub_ids = [ids[i] for i in keep]
    status = {r: bool(is_pos(r)) for r in sub_ids}
    labels = [
        ss.apply_huca_brugada_metadata(r, "Brugada-HUCA", set(base_labels[i]), status)
        for i, r in zip(keep, sub_ids)
    ]
    subjects = [ss.subject_of_apr28_id(r, "Brugada-HUCA", {}, {}) for r in sub_ids]

    ds = ss.SourceSafeDataset(
        X=X[keep],
        recording_ids=sub_ids,
        sources=np.asarray(["Brugada-HUCA"] * len(sub_ids), dtype=object),
        subjects=np.asarray(subjects, dtype=object),
        labels=labels,
        is_clean_normal=np.asarray([ss.is_clean_apr28_normal(l) for l in labels], dtype=bool),
        is_apr28=np.ones(len(sub_ids), dtype=bool),
    )
    y = np.asarray([DX in l for l in labels], dtype=int)
    return ds, y, sub_ids


# -------------------------------------------------------------------------- stats

def f1_optimal_threshold(y, s):
    best = (-1.0, 0.5)
    for thr in np.unique(s):
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, float(thr))
    return best[1], best[0]


def confusion_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    d = lambda a, b: float(a / b) if b else float("nan")
    return {
        "threshold": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": d(tp, tp + fn), "specificity": d(tn, tn + fp),
        "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn),
        "f1": d(2 * tp, 2 * tp + fp + fn),
    }


def bootstrap_ci(y, s, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = np.empty(n)
    for i in range(n):
        idx = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        yy = y[idx]
        out[i] = roc_auc_score(yy, s[idx]) if len(np.unique(yy)) == 2 else np.nan
    out = out[np.isfinite(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run_once(ds, y, seed, log_folds=False):
    cfg = ss.SourceSafeConfig(excluded_train_sources=(), random_state=seed)
    res, scores = ss.run_source_safe_diagnosis(ds, DX, cfg)
    if log_folds:
        for a in res.fold_audits:
            log(f"    fold {a.fold}: train={a.n_train} (pos {a.n_train_pos}) "
                f"test={a.n_test} (pos {a.n_test_pos}) subj_overlap={a.n_subject_overlap} "
                f"feats={a.top_feature_count}")
    if not np.isfinite(scores).all():
        raise SystemExit("unscored rows present")
    return res, scores


# --------------------------------------------------------------------------- main

def main():
    X, ids, meta, base_labels = load_huca()
    log(f"metadata brugada counts: {Counter(meta[r]['brugada'] for r in ids).most_common()}")

    # cross-tab: which cases are even visible on a resting ECG
    ct = Counter((meta[r]["brugada"], meta[r]["basal_pattern"]) for r in ids)
    log(f"brugada x basal_pattern: {sorted(ct.items())}")

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": {}}

    for variant in ("confirmed", "inclusive"):
        log("=" * 78)
        log(f"VARIANT {variant}")
        ds, y, sub_ids = build_dataset(X, ids, meta, base_labels, variant)
        log(f"  rows={len(y)}  cases={int(y.sum())}  controls={int((1 - y).sum())}  "
            f"unique subjects={len(set(ds.subjects))}")

        # --- headline run, seed 0
        t0 = time.time()
        res, scores = run_once(ds, y, 0, log_folds=True)
        log(f"  seed 0 done in {time.time() - t0:.1f}s")

        auc = roc_auc_score(y, scores)
        lo, hi = bootstrap_ci(y, scores)
        thr, f1 = f1_optimal_threshold(y, scores)
        cm = confusion_at(y, scores, thr)
        log(f"  OOF AUROC = {auc:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
        log(f"  harness primary_auc={res.primary_auc:.4f} final_full_auc={res.final_full_auc:.4f}")
        log(f"  F1-optimal thr={thr:.4f} F1={f1:.3f} sens={cm['sensitivity']:.3f} "
            f"spec={cm['specificity']:.3f} ppv={cm['ppv']:.3f}")
        log(f"  confusion: TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")

        # --- basal-pattern decomposition (only for real cases)
        bp = np.asarray([meta[r]["basal_pattern"] for r in sub_ids])
        neg = y == 0
        strat = {}
        for name, m in (("cases_basal_pattern_1", (y == 1) & (bp == 1)),
                        ("cases_basal_pattern_0", (y == 1) & (bp == 0))):
            if m.sum() == 0:
                continue
            strat[name] = {
                "n_cases": int(m.sum()),
                "auroc_vs_all_controls": ss.auc_from_scores(scores[m], scores[neg]),
                "median_score": float(np.median(scores[m])),
            }
            log(f"  {name}: n={int(m.sum())} AUROC vs controls = "
                f"{strat[name]['auroc_vs_all_controls']:.4f}")
        strat["controls"] = {
            "n": int(neg.sum()),
            "n_basal_pattern_1": int((neg & (bp == 1)).sum()),
            "median_score": float(np.median(scores[neg])),
        }
        log(f"  controls: n={int(neg.sum())} ({int((neg & (bp == 1)).sum())} with pathological "
            f"baseline) median score={np.median(scores[neg]):.4f}")

        # --- seed stability
        seed_aucs = [auc]
        for seed in range(1, N_SEEDS):
            _, sc = run_once(ds, y, seed)
            seed_aucs.append(float(roc_auc_score(y, sc)))
        log(f"  seeds 0-{N_SEEDS - 1}: mean={np.mean(seed_aucs):.4f} sd={np.std(seed_aucs):.4f} "
            f"min={min(seed_aucs):.4f} max={max(seed_aucs):.4f}")

        # --- permutation control: labels shuffled, whole protocol rerun
        perm_aucs = []
        rng = np.random.default_rng(12345)
        for p in range(N_PERM):
            yp = rng.permutation(y)
            plabels = []
            for i, lab in enumerate(ds.labels):
                l = set(lab)
                l.discard(DX)
                if yp[i] == 1:
                    l.add(DX)
                plabels.append(l)
            pds = ss.SourceSafeDataset(
                X=ds.X, recording_ids=ds.recording_ids, sources=ds.sources,
                subjects=ds.subjects, labels=plabels,
                is_clean_normal=ds.is_clean_normal, is_apr28=ds.is_apr28,
            )
            _, sc = run_once(pds, yp, p)
            perm_aucs.append(float(roc_auc_score(yp, sc)))
        log(f"  permutation control ({N_PERM}x): mean={np.mean(perm_aucs):.4f} "
            f"sd={np.std(perm_aucs):.4f} max={max(perm_aucs):.4f}   (expect ~0.50)")

        # --- per-record OOF scores
        with open(OUT / f"oof_scores_{variant}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["recording_id", "brugada", "basal_pattern", "sudden_death", "y", "oof_score"])
            for i, r in enumerate(sub_ids):
                w.writerow([r, meta[r]["brugada"], meta[r]["basal_pattern"],
                            meta[r]["sudden_death"], int(y[i]), f"{scores[i]:.6f}"])

        report["variants"][variant] = {
            "positive_rule": "brugada == 1 (confirmed only; 7 atypical rows dropped)"
            if variant == "confirmed" else "brugada > 0 (confirmed + atypical, shipped loader rule)",
            "n_rows": len(y), "n_cases": int(y.sum()), "n_controls": int((1 - y).sum()),
            "n_unique_subjects": len(set(ds.subjects)),
            "oof_auroc": float(auc), "ci95_low": lo, "ci95_high": hi,
            "harness_primary_auc": res.primary_auc,
            "harness_final_full_auc": res.final_full_auc,
            "harness_final_clean_normal_auc": res.final_clean_normal_auc,
            "train_sources": res.train_sources,
            "n_external_pos": res.n_external_pos,
            "f1_optimal": {**cm, "f1_at_threshold": f1},
            "seed_stability": {
                "n_seeds": N_SEEDS, "aucs": seed_aucs,
                "mean": float(np.mean(seed_aucs)), "sd": float(np.std(seed_aucs)),
                "min": float(min(seed_aucs)), "max": float(max(seed_aucs)),
            },
            "permutation_control": {
                "n": N_PERM, "aucs": perm_aucs,
                "mean": float(np.mean(perm_aucs)), "sd": float(np.std(perm_aucs)),
                "max": float(max(perm_aucs)),
            },
            "basal_pattern_decomposition": strat,
            "fold_audits": [
                {"fold": a.fold, "n_train": a.n_train, "n_test": a.n_test,
                 "n_train_pos": a.n_train_pos, "n_test_pos": a.n_test_pos,
                 "n_subject_overlap": a.n_subject_overlap,
                 "top_feature_count": a.top_feature_count}
                for a in res.fold_audits
            ],
        }

    with open(OUT / "huca_brugada_refit_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log("=" * 78)
    log(f"wrote {OUT / 'huca_brugada_refit_results.json'}")
    for v, r in report["variants"].items():
        log(f"  {v}: {r['n_cases']} cases vs {r['n_controls']} controls -> "
            f"OOF AUROC {r['oof_auroc']:.4f} (95% CI {r['ci95_low']:.3f}-{r['ci95_high']:.3f})")


if __name__ == "__main__":
    main()
