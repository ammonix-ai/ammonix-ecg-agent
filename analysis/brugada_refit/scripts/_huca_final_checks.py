"""Final two checks on the HUCA Brugada refit.

A. #Dx leak test. The derived 500 Hz headers carry '#Dx: 418818005' (SNOMED Brugada)
   on all 76 cases AND on 23 brugada=0 controls. If that tag reached the features, those
   23 controls would score like cases. They are the perfect natural probe.

B. Honest operating point. The F1-optimal threshold in the main run was chosen on the
   same OOF scores it was scored against. Here the threshold for each fold is chosen on
   the OOF scores of the OTHER four folds only, so no fold's own labels inform its
   threshold.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
OUT = Path("/path/to/huca_brugada_refit_workdir")


def p(m=""):
    print(m, flush=True)


man = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
out = {}

for variant in ("confirmed", "inclusive"):
    rows = list(csv.DictReader(open(OUT / f"oof_scores_{variant}.csv", encoding="utf-8")))
    ids = [r["recording_id"] for r in rows]
    y = np.asarray([int(r["y"]) for r in rows])
    s = np.asarray([float(r["oof_score"]) for r in rows])
    bp = np.asarray([int(r["basal_pattern"]) for r in rows])
    has_dx = np.asarray([
        "#Dx" in Path(man[r]).read_text(encoding="utf-8", errors="replace") for r in ids
    ])

    p("=" * 78)
    p(f"VARIANT {variant}   n={len(y)} cases={int(y.sum())} controls={int((1 - y).sum())}")
    p("=" * 78)
    p(f"  overall OOF AUROC {roc_auc_score(y, s):.4f}")

    # ---------------- A. #Dx leak probe
    p("\n  A. #Dx LEAK PROBE")
    ctrl = y == 0
    a, b = ctrl & has_dx, ctrl & ~has_dx
    p(f"     cases carrying #Dx            : {int(((y == 1) & has_dx).sum())}/{int((y == 1).sum())}")
    p(f"     controls carrying #Dx         : {int(a.sum())}   controls without: {int(b.sum())}")
    p(f"     median score  cases           : {np.median(s[y == 1]):.4f}")
    p(f"     median score  controls +#Dx   : {np.median(s[a]):.4f}")
    p(f"     median score  controls -#Dx   : {np.median(s[b]):.4f}")
    auc_dx_ctrl = roc_auc_score(has_dx[ctrl].astype(int), s[ctrl])
    p(f"     AUROC of score for #Dx status, CONTROLS ONLY: {auc_dx_ctrl:.4f}")
    p(f"       (a live header leak would push this toward 1.0; these 23 also genuinely")
    p(f"        have a pathological baseline, so some elevation is physiological)")
    # cases vs #Dx-carrying controls = the tag-matched comparison
    auc_matched = ss.auc_from_scores(s[y == 1], s[a])
    auc_plain = ss.auc_from_scores(s[y == 1], s[b])
    p(f"     cases vs #Dx-carrying controls (tag-matched): {auc_matched:.4f}")
    p(f"     cases vs non-#Dx controls                   : {auc_plain:.4f}")
    p(f"       -> if the tag drove the model these two would differ enormously")

    # ---------------- B. honest operating point
    p("\n  B. OPERATING POINT, threshold chosen out-of-fold")
    groups = np.asarray([f"Brugada-HUCA:{r}" for r in ids], dtype=object)
    X_dummy = np.zeros((len(y), 1), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    fold_of = np.full(len(y), -1)
    for f, (_, te) in enumerate(splitter.split(X_dummy, y, groups)):
        fold_of[te] = f
    assert (fold_of >= 0).all()

    def f1_thr(yy, ss_):
        best = (-1.0, 0.5)
        for t in np.unique(ss_):
            pred = (ss_ >= t).astype(int)
            tp = int(((pred == 1) & (yy == 1)).sum())
            fp = int(((pred == 1) & (yy == 0)).sum())
            fn = int(((pred == 0) & (yy == 1)).sum())
            f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
            if f1 > best[0]:
                best = (f1, float(t))
        return best[1]

    pred = np.zeros(len(y), dtype=int)
    thrs = []
    for f in range(5):
        te = fold_of == f
        tr = ~te
        t = f1_thr(y[tr], s[tr])
        thrs.append(t)
        pred[te] = (s[te] >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    d = lambda x, n: float(x / n) if n else float("nan")
    p(f"     per-fold thresholds: {[round(t, 4) for t in thrs]}")
    p(f"     TP={tp} FP={fp} TN={tn} FN={fn}")
    p(f"     sensitivity {d(tp, tp + fn):.3f}  specificity {d(tn, tn + fp):.3f}  "
      f"PPV {d(tp, tp + fp):.3f}  NPV {d(tn, tn + fn):.3f}  F1 {d(2 * tp, 2 * tp + fp + fn):.3f}")

    # in-sample comparison, to show the size of the optimism
    t_in = f1_thr(y, s)
    pin = (s >= t_in).astype(int)
    tp2 = int(((pin == 1) & (y == 1)).sum()); fp2 = int(((pin == 1) & (y == 0)).sum())
    fn2 = int(((pin == 0) & (y == 1)).sum())
    p(f"     [in-sample threshold {t_in:.4f} would have given F1 "
      f"{d(2 * tp2, 2 * tp2 + fp2 + fn2):.3f} vs {d(2 * tp, 2 * tp + fp + fn):.3f} out-of-fold]")

    out[variant] = {
        "oof_auroc": float(roc_auc_score(y, s)),
        "dx_probe": {
            "n_controls_with_dx": int(a.sum()), "n_controls_without_dx": int(b.sum()),
            "median_score_cases": float(np.median(s[y == 1])),
            "median_score_controls_with_dx": float(np.median(s[a])),
            "median_score_controls_without_dx": float(np.median(s[b])),
            "auroc_of_score_for_dx_status_controls_only": float(auc_dx_ctrl),
            "auroc_cases_vs_dx_controls": float(auc_matched),
            "auroc_cases_vs_non_dx_controls": float(auc_plain),
        },
        "operating_point_out_of_fold": {
            "per_fold_thresholds": [float(t) for t in thrs],
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": d(tp, tp + fn), "specificity": d(tn, tn + fp),
            "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn),
            "f1": d(2 * tp, 2 * tp + fp + fn),
        },
        "in_sample_threshold_f1": d(2 * tp2, 2 * tp2 + fp2 + fn2),
    }
    p()

with open(OUT / "final_checks.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
p(f"wrote {OUT / 'final_checks.json'}")
