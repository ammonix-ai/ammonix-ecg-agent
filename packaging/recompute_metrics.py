"""Recompute per-class AUROC before and after the label recovery.

WHAT THIS IS, AND IS NOT
------------------------
This holds the SCORES FIXED and varies only the LABELS, so the delta isolates
the effect of the recovery and nothing else.

It does NOT reproduce the published `primary_auc` / `final_full_auc`. Those were
computed in the private CV from a scores array that is single-fold out-of-fold
for the Apr28 training rows and a 5-model mean for the external rows. The
shipped `probabilities.npy` is the 5-fold mean for EVERY row, so for Apr28 rows
it is partly in-sample and reads optimistically. Absolute values here are
therefore not comparable to the model card; the before/after difference is.

Two negative pools, mirroring the private CV:
  in-cohort   positives vs non-positives from the Apr28 multi-source cohort
              — the hard, same-source comparison behind `primary_auc`
  full        positives vs every non-positive in the universe, which adds the
              clean SR normals and the LVEF controls — behind `final_full_auc`

Caveat carried from packaging: the `__mi_family_quarantine__` marker was
stripped when the universe was made public, so the 439 rows the private pipeline
excluded from MI-family evaluation are included here. MI-family numbers are
slightly pessimistic as a result.

    python recompute_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UNIVERSE = REPO / "_staging" / "universe"
PATIENTS = UNIVERSE / "patients.jsonl"
BACKUP = UNIVERSE / "patients.jsonl.pre_recovery_backup"
PROBS = UNIVERSE / "probabilities.npy"
METADATA = UNIVERSE / "metadata.json"

APR28 = "Apr28 multi-source"


def auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """Mann-Whitney AUROC. None when either side is empty."""
    if pos.size == 0 or neg.size == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties, or tied scores bias the statistic
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def load_labels(path: Path) -> tuple[list[set[str]], list[str]]:
    labels, cohorts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            labels.append({str(d).lower() for d in (row.get("labels") or [])})
            cohorts.append(row.get("cohort") or "")
    return labels, cohorts


def main() -> int:
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    classes = list(meta["classes"])
    probs = np.load(PROBS, mmap_mode="r")

    new_labels, cohorts = load_labels(PATIENTS)
    old_labels, _ = load_labels(BACKUP)
    n = len(new_labels)
    assert probs.shape[0] == n, f"{probs.shape[0]} scores vs {n} rows"

    apr28 = np.array([c == APR28 for c in cohorts])

    rows = []
    for j, dx in enumerate(classes):
        s = np.asarray(probs[:, j], dtype=float)
        key = dx.lower()
        out = {"dx": dx}
        for tag, labelset in (("old", old_labels), ("new", new_labels)):
            pos_mask = np.array([key in L for L in labelset])
            neg_mask = ~pos_mask
            out[f"n_pos_{tag}"] = int(pos_mask.sum())
            out[f"full_{tag}"] = auroc(s[pos_mask], s[neg_mask])
            out[f"cohort_{tag}"] = auroc(s[pos_mask & apr28], s[neg_mask & apr28])
        rows.append(out)

    def macro(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("=" * 96)
    print("PER-CLASS AUROC — same scores, old labels vs recovered labels")
    print("=" * 96)
    print(f"{'diagnosis':<44}{'n+ old':>8}{'n+ new':>8}{'full old':>10}{'full new':>10}{'delta':>9}")
    for r in sorted(rows, key=lambda r: -(r["n_pos_new"] - r["n_pos_old"])):
        fo, fn = r["full_old"], r["full_new"]
        d = f"{fn - fo:+.3f}" if (fo is not None and fn is not None) else "—"
        print(f"{r['dx'][:43]:<44}{r['n_pos_old']:>8,}{r['n_pos_new']:>8,}"
              f"{(f'{fo:.3f}' if fo is not None else '—'):>10}"
              f"{(f'{fn:.3f}' if fn is not None else '—'):>10}{d:>9}")

    print()
    print(f"MACRO  full negative pool     old {macro('full_old'):.4f}   new {macro('full_new'):.4f}"
          f"   delta {macro('full_new') - macro('full_old'):+.4f}")
    print(f"MACRO  in-cohort (Apr28)      old {macro('cohort_old'):.4f}   new {macro('cohort_new'):.4f}"
          f"   delta {macro('cohort_new') - macro('cohort_old'):+.4f}")
    print()
    print("Reminder: absolute values are NOT the published primary_auc/final_full_auc —")
    print("the shipped probabilities are a 5-fold mean for every row. The deltas are the result.")

    out = UNIVERSE / "metrics_recomputed.json"
    out.write_text(json.dumps({"classes": rows,
                               "macro_full_old": macro("full_old"),
                               "macro_full_new": macro("full_new"),
                               "macro_cohort_old": macro("cohort_old"),
                               "macro_cohort_new": macro("cohort_new")},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
