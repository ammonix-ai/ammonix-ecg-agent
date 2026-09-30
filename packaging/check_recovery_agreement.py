"""Is the recovered-label agreement real, or just a prevalence artefact?

Adding labels to a record can only raise the number of predictions that score as
true positives, so 'agreement went up' is not by itself evidence that the
mapping is right.  Two controls:

1. PERMUTATION.  Shuffle the recovered label sets between records *within a
   cohort*, keeping the exact same label prevalence, and recompute agreement.
   If the mapping carries per-record information, the real TP count must sit far
   above the permuted one.  If they are close, the mapping is only tracking base
   rates and the recovery is worthless.

2. PER CLASS.  Precision per scored class before and after, next to how often
   the class was recovered.  A class where the classifier fires often and the
   12SL read agrees rarely is where the mapping (or the classifier) is wrong.

Reads the scratchpad outputs of recover_machine_labels.py.  Writes nothing.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from recover_machine_labels import METADATA, OUT_JSONL, PATIENTS  # noqa: E402


def main() -> int:
    scored = set(json.loads(METADATA.read_text(encoding="utf-8"))["classes"])

    preds_by_id: dict[str, set[str]] = {}
    with PATIENTS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            preds_by_id[r["display_id"]] = set(r.get("predictions") or [])

    recs = []
    with OUT_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            recs.append((
                r["cohort"],
                r["display_id"],
                {x for x in r["existing_labels"] if x in scored},
                {x for x in r["recovered_labels"] if x in scored},
            ))

    # ---- per class -----------------------------------------------------------
    pred_n: Counter[str] = Counter()
    tp_before: Counter[str] = Counter()
    tp_after: Counter[str] = Counter()
    rec_n: Counter[str] = Counter()
    for _c, did, before, rec in recs:
        p = preds_by_id.get(did, set())
        after = before | rec
        for x in p:
            pred_n[x] += 1
            if x in before:
                tp_before[x] += 1
            if x in after:
                tp_after[x] += 1
        for x in rec:
            rec_n[x] += 1

    print("=" * 92)
    print("PER-CLASS AGREEMENT ON THE 48,958 RECOVERED RECORDS")
    print("=" * 92)
    print(f"{'class':<44}{'fired':>8}{'recovered':>11}{'TP bef':>8}{'TP aft':>8}"
          f"{'prec bef':>10}{'prec aft':>10}")
    for cls, n in pred_n.most_common():
        pb = 100.0 * tp_before[cls] / n
        pa = 100.0 * tp_after[cls] / n
        print(f"{cls:<44}{n:>8,}{rec_n[cls]:>11,}{tp_before[cls]:>8,}"
              f"{tp_after[cls]:>8,}{pb:>9.1f}%{pa:>9.1f}%")
    only_recovered = [c for c in rec_n if c not in pred_n]
    if only_recovered:
        print("\nrecovered but never predicted on these records:",
              ", ".join(sorted(only_recovered)))

    # ---- permutation control -------------------------------------------------
    by_cohort: dict[str, list[int]] = {}
    for i, (c, *_rest) in enumerate(recs):
        by_cohort.setdefault(c, []).append(i)

    real = sum(len(preds_by_id.get(d, set()) & (b | r)) for _c, d, b, r in recs)
    base = sum(len(preds_by_id.get(d, set()) & b) for _c, d, b, _r in recs)

    print()
    print("=" * 92)
    print("PERMUTATION CONTROL  (recovered label sets shuffled within cohort)")
    print("=" * 92)
    rng = random.Random(20260813)
    trials = []
    for t in range(5):
        perm_tp = 0
        for cohort, idxs in by_cohort.items():
            shuffled = idxs[:]
            rng.shuffle(shuffled)
            for i, j in zip(idxs, shuffled):
                _c, d, b, _r = recs[i]
                fake = recs[j][3]
                perm_tp += len(preds_by_id.get(d, set()) & (b | fake))
        trials.append(perm_tp)
        print(f"  trial {t + 1}: TP {perm_tp:,}")
    mean_perm = sum(trials) / len(trials)
    print(f"\n  before recovery      TP {base:,}")
    print(f"  shuffled labels      TP {mean_perm:,.0f}   "
          f"(+{mean_perm - base:,.0f} from prevalence alone)")
    print(f"  REAL recovery        TP {real:,}   (+{real - base:,} over baseline)")
    gain_real = real - base
    gain_perm = mean_perm - base
    print(f"\n  share of the gain that is record-specific: "
          f"{100.0 * (gain_real - gain_perm) / gain_real:.1f}%")
    print(f"  real / shuffled TP ratio: {real / mean_perm:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
