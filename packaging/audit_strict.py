"""Strict tests: hedge sensitivity of the agreement rise, technical-leak, bug impact."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRATCH = Path(
    r"/path/to/scratch"
    r""
    r"/37adc070-feaf-456c-9694-b1d7ee490af5/scratchpad"
)
UNI = Path(r"/path/to/ammonix-ecg-agent/_staging/universe")
sys.path.insert(0, str(Path(__file__).parent))
from recover_machine_labels import (  # noqa: E402
    NO_FINDING_RE, TECHNICAL_RE, map_statement, norm,
)

scored = set(json.loads((UNI / "metadata.json").read_text(encoding="utf-8"))["classes"])
rows = {}
for line in (UNI / "patients.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    rows[r["display_id"]] = r
recs = [json.loads(l) for l in (SCRATCH / "recovered_labels.jsonl").open(encoding="utf-8")]

HEDGE = re.compile(
    r"\bpossible\b|\bpossibly\b|\bprobable\b|\bprobably\b|\bborderline\b|"
    r"\bconsider\b|may be due to|cannot rule out|\bsuggest|\bmay also\b|\?"
)

print("=" * 78)
print("1. HEDGE SENSITIVITY OF THE AGREEMENT RISE")
print("=" * 78)
tpb = tp_all = tp_conf = 0
npred = 0
add_all = Counter()
add_conf = Counter()
for o in recs:
    preds = set(rows.get(o["display_id"], {}).get("predictions") or [])
    before = {e for e in o["existing_labels"] if e in scored}
    conf = set()
    for line in o["raw_reports"]:
        s = norm(line)
        if not s or HEDGE.search(s):
            continue
        conf |= map_statement(s)[0]
    after_all = before | {x for x in o["recovered_labels"] if x in scored}
    after_conf = before | {x for x in conf if x in scored}
    npred += len(preds)
    tpb += len(preds & before)
    tp_all += len(preds & after_all)
    tp_conf += len(preds & after_conf)
    for x in o["recovered_labels"]:
        add_all[x] += 1
    for x in conf:
        add_conf[x] += 1
print(f"  predictions            {npred:,}")
print(f"  TP before              {tpb:>7,}   {100*tpb/npred:6.2f}%")
print(f"  TP after (all stmts)   {tp_all:>7,}   {100*tp_all/npred:6.2f}%")
print(f"  TP after (UNHEDGED)    {tp_conf:>7,}   {100*tp_conf/npred:6.2f}%")
print(f"  share of the rise that survives dropping every hedged statement: "
      f"{100*(tp_conf-tpb)/(tp_all-tpb):.1f}%")
print("\n  scored classes: records labelled from all stmts vs unhedged only")
for c in sorted(scored, key=lambda c: -add_all[c]):
    if add_all[c]:
        print(f"    {add_all[c]:>7,} -> {add_conf[c]:>7,}   "
              f"({100*add_conf[c]/add_all[c]:5.1f}% unhedged)  {c}")

print()
print("=" * 78)
print("2. TECHNICAL / NO-FINDING LEAK  (statements a rule labelled although a")
print("   technical or no-finding pattern also matches -- rules win, by design)")
print("=" * 78)
st = Counter()
for o in recs:
    for line in o["raw_reports"]:
        s = norm(line)
        if s:
            st[s] += 1
leak = []
for s, c in st.items():
    lab, d = map_statement(s)
    if d == "labelled":
        if any(p.search(s) for p in TECHNICAL_RE):
            leak.append(("TECH", c, s, sorted(lab)))
        elif any(p.search(s) for p in NO_FINDING_RE):
            leak.append(("NOFIND", c, s, sorted(lab)))
for k, c, s, lab in sorted(leak, key=lambda x: -x[1])[:25]:
    print(f"  {k:<7}{c:>6}  {lab}\n           {s}")
print(f"  total leaking statements {len(leak)} distinct, "
      f"{sum(x[1] for x in leak):,} instances")

print()
print("=" * 78)
print("3. IMPACT OF THE 'supraventricular' SUBSTRING COLLISIONS")
print("=" * 78)
bad_vt = bad_bg = 0
bad_vt_ids, bad_bg_ids = [], []
for o in recs:
    has_svt_only = False
    has_svbig = False
    real_vt = False
    for line in o["raw_reports"]:
        s = norm(line)
        if "supraventricular tachycardia" in s and "non-sustained ventricular" not in s:
            has_svt_only = True
        elif re.search(r"(?<!supra)ventricular tachycardia", s):
            real_vt = True
        if "supraventricular bigeminy" in s:
            has_svbig = True
    if has_svt_only and not real_vt and "ventricular tachycardia" in o["recovered_labels"]:
        bad_vt += 1
        bad_vt_ids.append(o["display_id"])
    if has_svbig and "ventricular bigeminy" in o["recovered_labels"]:
        bad_bg += 1
        bad_bg_ids.append(o["display_id"])
vt_tot = sum(1 for o in recs if "ventricular tachycardia" in o["recovered_labels"])
bg_tot = sum(1 for o in recs if "ventricular bigeminy" in o["recovered_labels"])
print(f"  records labelled 'ventricular tachycardia' {vt_tot}; "
      f"of these {bad_vt} come only from a SUPRAventricular tachycardia statement")
print(f"  records labelled 'ventricular bigeminy'    {bg_tot}; "
      f"of these {bad_bg} come from 'supraventricular bigeminy'")
print(f"  examples VT: {bad_vt_ids[:6]}")
print(f"  examples VB: {bad_bg_ids[:6]}")

print()
print("=" * 78)
print("4. STEMI PROVENANCE")
print("=" * 78)
stemi_recs = [o for o in recs if "STEMI" in o["recovered_labels"]]
hedged_only = 0
no_elev = 0
for o in stemi_recs:
    lines = [norm(x) for x in o["raw_reports"]]
    src = [s for s in lines if "STEMI" in map_statement(s)[0]]
    if all(re.search(r"possibl|probabl|consider", s) for s in src):
        hedged_only += 1
    if not any("st elev" in s for s in lines):
        no_elev += 1
print(f"  records with STEMI                                    {len(stemi_recs):,}")
print(f"  ... whose every STEMI-source statement is hedged       {hedged_only:,}")
print(f"  ... with NO 'st elevation' anywhere in the report      {no_elev:,}")
print(f"  records with nstemi                                    "
      f"{sum(1 for o in recs if 'nstemi' in o['recovered_labels']):,}")

print()
print("=" * 78)
print("5. 1st AV-BLOCK PROVENANCE")
print("=" * 78)
avb = [o for o in recs if "1st av-block" in o["recovered_labels"]]
borderline_only = 0
for o in avb:
    src = [norm(x) for x in o["raw_reports"] if "1st av-block" in map_statement(norm(x))[0]]
    if all("borderline" in s for s in src):
        borderline_only += 1
print(f"  records with 1st av-block                {len(avb):,}")
print(f"  ... sourced ONLY from 'borderline 1st degree a-v block'  {borderline_only:,}"
      f"  ({100*borderline_only/len(avb):.1f}%)")

print()
print("=" * 78)
print("6. WHICH SCORED CLASSES DRIVE THE TP GAIN")
print("=" * 78)
gain = Counter()
fp_after = Counter()
for o in recs:
    preds = set(rows.get(o["display_id"], {}).get("predictions") or [])
    before = {e for e in o["existing_labels"] if e in scored}
    add = {x for x in o["recovered_labels"] if x in scored}
    for c in preds & add - before:
        gain[c] += 1
    for c in preds - (before | add):
        fp_after[c] += 1
tot = sum(gain.values())
print(f"  {'newly-true preds':>17}  {'still false':>12}  class")
for c, v in gain.most_common():
    print(f"  {v:>17,}  {fp_after[c]:>12,}  {c}")
print(f"  total newly-true {tot:,}")
print("\n  largest remaining false positives:")
for c, v in fp_after.most_common(10):
    print(f"    {v:>7,}  {c}")
