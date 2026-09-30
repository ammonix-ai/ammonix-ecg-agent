"""Independent validation of recover_machine_labels.py output.

Re-derives every headline number from the primary inputs (reports.json,
affected_ids.json, patients.jsonl, metadata.json) without trusting the
mapper's own recovery_report.json.  Read-only everywhere.
"""

from __future__ import annotations

import json
import random
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
from recover_machine_labels import map_statement, norm  # noqa: E402

affected = json.loads((SCRATCH / "affected_ids.json").read_text(encoding="utf-8"))
reports = json.loads((SCRATCH / "reports.json").read_text(encoding="utf-8"))
scored = set(json.loads((UNI / "metadata.json").read_text(encoding="utf-8"))["classes"])

rows = {}
for line in (UNI / "patients.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    rows[r["display_id"]] = r

rec = {}
dup = 0
for line in (SCRATCH / "recovered_labels.jsonl").open(encoding="utf-8"):
    o = json.loads(line)
    if o["display_id"] in rec:
        dup += 1
    rec[o["display_id"]] = o

print("=" * 78)
print("A. FILE INTEGRITY / JOIN")
print("=" * 78)
exp = {(c, s, d) for c, m in affected.items() for s, d in m.items()}
print(f"  affected_ids entries        {len(exp):,}")
print(f"  recovered_labels rows       {len(rec):,}   duplicate display_ids {dup}")
print(f"  reports.json study_ids      {len(reports):,}")

bad_cohort = bad_raw = missing_row = 0
for c, s, d in exp:
    o = rec.get(d)
    if o is None:
        missing_row += 1
        continue
    if o["cohort"] != c or o["study_id"] != s:
        bad_cohort += 1
    if o["raw_reports"] != reports.get(s, []):
        bad_raw += 1
print(f"  rows missing from output    {missing_row}")
print(f"  cohort/study_id mismatches  {bad_cohort}")
print(f"  raw_reports != reports.json {bad_raw}")

# Does the file's recovered_labels equal a fresh run of map_statement?
drift = 0
for d, o in rec.items():
    fresh = set()
    for line in o["raw_reports"]:
        s = norm(line)
        if s:
            fresh |= map_statement(s)[0]
    if sorted(fresh) != o["recovered_labels"]:
        drift += 1
print(f"  rows whose labels != rerun  {drift}   (0 = file is exactly the code's output)")

print()
print("=" * 78)
print("B. STATEMENT COVERAGE  (re-derived from reports.json)")
print("=" * 78)
disp = Counter()
distinct = Counter()
unmapped = Counter()
per_cohort_stmt = defaultdict(Counter)
sid2cohort = {s: c for c, m in affected.items() for s in m}
for s, lines in reports.items():
    coh = sid2cohort.get(s, "?")
    for line in lines:
        t = norm(line)
        if not t:
            continue
        distinct[t] += 1
        _, dd = map_statement(t)
        disp[dd] += 1
        per_cohort_stmt[coh][dd] += 1
        per_cohort_stmt[coh]["total"] += 1
        if dd == "unmapped":
            unmapped[t] += 1
tot = sum(disp.values())
print(f"  statement instances     {tot:,}      distinct strings {len(distinct):,}")
for k in ("labelled", "technical", "no-finding", "unmapped"):
    print(f"  {k:<22} {disp[k]:>8,}   {100*disp[k]/tot:6.2f}%")
print(f"  {'COVERED':<22} {tot-disp['unmapped']:>8,}   {100*(tot-disp['unmapped'])/tot:6.2f}%")
print(f"  distinct unmapped strings {len(unmapped):,}")
print("  per cohort:")
for c, cc in per_cohort_stmt.items():
    t = cc["total"]
    print(f"    {c:<22} stmts {t:>7,}  covered {100*(t-cc['unmapped'])/t:6.2f}%"
          f"  labelled {100*cc['labelled']/t:6.2f}%  unmapped {cc['unmapped']:,}")
print("  top 25 unmapped:")
for s, c in unmapped.most_common(25):
    print(f"    {c:>5}  {s}")

print()
print("=" * 78)
print("C. RECORDS GAINING LABELS  (re-derived)")
print("=" * 78)
SUMMARY = {"normal ecg", "abnormal ecg", "borderline ecg", "sinus rhythm"}
gain = Counter()
gain_ns = Counter()
nrec = Counter()
nolabel = Counter()
scored_hit = Counter()
dx_all = Counter()
dx_cohort = defaultdict(Counter)
for d, o in rec.items():
    c = o["cohort"]
    nrec[c] += 1
    ex = {e.lower() for e in o["existing_labels"]}
    rl = o["recovered_labels"]
    for x in rl:
        dx_all[x] += 1
        dx_cohort[c][x] += 1
    new = [x for x in rl if x.lower() not in ex]
    if new:
        gain[c] += 1
    if [x for x in new if x.lower() not in SUMMARY]:
        gain_ns[c] += 1
    if not rl:
        nolabel[c] += 1
    if {x for x in rl if x in scored} | {e for e in o["existing_labels"] if e in scored}:
        scored_hit[c] += 1
print(f"  total records {sum(nrec.values()):,}  gained>=1 {sum(gain.values()):,}"
      f" ({100*sum(gain.values())/sum(nrec.values()):.1f}%)"
      f"  gained non-summary {sum(gain_ns.values()):,}"
      f" ({100*sum(gain_ns.values())/sum(nrec.values()):.1f}%)")
print(f"  records with >=1 SCORED class after {sum(scored_hit.values()):,}"
      f" ({100*sum(scored_hit.values())/sum(nrec.values()):.1f}%)")
print(f"  records with no label at all        {sum(nolabel.values()):,}")
for c in nrec:
    print(f"    {c:<22} n {nrec[c]:>6,}  gained {gain[c]:>6,} ({100*gain[c]/nrec[c]:5.1f}%)"
          f"  non-summary {gain_ns[c]:>6,} ({100*gain_ns[c]/nrec[c]:5.1f}%)"
          f"  none {nolabel[c]}")
print("\n  recovered diagnosis distribution (all cohorts):")
for k, v in dx_all.most_common():
    print(f"    {v:>7,}  {100*v/len(rec):5.2f}%  {k}")
for c in dx_cohort:
    print(f"\n  {c} top 15:")
    for k, v in dx_cohort[c].most_common(15):
        print(f"    {v:>7,}  {100*v/nrec[c]:5.1f}%  {k}")

print()
print("=" * 78)
print("D. CLASSIFIER AGREEMENT  (re-derived; 32 scored classes only)")
print("=" * 78)
tpb = tpa = npred = 0
lb = la = 0
pcb = defaultdict(lambda: [0, 0, 0])
recovered_by_cohort = defaultdict(list)
for d, o in rec.items():
    row = rows.get(d, {})
    preds = set(row.get("predictions") or [])
    before = {e for e in o["existing_labels"] if e in scored}
    add = {x for x in o["recovered_labels"] if x in scored}
    after = before | add
    npred += len(preds)
    tpb += len(preds & before)
    tpa += len(preds & after)
    lb += len(before)
    la += len(after)
    p = pcb[o["cohort"]]
    p[0] += len(preds)
    p[1] += len(preds & before)
    p[2] += len(preds & after)
    recovered_by_cohort[o["cohort"]].append(add)
print(f"  predictions emitted  {npred:,}")
print(f"  TP before {tpb:>7,}   precision {100*tpb/npred:6.2f}%")
print(f"  TP after  {tpa:>7,}   precision {100*tpa/npred:6.2f}%")
print(f"  scored labels before {lb:,}  recall {100*tpb/lb:5.2f}%")
print(f"  scored labels after  {la:,}  recall {100*tpa/la:5.2f}%")
for c, p in pcb.items():
    print(f"    {c:<22} preds {p[0]:>7,}  TP {p[1]:>7,} -> {p[2]:>7,}"
          f"  {100*p[1]/p[0]:6.2f}% -> {100*p[2]/p[0]:6.2f}%")

print()
print("=" * 78)
print("E. PERMUTATION CONTROL  (own shuffle, within cohort, 5 trials)")
print("=" * 78)
order = defaultdict(list)
for d, o in rec.items():
    order[o["cohort"]].append(d)
trials = []
for seed in range(5):
    rnd = random.Random(1000 + seed)
    tp = 0
    for c, ids in order.items():
        pool = list(recovered_by_cohort[c])
        rnd.shuffle(pool)
        for d, add in zip(ids, pool):
            row = rows.get(d, {})
            preds = set(row.get("predictions") or [])
            before = {e for e in rec[d]["existing_labels"] if e in scored}
            tp += len(preds & (before | add))
    trials.append(tp)
    print(f"  trial {seed}: shuffled TP {tp:,}")
mean = sum(trials) / len(trials)
print(f"  before {tpb:,}   shuffled mean {mean:,.0f}   real {tpa:,}")
print(f"  gain shuffled {mean-tpb:,.0f}   gain real {tpa-tpb:,}"
      f"   ratio {(tpa-tpb)/(mean-tpb):.2f}x"
      f"   record-specific share {100*(1-(mean-tpb)/(tpa-tpb)):.1f}%")
