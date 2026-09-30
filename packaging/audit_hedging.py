"""Statement-level audit of the recovery mapping: hedges, risky regexes, spot-check."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

SCRATCH = Path(
    r"/path/to/scratch"
    r""
    r"/37adc070-feaf-456c-9694-b1d7ee490af5/scratchpad"
)
sys.path.insert(0, str(Path(__file__).parent))
from recover_machine_labels import map_statement, norm  # noqa: E402

reports = json.loads((SCRATCH / "reports.json").read_text(encoding="utf-8"))
stmts = Counter()
for lines in reports.values():
    for line in lines:
        t = norm(line)
        if t:
            stmts[t] += 1

mode = sys.argv[1] if len(sys.argv) > 1 else "hedge"

if mode == "hedge":
    for kw in ["cannot rule out", "probable normal variant", "possible", "probabl",
               "may be due to", "consider", "suggest", "borderline", "?", "vs ",
               "nondiagnostic", "equivocal", "no evidence", "cannot exclude",
               "unconfirmed", "artifact", "\u2014"]:
        hits = [(s, c) for s, c in stmts.items() if kw in s]
        if not hits:
            continue
        vol = sum(c for _, c in hits)
        print("=" * 78)
        print(f"HEDGE '{kw}'   {len(hits)} distinct   {vol:,} instances")
        print("=" * 78)
        for s, c in sorted(hits, key=lambda x: -x[1])[:22]:
            lab, disp = map_statement(s)
            print(f"  {c:>6}  [{disp:<10}] {sorted(lab)}")
            print(f"          {s}")

elif mode == "regex":
    import re
    for name, pat in [("lad", r"\blad\b"), ("rad", r"\brad\b"), ("q wave", r"\bq waves?\b|^q in "),
                      ("old", r"\bold\b"), ("infarct", r"infarct"), ("paced", r"\bpacemaker\b|\bpacing\b|\bpaced\b"),
                      ("early repol", r"early repol"), ("normal variant", r"normal variant"),
                      ("t wave", r"t wave|t changes|t abnrm"), ("prwp", r"r-?wave progression|r wave progression")]:
        r = re.compile(pat)
        hits = [(s, c) for s, c in stmts.items() if r.search(s)]
        vol = sum(c for _, c in hits)
        print("=" * 78)
        print(f"REGEX {name}  {pat}   {len(hits)} distinct   {vol:,} instances")
        print("=" * 78)
        for s, c in sorted(hits, key=lambda x: -x[1])[:25]:
            lab, disp = map_statement(s)
            print(f"  {c:>6}  [{disp:<10}] {sorted(lab)}")
            print(f"          {s}")

elif mode == "top":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    cum = 0
    tot = sum(stmts.values())
    for s, c in stmts.most_common(n):
        lab, disp = map_statement(s)
        cum += c
        print(f"{c:>6} {100*cum/tot:5.1f}%  [{disp:<10}] {sorted(lab)}")
        print(f"        {s}")

elif mode == "spot":
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    rnd = random.Random(seed)
    rows = [json.loads(l) for l in (SCRATCH / "recovered_labels.jsonl").open(encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["cohort"], []).append(r)
    picked = []
    for c, rs in by.items():
        picked += rnd.sample(rs, 7 if c != "MIMIC SR normals" else 6)
    for i, r in enumerate(picked, 1):
        print("=" * 78)
        print(f"[{i}] {r['display_id']}  study {r['study_id']}  {r['cohort']}")
        print("  RAW:")
        for line in r["raw_reports"]:
            lab, disp = map_statement(norm(line))
            print(f"    | {line}")
            print(f"    |      -> [{disp}] {sorted(lab)}")
        print(f"  EXISTING : {r['existing_labels']}")
        print(f"  RECOVERED: {r['recovered_labels']}")
        if r["unmapped_statements"]:
            print(f"  UNMAPPED : {r['unmapped_statements']}")

elif mode == "spot_interesting":
    # records with the richest reports, where mapping errors would show
    rnd = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    rows = [json.loads(l) for l in (SCRATCH / "recovered_labels.jsonl").open(encoding="utf-8")]
    rich = [r for r in rows if len(r["raw_reports"]) >= 6]
    for i, r in enumerate(rnd.sample(rich, 12), 1):
        print("=" * 78)
        print(f"[{i}] {r['display_id']}  {r['cohort']}")
        for line in r["raw_reports"]:
            lab, disp = map_statement(norm(line))
            print(f"    | {line}")
            print(f"    |      -> [{disp}] {sorted(lab)}")
        print(f"  EXISTING : {r['existing_labels']}")
        print(f"  RECOVERED: {r['recovered_labels']}")
