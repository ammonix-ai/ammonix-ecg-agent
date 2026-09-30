"""Record the deliberate high-sensitivity operating point in the shipped artifacts.

Decision 2026-08-14: keep primary_f1 (0.2738) despite 6.3% of the universe firing
and a 1.4% PPV. Rationale: Brugada carries sudden-death risk, real suspicion cases are
common, and the false positives are conduction disease rather than normal ECGs - the model
fires on only 0.1% of clean sinus-rhythm normals. Missing a real case costs more than
flagging an abnormal one.

Without this written down, the next person reads "PPV 1.4%" as a defect and raises the
threshold, silently trading away the sensitivity the choice was made to protect.
"""
from __future__ import annotations

import json
from pathlib import Path

NOTE = (
    "OPERATING POINT IS DELIBERATE - DO NOT RAISE THE THRESHOLD TO 'FIX' PPV. "
    "default_threshold 0.2738 (primary_f1) flags ~6.3% of the universe at ~1.4% PPV "
    "against the 76 validated positives. This is an intentional high-sensitivity choice "
    "for a sudden-death diagnosis: Brugada suspicion is common, and the false positives "
    "are conduction disease (RBBB/LBBB/fascicular block/IVCD/pacing - the recognised "
    "phenocopies), not normal ECGs. The model fires on only ~0.1% of clean sinus-rhythm "
    "normals, so the cost is surfacing abnormal ECGs for review, not alarming healthy "
    "people. Raising to full_f1 (0.8877) would cut firing to 0.16% but drop sensitivity "
    "from 0.72 to 0.18, i.e. miss four of every five real cases."
)

TARGETS = [
    Path("/path/to/ammonix-ecg-agent/models/source_safe_canonical_v1"),
    Path("/path/to/private_training_repo/domains/ecg-12lead/WaveMedix/models/source_safe_canonical_v1"),
]
SAFE = "brugada_syndrome"
DX = "brugada syndrome"


def log(m=""):
    print(m, flush=True)


def add_note(notes):
    notes = list(notes or [])
    if NOTE not in notes:
        notes.append(NOTE)
    return notes


for pkg in TARGETS:
    p = pkg / "diagnoses" / SAFE / "metadata.json"
    if not p.exists():
        log(f"skip (missing): {p}")
        continue
    d = json.load(open(p, encoding="utf-8"))
    d["notes"] = add_note(d.get("notes"))
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
    log(f"noted: {p}")

    mp = pkg / "manifest.json"
    man = json.load(open(mp, encoding="utf-8"))
    for e in man["diagnoses"]:
        if e.get("safe_name") == SAFE and "notes" in e:
            e["notes"] = add_note(e.get("notes"))
    json.dump(man, open(mp, "w", encoding="utf-8"), indent=2)
    log(f"noted: {mp}")

# universe metadata: per-class diagnoses block, if it carries notes
up = Path("/path/to/ammonix-ecg-agent/_staging/universe/metadata.json")
m = json.load(open(up, encoding="utf-8"))
hit = False
for d in m.get("diagnoses", []):
    if isinstance(d, dict) and d.get("diagnosis") == DX and "notes" in d:
        d["notes"] = add_note(d.get("notes"))
        hit = True
m.setdefault("notes", [])
if NOTE not in m["notes"]:
    m["notes"].append(NOTE)
json.dump(m, open(up, "w", encoding="utf-8"), indent=2)
log(f"noted: {up}  (per-class block: {'yes' if hit else 'top-level only'})")

chk = json.load(open(TARGETS[0] / "diagnoses" / SAFE / "metadata.json", encoding="utf-8"))
log(f"\nverified: brugada notes = {len(chk['notes'])} entries, "
    f"default_threshold {chk['default_threshold']:.4f} (unchanged)")
