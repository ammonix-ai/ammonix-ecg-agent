"""Restore real source names and proper attribution to the open-repo brugada class.

The HUCA Brugada dataset is published on PhysioNet under CC BY-SA 4.0, which REQUIRES
attribution. Folding it to "clinical-partner" removed credit the licence obliges us to
give, and removed provenance a reader needs. MIMIC-IV ECG is likewise a public
(credentialed) release. Naming a source is citation, not redistribution.

private-STEMI is left folded: it is the one cohort whose own name asserts restricted use,
and it is not part of this class's training or evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

OPEN = Path("/path/to/ammonix-ecg-agent")
PKG = OPEN / "models" / "source_safe_canonical_v1"
STAGE = Path("/path/to/huca_brugada_refit_workdir/model_package")
SAFE = "brugada_syndrome"

CITED_NOTES = [
    "Refit 2026-08-14 on the HUCA Brugada cohort only: 76 cases vs 287 controls from the "
    "same study. Labels taken from that dataset's own metadata.csv 'brugada' column "
    "(0=healthy, 1=confirmed, 2=other/atypical), which is the source of truth.",
    "SOURCE / ATTRIBUTION: 'HUCA 12-lead ECG recordings for the study of Brugada syndrome' "
    "v1.0.0, PhysioNet, licensed CC BY-SA 4.0. Hospital Universitario Central de Asturias. "
    "Only derived q_psi features are distributed here, never the raw recordings; the raw "
    "dataset is publicly downloadable from PhysioNet under its own licence.",
    "Supersedes a build that trained on MIMIC (57 positives) with the HUCA cohort excluded "
    "from training, and counted all 363 HUCA records as positive (n_total_pos 420) - which "
    "placed 287 of that study's own healthy controls in the positive class.",
    "brugada=2 (7 atypical cases) counted positive, matching load_huca_brugada_metadata. "
    "The confirmed-only variant (69 cases) gives an out-of-fold AUROC of 0.889.",
    "The 57 MIMIC brugada-labelled records are treated as negatives. Transfer between HUCA "
    "and MIMIC is at chance in both directions (HUCA->MIMIC 0.519, MIMIC->HUCA 0.546) even "
    "though each cohort is internally learnable (MIMIC-internal 0.710), indicating the two "
    "label sets capture different constructs. MIMIC's Brugada label is discharge-letter "
    "derived and not echo/challenge validated.",
    "SCOPE: false positives concentrate in conduction disease - 83% of flagged MIMIC "
    "controls carry RBBB, LBBB, fascicular block, IVCD or pacing, the recognised Brugada "
    "phenocopies. The model is near-silent on clean normals (22 of 19,805 sinus-rhythm "
    "normals fire at the default threshold). It is not a general-population Brugada "
    "detector without a conduction-abnormality differential.",
]


def log(m=""):
    print(m, flush=True)


meta = json.load(open(STAGE / "metadata.json", encoding="utf-8"))   # unscrubbed original
meta["notes"] = CITED_NOTES

dest = PKG / "diagnoses" / SAFE
json.dump(meta, open(dest / "metadata.json", "w", encoding="utf-8"), indent=2)
log(f"restored {dest / 'metadata.json'}")
log(f"  train_sources = {meta['train_sources']}")
log(f"  pos_by_source = { {k: v for k, v in meta['pos_by_source'].items() if v} }")
log(f"  neg_by_source(HUCA) = {meta['neg_by_source'].get('Brugada-HUCA')}")

mpath = PKG / "manifest.json"
man = json.load(open(mpath, encoding="utf-8"))
idx = next(i for i, d in enumerate(man["diagnoses"]) if d.get("safe_name") == SAFE)
keys = list(man["diagnoses"][idx].keys())
man["diagnoses"][idx] = {k: meta[k] for k in keys if k in meta}
note = ("brugada syndrome (refit 2026-08-14) names its data source directly: the HUCA "
        "Brugada dataset is a public PhysioNet release under CC BY-SA 4.0, which requires "
        "attribution. The other classes still carry the older 'clinical-partner' folding "
        "from scripts/scrub_model_manifest.py; that policy needs review, since folding a "
        "CC BY-SA source removes required credit.")
man["notes"] = [n for n in man["notes"] if "clinical-partner cohort alone" not in n]
if note not in man["notes"]:
    man["notes"].insert(0, note)
tmp = mpath.with_suffix(".json.tmp")
json.dump(man, open(tmp, "w", encoding="utf-8"), indent=2)
tmp.replace(mpath)
log(f"restored manifest entry + note")

e = json.load(open(mpath, encoding="utf-8"))["diagnoses"][idx]
assert e["train_sources"] == ["Brugada-HUCA"] and e["n_total_pos"] == 76
log(f"\nverified: train_sources={e['train_sources']} n_total_pos={e['n_total_pos']} "
    f"primary_auc={e['primary_auc']:.4f}")
