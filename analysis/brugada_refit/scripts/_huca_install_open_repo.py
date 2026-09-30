"""Install the rebuilt brugada class into the Ammonix ECG Agent repo, scrubbed for release.

The open package ships publicly (Hugging Face, per ammonix-ecg-agent/.gitignore), so the
partner hospital must not be identifiable. This reuses the repo's OWN scrub policy by
importing scripts/scrub_model_manifest.py, so it cannot drift from it.

One thing that script cannot do: it folds exact source-name strings, so free-text prose
naming the hospital passes straight through. The rebuilt notes are therefore written
public-safe here rather than scrubbed afterwards.

Also scrubs the per-diagnosis metadata.json, which the shipped scrub never touched.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

OPEN = Path("/path/to/ammonix-ecg-agent")
PKG = OPEN / "models" / "source_safe_canonical_v1"
STAGE = Path("/path/to/huca_brugada_refit_workdir/model_package")
BACKUP = Path("/path/to/huca_brugada_refit_workdir/open_repo_brugada_backup_20260814")
SAFE = "brugada_syndrome"

# public-safe restatement of the private notes; same science, no hospital identity
PUBLIC_NOTES = [
    "Refit 2026-08-14 on the clinical-partner Brugada cohort only: 76 cases vs 287 "
    "controls drawn from the same study, using that cohort's own per-patient "
    "diagnostic labels as source of truth.",
    "Supersedes a build that trained on MIMIC (57 positives) with the partner cohort "
    "excluded from training, and counted all 363 partner records as positive "
    "(n_total_pos 420) - which placed 287 of that study's own controls in the "
    "positive class.",
    "7 atypical cases are counted positive here; the confirmed-only variant (69 cases) "
    "gives an out-of-fold AUROC of 0.889.",
    "The 57 MIMIC brugada-labelled records are treated as negatives. Transfer between "
    "the partner cohort and MIMIC is at chance in both directions (0.52 and 0.55) even "
    "though each cohort is internally learnable, indicating the two label sets capture "
    "different constructs.",
    "SCOPE: false positives concentrate in conduction disease - 83% of flagged MIMIC "
    "controls carry RBBB, LBBB, fascicular block, IVCD or pacing. The model is nearly "
    "silent on clean normals (22 of 19,805 sinus-rhythm normals fire). It is not a "
    "general-population Brugada detector without a conduction-abnormality differential.",
]


def log(m=""):
    print(m, flush=True)


spec = importlib.util.spec_from_file_location("scrub", OPEN / "scripts" / "scrub_model_manifest.py")
scrub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scrub)
log(f"using repo scrub policy: {sorted(scrub.PRIVATE_SOURCES)} -> {scrub.PRIVATE_LABEL}")

dest = PKG / "diagnoses" / SAFE
BACKUP.mkdir(parents=True, exist_ok=True)
for f in dest.iterdir():
    shutil.copy2(f, BACKUP / f.name)
shutil.copy2(PKG / "manifest.json", BACKUP / "manifest.json.orig")
log(f"backed up open-repo brugada package -> {BACKUP}")

# ---------------------------------------------------------------- scrub metadata
meta = json.load(open(STAGE / "metadata.json", encoding="utf-8"))
meta["notes"] = PUBLIC_NOTES
counter = {"mappings": 0, "lists": 0, "scalars": 0}
meta = scrub.scrub_sources(meta, counter)
log(f"scrubbed metadata: {counter}")

blob = json.dumps(meta)
for bad in ("HUCA", "huca", "Huca"):
    assert bad not in blob, f"{bad!r} still present in metadata"
log("  metadata.json is free of the partner name")

log("\ninstalling ...")
for f in sorted(STAGE.iterdir()):
    if f.name == "metadata.json":
        continue
    shutil.copy2(f, dest / f.name)
    log(f"  {f.name}")
json.dump(meta, open(dest / "metadata.json", "w", encoding="utf-8"), indent=2)
log("  metadata.json (scrubbed)")

# ---------------------------------------------------------------- manifest
mpath = PKG / "manifest.json"
man = json.load(open(mpath, encoding="utf-8"))
idx = next(i for i, d in enumerate(man["diagnoses"]) if d.get("safe_name") == SAFE)
old_keys = list(man["diagnoses"][idx].keys())
man["diagnoses"][idx] = {k: meta[k] for k in old_keys if k in meta}
acc = [d for d in man["diagnoses"] if d.get("status") == "accepted"]
op, of = man["macro_primary_auc"], man["macro_final_auc"]
man["macro_primary_auc"] = float(np.mean([d["primary_auc"] for d in acc]))
man["macro_final_auc"] = float(np.mean([d["final_full_auc"] for d in acc]))
note = ("brugada syndrome was refit 2026-08-14 on the clinical-partner cohort alone "
        "(76 cases vs 287 same-study controls). It is the one class trained on that "
        "cohort; the config block describes the other 31.")
if note not in man["notes"]:
    man["notes"].insert(0, note)
man = scrub.scrub_sources(man, {"mappings": 0, "lists": 0, "scalars": 0})
tmp = mpath.with_suffix(".json.tmp")
json.dump(man, open(tmp, "w", encoding="utf-8"), indent=2)
tmp.replace(mpath)
log(f"\nmanifest: macro_primary {op:.6f} -> {man['macro_primary_auc']:.6f}, "
    f"macro_final {of:.6f} -> {man['macro_final_auc']:.6f}")

# ------------------------------------------------- leak sweep over the whole package
log("\nleak sweep across the installed package ...")
hits = []
for f in PKG.rglob("*"):
    if not f.is_file():
        continue
    try:
        txt = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if "HUCA" in txt or "huca" in txt:
        hits.append(f.relative_to(PKG))
log(f"  brugada class + manifest: {'CLEAN' if not any(str(h).startswith(('manifest', f'diagnoses/{SAFE}')) for h in hits) else 'LEAK'}")
if hits:
    log(f"  PRE-EXISTING leaks in {len(hits)} other files (never scrubbed by "
        f"scrub_model_manifest.py, which only reads manifest.json):")
    for h in sorted(hits)[:6]:
        log(f"    {h}")
    if len(hits) > 6:
        log(f"    ... and {len(hits) - 6} more")

import xgboost as xgb
chk = json.load(open(dest / "metadata.json", encoding="utf-8"))
assert chk["n_total_pos"] == 76 and chk["train_sources"] == ["clinical-partner"]
vs = np.load(dest / "validation_scores.npy")
assert vs.shape == (34103,)
for i in range(5):
    b = xgb.Booster(); b.load_model(str(dest / f"fold_{i}.xgb.json"))
    assert b.num_features() == 500
log(f"\nverified: n_total_pos={chk['n_total_pos']}, train_sources={chk['train_sources']}, "
    f"primary_auc={chk['primary_auc']:.4f}, threshold={chk['default_threshold']:.4f}")
log("INSTALL OK (open repo)")
