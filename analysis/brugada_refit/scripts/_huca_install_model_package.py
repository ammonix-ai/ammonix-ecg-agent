"""Install the rebuilt brugada_syndrome class into source_safe_canonical_v1.

Touches ONLY:
  diagnoses/brugada_syndrome/*      (6 files replaced)
  manifest.json                     (brugada entry + macro AUCs + one note)

The global config block is deliberately left alone: it records how the other 31 classes
were built (with Brugada-HUCA excluded), which is still true of them. The brugada
exception is recorded in the manifest notes instead.

Backup of the shipped version: huca_brugada_refit/shipped_brugada_backup_20260814/
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

LIVE = Path("/path/to/private_training_repo/domains/ecg-12lead/WaveMedix/models/source_safe_canonical_v1")
STAGE = Path("/path/to/huca_brugada_refit_workdir/model_package")
BACKUP = Path("/path/to/huca_brugada_refit_workdir/shipped_brugada_backup_20260814")
SAFE = "brugada_syndrome"
NOTE = ("brugada syndrome was refit 2026-08-14 on Brugada-HUCA only (76 cases vs 287 "
        "same-study controls, labels from HUCA metadata.csv). It is the one class whose "
        "excluded_train_sources is [] rather than ['Brugada-HUCA']; the config block below "
        "describes the other 31 classes. Its prior build counted all 363 HUCA rows as "
        "positive (n_total_pos 420) and trained on MIMIC.")


def log(m=""):
    print(m, flush=True)


assert BACKUP.exists() and (BACKUP / "metadata.json").exists(), "backup missing - refusing"
assert (STAGE / "metadata.json").exists(), "staged package missing"

dest = LIVE / "diagnoses" / SAFE
new_meta = json.load(open(STAGE / "metadata.json", encoding="utf-8"))

log("installing model files ...")
for f in sorted(STAGE.iterdir()):
    shutil.copy2(f, dest / f.name)
    log(f"  {f.name}")

# ------------------------------------------------------------------ manifest
mpath = LIVE / "manifest.json"
man = json.load(open(mpath, encoding="utf-8"))
idx = next(i for i, d in enumerate(man["diagnoses"]) if d.get("safe_name") == SAFE)
old = man["diagnoses"][idx]
man["diagnoses"][idx] = {k: new_meta[k] for k in old.keys() if k in new_meta}
missing = [k for k in old.keys() if k not in new_meta]
assert not missing, f"manifest keys missing from new metadata: {missing}"

acc = [d for d in man["diagnoses"] if d.get("status") == "accepted"]
old_p, old_f = man["macro_primary_auc"], man["macro_final_auc"]
man["macro_primary_auc"] = float(np.mean([d["primary_auc"] for d in acc]))
man["macro_final_auc"] = float(np.mean([d["final_full_auc"] for d in acc]))
if NOTE not in man["notes"]:
    man["notes"].insert(0, NOTE)

tmp = mpath.with_suffix(".json.tmp")
json.dump(man, open(tmp, "w", encoding="utf-8"), indent=2)
tmp.replace(mpath)
log(f"\nmanifest updated:")
log(f"  macro_primary_auc {old_p:.6f} -> {man['macro_primary_auc']:.6f}")
log(f"  macro_final_auc   {old_f:.6f} -> {man['macro_final_auc']:.6f}")

# ------------------------------------------------------------------ verify
log("\nverifying installed package ...")
import xgboost as xgb
chk = json.load(open(dest / "metadata.json", encoding="utf-8"))
assert chk["n_total_pos"] == 76 and chk["train_sources"] == ["Brugada-HUCA"]
vs = np.load(dest / "validation_scores.npy")
assert vs.shape == (34103,) and vs.dtype == np.float32
for i in range(5):
    b = xgb.Booster(); b.load_model(str(dest / f"fold_{i}.xgb.json"))
    fi = np.load(dest / f"fold_{i}.feature_indices.npy")
    assert b.num_features() == 500 and fi.shape == (500,) and fi.dtype == np.int32
man2 = json.load(open(mpath, encoding="utf-8"))
e = next(d for d in man2["diagnoses"] if d["safe_name"] == SAFE)
assert e["n_total_pos"] == 76 and abs(e["primary_auc"] - chk["primary_auc"]) < 1e-12
assert len(man2["diagnoses"]) == 32
log("  5 boosters load, 500 features each")
log(f"  validation_scores {vs.shape} float32, range {vs.min():.4f}-{vs.max():.4f}")
log(f"  manifest entry consistent, {len(man2['diagnoses'])} diagnoses intact")
log("\nINSTALL OK")
