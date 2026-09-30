"""Scout: verify HUCA rows in the Apr28 matrix and the metadata.csv join."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")

ids = ss.load_recording_ids(DATA / "apr28_ids.csv")
manifest = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = [ss.source_of_manifest_path(r, manifest.get(r, "")) for r in ids]

print(f"apr28 rows: {len(ids)}")
print("cohort source counts:", Counter(cohort).most_common())

huca_idx = [i for i, c in enumerate(cohort) if c == "Brugada-HUCA"]
huca_ids = [ids[i] for i in huca_idx]
print(f"\nHUCA rows in matrix: {len(huca_idx)}")
print("first 5 HUCA ids:", huca_ids[:5])
print("sample manifest path:", manifest.get(huca_ids[0], "") if huca_ids else "n/a")

# acquisition_source must keep HUCA separate
acq = [ss.acquisition_source(ids[i], cohort[i], {}, {}) for i in huca_idx]
print("acquisition_source values:", Counter(acq).most_common())

# subjects
subj = [ss.subject_of_apr28_id(ids[i], "Brugada-HUCA", {}, {}) for i in huca_idx]
print("unique subjects among HUCA rows:", len(set(subj)), "(expect == n rows)")

# metadata discovery + join
meta_path = ss.find_huca_metadata_path(manifest)
print("\nfind_huca_metadata_path ->", meta_path)

meta_rows = {}
if meta_path and meta_path.exists():
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta_rows[str(row["patient_id"]).strip()] = row

print(f"metadata.csv rows: {len(meta_rows)}")
print("brugada value counts:", Counter(str(r["brugada"]).strip() for r in meta_rows.values()).most_common())
print("basal_pattern counts:", Counter(str(r["basal_pattern"]).strip() for r in meta_rows.values()).most_common())
print("sudden_death counts:", Counter(str(r["sudden_death"]).strip() for r in meta_rows.values()).most_common())

matrix_set, meta_set = set(huca_ids), set(meta_rows)
print(f"\njoin: matrix&meta={len(matrix_set & meta_set)}  matrix-only={len(matrix_set - meta_set)}  meta-only={len(meta_set - matrix_set)}")
if matrix_set - meta_set:
    print("  matrix-only ids:", sorted(matrix_set - meta_set)[:10])
if meta_set - matrix_set:
    print("  meta-only ids:", sorted(meta_set - matrix_set)[:10])
print("duplicate ids in matrix:", [k for k, v in Counter(huca_ids).items() if v > 1])

# what the loader's own helper yields
status = ss.load_huca_brugada_metadata(meta_path)
pos_loader = sum(1 for r in huca_ids if status.get(r) is True)
print(f"\nload_huca_brugada_metadata (int>0) positives among matrix rows: {pos_loader}")
strict = sum(1 for r in huca_ids if str(meta_rows.get(r, {}).get("brugada", "")).strip() == "1")
atyp = sum(1 for r in huca_ids if str(meta_rows.get(r, {}).get("brugada", "")).strip() == "2")
ctrl = sum(1 for r in huca_ids if str(meta_rows.get(r, {}).get("brugada", "")).strip() == "0")
print(f"strict brugada==1: {strict}   atypical==2: {atyp}   controls==0: {ctrl}")

# what apr28_labels.jsonl says for HUCA rows (the buggy source-level label)
labels = ss.load_apr28_labels(DATA / "apr28_labels.jsonl")
file_pos = sum(1 for r in huca_ids if ss.HUCA_BRUGADA_LABEL in labels.get(r, set()))
print(f"\napr28_labels.jsonl says 'brugada syndrome' for {file_pos}/{len(huca_ids)} HUCA rows (source-level label)")
lab_counter = Counter()
for r in huca_ids:
    lab_counter.update(labels.get(r, set()))
print("top other labels on HUCA rows:", lab_counter.most_common(12))

# brugada-syndrome positives elsewhere in apr28
all_labels_applied = []
for i, r in enumerate(ids):
    src = ss.acquisition_source(r, cohort[i], {}, {})
    all_labels_applied.append(ss.apply_huca_brugada_metadata(r, src, set(labels.get(r, set())), status))
by_src = Counter()
for i, lab in enumerate(all_labels_applied):
    if ss.HUCA_BRUGADA_LABEL in lab:
        by_src[ss.acquisition_source(ids[i], cohort[i], {}, {})] += 1
print("\n'brugada syndrome' positives by source AFTER metadata fix:", by_src.most_common())

# matrix sanity on the HUCA slice
X = np.load(DATA / "apr28_X.npy", mmap_mode="r")
print(f"\nmatrix shape: {X.shape}")
sub = np.asarray(X[np.asarray(huca_idx)], dtype=np.float32)
print(f"HUCA slice: {sub.shape}  dtype={sub.dtype}  nan={int(np.isnan(sub).sum())}  inf={int(np.isinf(sub).sum())}")
print(f"all-zero rows: {int((sub == 0).all(axis=1).sum())}   constant cols in slice: {int((sub.max(0) == sub.min(0)).sum())}")

# sampling-rate check from the .hea headers of a few HUCA records
print("\nheader check (fs / n_samples):")
for r in huca_ids[:4]:
    p = Path(manifest.get(r, ""))
    if p.exists():
        print("  ", r, "->", p.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip())
    else:
        print("  ", r, "-> MISSING", p)
