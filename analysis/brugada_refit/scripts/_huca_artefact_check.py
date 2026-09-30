"""Is the HUCA Brugada AUROC physiology or an acquisition/file artefact?

Three independent checks, none of which require the classifier to be trustworthy:
  1. header artefact  - do case and control .hea files differ systematically?
  2. patient_id leak  - does the bare patient_id separate cases from controls?
  3. feature attribution - which features does the in-fold screen actually pick,
     and do they live in V1-V3 where Brugada lives?
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/path/to/private_training_repo/domains/ecg-12lead/WaveMedix")

import numpy as np
from sklearn.metrics import roc_auc_score

from ammonix.diagnostics import source_safe_cv as ss

DATA = Path("/path/to/private_training_repo/domains/ecg-12lead/data/qpsi_jun3")
OUT = Path("/path/to/huca_brugada_refit_workdir")
OUT.mkdir(parents=True, exist_ok=True)
DX = ss.HUCA_BRUGADA_LABEL


def p(m=""):
    print(m, flush=True)


ids_all = ss.load_recording_ids(DATA / "apr28_ids.csv")
manifest = ss.load_apr28_manifest(DATA / "apr28_manifest.csv")
cohort = [ss.source_of_manifest_path(r, manifest.get(r, "")) for r in ids_all]
idx = np.asarray([i for i, c in enumerate(cohort) if c == "Brugada-HUCA"])
ids = [ids_all[i] for i in idx]

meta_path = ss.find_huca_metadata_path(manifest)
meta = {}
with open(meta_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        meta[str(row["patient_id"]).strip()] = {
            k: int(float(str(row[k]).strip())) for k in ("basal_pattern", "sudden_death", "brugada")
        }

# confirmed-variant labelling: drop the 7 atypical
keep = [i for i, r in enumerate(ids) if meta[r]["brugada"] in (0, 1)]
sub_ids = [ids[i] for i in keep]
y = np.asarray([meta[r]["brugada"] == 1 for r in sub_ids], dtype=int)
p(f"confirmed variant: {len(sub_ids)} rows, {int(y.sum())} cases, {int((1 - y).sum())} controls")

# ---------------------------------------------------------------- 1. header artefact
p("\n" + "=" * 78)
p("1. HEADER ARTEFACT CHECK  (.hea of cases vs controls)")
p("=" * 78)

fields = defaultdict(lambda: defaultdict(Counter))  # field -> class -> values
first_line = defaultdict(Counter)
comments = defaultdict(Counter)
missing = []
for r, lab in zip(sub_ids, y):
    hea = Path(manifest.get(r, ""))
    if not hea.exists():
        missing.append(r)
        continue
    lines = hea.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        continue
    head = lines[0].split()
    # name nsig fs nsamp [basetime basedate]
    first_line[int(lab)][" ".join(head[1:4])] += 1
    if len(head) > 4:
        first_line[int(lab)]["EXTRA:" + " ".join(head[4:])] += 1
    for sig in lines[1:]:
        s = sig.strip()
        if not s:
            continue
        if s.startswith("#"):
            comments[int(lab)][s[:60]] += 1
            continue
        parts = s.split()
        if len(parts) >= 3:
            fields["format"][int(lab)][parts[1]] += 1
            fields["gain"][int(lab)][parts[2]] += 1
        if len(parts) >= 5:
            fields["adc_res"][int(lab)][parts[3]] += 1
            fields["adc_zero"][int(lab)][parts[4]] += 1
        if len(parts) >= 6:
            fields["init_value_sign"][int(lab)]["neg" if parts[5].startswith("-") else "nonneg"] += 1

if missing:
    p(f"  WARNING: {len(missing)} header files missing")
p(f"  first-line 'nsig fs nsamp'  controls: {dict(first_line[0])}")
p(f"  first-line 'nsig fs nsamp'  cases   : {dict(first_line[1])}")
for f_name in ("format", "gain", "adc_res", "adc_zero"):
    c0, c1 = fields[f_name][0], fields[f_name][1]
    same = set(c0) == set(c1)
    p(f"  {f_name:14s} controls={dict(list(c0.most_common(4)))} cases={dict(list(c1.most_common(4)))} "
      f"-> {'IDENTICAL value set' if same else 'DIFFERENT value set'}")
p(f"  comment lines controls={sum(comments[0].values())} cases={sum(comments[1].values())}")
if comments[0] or comments[1]:
    p(f"    controls sample: {comments[0].most_common(3)}")
    p(f"    cases sample   : {comments[1].most_common(3)}")

# file sizes
sizes = {0: [], 1: []}
for r, lab in zip(sub_ids, y):
    dat = Path(manifest.get(r, "")).with_suffix(".dat")
    if dat.exists():
        sizes[int(lab)].append(dat.stat().st_size)
for k in (0, 1):
    if sizes[k]:
        a = np.asarray(sizes[k])
        p(f"  .dat size class{k}: n={len(a)} unique={len(set(a.tolist()))} min={a.min()} max={a.max()}")

# ---------------------------------------------------------------- 2. patient_id leak
p("\n" + "=" * 78)
p("2. PATIENT_ID LEAK CHECK")
p("=" * 78)
pid = np.asarray([float(r) for r in sub_ids])
auc_pid = roc_auc_score(y, pid)
p(f"  AUROC of raw patient_id as a score : {auc_pid:.4f}  (0.50 = ids carry no class info)")
p(f"  AUROC of -patient_id               : {1 - auc_pid:.4f}")
p(f"  case id range   : {pid[y == 1].min():.0f} - {pid[y == 1].max():.0f} (median {np.median(pid[y == 1]):.0f})")
p(f"  control id range: {pid[y == 0].min():.0f} - {pid[y == 0].max():.0f} (median {np.median(pid[y == 0]):.0f})")

# ---------------------------------------------------------------- 3. feature attribution
p("\n" + "=" * 78)
p("3. FEATURE ATTRIBUTION  (top screen features per fold, train rows only)")
p("=" * 78)

cols = (DATA / "apr28_cols.txt").read_text(encoding="utf-8").splitlines()
p(f"  feature space: {len(cols)} columns")

X_all = np.load(DATA / "apr28_X.npy", mmap_mode="r")
X = np.ascontiguousarray(np.asarray(X_all[idx[np.asarray(keep)]], dtype=np.float32))
del X_all
p(f"  HUCA slice: {X.shape}")

from sklearn.model_selection import StratifiedGroupKFold

groups = np.asarray([f"Brugada-HUCA:{r}" for r in sub_ids], dtype=object)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

rank_acc: Counter[str] = Counter()
top_gain: Counter[str] = Counter()
for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
    screen = ss.default_xgb_screen_factory()
    screen.fit(X[tr], y[tr])
    top = ss._top_features_from_screen(screen, 500, X.shape[1])
    for rank, j in enumerate(top[:50]):
        rank_acc[cols[j]] += 50 - rank
    clf = ss.default_xgb_classifier_factory()
    clf.fit(X[tr][:, top], y[tr])
    imp = np.asarray(clf.feature_importances_)
    for j in np.argsort(imp)[::-1][:20]:
        top_gain[cols[top[j]]] += float(imp[j])

p("\n  Most consistently top-ranked screen features (across the 5 folds):")
for name, score in rank_acc.most_common(25):
    p(f"    {score:5d}  {name}")

p("\n  Highest classifier importance (summed over folds):")
for name, score in top_gain.most_common(15):
    p(f"    {score:6.3f}  {name}")

# lead attribution over the union of per-fold top-50
lead_re = re.compile(r"(?:^|::|_)(V[1-6]|I{1,3}|aVR|aVL|aVF)(?:$|::|_)", re.IGNORECASE)
lead_counts: Counter[str] = Counter()
for name, _ in rank_acc.most_common(200):
    m = lead_re.search(name)
    lead_counts[m.group(1).upper() if m else "no-lead"] += 1
p(f"\n  Lead attribution of the top 200 screen features: {lead_counts.most_common()}")
rp = sum(v for k, v in lead_counts.items() if k in {"V1", "V2", "V3"})
p(f"  right-precordial (V1-V3) share: {rp}/{sum(lead_counts.values())} "
  f"= {rp / max(1, sum(lead_counts.values())):.1%}   (Brugada lives in V1-V3)")

with open(OUT / "feature_attribution_confirmed.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["feature", "rank_score_sum_top50_5folds"])
    for name, score in rank_acc.most_common(300):
        w.writerow([name, score])
p(f"\n  wrote {OUT / 'feature_attribution_confirmed.csv'}")
