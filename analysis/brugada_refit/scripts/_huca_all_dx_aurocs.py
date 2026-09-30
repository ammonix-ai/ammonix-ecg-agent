"""Compute the three AUROCs the UI shows, for all 32 diagnoses.

  primary   model's held-out AUROC vs same-source controls only
  final     model's held-out AUROC vs every negative in its own dataset
  area      AUROC on the displayed universe (what the ROC panel draws)

Emits a markdown table for docs/auroc-metrics.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

OPEN = Path("/path/to/ammonix-ecg-agent")
U = OPEN / "_staging" / "universe"
OUT = Path("/path/to/huca_brugada_refit_workdir/all_dx_aurocs.md")

meta = json.load(open(U / "metadata.json", encoding="utf-8"))
classes = meta["classes"]
P = np.load(U / "probabilities.npy", mmap_mode="r")

labels = []
for line in open(U / "patients.jsonl", encoding="utf-8"):
    labels.append(set(json.loads(line).get("labels") or []))
n = len(labels)
print(f"universe rows {n}, classes {len(classes)}")

held = {d["diagnosis"]: d for d in meta.get("diagnoses", []) if isinstance(d, dict)}

rows = []
for j, dx in enumerate(classes):
    y = np.fromiter((dx in l for l in labels), dtype=bool, count=n)
    s = np.asarray(P[:, j], dtype=np.float64)
    npos = int(y.sum())
    area = roc_auc_score(y.astype(int), s) if 0 < npos < n else float("nan")
    h = held.get(dx, {})
    rows.append({
        "dx": dx,
        "primary": h.get("primary_auc"),
        "final": h.get("final_full_auc"),
        "area": area,
        "n_model": h.get("n_total_pos"),
        "n_universe": npos,
        "train": "+".join(h.get("train_sources") or []) or "-",
    })

rows.sort(key=lambda r: -(r["area"] if r["area"] == r["area"] else -1))

lines = []
lines.append("| diagnosis | primary | final | area | n+ (model) | n+ (universe) | trained on |")
lines.append("|---|---:|---:|---:|---:|---:|---|")
def f(v):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}"
for r in rows:
    lines.append(
        f"| {r['dx']} | {f(r['primary'])} | {f(r['final'])} | {f(r['area'])} | "
        f"{r['n_model'] if r['n_model'] is not None else '—'} | {r['n_universe']} | {r['train']} |"
    )

table = "\n".join(lines)
OUT.write_text(table + "\n", encoding="utf-8")
print(table)

fa = np.array([r["area"] for r in rows if r["area"] == r["area"]])
fp = np.array([r["primary"] for r in rows if r["primary"] is not None])
ff = np.array([r["final"] for r in rows if r["final"] is not None])
print()
print(f"macro primary {fp.mean():.4f} | macro final {ff.mean():.4f} | macro area {fa.mean():.4f}")
higher = sum(1 for r in rows if r["primary"] is not None and r["area"] == r["area"] and r["area"] > r["primary"])
print(f"classes where area > primary: {higher}/{len(rows)}")
