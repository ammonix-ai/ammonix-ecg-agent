#!/usr/bin/env python3
"""End-to-end check of the frozen inference runtime on shipped recordings.

    raw WFDB -> QPSI tokenizer -> feat_order vector -> 5-swarm ensemble
             -> 32 probabilities -> thresholded predictions -> universe placement

For each record it prints the live result next to the universe's stored row so
the two can be compared directly, and a leave-self-out k-NN placement, which is
what a recording that is *not* already in the universe would get.

Usage (from the repo root, with the venv active):

    python scripts/verify_inference.py                    # PTB-XL/HR16370
    python scripts/verify_inference.py HR16370 JS01160    # named records
    python scripts/verify_inference.py --sample 5         # one per source

Set PYTHONIOENCODING=utf-8 on Windows: several diagnosis names are non-ASCII.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from services import projection as projection_service  # noqa: E402
from services.inference import (  # noqa: E402
    STORED_SCORE_KIND,
    classify,
    extract_features,
    load_runtime,
    rank_probabilities,
    top_feature_contributions,
)
from services.projection import load_placement_model, place_point  # noqa: E402
from services.signals import load_record, manifest_rows  # noqa: E402
from services.source_safe_universe import universe_data_dir  # noqa: E402

DEFAULT_RECORD = "HR16370"


def stored_rows() -> dict[str, dict]:
    """recording_id -> universe row, for the 10,876 rows that carry one."""
    path = universe_data_dir() / "patients.jsonl"
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.get("recording_id")
            if rid:
                out[str(rid)] = row
    return out


def pick_sample(n: int) -> list[str]:
    """One record per source, up to n."""
    seen: dict[str, str] = {}
    for row in manifest_rows():
        source = row.get("source", "")
        if source not in seen:
            seen[source] = row["recording_id"]
        if len(seen) >= n:
            break
    return list(seen.values())[:n]


def leave_self_out_placement(model, vector: np.ndarray, self_idx: int | None, k: int) -> dict:
    """k-NN placement with the record's own universe row removed.

    A shipped record is already a point in the universe, so a plain placement
    would find itself at distance ~0 and return its own coordinates. Dropping
    that row shows what an unseen recording's placement actually looks like.
    """
    diff = model.probabilities.astype(np.float64, copy=False) - vector[None, :]
    distances = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    if self_idx is not None:
        distances[self_idx] = np.inf
    idx = np.argpartition(distances, k - 1)[:k]
    idx = idx[np.argsort(distances[idx])]
    weights = 1.0 / (distances[idx] + 1e-9)
    weights /= weights.sum()
    out = {}
    for method, coords in model.coords.items():
        out[method] = [float(v) for v in (coords[idx] * weights[:, None]).sum(axis=0)]
    out["neighbors"] = [
        {"displayId": model.display_ids[i], "distance": round(float(distances[i]), 4),
         "primary": model.primaries[i]}
        for i in idx.tolist()
    ]
    return out


def verify(recording_id: str, stored: dict[str, dict], *, explain: bool = True) -> dict:
    runtime = load_runtime()
    model = load_placement_model()

    print("=" * 78)
    print(f"RECORD {recording_id}")
    print("=" * 78)

    signals = load_record(recording_id)
    print(
        f"signal      : {signals.signal.shape[0]} leads x {signals.signal.shape[1]} samples "
        f"@ {signals.sampling_rate} Hz ({signals.duration_sec:.1f} s), source {signals.source}, "
        f"display {signals.display_id}"
    )

    t0 = time.perf_counter()
    features = extract_features(signals.signal, signals.sampling_rate, recording_id=recording_id)
    t_extract = time.perf_counter() - t0

    n_features = len(features)
    n_expected = runtime.feature_count
    unknown = [name for name in features if name not in runtime.feature_index]
    absent = [name for name in runtime.feature_names if name not in features]
    print(f"features    : {n_features} extracted in {t_extract:.1f}s")
    print(f"feat_order  : {n_expected} (from the package's feature_names.json)")
    print(
        f"match       : {'YES' if n_features == n_expected and not unknown and not absent else 'NO'}"
        f"  (missing {len(absent)}, unexpected {len(unknown)})"
    )

    t0 = time.perf_counter()
    result = classify(features)
    t_classify = time.perf_counter() - t0
    print(f"classify    : {t_classify:.2f}s, scoreKind={result['scoreKind']}")

    print("\ntop 5 (live, 5-fold mean):")
    for dx, p in rank_probabilities(result["probabilities"], 5):
        thr = result["thresholds"][dx]
        flag = "  <- over threshold" if p >= thr else ""
        print(f"    {dx:<48s} {p:.6f}   thr {thr:.4f}{flag}")
    print(f"\npredictions : {result['predictions']}")

    row = stored.get(recording_id)
    comparison: dict = {}
    if row is None:
        print("stored      : this record has no universe row")
    else:
        stored_top = row.get("top_prediction")
        stored_max = float(row.get("max_score", float("nan")))
        idx = int(row["idx"])
        stored_vec = model.probabilities[idx].astype(np.float64)
        live_vec = np.array([result["probabilities"][dx] for dx in model.classes])
        delta = np.abs(stored_vec - live_vec)
        print("\nstored universe row vs live run")
        print(f"    stored top  : {stored_top:<40s} {stored_max:.6f}")
        print(f"    live top    : {result['topPrediction']:<40s} {result['maxScore']:.6f}")
        print(f"    agree       : {'YES' if stored_top == result['topPrediction'] else 'NO'}")
        print(f"    stored preds: {row.get('predictions')}")
        print(f"    live preds  : {result['predictions']}")
        print(f"    max |delta| over all 32 classes: {float(delta.max()):.3e}")
        print(f"    labels      : {row.get('labels')}")
        comparison = {
            "storedTopPrediction": stored_top,
            "storedMaxScore": stored_max,
            "storedScoreKind": STORED_SCORE_KIND,
            "liveTopPrediction": result["topPrediction"],
            "liveMaxScore": result["maxScore"],
            "maxAbsDelta": float(delta.max()),
            "agree": stored_top == result["topPrediction"],
        }

    placement = place_point(result["probabilities"])
    print("\nplacement into the fixed universe")
    for method in ("pca", "umap", "tsne"):
        coord = placement.get(method)
        how = placement.get(f"{method}Placement", "n/a")
        if coord is None:
            print(f"    {method:<5s}: unavailable")
        else:
            print(f"    {method:<5s}: [{coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}]   ({how})")
    residual = placement.get("pcaResidualRms")
    if residual is not None:
        print(f"    pca residual rms of the re-derived linear map: {residual:.3e}")
    print("    neighbours (k=6, probability space):")
    for nb in placement["neighbors"]:
        print(f"        {nb['displayId']}  d={nb['distance']:.4f}  {nb['primary']}")

    if row is not None:
        self_idx = int(row["idx"])
        print("\n    stored coordinates of this record's own universe point:")
        for method in ("pca", "umap", "tsne"):
            if method in model.coords:
                c = model.coords[method][self_idx]
                print(f"        {method:<5s}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
        live_vec = np.array([result["probabilities"][dx] for dx in model.classes])
        loo = leave_self_out_placement(model, live_vec, self_idx, projection_service.DEFAULT_K)
        print("\n    leave-self-out k-NN placement (what an unseen record would get):")
        for method in ("pca", "umap", "tsne"):
            if method in loo:
                c = loo[method]
                print(f"        {method:<5s}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
        print("        neighbours:", ", ".join(
            f"{nb['displayId']}(d={nb['distance']}, {nb['primary']})" for nb in loo["neighbors"]
        ))

    if explain:
        print(f"\ntop features driving '{result['topPrediction']}' (mean SHAP over 5 folds, log-odds):")
        for item in top_feature_contributions(features, result["topPrediction"], k=5):
            print(f"    {item['contribution']:+.4f}  {item['name']} = {item['value']}")

    print()
    return {"recordingId": recording_id, "result": result, "comparison": comparison,
            "placement": placement}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="*", help="recording ids (default: HR16370)")
    parser.add_argument("--sample", type=int, default=0, help="verify one record per source")
    parser.add_argument("--quiet", action="store_true", help="silence pipeline INFO logs")
    args = parser.parse_args()

    if args.quiet:
        logging.disable(logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    records = args.records or (pick_sample(args.sample) if args.sample else [DEFAULT_RECORD])

    t0 = time.perf_counter()
    runtime = load_runtime()
    print(
        f"runtime     : package {runtime.version}, {len(runtime.models)} diagnoses x "
        f"{len(runtime.models[0].folds)} folds, {runtime.feature_count} features, "
        f"loaded in {time.perf_counter() - t0:.1f}s"
    )
    try:
        import qpsi_native  # noqa: F401

        print("qpsi_native : present (Rust kernels active)")
    except ImportError:
        print("qpsi_native : absent (pure-Python path; same values, ~16x slower)")

    stored = stored_rows()
    print(f"universe    : {len(stored)} rows carry a recording_id\n")

    disagreements: list[str] = []
    for rid in records:
        out = verify(rid, stored)
        comparison = out["comparison"]
        if comparison and not comparison["agree"]:
            disagreements.append(rid)

    if disagreements:
        print(f"LEADING-DIAGNOSIS DISAGREEMENT on: {', '.join(disagreements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
