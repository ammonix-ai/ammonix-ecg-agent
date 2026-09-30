"""Serve the pre-built source-safe ECG universe (63,256 scored records).

SERVE PATH ONLY. The published universe is a static, pre-computed artifact:
metadata, per-record rows, per-class probabilities and 3-D projections are all
baked on disk. Nothing here trains, scores or re-projects anything, so no
model package, no feature matrices, no XGBoost and no scikit-learn/UMAP/t-SNE
runtime are involved. If a requested projection is not on disk, the request
fails with 404 rather than computing it.

Data layout (see `universe_data_dir()`):

    metadata.json                     model + cohort summary, class list, thresholds
    patients.jsonl                    one JSON row per record (63,256 lines)
    probabilities.npy                 (n_patients, n_classes) float scores  [ROC only]
    projections/pca.npy               (n_patients, 3) float32 coordinates
    projections/pca.json              method + hyper-parameters for that file
    projections/clinical_pca.npy|json
    projections/tsne_perplexity_<P>.npy|json
    projections/umap_neighbors_<K>.npy|json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from diagnosis_canon import canonicalise_all
from diagnosis_severity import primary_diagnosis
from diagnosis_style import DIAGNOSIS_STYLE

logger = logging.getLogger(__name__)

# Negative id keeps the published universe distinct from any locally trained
# cohort; the frontend selects it by this exact value.
SOURCE_SAFE_COHORT_ID = -60603
SOURCE_SAFE_COHORT_NAME = "Source-safe Jun3 universe"
SOURCE_SAFE_VERSION = 1

PROBABILITIES_NAME = "probabilities.npy"
PATIENTS_NAME = "patients.jsonl"
METADATA_NAME = "metadata.json"
PROJECTIONS_DIRNAME = "projections"

SUPPORTED_METHODS = ("pca", "clinical", "umap", "tsne")

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
# Repo-relative default; override with UNIVERSE_DATA_DIR for a deployment that
# keeps the ~40 MB payload outside the checkout.
_DEFAULT_DATA_SUBPATH = ("_staging", "universe")


class UniverseDataMissing(FileNotFoundError):
    """The published universe payload is absent or incomplete."""


def universe_data_dir() -> Path:
    override = os.environ.get("UNIVERSE_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _REPO_ROOT.joinpath(*_DEFAULT_DATA_SUBPATH)


def _display_path(path: Path) -> str:
    """Path for error messages: repo-relative when possible, never absolute.

    Error strings reach the browser, so an operator's directory layout must not
    travel with them.
    """
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return "$UNIVERSE_DATA_DIR"


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise UniverseDataMissing(
            f"Universe {what} not found at {_display_path(path)}. "
            "Download the published universe payload into the repo, or point "
            "UNIVERSE_DATA_DIR at the directory that holds it."
        )
    return path


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

_METADATA_CACHE: dict[str, Any] = {}


def load_universe_metadata() -> dict[str, Any]:
    """Parsed metadata.json, cached on (path, mtime)."""
    path = _require(universe_data_dir() / METADATA_NAME, "metadata")
    key = f"{path}:{path.stat().st_mtime_ns}"
    if _METADATA_CACHE.get("key") == key:
        return _METADATA_CACHE["value"]
    with open(path, encoding="utf-8") as f:
        metadata = json.load(f)
    _METADATA_CACHE.update({"key": key, "value": metadata})
    return metadata


def universe_ready() -> bool:
    root = universe_data_dir()
    return all((root / name).exists() for name in (METADATA_NAME, PATIENTS_NAME))


def source_safe_lattice_status() -> dict[str, Any]:
    """The single cohort row the Lattice/Universe page selects on load.

    Built purely from the cached metadata. The private original derived this
    from the model package plus a 166 GB feature-matrix directory and reported
    the matrix directory as `qpsi_file`; that field is deliberately not emitted
    here — it carried an absolute developer path and there is no matrix
    directory in an open deployment.
    """
    metadata = load_universe_metadata()
    diagnoses = [row for row in metadata.get("diagnoses", []) if row.get("status") == "accepted"]
    n_patients = int(metadata.get("n_patients", 0))
    return {
        "cohort_id": SOURCE_SAFE_COHORT_ID,
        "cohort_name": SOURCE_SAFE_COHORT_NAME,
        "record_count": n_patients,
        "has_qpsi": True,
        "qpsi_records": n_patients,
        "has_training": bool(diagnoses),
        "source_safe": True,
        "cache_built": True,
        "training": {
            "cohort_id": SOURCE_SAFE_COHORT_ID,
            "cohort_name": SOURCE_SAFE_COHORT_NAME,
            "n_patients": n_patients,
            "n_classes": len(diagnoses) or int(metadata.get("n_classes", 0)),
            "n_folds": 5,
            "holdout_size": 0,
            "status": "completed",
            "trained_at": metadata.get("model_created_at"),
            "macro_f1": None,
            "macro_auroc": metadata.get("macro_final_auc"),
            "classes": [row["diagnosis"] for row in diagnoses] or list(metadata.get("classes", [])),
            "model_family": metadata.get("model_family"),
            "version": metadata.get("model_version"),
        },
    }


# ---------------------------------------------------------------------------
# patients + projections
# ---------------------------------------------------------------------------

def _iter_patient_rows() -> Iterator[dict[str, Any]]:
    path = _require(universe_data_dir() / PATIENTS_NAME, "patient rows")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _projection_path(method: str, *, perplexity: int, n_neighbors: int) -> Path:
    projections = universe_data_dir() / PROJECTIONS_DIRNAME
    if method == "tsne":
        return projections / f"tsne_perplexity_{int(perplexity)}.npy"
    if method == "umap":
        return projections / f"umap_neighbors_{int(n_neighbors)}.npy"
    if method == "clinical":
        return projections / "clinical_pca.npy"
    return projections / "pca.npy"


def available_projections() -> list[str]:
    projections = universe_data_dir() / PROJECTIONS_DIRNAME
    if not projections.is_dir():
        return []
    return sorted(p.stem for p in projections.glob("*.npy"))


def load_projection(method: str, *, perplexity: int, n_neighbors: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load pre-computed (n_patients, 3) coordinates + their sidecar metadata.

    Never computes: a missing variant is a 404, not an hour of t-SNE.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown projection method '{method}'. Supported: {', '.join(SUPPORTED_METHODS)}."
        )
    path = _projection_path(method, perplexity=perplexity, n_neighbors=n_neighbors)
    if not path.exists():
        raise UniverseDataMissing(
            f"Projection '{path.stem}' is not part of the published universe. "
            f"Available: {', '.join(available_projections()) or 'none'}."
        )
    meta_path = path.with_suffix(".json")
    projection_meta: dict[str, Any] = {"method": method, "params": {}}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            projection_meta = json.load(f)
    return np.load(path), projection_meta


def _diagnosis_colors(classes: Iterable[str]) -> dict[str, str]:
    """Class -> hex colour, styled table first, cycling palette for the rest."""
    palette = [
        "#E53935", "#D81B60", "#8E24AA", "#5E35B1", "#3949AB", "#1E88E5",
        "#039BE5", "#00ACC1", "#00897B", "#43A047", "#7CB342", "#C0CA33",
        "#FDD835", "#FFB300", "#FB8C00", "#F4511E", "#6D4C41", "#546E7A",
        "#AB47BC", "#5C6BC0", "#26A69A", "#9CCC65", "#FF7043", "#78909C",
        "#EC407A", "#42A5F5", "#66BB6A", "#FFCA28",
    ]
    out: dict[str, str] = {}
    for i, dx in enumerate(classes):
        style = DIAGNOSIS_STYLE.get(str(dx).lower())
        out[str(dx)] = style[0] if style else palette[i % len(palette)]
    out.setdefault("unknown", "#78909C")
    return out


def get_source_safe_projection(
    method: str,
    *,
    perplexity: int = 100,
    n_neighbors: int = 15,
) -> dict[str, Any]:
    """The whole scene: one point per record plus the model summary block."""
    metadata = load_universe_metadata()
    coords, projection_meta = load_projection(method, perplexity=perplexity, n_neighbors=n_neighbors)

    classes = list(metadata.get("classes", []))
    thresholds = dict(metadata.get("thresholds", {}))
    colors = _diagnosis_colors(classes)
    # The model's own scored classes are accepted verbatim by the canonicaliser:
    # several (bifascicular block, pacing rhythm, low qrs voltages) predate
    # DIAG_CANON and have no entry in it, so a lookup alone would drop them.
    canon_extra = {c.lower(): c for c in classes}

    points: list[dict[str, Any]] = []
    for row in _iter_patient_rows():
        i = int(row["idx"])
        if i >= len(coords):
            continue
        raw_labels = row.get("labels") or []
        # Canonicalise before anything else. Labels arrive from three different
        # sources — PhysioNet #Dx SNOMED, the Apr28 label file, and the 12SL
        # machine reads — with different spellings and different ideas of what
        # counts as a diagnosis. Uncanonicalised, the picker offered 98 entries
        # including 'abnormal ecg', 'borderline ecg' and the bare code
        # '73795002'. This collapses them to the project vocabulary and drops
        # what is not a finding; 'normal ecg' folds to 'sinus rhythm' rather
        # than vanishing, matching normalize_diagnosis().
        labels = canonicalise_all(raw_labels, canon_extra)
        # The stored 'primary' is alphabetical, not clinical — for 47% of
        # records it is simply whichever label sorted first, and uppercase
        # labels jumped the queue on ASCII order. Recompute it by severity so
        # the diagnosis representing a record is the one that matters most.
        primary = primary_diagnosis(labels) if labels else "unknown"
        # Canonicalised with the SAME function as the labels. One of the merges
        # renames a scored class ('lvef <40% expanded MIMIC EF>=55 controls' ->
        # 'lvef ≤45%'), so renaming only one side would silently break every
        # comparison between predictions and gold.
        predictions = canonicalise_all(row.get("predictions") or [], canon_extra)
        # recording_id exists only for the openly redistributable PhysioNet
        # records; every row has the stable public display_id (SS-NNNNN).
        patient_id = row.get("recording_id") or row.get("display_id") or f"idx-{i}"
        points.append(
            {
                "patientId": patient_id,
                "displayId": row.get("display_id"),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "primaryDiagnosis": primary,
                "diagnoses": labels,
                "cohort": row.get("cohort", SOURCE_SAFE_COHORT_NAME),
                "color": colors.get(primary, "#78909C"),
                # One shape for every record. The published universe carries no
                # fold membership — `probabilities.npy` is the mean of all five
                # folds for every row — so there is no held-out set to draw
                # differently. The `cohort` field is the honest way to see where
                # a record came from.
                "symbol": "sphere",
                "xgbCorrect": bool(row.get("xgb_correct", False)),
                "xgbPredictions": predictions,
                "pipelineCorrect": bool(row.get("xgb_correct", False)),
                "pipelinePredictions": predictions,
                "tribes": [],
                "multiTribe": False,
                "age": None,
                "sex": None,
                "modelClassClean": len([dx for dx in labels if dx in thresholds]) == 1,
                "isSupplement": False,
                "sourceCohort": row.get("source", ""),
                "source": row.get("source", ""),
                "topPrediction": row.get("top_prediction"),
                "maxScore": row.get("max_score"),
            }
        )

    hyperparams = projection_meta.get("params", {})
    return {
        "method": method,
        "points": points,
        "totalPatients": len(points),
        "cohortsIncluded": list(metadata.get("cohort_counts", {}).keys()) or [SOURCE_SAFE_COHORT_NAME],
        "hyperparameters": hyperparams,
        "diagnosisColors": colors,
        "cached": True,
        "axisLabels": hyperparams.get("axis_labels", []),
        # No dataDir / packageDir here: the private build emitted the absolute
        # paths of the feature matrices and the model package on every response.
        "sourceSafe": {
            "modelFamily": metadata.get("model_family"),
            "modelVersion": metadata.get("model_version"),
            "modelHeadline": metadata.get("model_headline"),
            "macroPrimaryAuc": metadata.get("macro_primary_auc"),
            "macroFinalAuc": metadata.get("macro_final_auc"),
            "featureCount": metadata.get("feature_count"),
            "nClasses": metadata.get("n_classes"),
            "thresholds": thresholds,
            "diagnoses": metadata.get("diagnoses", []),
            "sourceCounts": metadata.get("source_counts", {}),
            "cohortCounts": metadata.get("cohort_counts", {}),
            "licenseNote": metadata.get("license_note"),
        },
    }


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

def _downsample_roc(fpr: np.ndarray, tpr: np.ndarray, max_points: int) -> list[dict[str, float]]:
    """Thin a full staircase ROC to ~max_points for the wire, keeping the
    endpoints. Uniform stride preserves the monotone curve's shape; the AUROC is
    computed on the FULL arrays by the caller, so this affects only the drawn
    polyline, not the reported area."""
    n = int(fpr.shape[0])
    if n <= max_points:
        idx: Iterable[int] = range(n)
    else:
        step = max(1, n // max_points)
        keep = set(range(0, n, step))
        keep.add(0)
        keep.add(n - 1)
        idx = sorted(keep)
    return [{"fpr": round(float(fpr[i]), 5), "tpr": round(float(tpr[i]), 5)} for i in idx]


def get_source_safe_roc(diagnosis: str, *, max_points: int = 400) -> dict[str, Any]:
    """ROC for one diagnosis computed ON THE DISPLAYED UNIVERSE.

    Positives = records whose gold labels include `diagnosis`; scores = the
    model's per-class probability from the published probability matrix. The
    returned `auroc` is the area under THIS curve over THIS population — it
    deliberately differs from the package's held-out `final_full_auc` /
    `primary_auc`, both returned alongside so a caller can label them distinctly
    and never conflate the two numbers.
    """
    metadata = load_universe_metadata()
    classes = list(metadata.get("classes", []))
    if diagnosis not in classes:
        raise KeyError(diagnosis)
    dx_idx = classes.index(diagnosis)
    thr = float(dict(metadata.get("thresholds", {})).get(diagnosis, 0.5))
    held = next(
        (d for d in metadata.get("diagnoses", []) if d.get("diagnosis") == diagnosis),
        {},
    )

    prob_path = _require(universe_data_dir() / PROBABILITIES_NAME, "probability matrix")
    probs = np.load(prob_path, mmap_mode="r")
    rows = list(_iter_patient_rows())
    n = len(rows)
    y_true = np.zeros(n, dtype=np.int64)
    y_score = np.zeros(n, dtype=np.float64)
    for k, row in enumerate(rows):
        i = int(row["idx"])
        if i >= probs.shape[0]:
            continue
        y_true[k] = 1 if diagnosis in (row.get("labels") or []) else 0
        y_score[k] = float(probs[i, dx_idx])

    total_pos = int(y_true.sum())
    total_neg = int(n - total_pos)

    base = {
        "diagnosis": diagnosis,
        "threshold": thr,
        "nPositive": total_pos,
        "nNegative": total_neg,
        "heldOutFinalAuroc": held.get("final_full_auc"),
        "heldOutPrimaryAuroc": held.get("primary_auc"),
    }

    if total_pos == 0 or total_neg == 0:
        # No ROC is defined without both classes present in the universe.
        return {
            **base,
            "points": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}],
            "auroc": 0.5,
            "operatingFpr": None,
            "operatingTpr": None,
        }

    order = np.argsort(-y_score, kind="mergesort")  # stable: deterministic ties
    labels_sorted = y_true[order]
    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1 - labels_sorted)
    tpr = tp_cum / total_pos
    fpr = fp_cum / total_neg
    fpr_full = np.concatenate(([0.0], fpr))  # the staircase ends at (1, 1)
    tpr_full = np.concatenate(([0.0], tpr))
    # Trapezoidal area under the staircase (np.trapz removed in NumPy 2.0).
    auroc = float(np.sum(np.diff(fpr_full) * (tpr_full[1:] + tpr_full[:-1]) / 2.0))

    pred_pos = y_score >= thr
    op_tp = int(np.sum(pred_pos & (y_true == 1)))
    op_fp = int(np.sum(pred_pos & (y_true == 0)))

    return {
        **base,
        "points": _downsample_roc(fpr_full, tpr_full, max_points),
        "auroc": round(auroc, 4),
        "operatingFpr": round(op_fp / total_neg, 5),
        "operatingTpr": round(op_tp / total_pos, 5),
    }
