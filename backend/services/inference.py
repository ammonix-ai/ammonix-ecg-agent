"""Frozen inference runtime: raw 12-lead waveform -> QPSI features -> probabilities.

    raw 12-lead (12, n) + fs
        -> qpsi.run_pipeline(compute_direct_measurements=True)      extract_features()
        -> feature dict keyed by the package's feature_names.json
        -> 32 diagnoses x 5 XGBoost folds, mean over folds          classify()
        -> {"probabilities": {dx: p}, "scoreKind": "ensemble_5fold"}

THE CLASSIFIER IS FROZEN. This module loads and evaluates; it never fits.
There is no `.fit()` here, no threshold search, no calibration step — the
thresholds come from the package's `manifest.json` exactly as exported.

**Why the five folds are averaged.** Each diagnosis was trained as a 5-fold
subject-grouped ensemble. For a recording the package has never seen, all five
folds are out-of-sample, so their mean is the honest score, reported as
`scoreKind: "ensemble_5fold"`.

**What the universe's stored score actually is.** The published universe was
scored with this same rule — mean over all five folds — not with the
single held-out fold. Verified numerically: on records from all five sources
the live pipeline reproduces `probabilities.npy` to within float32 rounding
(max |delta| 2.4e-8). So for a record that ships in the universe, `stored` and
`live` are the same quantity and will agree to eight decimals.

That agreement is a reproducibility check, not an out-of-sample result. Most
shipped PhysioNet records were in the training pool for at least some
diagnoses, so four of the five folds had seen them; their per-fold spread shows
it (e.g. HR16370 / sinus irregularity: 0.16, 0.79, 0.81, 0.80, 0.83 — one fold
held it out, four did not). `foldProbabilities` is returned so a caller can see
that spread rather than infer it. A genuine single-fold out-of-sample score
cannot be served from what ships: the package's per-diagnosis
`validation_scores.npy` holds it, but the record ordering that indexes it lives
in the private training corpus.

Feature order, feature count, thresholds and the class list are all read from
the package on disk. Nothing about the model's shape is hardcoded here.

Environment:
    MODEL_PACKAGE_DIR   frozen classifier package
                        (default: <repo>/models/source_safe_canonical_v1)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
_DEFAULT_PACKAGE_SUBPATH = ("models", "source_safe_canonical_v1")


def _ensure_qpsi_importable() -> None:
    """Put the vendored tokenizer on sys.path when it is not installed.

    `uvicorn main:app` runs with the backend directory on sys.path, not the repo
    root, so `import qpsi` would fail even though the package sits one level up.
    A pip-installed qpsi wins: this only adds the repo root as a fallback.
    """
    if importlib.util.find_spec("qpsi") is not None:
        return
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

MANIFEST_NAME = "manifest.json"
FEATURE_NAMES_NAME = "feature_names.json"

#: Value of `scoreKind` for everything this module produces.
ENSEMBLE_SCORE_KIND = "ensemble_5fold"
#: What the universe's stored per-record score really is. The API contract
#: labels it `oof_single_fold`; the shipped `probabilities.npy` was in fact
#: built by averaging the same five folds (see the module docstring), so it is
#: the same quantity this module computes. Kept as its own constant so callers
#: label the stored number deliberately instead of assuming.
STORED_SCORE_KIND = "ensemble_5fold"
#: The label the API contract currently specifies for the stored score. It does
#: not match how the universe was built; surface `STORED_SCORE_KIND` instead
#: unless the contract is amended.
CONTRACT_STORED_SCORE_KIND = "oof_single_fold"


class ModelPackageMissing(FileNotFoundError):
    """The frozen classifier package is absent or incomplete."""


class FeatureExtractionError(RuntimeError):
    """QPSI could not turn this waveform into a feature vector."""


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldModel:
    """One fold of one diagnosis: 500 selected columns + its booster."""

    fold: int
    feature_indices: np.ndarray          # int32 (k,) indices into feat_order
    booster: Any                         # xgboost.Booster


@dataclass(frozen=True)
class DiagnosisModel:
    """The five folds of one diagnosis, plus its frozen operating point."""

    diagnosis: str
    safe_name: str
    threshold: float
    threshold_kind: str
    folds: tuple[FoldModel, ...]
    primary_auc: float | None = None
    final_full_auc: float | None = None

    def fold_scores(self, x: np.ndarray) -> np.ndarray:
        """Positive-class probability from each fold for one feature row."""
        out = np.empty(len(self.folds), dtype=np.float64)
        for i, fold in enumerate(self.folds):
            sub = np.ascontiguousarray(x[:, fold.feature_indices])
            out[i] = float(np.asarray(fold.booster.inplace_predict(sub)).ravel()[0])
        return out


@dataclass(frozen=True)
class Runtime:
    """Everything needed to score a recording, loaded once."""

    package_dir: Path
    version: str
    model_family: str
    created_at: str | None
    feature_names: tuple[str, ...]
    feature_index: Mapping[str, int]
    models: tuple[DiagnosisModel, ...]
    default_threshold_kind: str
    load_seconds: float = 0.0
    _by_name: dict[str, DiagnosisModel] = field(default_factory=dict, repr=False)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(m.diagnosis for m in self.models)

    @property
    def thresholds(self) -> dict[str, float]:
        return {m.diagnosis: m.threshold for m in self.models}

    def model(self, diagnosis: str) -> DiagnosisModel:
        return self._by_name[diagnosis]


_RUNTIME: Runtime | None = None
_RUNTIME_LOCK = threading.Lock()


def model_package_dir() -> Path:
    override = os.environ.get("MODEL_PACKAGE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _REPO_ROOT.joinpath(*_DEFAULT_PACKAGE_SUBPATH)


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise ModelPackageMissing(
            f"Frozen classifier {what} not found ({path.name} missing). The 167 MB "
            "package is distributed separately from the source; fetch it and point "
            "MODEL_PACKAGE_DIR at it."
        )
    return path


def load_runtime(*, force: bool = False) -> Runtime:
    """Load feat_order + the 5 folds x 32 diagnoses boosters. Cached process-wide."""
    global _RUNTIME
    if _RUNTIME is not None and not force:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None and not force:
            return _RUNTIME
        _RUNTIME = _load_runtime_uncached()
    return _RUNTIME


def runtime_loaded() -> bool:
    """True once the frozen package is resident. Never triggers a load itself —
    for health checks that must not cost 8 seconds and 460 MB."""
    return _RUNTIME is not None


def _load_runtime_uncached() -> Runtime:
    import xgboost as xgb  # imported here so the module imports without xgboost

    started = time.perf_counter()
    root = model_package_dir()
    manifest_path = _require(root / MANIFEST_NAME, "manifest")
    names_path = _require(root / FEATURE_NAMES_NAME, "feature name list")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(names_path, encoding="utf-8") as f:
        feature_names = tuple(json.load(f))

    # The manifest states the width it was exported at; trust feature_names.json
    # as the operative order and refuse to run if the two disagree.
    declared = manifest.get("feature_count")
    if declared is not None and int(declared) != len(feature_names):
        raise ModelPackageMissing(
            f"Package is inconsistent: manifest feature_count={declared} but "
            f"{FEATURE_NAMES_NAME} lists {len(feature_names)} names."
        )

    default_kind = str(manifest.get("default_threshold_kind", "primary_f1"))
    models: list[DiagnosisModel] = []
    for entry in manifest.get("diagnoses", []):
        if entry.get("status") != "accepted":
            continue
        folds: list[FoldModel] = []
        for fold_entry in entry.get("folds", []):
            model_path = _require(root / fold_entry["model_path"], "fold model")
            index_path = _require(root / fold_entry["feature_index_path"], "fold feature index")
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            indices = np.load(index_path).astype(np.int64, copy=False)
            if indices.size and int(indices.max()) >= len(feature_names):
                raise ModelPackageMissing(
                    f"Fold {fold_entry.get('fold')} of '{entry.get('diagnosis')}' selects "
                    f"column {int(indices.max())}, beyond the {len(feature_names)}-wide "
                    "feature order."
                )
            folds.append(
                FoldModel(
                    fold=int(fold_entry.get("fold", len(folds))),
                    feature_indices=indices,
                    booster=booster,
                )
            )
        if not folds:
            logger.warning("Diagnosis '%s' has no persisted folds; skipped.", entry.get("diagnosis"))
            continue
        models.append(
            DiagnosisModel(
                diagnosis=str(entry["diagnosis"]),
                safe_name=str(entry.get("safe_name", "")),
                threshold=float(entry.get("default_threshold", 0.5)),
                threshold_kind=str(entry.get("default_threshold_kind", default_kind)),
                folds=tuple(sorted(folds, key=lambda f: f.fold)),
                primary_auc=entry.get("primary_auc"),
                final_full_auc=entry.get("final_full_auc"),
            )
        )

    if not models:
        raise ModelPackageMissing("Package contains no accepted diagnoses.")

    elapsed = time.perf_counter() - started
    runtime = Runtime(
        package_dir=root,
        version=str(manifest.get("version", "unknown")),
        model_family=str(manifest.get("model_family", "unknown")),
        created_at=manifest.get("created_at"),
        feature_names=feature_names,
        feature_index={name: i for i, name in enumerate(feature_names)},
        models=tuple(models),
        default_threshold_kind=default_kind,
        load_seconds=elapsed,
        _by_name={m.diagnosis: m for m in models},
    )
    logger.info(
        "Loaded frozen package %s: %d diagnoses x %d folds, %d features, %.1fs",
        runtime.version, len(models), len(models[0].folds), runtime.feature_count, elapsed,
    )
    return runtime


def runtime_summary() -> dict[str, Any]:
    """Small dict for /api/status — no model objects, no absolute paths."""
    rt = load_runtime()
    return {
        "modelFamily": rt.model_family,
        "modelVersion": rt.version,
        "createdAt": rt.created_at,
        "featureCount": rt.feature_count,
        "nClasses": len(rt.models),
        "nFolds": len(rt.models[0].folds),
        "scoreKind": ENSEMBLE_SCORE_KIND,
        "thresholdKind": rt.default_threshold_kind,
        "frozen": True,
    }


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def extract_features(
    signal: np.ndarray,
    fs: int,
    *,
    recording_id: str = "(upload)",
) -> dict[str, float]:
    """Run the QPSI tokenizer over one recording and return its feature dict.

    `signal` is (12, n_samples) in millivolts, canonical lead order
    (I, II, III, aVR, aVL, aVF, V1-V6) — see services.signals.

    The returned keys are the direct-measurement track only
    (`ml_direct::*` + `plane_direct::*`), which is exactly what the frozen
    classifier consumes. The Gaussian track still runs (the pipeline is one
    unit) but its output is not part of the model's feature space.

    The Rust kernels (`qpsi_native`) are an optional accelerator: with them a
    record takes ~1.3 s, without them ~21 s, and the direct-measurement values
    are identical either way.
    """
    _ensure_qpsi_importable()
    from qpsi.config import QPSIConfig
    from qpsi.pipeline import run_pipeline

    arr = np.ascontiguousarray(np.asarray(signal, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[0] != 12:
        raise FeatureExtractionError(
            f"Expected a (12, n_samples) array of millivolts, got {arr.shape}."
        )

    # Same call the universe was extracted with: direct-measurement track ON,
    # plots off, everything else at pipeline defaults.
    result = run_pipeline(
        arr,
        fs=int(fs),
        recording_id=str(recording_id),
        plot_enabled=False,
        qpsi_config=QPSIConfig(compute_direct_measurements=True),
    )
    if not result:
        raise FeatureExtractionError(
            f"QPSI found no usable beats in '{recording_id}'."
        )

    final_json = result.get("_final_json") or {}
    features: dict[str, float] = {}
    for key in ("direct_features", "plane_direct_features"):
        block = final_json.get(key) or {}
        if not block:
            raise FeatureExtractionError(
                f"QPSI produced no '{key}' for '{recording_id}'. The "
                "direct-measurement track must be enabled for classification."
            )
        features.update(_numeric(block))
    return features


def _numeric(block: Mapping[str, Any]) -> dict[str, float]:
    """Coerce a QPSI feature block to floats, NaN for anything non-numeric.

    Matches how the training matrices were built: booleans became 1.0/0.0 and
    non-numeric cells stayed NaN, which XGBoost treats as missing.
    """
    out: dict[str, float] = {}
    for name, value in block.items():
        if isinstance(value, bool):
            out[name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, complex):
            out[name] = float(value)
        else:
            out[name] = math.nan
    return out


def feature_vector(features: Mapping[str, float], runtime: Runtime | None = None) -> np.ndarray:
    """Order a feature dict into the package's feat_order as (1, n_features).

    Names the package wants but the extraction did not produce become NaN
    (XGBoost's missing value) rather than 0.0, so a gap never masquerades as a
    measurement of zero.
    """
    rt = runtime or load_runtime()
    x = np.full((1, rt.feature_count), np.nan, dtype=np.float32)
    hits = 0
    for name, value in features.items():
        i = rt.feature_index.get(name)
        if i is None:
            continue
        x[0, i] = value
        hits += 1
    if hits < rt.feature_count:
        logger.warning(
            "Feature vector is short: %d of %d names present; the rest scored as missing.",
            hits, rt.feature_count,
        )
    return x


def missing_feature_names(features: Mapping[str, float], runtime: Runtime | None = None) -> list[str]:
    """Names in feat_order that the extraction did not supply."""
    rt = runtime or load_runtime()
    return [name for name in rt.feature_names if name not in features]


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def classify(features: Mapping[str, float], runtime: Runtime | None = None) -> dict[str, Any]:
    """Score one feature dict against all 32 frozen diagnoses.

    Returns probabilities (mean of the 5 folds), the thresholded prediction
    list, and the per-fold spread so a caller can show how much the swarm
    disagreed.
    """
    rt = runtime or load_runtime()
    x = feature_vector(features, rt)

    probabilities: dict[str, float] = {}
    fold_probabilities: dict[str, list[float]] = {}
    for model in rt.models:
        scores = model.fold_scores(x)
        probabilities[model.diagnosis] = float(scores.mean())
        fold_probabilities[model.diagnosis] = [float(s) for s in scores]

    thresholds = rt.thresholds
    predictions = [dx for dx, p in probabilities.items() if p >= thresholds[dx]]
    predictions.sort(key=lambda dx: -probabilities[dx])
    top = max(probabilities.items(), key=lambda kv: kv[1])

    return {
        "probabilities": probabilities,
        "scoreKind": ENSEMBLE_SCORE_KIND,
        "predictions": predictions,
        "thresholds": thresholds,
        "thresholdKind": rt.default_threshold_kind,
        "topPrediction": top[0],
        "maxScore": top[1],
        "foldProbabilities": fold_probabilities,
        "featureCount": rt.feature_count,
        "modelVersion": rt.version,
    }


def top_feature_contributions(
    features: Mapping[str, float],
    diagnosis: str,
    *,
    k: int = 10,
    runtime: Runtime | None = None,
) -> list[dict[str, Any]]:
    """Which features drove one diagnosis's score, most influential first.

    Per-fold SHAP contributions (XGBoost `pred_contribs`) averaged over the five
    folds, in log-odds units. Reading only, no refit. `contribution` is signed:
    positive pushed the score up.
    """
    rt = runtime or load_runtime()
    model = rt.model(diagnosis)
    x = feature_vector(features, rt)

    import xgboost as xgb

    totals = np.zeros(rt.feature_count, dtype=np.float64)
    for fold in model.folds:
        sub = np.ascontiguousarray(x[:, fold.feature_indices])
        contribs = fold.booster.predict(xgb.DMatrix(sub), pred_contribs=True)
        # Last column is the bias term; drop it.
        totals[fold.feature_indices] += np.asarray(contribs)[0, :-1]
    totals /= len(model.folds)

    order = np.argsort(-np.abs(totals))[:k]
    out: list[dict[str, Any]] = []
    for i in order:
        if totals[i] == 0.0:
            continue
        name = rt.feature_names[i]
        value = features.get(name, math.nan)
        out.append(
            {
                "name": name,
                "value": None if (isinstance(value, float) and math.isnan(value)) else float(value),
                "contribution": float(totals[i]),
            }
        )
    return out


def rank_probabilities(probabilities: Mapping[str, float], k: int | None = None) -> list[tuple[str, float]]:
    """Probabilities as a descending (diagnosis, p) list; `k` truncates."""
    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
    return ranked[:k] if k else ranked


def probability_array(
    probabilities: Mapping[str, float],
    classes: Sequence[str],
) -> np.ndarray:
    """Lay a probability dict out along an explicit class order.

    The universe's `probabilities.npy` columns follow `metadata.json["classes"]`,
    which is not the package's manifest order — anything comparing the two must
    pass the universe's order explicitly.
    """
    return np.array([float(probabilities.get(dx, 0.0)) for dx in classes], dtype=np.float64)
