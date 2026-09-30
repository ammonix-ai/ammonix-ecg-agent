"""Place a newly scored recording into the fixed 63,256-point universe.

THE EMBEDDING IS NEVER REFIT. The shipped coordinates are the universe; a new
recording is placed *into* them, so every user sees the same scene with one
extra point.

What is actually available, per method:

* **PCA** — a projection, therefore linear, therefore invertible from what
  ships. No fitted `PCA` object was persisted with the universe, so the map is
  re-derived once at load time by least squares from the shipped
  `probabilities.npy` (63,256 x 32) onto `projections/pca.npy` (63,256 x 3).
  The fit's residual is measured and reported in the payload
  (`pcaResidualRms`); on the shipped data it is at float32 noise level, which
  is what a linear map recovering a linear map looks like. This is a
  re-derivation of the stored basis, not a refit of the embedding: the 63,256
  points keep the coordinates they shipped with.
* **UMAP** — a fitted UMAP model has an out-of-sample `transform()`, but no
  fitted model was persisted with this universe, only its output coordinates.
  So UMAP falls back to the same k-NN placement as t-SNE and says so
  (`umapPlacement: "knn_approx"`). Ship `projections/umap_model.joblib` and
  this module will use its `transform()` instead.
* **t-SNE** — has no out-of-sample operator at all, by construction. The point
  is placed at the k-NN centroid in probability space
  (`tsnePlacement: "knn_approx"`).

k-NN placement: k = 6 nearest universe rows in 32-dimensional probability
space (Euclidean), coordinates averaged with 1/distance weights. The same
neighbours answer the contract's `neighbors` field, so a viewer can see which
records the placement was derived from.

Environment:
    UNIVERSE_DATA_DIR   universe payload (default: <repo>/_staging/universe)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from services.source_safe_universe import (
    load_projection,
    load_universe_metadata,
    universe_data_dir,
)

logger = logging.getLogger(__name__)

PROBABILITIES_NAME = "probabilities.npy"
PATIENTS_NAME = "patients.jsonl"
UMAP_MODEL_NAMES = ("umap_model.joblib", "umap_model.pkl")

#: Neighbour count for the k-NN placement and the `neighbors` payload.
DEFAULT_K = 6
#: t-SNE variant the contract pins server-side.
TSNE_PERPLEXITY = 100
#: UMAP variant the contract pins server-side.
UMAP_N_NEIGHBORS = 15

KNN_APPROX = "knn_approx"
EXACT = "transform"
LINEAR_REDERIVED = "linear_least_squares"


class UniverseDataMissing(FileNotFoundError):
    """The universe payload needed for placement is absent."""


@dataclass(frozen=True)
class PlacementModel:
    """The fixed universe, plus the operators that place a point into it."""

    classes: tuple[str, ...]
    probabilities: np.ndarray            # (n, 32) float32, the shipped matrix
    coords: dict[str, np.ndarray]        # method -> (n, 3)
    display_ids: tuple[str, ...]
    primaries: tuple[str, ...]
    pca_map: np.ndarray | None           # (33, 3) affine map, probabilities -> pca
    pca_residual_rms: float | None
    umap_transformer: Any | None

    @property
    def n_points(self) -> int:
        return int(self.probabilities.shape[0])


_MODEL: PlacementModel | None = None
_MODEL_LOCK = threading.Lock()


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise UniverseDataMissing(
            f"Universe {what} not found ({path.name} missing). Download the published "
            "universe payload, or point UNIVERSE_DATA_DIR at the directory holding it."
        )
    return path


def load_placement_model(*, force: bool = False) -> PlacementModel:
    """Load the universe matrices and derive the placement operators once."""
    global _MODEL
    if _MODEL is not None and not force:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and not force:
            return _MODEL
        _MODEL = _load_placement_model_uncached()
    return _MODEL


def _load_placement_model_uncached() -> PlacementModel:
    root = universe_data_dir()
    metadata = load_universe_metadata()
    classes = tuple(metadata.get("classes", []))
    if not classes:
        raise UniverseDataMissing("Universe metadata lists no classes.")

    probabilities = np.load(_require(root / PROBABILITIES_NAME, "probability matrix"))
    if probabilities.shape[1] != len(classes):
        raise UniverseDataMissing(
            f"Probability matrix has {probabilities.shape[1]} columns but metadata "
            f"lists {len(classes)} classes."
        )

    coords: dict[str, np.ndarray] = {}
    for method, kwargs in (
        ("pca", {}),
        ("umap", {"n_neighbors": UMAP_N_NEIGHBORS}),
        ("tsne", {"perplexity": TSNE_PERPLEXITY}),
    ):
        try:
            xyz, _meta = load_projection(
                method,
                perplexity=int(kwargs.get("perplexity", TSNE_PERPLEXITY)),
                n_neighbors=int(kwargs.get("n_neighbors", UMAP_N_NEIGHBORS)),
            )
        except FileNotFoundError as exc:
            logger.warning("Projection '%s' unavailable: %s", method, exc)
            continue
        coords[method] = np.asarray(xyz, dtype=np.float64)

    display_ids: list[str] = []
    primaries: list[str] = []
    with open(_require(root / PATIENTS_NAME, "patient rows"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            display_ids.append(str(row.get("display_id") or f"idx-{row.get('idx')}"))
            primaries.append(str(row.get("primary") or "unknown"))

    pca_map, pca_residual = _derive_pca_map(probabilities, coords.get("pca"))
    umap_transformer = _load_umap_transformer(root)

    model = PlacementModel(
        classes=classes,
        probabilities=probabilities,
        coords=coords,
        display_ids=tuple(display_ids),
        primaries=tuple(primaries),
        pca_map=pca_map,
        pca_residual_rms=pca_residual,
        umap_transformer=umap_transformer,
    )
    logger.info(
        "Placement model ready: %d points, projections=%s, pca residual rms=%s, umap transform=%s",
        model.n_points, sorted(coords), pca_residual, umap_transformer is not None,
    )
    return model


def _derive_pca_map(
    probabilities: np.ndarray,
    pca_coords: np.ndarray | None,
) -> tuple[np.ndarray | None, float | None]:
    """Least-squares affine map from probability space to the stored PCA coords.

    PCA is `(p - mean) @ components.T`: affine in p. Fitting an affine map to
    63,256 (probability, coordinate) pairs therefore recovers the stored basis
    rather than approximating it, and the residual says so out loud.
    """
    if pca_coords is None:
        return None, None
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(pca_coords, dtype=np.float64)
    design = np.hstack([p, np.ones((p.shape[0], 1))])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = float(np.sqrt(np.mean((design @ solution - y) ** 2)))
    scale = float(np.sqrt(np.mean(y ** 2))) or 1.0
    if residual / scale > 1e-3:
        logger.warning(
            "PCA re-derivation residual is %.3g (%.2f%% of coordinate scale) — the "
            "stored PCA may not be a plain linear map of the probability matrix.",
            residual, 100.0 * residual / scale,
        )
    return solution, residual


def _load_umap_transformer(root: Path) -> Any | None:
    """Use a persisted fitted UMAP model if one ships; otherwise None."""
    for name in UMAP_MODEL_NAMES:
        path = root / "projections" / name
        if not path.exists():
            continue
        try:
            import joblib  # noqa: PLC0415 - optional dependency

            transformer = joblib.load(path)
        except Exception as exc:  # pragma: no cover - depends on shipped artefact
            logger.warning("Fitted UMAP model at %s could not be loaded: %s", name, exc)
            return None
        if not hasattr(transformer, "transform"):
            logger.warning("Artefact %s has no transform(); ignoring.", name)
            return None
        return transformer
    return None


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------

def _probability_vector(
    probabilities: Mapping[str, float] | Sequence[float],
    classes: Sequence[str],
) -> np.ndarray:
    if isinstance(probabilities, Mapping):
        missing = [dx for dx in classes if dx not in probabilities]
        if missing:
            logger.warning(
                "Probabilities missing %d of %d universe classes (e.g. %s); scored as 0.",
                len(missing), len(classes), missing[0],
            )
        return np.array([float(probabilities.get(dx, 0.0)) for dx in classes], dtype=np.float64)
    vector = np.asarray(probabilities, dtype=np.float64).ravel()
    if vector.size != len(classes):
        raise ValueError(
            f"Expected {len(classes)} probabilities in universe class order, got {vector.size}."
        )
    return vector


def _knn(model: PlacementModel, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the k nearest universe rows in probability space."""
    diff = model.probabilities.astype(np.float64, copy=False) - vector[None, :]
    distances = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    k = max(1, min(int(k), distances.shape[0]))
    idx = np.argpartition(distances, k - 1)[:k]
    idx = idx[np.argsort(distances[idx])]
    return idx, distances[idx]


def _weighted_centroid(coords: np.ndarray, idx: np.ndarray, distances: np.ndarray) -> list[float]:
    weights = 1.0 / (distances + 1e-9)
    weights /= weights.sum()
    return [float(v) for v in (coords[idx] * weights[:, None]).sum(axis=0)]


def nearest_neighbors(
    probabilities: Mapping[str, float] | Sequence[float],
    *,
    k: int = DEFAULT_K,
    model: PlacementModel | None = None,
) -> list[dict[str, Any]]:
    """The k nearest universe records in probability space, nearest first."""
    m = model or load_placement_model()
    vector = _probability_vector(probabilities, m.classes)
    idx, distances = _knn(m, vector, k)
    return [
        {
            "displayId": m.display_ids[i] if i < len(m.display_ids) else f"idx-{int(i)}",
            "distance": float(d),
            "primary": m.primaries[i] if i < len(m.primaries) else "unknown",
        }
        for i, d in zip(idx.tolist(), distances.tolist())
    ]


def place_point(
    probabilities: Mapping[str, float] | Sequence[float],
    *,
    k: int = DEFAULT_K,
    model: PlacementModel | None = None,
) -> dict[str, Any]:
    """Place one scored recording into the fixed universe.

    Returns the three coordinate triples plus, for each method, how the point
    got there. Nothing in here modifies the universe.
    """
    m = model or load_placement_model()
    vector = _probability_vector(probabilities, m.classes)
    idx, distances = _knn(m, vector, k)

    out: dict[str, Any] = {
        "pca": None,
        "umap": None,
        "tsne": None,
        "k": int(len(idx)),
        "neighborSpace": "probability",
        "universePoints": m.n_points,
    }

    # PCA — exact under the re-derived linear map.
    if m.pca_map is not None:
        design = np.concatenate([vector, [1.0]])
        out["pca"] = [float(v) for v in design @ m.pca_map]
        out["pcaPlacement"] = LINEAR_REDERIVED
        out["pcaResidualRms"] = m.pca_residual_rms
    elif "pca" in m.coords:
        out["pca"] = _weighted_centroid(m.coords["pca"], idx, distances)
        out["pcaPlacement"] = KNN_APPROX

    # UMAP — fitted transform() when one ships, k-NN otherwise.
    if m.umap_transformer is not None:
        transformed = np.asarray(m.umap_transformer.transform(vector[None, :]), dtype=np.float64)
        out["umap"] = [float(v) for v in transformed.ravel()[:3]]
        out["umapPlacement"] = EXACT
    elif "umap" in m.coords:
        out["umap"] = _weighted_centroid(m.coords["umap"], idx, distances)
        out["umapPlacement"] = KNN_APPROX
        out["umapNote"] = (
            "No fitted UMAP model ships with the universe, only its coordinates, "
            "so this point is placed by the same k-NN rule as t-SNE."
        )

    # t-SNE — k-NN centroid, always. No out-of-sample operator exists.
    if "tsne" in m.coords:
        out["tsne"] = _weighted_centroid(m.coords["tsne"], idx, distances)
    out["tsnePlacement"] = KNN_APPROX

    out["neighbors"] = [
        {
            "displayId": m.display_ids[i] if i < len(m.display_ids) else f"idx-{int(i)}",
            "distance": float(d),
            "primary": m.primaries[i] if i < len(m.primaries) else "unknown",
        }
        for i, d in zip(idx.tolist(), distances.tolist())
    ]
    return out
