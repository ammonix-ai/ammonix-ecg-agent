"""
Template distance features: per-patient distance to each diagnosis template.

For each patient, computes four distance metrics to each diagnosis template:
    1. Euclidean distance (L2 norm of difference)
    2. Cosine similarity (dot product of unit vectors)
    3. Max z-score (largest deviation in units of template std)
    4. Mahalanobis distance (diagonal approximation using template std)

With 33 diagnoses x 4 metrics = 132 new additive features for XGBoost.

These features are ADDITIVE -- they do not replace or modify the existing
566 features.  They are designed to be appended to the feature vector
before XGBoost training/inference.

Usage:
    from ammonix.templates import TemplateSet
    from qpsi.features.template_distances import extract_template_distance_features

    template_set = TemplateSet.load("layer-2/waveform_templates.json")
    patient_features = extract_all_features(patient)
    patient_params = extract_gaussian_params(patient_features)
    distance_features = extract_template_distance_features(patient_params, template_set)
    # distance_features: {"afib_template_euclidean": 12.3, ...}

.. note::
   **WaveMedix-canonical (V6 extension).** No counterpart in the Feb14
   or Apr28 notebooks. Per K3 (Phase 2 lock, 2026-05-18), this module
   stays WaveMedix-only — one-way distillation into the platform; no
   ship-back to the notebook. See ``qpsi/FIDELITY_REPORT.md`` section
   "WaveMedix-canonical extensions (V5/V6)".
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Avoid division by zero when computing cosine similarity or z-scores.
_EPS = 1e-12

# Maximum z-score cap to prevent extreme outliers from dominating.
_ZSCORE_CAP = 20.0


def _safe_slug(diagnosis: str) -> str:
    """Convert a diagnosis name to a safe feature key slug.

    Replaces spaces with underscores, strips special characters,
    and lowercases.  E.g. "left bundle branch block / variations"
    -> "left_bundle_branch_block_variations".

    Args:
        diagnosis: Canonical diagnosis name.

    Returns:
        Slug suitable for use as a feature key component.
    """
    slug = diagnosis.lower().strip()
    # Replace common separators
    for ch in "/\\-()":
        slug = slug.replace(ch, " ")
    # Collapse whitespace to single underscore
    parts = slug.split()
    slug = "_".join(p for p in parts if p)
    # Strip any remaining non-alphanumeric/underscore chars
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    return slug


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    Handles NaN by treating NaN positions as zero difference.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        L2 norm of (a - b).
    """
    diff = np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0)
    return float(np.sqrt(np.sum(diff ** 2)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is all-zero (undefined case).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if undefined.
    """
    a_clean = np.nan_to_num(a, nan=0.0)
    b_clean = np.nan_to_num(b, nan=0.0)

    norm_a = np.linalg.norm(a_clean)
    norm_b = np.linalg.norm(b_clean)

    if norm_a < _EPS or norm_b < _EPS:
        return 0.0

    return float(np.dot(a_clean, b_clean) / (norm_a * norm_b))


def max_zscore(
    patient_params: np.ndarray,
    template_mean: np.ndarray,
    template_std: np.ndarray,
) -> float:
    """Compute the maximum absolute z-score of patient params vs template.

    For each dimension, computes |patient - mean| / std, then returns
    the maximum across all dimensions.  Dimensions with zero std (constant
    across training patients) are skipped.

    Args:
        patient_params: Patient's Gaussian parameter vector.
        template_mean: Template mean vector.
        template_std: Template std vector.

    Returns:
        Maximum absolute z-score, capped at :data:`_ZSCORE_CAP`.
        Returns 0.0 if no dimensions have non-zero std.
    """
    patient_clean = np.nan_to_num(patient_params, nan=0.0)
    mean_clean = np.nan_to_num(template_mean, nan=0.0)
    std_clean = np.nan_to_num(template_std, nan=0.0)

    # Mask: dimensions where std > 0 (meaningful variation)
    valid = std_clean > _EPS
    if not np.any(valid):
        return 0.0

    zscores = np.abs(patient_clean[valid] - mean_clean[valid]) / std_clean[valid]
    max_z = float(np.max(zscores))
    return min(max_z, _ZSCORE_CAP)


def mean_zscore(
    patient_params: np.ndarray,
    template_mean: np.ndarray,
    template_std: np.ndarray,
) -> float:
    """Compute the mean absolute z-score of patient params vs template.

    Like :func:`max_zscore` but returns the average instead of the maximum.
    More robust to individual outlier dimensions.

    Args:
        patient_params: Patient's Gaussian parameter vector.
        template_mean: Template mean vector.
        template_std: Template std vector.

    Returns:
        Mean absolute z-score across valid dimensions.
        Returns 0.0 if no dimensions have non-zero std.
    """
    patient_clean = np.nan_to_num(patient_params, nan=0.0)
    mean_clean = np.nan_to_num(template_mean, nan=0.0)
    std_clean = np.nan_to_num(template_std, nan=0.0)

    valid = std_clean > _EPS
    if not np.any(valid):
        return 0.0

    zscores = np.abs(patient_clean[valid] - mean_clean[valid]) / std_clean[valid]
    mean_z = float(np.mean(zscores))
    return min(mean_z, _ZSCORE_CAP)


def _count_above_threshold(
    patient_params: np.ndarray,
    template_mean: np.ndarray,
    template_std: np.ndarray,
    threshold: float = 2.0,
) -> float:
    """Count the number of features with |z-score| > threshold.

    Returns a float (not int) for consistency with other metrics
    and XGBoost feature compatibility.

    Args:
        patient_params: Patient's Gaussian parameter vector.
        template_mean: Template mean vector.
        template_std: Template std vector.
        threshold: Z-score threshold (default 2.0 = 2 sigma).

    Returns:
        Count of dimensions where |z| > threshold.
        Returns 0.0 if no dimensions have non-zero std.
    """
    patient_clean = np.nan_to_num(patient_params, nan=0.0)
    mean_clean = np.nan_to_num(template_mean, nan=0.0)
    std_clean = np.nan_to_num(template_std, nan=0.0)

    valid = std_clean > _EPS
    if not np.any(valid):
        return 0.0

    zscores = np.abs(patient_clean[valid] - mean_clean[valid]) / std_clean[valid]
    return float(np.sum(zscores > threshold))


def _mahalanobis_distance(
    patient_vec: np.ndarray,
    mean_vec: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """Compute Mahalanobis distance: sqrt((x-mu)^T @ cov_inv @ (x-mu)).

    Args:
        patient_vec: Patient's Gaussian parameter vector.
        mean_vec: Template mean vector.
        cov_inv: Inverse covariance matrix (or diagonal approximation).

    Returns:
        Mahalanobis distance as a scalar.  Returns 0.0 if the computation
        yields NaN or Inf (degenerate case).
    """
    diff = np.nan_to_num(patient_vec, nan=0.0) - np.nan_to_num(mean_vec, nan=0.0)
    # (x - mu)^T @ cov_inv @ (x - mu)
    val = float(diff @ cov_inv @ diff)
    if not math.isfinite(val) or val < 0.0:
        return 0.0
    return math.sqrt(val)


def _build_diagonal_cov_inv(std_vec: np.ndarray) -> np.ndarray:
    """Build inverse covariance matrix from std vector (diagonal approximation).

    Uses diag(1 / std^2) as the inverse covariance, which assumes feature
    independence.  Dimensions with near-zero std are zeroed out to avoid
    division-by-zero artifacts.

    NOTE: A full Mahalanobis distance requires storing per-diagnosis covariance
    matrices in the template JSON (e.g. via LedoitWolf shrinkage estimation).
    The diagonal approximation here is a practical starting point that still
    captures per-dimension scaling.  Future enhancement: store full covariance
    matrices in waveform_templates.json and use them here.

    Args:
        std_vec: Standard deviation vector from the template.

    Returns:
        Diagonal matrix of shape (n, n) representing the inverse covariance.
    """
    std_clean = np.nan_to_num(std_vec, nan=0.0)
    # Zero out near-zero std to avoid 1/0
    valid = std_clean > _EPS
    inv_diag = np.zeros_like(std_clean)
    inv_diag[valid] = 1.0 / (std_clean[valid] ** 2)
    return np.diag(inv_diag)


def extract_template_distance_features(
    patient_params: np.ndarray,
    template_set: Any,
) -> Dict[str, float]:
    """Compute distance features from a patient to all diagnosis templates.

    For each diagnosis template, computes:
        - ``{slug}_template_euclidean``: Euclidean distance
        - ``{slug}_template_cosine``: Cosine similarity
        - ``{slug}_template_zscore_max``: Maximum absolute z-score
        - ``{slug}_template_mahalanobis``: Mahalanobis distance (diagonal approx)

    Args:
        patient_params: Patient's Gaussian parameter vector (from
            :func:`~ammonix.templates.waveform_templates.extract_gaussian_params`).
            Shape ``(VECTOR_LENGTH,)``.
        template_set: A :class:`~ammonix.templates.waveform_templates.TemplateSet`
            instance with loaded diagnosis templates.

    Returns:
        Dict mapping feature names to float values.
        With 33 diagnoses, produces 198 features (33 x 6 metrics).
        Returns empty dict if patient_params is all-zero (no MLP data).
    """
    features: Dict[str, float] = {}

    # Skip patients with no Gaussian data
    if np.count_nonzero(np.nan_to_num(patient_params, nan=0.0)) == 0:
        # Return all features as 0.0 so XGBoost gets a complete vector
        for dx in template_set.diagnoses:
            slug = _safe_slug(dx)
            features[f"{slug}_template_euclidean"] = 0.0
            features[f"{slug}_template_cosine"] = 0.0
            features[f"{slug}_template_zscore_max"] = 0.0
            features[f"{slug}_template_mahalanobis"] = 0.0
            features[f"{slug}_template_zscore_mean"] = 0.0
            features[f"{slug}_template_zscore_gt2sigma"] = 0.0
        return features

    for dx in template_set.diagnoses:
        tmpl = template_set.get(dx)
        if tmpl is None:
            continue

        slug = _safe_slug(dx)

        features[f"{slug}_template_euclidean"] = euclidean_distance(
            patient_params, tmpl.mean_params,
        )
        features[f"{slug}_template_cosine"] = cosine_similarity(
            patient_params, tmpl.mean_params,
        )
        features[f"{slug}_template_zscore_max"] = max_zscore(
            patient_params, tmpl.mean_params, tmpl.std_params,
        )

        # Mahalanobis distance (diagonal covariance approximation).
        # Skip if too few patients contributed to the template (std is
        # all-zero), which would produce a degenerate zero distance.
        if tmpl.n_patients >= 3 and np.any(tmpl.std_params > _EPS):
            cov_inv = _build_diagonal_cov_inv(tmpl.std_params)
            features[f"{slug}_template_mahalanobis"] = _mahalanobis_distance(
                patient_params, tmpl.mean_params, cov_inv,
            )
        else:
            # Not enough data for meaningful Mahalanobis — use 0.0 sentinel
            features[f"{slug}_template_mahalanobis"] = 0.0

        # Mean z-score: average deviation across all features (diffuse conditions)
        features[f"{slug}_template_zscore_mean"] = mean_zscore(
            patient_params, tmpl.mean_params, tmpl.std_params,
        )

        # Count > 2σ: number of features more than 2 std devs from template
        features[f"{slug}_template_zscore_gt2sigma"] = _count_above_threshold(
            patient_params, tmpl.mean_params, tmpl.std_params, threshold=2.0,
        )

    return features


def extract_template_distance_features_from_patient(
    patient: Dict[str, Any],
    template_set: Any,
) -> Dict[str, float]:
    """Convenience wrapper: extract distance features directly from patient record.

    Combines :func:`~ammonix.templates.waveform_templates.extract_gaussian_params`
    and :func:`extract_template_distance_features` in one call.

    Args:
        patient: Raw patient record dict.
        template_set: Loaded :class:`~ammonix.templates.waveform_templates.TemplateSet`.

    Returns:
        Dict of template distance features.
    """
    from ammonix.domain import extract_all_features
    from ammonix.templates.waveform_templates import extract_gaussian_params

    features = extract_all_features(patient)
    params = extract_gaussian_params(features)
    return extract_template_distance_features(params, template_set)


def get_template_distance_feature_names(
    template_set: Any,
) -> List[str]:
    """Return the ordered list of feature names this module produces.

    Useful for appending to a model's feat_order.

    Args:
        template_set: Loaded TemplateSet.

    Returns:
        Sorted list of feature name strings.
    """
    names: List[str] = []
    for dx in template_set.diagnoses:
        slug = _safe_slug(dx)
        names.append(f"{slug}_template_euclidean")
        names.append(f"{slug}_template_cosine")
        names.append(f"{slug}_template_zscore_max")
        names.append(f"{slug}_template_mahalanobis")
        names.append(f"{slug}_template_zscore_mean")
        names.append(f"{slug}_template_zscore_gt2sigma")
    return names
