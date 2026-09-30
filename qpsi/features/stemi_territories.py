"""
STEMI coronary territory feature engineering -- ~45 features per patient.

Maps per-lead ST segment measurements to 5 coronary artery territories
(LAD, LCx, RCA, Posterior, Right Ventricle) and computes territory-level
aggregate features for downstream STEMI classification.

This module operates on the **flattened semantic features dict** produced by
``ammonix.domain.extract_all_features()`` (keys like ``Sem_ST_elev_lead_V1``,
``Sem_ST_dep_lead_II``, ``CP_*`` computed metrics, etc.).  It also provides
a lower-level function that works directly with ``STSegmentFeatures`` dicts
from the QPSI pipeline.

Public API
----------
- ``TERRITORIES``                              -- coronary territory definitions
- ``extract_stemi_territory_features()``       -- main entry (flat features dict)
- ``extract_territories_from_st_results()``    -- pipeline-level entry (STSegmentFeatures)
- ``STEMI_TERRITORY_FEATURE_NAMES``            -- ordered list of all output feature names

Dependencies: none beyond stdlib (pure dict -> dict transform).

.. note::
   **WaveMedix-canonical (V6 extension).** No counterpart in the Feb14
   or Apr28 notebooks. Per K3 (Phase 2 lock, 2026-05-18), this module
   stays WaveMedix-only — one-way distillation into the platform; no
   ship-back to the notebook. See ``qpsi/FIDELITY_REPORT.md`` section
   "WaveMedix-canonical extensions (V5/V6)".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Coronary territory definitions
# ---------------------------------------------------------------------------

TERRITORIES: Dict[str, Dict[str, List[str]]] = {
    "LAD": {
        "primary": ["V1", "V2", "V3", "V4"],
        "reciprocal": ["II", "III", "aVF"],
    },
    "LCx": {
        "primary": ["I", "aVL", "V5", "V6"],
        "reciprocal": ["V1", "V2", "V3"],
    },
    "RCA": {
        "primary": ["II", "III", "aVF"],
        "reciprocal": ["I", "aVL"],
    },
    "Posterior": {
        "primary": [],  # detected via reciprocal changes (tall R, ST depression in V1-V3)
        "reciprocal": ["V1", "V2", "V3"],
    },
    "RV": {
        "primary": ["V1"],  # V1 ST elevation + inferior STEMI pattern
        "reciprocal": [],
    },
}

ALL_LEADS: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Territory pairs for reciprocal analysis
# (territory, its typical reciprocal territory)
_RECIPROCAL_PAIRS: List[tuple] = [
    ("LAD", "RCA"),    # anterior <-> inferior
    ("RCA", "LAD"),    # inferior <-> anterior
    ("LCx", "LAD"),    # lateral <-> anterior/septal
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_lead_elev(features: Dict[str, float], lead: str) -> float:
    """Get whether a lead has ST elevation (binary from semantic features)."""
    return features.get(f"Sem_ST_elev_lead_{lead}", 0.0)


def _get_lead_dep(features: Dict[str, float], lead: str) -> float:
    """Get whether a lead has ST depression (binary from semantic features)."""
    return features.get(f"Sem_ST_dep_lead_{lead}", 0.0)


def _safe_div(num: float, den: float) -> float:
    """Safe division returning 0.0 on zero denominator."""
    return num / den if abs(den) > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Main entry point: flat features dict -> STEMI territory features
# ---------------------------------------------------------------------------

def extract_stemi_territory_features(
    semantic_features: Dict[str, float],
) -> Dict[str, float]:
    """Extract ~45 STEMI coronary territory features from a flat feature dict.

    Parameters
    ----------
    semantic_features:
        Flattened feature dict as produced by ``extract_all_features()``.
        Expected keys include:
        - ``Sem_ST_elev_lead_{lead}`` (binary, 12 leads)
        - ``Sem_ST_dep_lead_{lead}`` (binary, 12 leads)
        - ``Sem_ST_elev_lead_count``, ``Sem_ST_dep_lead_count``
        - ``Norm_ST_anterior_mean``, ``Norm_ST_inferior_mean``, etc. (optional)
        - ``Norm_Elev_Max``, ``Norm_Depr_Max``, ``Norm_Polarity_Balance`` (optional)

    Returns
    -------
    dict
        ~45 new features, all prefixed with ``STEMI_`` for namespace clarity.
        Missing input features are treated as 0.0 (graceful degradation).
    """
    out: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # 1. Per-territory elevation & depression counts
    # -----------------------------------------------------------------------
    territory_elev_counts: Dict[str, float] = {}
    territory_dep_counts: Dict[str, float] = {}
    territory_elev_fractions: Dict[str, float] = {}
    territory_dep_fractions: Dict[str, float] = {}

    for ter_name, ter_def in TERRITORIES.items():
        primary = ter_def["primary"]
        reciprocal = ter_def["reciprocal"]

        # Count elevated / depressed leads in primary zone
        n_prim = len(primary)
        elev_in_primary = sum(_get_lead_elev(semantic_features, ld) for ld in primary)
        dep_in_primary = sum(_get_lead_dep(semantic_features, ld) for ld in primary)

        territory_elev_counts[ter_name] = elev_in_primary
        territory_dep_counts[ter_name] = dep_in_primary
        territory_elev_fractions[ter_name] = _safe_div(elev_in_primary, n_prim) if n_prim > 0 else 0.0
        territory_dep_fractions[ter_name] = _safe_div(dep_in_primary, n_prim) if n_prim > 0 else 0.0

        out[f"STEMI_{ter_name}_elev_count"] = elev_in_primary
        out[f"STEMI_{ter_name}_dep_count"] = dep_in_primary
        out[f"STEMI_{ter_name}_elev_fraction"] = territory_elev_fractions[ter_name]
        out[f"STEMI_{ter_name}_dep_fraction"] = territory_dep_fractions[ter_name]

        # Count depression in reciprocal zone (supports STEMI diagnosis)
        n_recip = len(reciprocal)
        dep_in_reciprocal = sum(_get_lead_dep(semantic_features, ld) for ld in reciprocal)
        elev_in_reciprocal = sum(_get_lead_elev(semantic_features, ld) for ld in reciprocal)

        out[f"STEMI_{ter_name}_reciprocal_dep_count"] = dep_in_reciprocal
        out[f"STEMI_{ter_name}_reciprocal_elev_count"] = elev_in_reciprocal
        out[f"STEMI_{ter_name}_reciprocal_dep_fraction"] = (
            _safe_div(dep_in_reciprocal, n_recip) if n_recip > 0 else 0.0
        )

    # -----------------------------------------------------------------------
    # 2. Reciprocal change detection (binary per territory)
    # -----------------------------------------------------------------------
    # A territory has reciprocal changes if it has elevation in primary leads
    # AND depression in its reciprocal leads (or vice versa for posterior).
    for ter_name, ter_def in TERRITORIES.items():
        primary = ter_def["primary"]
        reciprocal = ter_def["reciprocal"]

        has_primary_elev = territory_elev_counts.get(ter_name, 0) >= 1
        has_reciprocal_dep = out.get(f"STEMI_{ter_name}_reciprocal_dep_count", 0) >= 1

        if ter_name == "Posterior":
            # Posterior STEMI: detected via ST depression (reciprocal) in V1-V3
            # with no primary leads -- reciprocal depression IS the signal
            has_reciprocal = has_reciprocal_dep
        elif ter_name == "RV":
            # RV: V1 elevation + inferior STEMI pattern (RCA territory active)
            rca_elev = territory_elev_counts.get("RCA", 0)
            has_reciprocal = has_primary_elev and rca_elev >= 2
        else:
            has_reciprocal = has_primary_elev and has_reciprocal_dep

        out[f"STEMI_{ter_name}_reciprocal_present"] = 1.0 if has_reciprocal else 0.0

    # -----------------------------------------------------------------------
    # 3. Territory involvement count (how many of the 5 territories are active)
    # -----------------------------------------------------------------------
    involvement_count = 0.0
    for ter_name in TERRITORIES:
        # A territory is "involved" if:
        # - It has >= 2 primary leads with elevation (standard STEMI), OR
        # - It has reciprocal changes present, OR
        # - For Posterior: reciprocal depression in >= 2 of V1-V3
        is_involved = False
        if ter_name == "Posterior":
            is_involved = out.get(f"STEMI_{ter_name}_reciprocal_dep_count", 0) >= 2
        elif ter_name == "RV":
            is_involved = out.get(f"STEMI_RV_reciprocal_present", 0) > 0
        else:
            primary_count = territory_elev_counts.get(ter_name, 0)
            is_involved = primary_count >= 2

        out[f"STEMI_{ter_name}_involved"] = 1.0 if is_involved else 0.0
        if is_involved:
            involvement_count += 1.0

    out["STEMI_territory_involvement_count"] = involvement_count

    # -----------------------------------------------------------------------
    # 4. ST magnitude features per territory (from normalized deviation metrics)
    # -----------------------------------------------------------------------
    # Use the Norm_ST_{territory}_mean/max features that st_deviation.py computes.
    # These map territory names: anterior -> LAD, inferior -> RCA, lateral -> LCx
    _norm_territory_map = {
        "LAD": "anterior",
        "RCA": "inferior",
        "LCx": "lateral",
    }
    for ter_name, norm_name in _norm_territory_map.items():
        norm_mean = semantic_features.get(f"CP_Norm_ST_{norm_name}_mean", 0.0)
        norm_max = semantic_features.get(f"CP_Norm_ST_{norm_name}_max", 0.0)
        out[f"STEMI_{ter_name}_norm_mean"] = norm_mean
        out[f"STEMI_{ter_name}_norm_max"] = norm_max

    # Posterior and RV don't have direct Norm_ST_ metrics, derive from V1-V3
    # For posterior: use the depression side of V1-V3 normalized deviation
    out["STEMI_Posterior_norm_mean"] = 0.0  # placeholder -- no direct mapping
    out["STEMI_Posterior_norm_max"] = 0.0
    out["STEMI_RV_norm_mean"] = 0.0
    out["STEMI_RV_norm_max"] = 0.0

    # -----------------------------------------------------------------------
    # 5. Global ST deviation features
    # -----------------------------------------------------------------------
    norm_elev_max = semantic_features.get("CP_Norm_Elev_Max", 0.0)
    norm_depr_max = semantic_features.get("CP_Norm_Depr_Max", 0.0)
    norm_elev_sum = semantic_features.get("CP_Norm_Elev_Sum", 0.0)
    norm_depr_sum = semantic_features.get("CP_Norm_Depr_Sum", 0.0)
    polarity_balance = semantic_features.get("CP_Norm_Polarity_Balance", 0.0)

    out["STEMI_max_elev_normalized"] = norm_elev_max
    out["STEMI_max_depr_normalized"] = abs(norm_depr_max)

    # Reciprocal ratio: magnitude of depression relative to elevation
    # High ratio with localized elevation = classic acute MI
    out["STEMI_reciprocal_ratio"] = _safe_div(abs(norm_depr_max), max(norm_elev_max, 1e-9))

    # Polarity balance (from st_deviation.py, pass-through for convenience)
    out["STEMI_polarity_balance"] = polarity_balance

    # -----------------------------------------------------------------------
    # 6. Territory concordance scores
    # -----------------------------------------------------------------------
    # How many leads within a territory agree on the same direction (elevation)?
    # High concordance = more leads showing same pattern = stronger signal
    for ter_name, ter_def in TERRITORIES.items():
        primary = ter_def["primary"]
        if not primary:
            out[f"STEMI_{ter_name}_concordance"] = 0.0
            continue
        n = len(primary)
        elev = territory_elev_counts.get(ter_name, 0)
        dep = territory_dep_counts.get(ter_name, 0)
        # Concordance: fraction of leads that agree on the dominant direction
        dominant = max(elev, dep)
        out[f"STEMI_{ter_name}_concordance"] = _safe_div(dominant, n)

    # -----------------------------------------------------------------------
    # 7. Cross-territory features
    # -----------------------------------------------------------------------
    # Maximum territory elevation fraction (which territory is most involved?)
    max_ter_frac = 0.0
    for ter_name in TERRITORIES:
        frac = territory_elev_fractions.get(ter_name, 0.0)
        max_ter_frac = max(max_ter_frac, frac)
    out["STEMI_max_territory_elev_fraction"] = max_ter_frac

    # Any reciprocal changes detected across all territories
    any_reciprocal = any(
        out.get(f"STEMI_{t}_reciprocal_present", 0) > 0 for t in TERRITORIES
    )
    out["STEMI_any_reciprocal"] = 1.0 if any_reciprocal else 0.0

    # Multi-territory involvement (>= 2 territories = extensive infarct)
    out["STEMI_multi_territory"] = 1.0 if involvement_count >= 2 else 0.0

    return out


# ---------------------------------------------------------------------------
# Pipeline-level entry point: STSegmentFeatures dict -> STEMI features
# ---------------------------------------------------------------------------

def extract_territories_from_st_results(
    st_results: Dict[str, Any],
    age: int = 50,
    sex: str = "Male",
) -> Dict[str, float]:
    """Extract STEMI territory features directly from per-lead STSegmentFeatures.

    This is the lower-level entry point for use within the QPSI pipeline
    where raw ``STSegmentFeatures`` objects are available (before flattening).

    Parameters
    ----------
    st_results:
        Dict mapping lead name -> ``STSegmentFeatures`` dataclass instance
        (as produced by ``analyze_st_segments_all_leads()["lead_features"]``).
    age:
        Patient age (for threshold computation).
    sex:
        Patient sex (for threshold computation).

    Returns
    -------
    dict
        Same ~45 STEMI territory features as ``extract_stemi_territory_features()``.
    """
    # Build a synthetic flat feature dict from the raw ST results,
    # then delegate to the main function.
    flat: Dict[str, float] = {}

    for lead in ALL_LEADS:
        f = st_results.get(lead)
        if f is None:
            flat[f"Sem_ST_elev_lead_{lead}"] = 0.0
            flat[f"Sem_ST_dep_lead_{lead}"] = 0.0
            continue

        elev = getattr(f, "st_elevation", 0.0) or 0.0
        depr = getattr(f, "st_depression", 0.0) or 0.0

        # Threshold-based binary flags (matching pipeline behavior)
        from qpsi.features.st_deviation import (
            get_lead_elevation_threshold,
            get_lead_depression_threshold,
        )
        elev_thr = get_lead_elevation_threshold(lead, age, sex)
        depr_thr = get_lead_depression_threshold(lead)

        flat[f"Sem_ST_elev_lead_{lead}"] = 1.0 if elev >= elev_thr else 0.0
        flat[f"Sem_ST_dep_lead_{lead}"] = 1.0 if depr <= depr_thr else 0.0

        # Normalized deviation
        nd = getattr(f, "normalized_deviation", 0.0) or 0.0
        flat[f"per_lead_norm_{lead}"] = nd

    # Aggregate normalized metrics (reconstruct from per-lead data)
    norm_elevations = [v for k, v in flat.items() if k.startswith("per_lead_norm_") and v > 0]
    norm_depressions = [v for k, v in flat.items() if k.startswith("per_lead_norm_") and v < 0]

    flat["CP_Norm_Elev_Max"] = max(norm_elevations) if norm_elevations else 0.0
    flat["CP_Norm_Depr_Max"] = min(norm_depressions) if norm_depressions else 0.0
    flat["CP_Norm_Elev_Sum"] = sum(norm_elevations)
    flat["CP_Norm_Depr_Sum"] = sum(norm_depressions)

    total_elev = sum(norm_elevations)
    total_depr = sum(abs(v) for v in norm_depressions)
    denom = max(total_elev + total_depr, 1e-6)
    flat["CP_Norm_Polarity_Balance"] = (total_elev - total_depr) / denom

    # Territory-level normalized means
    _territory_leads = {
        "anterior": ["V1", "V2", "V3", "V4"],
        "inferior": ["II", "III", "aVF"],
        "lateral": ["I", "aVL", "V5", "V6"],
    }
    for ter_name, ter_leads in _territory_leads.items():
        vals = [flat.get(f"per_lead_norm_{ld}", 0.0) for ld in ter_leads]
        flat[f"CP_Norm_ST_{ter_name}_mean"] = sum(vals) / max(len(vals), 1)
        flat[f"CP_Norm_ST_{ter_name}_max"] = max(vals, key=abs) if vals else 0.0

    return extract_stemi_territory_features(flat)


# ---------------------------------------------------------------------------
# Feature name registry (for downstream introspection)
# ---------------------------------------------------------------------------

def _enumerate_feature_names() -> List[str]:
    """Return the ordered list of all STEMI territory feature names.

    This uses a dummy input to discover feature names deterministically.
    """
    dummy = extract_stemi_territory_features({})
    return sorted(dummy.keys())


# Computed at module load time for O(1) access
STEMI_TERRITORY_FEATURE_NAMES: List[str] = _enumerate_feature_names()
