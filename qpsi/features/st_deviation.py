"""
ST-segment deviation analysis -- the largest single feature module.

Enhanced with lead-specific AHA/ACC/ESC guideline thresholds, age/sex-dependent
V2/V3 elevation criteria, high take-off discrimination, LBBB context awareness,
and threshold-normalized deviation scores for cross-lead comparison.

Source: Cell 9A3 (Cell 11) of the QPSI notebook (Feb14 revision).

This module does NOT use the standard ``(ctx: FeatureContext) -> dict`` registry
pattern.  It exposes its own ``STSegmentFeatures`` dataclass and is called
separately by the pipeline.

Public API
----------
- ``STSegmentFeatures``               -- per-lead result dataclass
- ``analyze_st_segments_all_leads()`` -- main multi-lead entry point
- ``detect_st_patterns()``            -- multi-lead STEMI/pattern detection
- ``get_lead_elevation_threshold()``  -- AHA/ACC/ESC elevation threshold (Feb14)
- ``get_lead_depression_threshold()`` -- AHA/ACC/ESC depression threshold (Feb14)
- ``compute_normalized_deviation()``  -- threshold-relative score (Feb14)
- ``estimate_st_axis()``              -- ST vector axis from Lead I + aVF (Feb14)
"""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate, signal

from qpsi.compat import trapezoid as _trapezoid

from qpsi.features.context import FeatureContext, _f
from qpsi.features.helpers import _t_amp_and_duration
from qpsi.features.qrs_complex import calculate_qrs_axis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ST-Segment Global Hyperparameters
# ---------------------------------------------------------------------------

ST_J60_WINDOW_MS: float = 120.0        # J-point offset for standard measurement (ms)
ST_ELEVATION_THR_MV: float = 0.10      # Baseline minimal elevation to call ST elevation (mV)
ST_DEPRESSION_THR_MV: float = -0.05    # Minimal depression to call ST depression (mV) -- tightened from -0.04
ST_SLOPE_THR_UV_PER_MS: float = 5.0    # Minimal slope magnitude (uV/ms)
ST_FLAT_THR_MV: float = 0.02           # "Isoelectric" range threshold (mV)
ST_MIN_VALID_SEG_MS: float = 40.0      # Minimal ST-segment duration considered valid (ms)

# Thresholds for V1/V2 specifically
ST_ELEVATION_V1V2_HIGH: float = 0.20   # Threshold when "High Take-off" context is suspected
ST_ELEVATION_V1V2_LOW: float = 0.10    # Standard threshold when no depression is present

# AHA/ACC/ESC age- and sex-specific ST-elevation thresholds for V2 and V3.
# Extracted from inline ``get_lead_elevation_threshold`` body (Phase 2 §4.6
# cleanup, C2.EP) so the ``HyperParams`` panel can override them at runtime
# via monkey-patch or import-and-reassign. Aligned with the [[feedback-qpsi-
# unbiased-measurements]] principle (toggle-friendly thresholds vs hardcoded
# clinical definitions) and mirrors qpsi_direct's same extraction.
ST_ELEV_V2V3_MAN_YOUNG: float = 0.25   # Men < 40 years, leads V2/V3 (mV)
ST_ELEV_V2V3_MAN_OLD:   float = 0.20   # Men >= 40 years, leads V2/V3 (mV)
ST_ELEV_V2V3_WOMAN:     float = 0.15   # Women, leads V2/V3 (mV)
ST_ELEV_OTHER:          float = 0.10   # All other leads (mV) -- same as ST_ELEVATION_THR_MV
ST_DEPR_OTHER:          float = -0.05  # All other leads (mV) -- same as ST_DEPRESSION_THR_MV
ST_BASELINE_LOOKBACK_OFFSET_MS: float = 40.0  # ms before P-onset for TP-baseline window


# ---------------------------------------------------------------------------
# Lead-Specific Clinical Threshold Functions (AHA/ACC/ESC)  -- Feb14
# ---------------------------------------------------------------------------

def get_lead_elevation_threshold(lead_name: str, age: int = 50, sex: str = "Male") -> float:
    """Return the clinically significant ST-elevation threshold (mV) for a given lead,
    per AHA/ACC/ESC guidelines.

    - V2, V3: age/sex dependent (men < 40: ``ST_ELEV_V2V3_MAN_YOUNG``,
      men >= 40: ``ST_ELEV_V2V3_MAN_OLD``, women: ``ST_ELEV_V2V3_WOMAN``)
    - All other leads: ``ST_ELEV_OTHER``
    """
    if lead_name in ("V2", "V3"):
        if sex and "FEMALE" in sex.upper():
            return ST_ELEV_V2V3_WOMAN
        else:
            return ST_ELEV_V2V3_MAN_YOUNG if age < 40 else ST_ELEV_V2V3_MAN_OLD
    return ST_ELEV_OTHER


def get_lead_depression_threshold(lead_name: str) -> float:
    """Return the clinically significant ST-depression threshold (mV, negative value)
    per AHA/ACC/ESC guidelines.

    - aVR excluded (reciprocal lead, often has physiological elevation)
    - All other leads: ``ST_DEPR_OTHER`` (-0.05 mV / 0.5 mm)
    """
    if lead_name == "aVR":
        return -999.0  # effectively never flag aVR depression
    return ST_DEPR_OTHER


def compute_normalized_deviation(
    amplitude_mv: float,
    lead_name: str,
    age: int = 50,
    sex: str = "Male",
) -> float:
    """Compute a *threshold-normalized* ST deviation score for a single lead.

    Returns a signed float:
      positive = fraction of lead-specific elevation threshold met  (1.0 = exactly at threshold)
      negative = fraction of lead-specific depression threshold met (-1.0 = exactly at threshold)

    This allows cross-lead comparison despite differing mV thresholds.

    Examples::

        0.15 mV in V3 for a man >= 40  ->  0.15 / 0.20 = +0.75
        0.15 mV in V5                  ->  0.15 / 0.10 = +1.50  (above threshold)
       -0.08 mV in V4                  -> -0.08 / 0.05 = -1.60  (above threshold)
    """
    if amplitude_mv >= 0:
        thr = get_lead_elevation_threshold(lead_name, age, sex)
        return amplitude_mv / max(thr, 1e-6)
    else:
        thr = abs(get_lead_depression_threshold(lead_name))
        if thr > 900:
            return 0.0  # aVR: never counts
        return amplitude_mv / max(thr, 1e-6)  # returns negative


# ---------------------------------------------------------------------------
# STSegmentFeatures dataclass
# ---------------------------------------------------------------------------

@dataclass
class STSegmentFeatures:
    """Per-lead ST-segment feature bundle.

    Contains standard J-point amplitude ladder, morphology descriptors,
    and Feb14 additions (QRS duration, T-wave polarity/amplitude,
    normalized deviation, lead-specific thresholds).
    """

    # --- Standard fields (restored to prevent pipeline errors) ---
    j_point_ms: float
    j_point_amplitude: float
    j_plus_20_amplitude: float
    j_plus_40_amplitude: float
    j_plus_60_amplitude: float
    j_plus_80_amplitude: float
    j_plus_100_amplitude: float = 0.0
    j_plus_120_amplitude: float = 0.0
    j_plus_140_amplitude: float = 0.0
    j_plus_160_amplitude: float = 0.0
    j_plus_180_amplitude: float = 0.0

    st_slope: float = 0.0
    st_shape: str = ""
    st_elevation: float = 0.0
    st_depression: float = 0.0
    st_t_ratio: float = 0.0

    # --- Feb14 discriminator fields ---
    qrs_duration_ms: float = 80.0
    t_wave_polarity: str = "positive"
    t_wave_amplitude: float = 0.0

    # --- Feb14 lead-normalized deviation fields ---
    # Express ST deviation as a fraction of the lead-specific clinical
    # threshold (AHA/ACC/ESC).  +1.0 means "exactly at elevation threshold",
    # -1.0 means "exactly at depression threshold".  Allows downstream
    # model to compare leads with different mV thresholds on a common scale.
    normalized_deviation: float = 0.0       # signed, threshold-relative score
    lead_elev_threshold_mv: float = 0.10    # the elevation threshold used
    lead_depr_threshold_mv: float = -0.05   # the depression threshold used

    st_integral: float = 0.0
    t_wave_onset: float = 0.0
    clinical_interpretation: str = ""
    is_abnormal: bool = False
    confidence: float = 0.0
    estimated_qt_ms: float = 0.0
    st_primary_pattern: str = "none"
    st_consistency: str = "indeterminate"
    lead_name: str = "Unknown"
    interpretation_dict: Optional[Dict] = None

    # --- Apr28 Item 4.5.3 additions (Cell 11 expansion) ---
    # st_curvature: quadratic-fit curvature coefficient on the ST segment
    #   (coeffs[0] of the polyfit inside analyze_st_morphology). Exposed
    #   here so per_lead_morphology consumers can read it directly.
    # st_t_junction_angle: angle (degrees) between terminal ST slope and
    #   initial T-wave slope at the ST-T junction.
    # st_slope_progression: list of 4 instantaneous slopes (uV/ms) at
    #   J+0, J+40, J+80, J+120 ms. Length-4 list when populated; empty
    #   list when unavailable (matches notebook's `if len(sp) < 4: continue`).
    st_curvature: float = 0.0
    st_t_junction_angle: float = 0.0
    st_slope_progression: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def find_j_point_precise(
    trace: np.ndarray,
    time_ms: np.ndarray,
    qrs_fits: List[Dict],
    method: str = "derivative",
) -> Tuple[float, int]:
    """Find J-point (end of QRS) using multiple methods."""
    qrs_components = [f for f in qrs_fits if f["wave_type"] == "QRS"]
    if not qrs_components:
        return 40.0, np.argmin(np.abs(time_ms - 40))

    s_wave = None
    for comp in qrs_components:
        if comp.get("component") == "S":
            s_wave = comp
            break

    if s_wave is None:
        s_wave = max(qrs_components, key=lambda x: x["center_ms"])

    if method == "derivative":
        s_center = s_wave["center_ms"]
        s_sigma = s_wave["sigma_ms"]
        search_start = s_center + s_sigma
        search_end = s_center + 3 * s_sigma

        mask = (time_ms >= search_start) & (time_ms <= search_end)
        if mask.sum() > 5:
            dt = np.mean(np.diff(time_ms))
            deriv = np.gradient(trace, dt)
            from scipy.ndimage import gaussian_filter1d

            smooth_deriv = gaussian_filter1d(np.abs(deriv[mask]), sigma=2)
            min_deriv_idx = np.argmin(smooth_deriv)
            j_idx = np.where(mask)[0][min_deriv_idx]
            return float(time_ms[j_idx]), j_idx

    elif method == "tangent":
        s_center = s_wave["center_ms"]
        s_sigma = s_wave["sigma_ms"]
        t_tangent = s_center + 1.5 * s_sigma
        idx_tangent = np.argmin(np.abs(time_ms - t_tangent))
        window = 5
        if idx_tangent > window and idx_tangent < len(trace) - window:
            local_slope = np.polyfit(
                time_ms[idx_tangent - window : idx_tangent + window],
                trace[idx_tangent - window : idx_tangent + window],
                1,
            )[0]
            baseline = np.median(trace[time_ms > 200])
            current_amp = trace[idx_tangent]
            if abs(local_slope) > 1e-6:
                j_time = t_tangent + (baseline - current_amp) / local_slope
                j_idx = np.argmin(np.abs(time_ms - j_time))
                return float(time_ms[j_idx]), j_idx

    s_center = s_wave["center_ms"]
    s_sigma = s_wave["sigma_ms"]
    j_time = s_center + 2.0 * s_sigma
    j_idx = np.argmin(np.abs(time_ms - j_time))
    return float(time_ms[j_idx]), j_idx


def measure_baseline(
    trace: np.ndarray,
    time_ms: np.ndarray,
    p_wave_bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """Measure isoelectric baseline from multiple segments."""
    baselines: List[float] = []
    if p_wave_bounds and p_wave_bounds[0] is not None:
        tp_mask = (time_ms >= p_wave_bounds[0] - 40) & (time_ms < p_wave_bounds[0])
        if tp_mask.sum() > 10:
            baselines.append(np.median(trace[tp_mask]))

    pq_mask = (time_ms >= -80) & (time_ms <= -20)
    if pq_mask.sum() > 10:
        baselines.append(np.median(trace[pq_mask]))

    late_tp_mask = (time_ms >= 350) & (time_ms <= 450)
    if late_tp_mask.sum() > 10:
        baselines.append(np.median(trace[late_tp_mask]))

    if baselines:
        return float(np.median(baselines))
    else:
        return 0.0


def analyze_st_morphology(
    st_segment: np.ndarray,
    st_time: np.ndarray,
) -> Tuple[str, float, float]:
    """Classify ST segment morphology.

    Returns ``(shape, confidence, curvature)``. The curvature (quadratic
    coefficient) was always computed internally; Apr28 Item 4.5.3 surfaces
    it so per_lead_morphology consumers can read ``st_curvature`` without
    re-fitting.
    """
    if len(st_segment) < 5:
        return "indeterminate", 0.0, 0.0

    try:
        t_norm = (st_time - st_time[0]) / (st_time[-1] - st_time[0] + 1e-6)
        coeffs = np.polyfit(t_norm, st_segment, 2)
        fitted = np.polyval(coeffs, t_norm)

        linear_slope = (st_segment[-1] - st_segment[0]) / (st_time[-1] - st_time[0] + 1e-6)
        curvature = coeffs[0]
        residual = np.sqrt(np.mean((st_segment - fitted) ** 2))
        confidence = np.exp(-residual * 10)

        slope_uv_per_ms = linear_slope * 1000
        if abs(slope_uv_per_ms) < ST_SLOPE_THR_UV_PER_MS:
            shape = "horizontal"
        elif slope_uv_per_ms > ST_SLOPE_THR_UV_PER_MS:
            shape = "upsloping" if curvature >= -0.01 else "scooped"
        elif slope_uv_per_ms < -ST_SLOPE_THR_UV_PER_MS:
            shape = "downsloping" if curvature <= 0.01 else "convex"
        else:
            shape = "horizontal"

        return shape, float(confidence), float(curvature)

    except Exception:
        return "indeterminate", 0.0, 0.0


# ---------------------------------------------------------------------------
# Apr28 Item 4.5.3 — Cell 11 ST morphology additions
# (Item 4.5.2 expansion: detect_brugada_pattern + compute_qrs_st_concordance)
# ---------------------------------------------------------------------------

def compute_qrs_st_concordance(
    qrs_axis_deg: float,
    st_axis_deg: float,
) -> float:
    """Compute the QRS-ST concordance as ``cos(QRS_axis - ST_axis)``.

    +1 = concordant, -1 = discordant, 0 = perpendicular.

    Source: Apr28 notebook line 5524.
    """
    try:
        diff_rad = np.radians(qrs_axis_deg - st_axis_deg)
        return float(np.cos(diff_rad))
    except Exception:
        return 0.0


def detect_brugada_pattern(
    st_results: Dict[str, "STSegmentFeatures"],
) -> Dict[str, Any]:
    """Detect Brugada pattern in right precordial leads (V1, V2, V3).

    Returns a dict with:
      - ``detected``: bool
      - ``type``: ``"coved" | "saddleback" | "none"``
      - ``affected_leads``: list of leads showing the pattern
      - ``coved_leads`` / ``saddleback_leads``: per-pattern lead lists
      - ``per_lead_scores``: dict of raw numeric discriminators per lead

    Detection logic (per-lead, then aggregated):
      Coved (Type 1): convex/downsloping ST in V1-V3 + steep negative slope at
                      J+40 + negative T-wave + sustained negative slopes
      Saddleback (Type 2): initial rise, dip, partial recovery + positive T-wave

    Source: Apr28 notebook line 5540 (~100 lines verbatim port).
    """
    right_precordial = ["V1", "V2", "V3"]

    coved_leads: List[str] = []
    saddleback_leads: List[str] = []
    lead_scores: Dict[str, Dict[str, Any]] = {}

    for ld in right_precordial:
        if ld not in st_results:
            continue
        f = st_results[ld]
        sp = getattr(f, "st_slope_progression", [0.0] * 4)
        if len(sp) < 4:
            continue

        shape = getattr(f, "st_shape", "")
        t_pol = getattr(f, "t_wave_polarity", "positive")
        curvature = getattr(f, "st_curvature", 0.0)
        slope_raw = getattr(f, "st_slope", 0.0)
        slope_uv = slope_raw * 1000.0

        slope_j0 = sp[0]
        slope_j40 = sp[1]
        slope_j80 = sp[2]
        slope_j120 = sp[3]
        slope_reversal = slope_j0 - slope_j40

        lead_scores[ld] = {
            "slope_j0": round(slope_j0, 4),
            "slope_j40": round(slope_j40, 4),
            "slope_j80": round(slope_j80, 4),
            "slope_j120": round(slope_j120, 4),
            "slope_reversal": round(slope_reversal, 4),
            "shape": shape,
            "curvature": round(curvature, 4),
            "slope_uv_per_ms": round(slope_uv, 4),
            "t_wave_polarity": t_pol,
            "t_wave_amplitude": round(getattr(f, "t_wave_amplitude", 0.0), 4),
        }

        # Coved pattern indicators
        is_convex_or_down = shape in ("convex", "downsloping")
        has_steep_j40_descent = slope_j40 < -0.5
        has_negative_t = t_pol == "negative"
        has_sustained_negative = slope_j80 < 0 and slope_j120 < 0

        if is_convex_or_down and has_steep_j40_descent and has_negative_t:
            # V1 commonly has downsloping ST + negative T in normal ECGs;
            # require sustained steep descent (j80 < -0.8) for V1 specifically
            if ld == "V1":
                if slope_j80 < -0.8:
                    coved_leads.append(ld)
            else:
                coved_leads.append(ld)

        # Saddleback pattern indicators (tightened to reduce FPs)
        has_initial_rise = slope_j0 > 0.5
        has_dip = slope_j40 < -0.8
        has_recovery = slope_j120 > slope_j40 + 0.5
        has_positive_t = t_pol == "positive"

        if has_initial_rise and has_dip and has_recovery and has_positive_t:
            saddleback_leads.append(ld)

    if coved_leads:
        pattern_type = "coved"
        affected = coved_leads
    elif saddleback_leads:
        pattern_type = "saddleback"
        affected = saddleback_leads
    else:
        pattern_type = "none"
        affected = []

    return {
        "detected": len(affected) > 0,
        "type": pattern_type,
        "affected_leads": affected,
        "coved_leads": coved_leads,
        "saddleback_leads": saddleback_leads,
        "per_lead_scores": lead_scores,
    }


def estimate_qrs_axis(lead_fits: Dict[str, Dict]) -> float:
    """Estimate QRS electrical axis from Lead I and aVF net QRS amplitudes.

    Source: Apr28 notebook line 5899 (~17 lines verbatim).

    Distinct from ``qpsi.features.qrs_complex.calculate_qrs_axis``:
    this function uses a STRICT ``wave_type == "QRS"`` filter and returns
    ``float`` with 0.0 fallback. ``calculate_qrs_axis`` (and Item 4.5.1's
    ``compute_qrs_axis_features``) use the full QRS-family filter
    (``{"Q", "R", "S", "QRS"}``) and return ``Optional[float]`` / a 6-key
    dict respectively.

    The strict-QRS algorithm is functionally dead code in the deployed
    pipeline: ``analyze_st_segments_all_leads`` writes ``QRS_Axis_Deg``
    via this function, then Cell 9E's ``compute_qrs_axis_features``
    (wrapping ``build_final_record``) overwrites it with the full-family
    value before JSONL serialization. Ported here for full source-code
    parity with the Apr28 notebook.
    """
    try:
        def _net_qrs_amp(lead_name: str) -> float:
            if lead_name not in lead_fits:
                return 0.0
            qrs_waves = [
                w for w in lead_fits[lead_name]["avg"]
                if w["wave_type"] == "QRS"
            ]
            return sum(w.get("amp_mv", 0.0) for w in qrs_waves)

        amp_i = _net_qrs_amp("I")
        amp_avf = _net_qrs_amp("aVF")
        return float(np.degrees(np.arctan2(amp_avf, amp_i)))
    except Exception:
        return 0.0


def compute_st_t_junction_angle(
    trace: np.ndarray,
    time_ms: np.ndarray,
    st_end_ms: float,
    t_wave_onset_ms: float,
) -> float:
    """Compute the angle (degrees) between the terminal ST-segment slope and
    the initial T-wave slope at the ST-T junction.

    Source: Apr28 notebook ``q_psi_ai_for_ecg_Apr28.ipynb`` line 5454.
    """
    try:
        junction_ms = (
            min(st_end_ms, t_wave_onset_ms) if t_wave_onset_ms > 0 else st_end_ms
        )
        window = 20.0

        st_mask = (time_ms >= junction_ms - window) & (time_ms <= junction_ms)
        if st_mask.sum() < 3:
            return 0.0
        st_t = time_ms[st_mask]
        st_v = trace[st_mask]
        st_slope = np.polyfit(st_t, st_v, 1)[0]

        tw_mask = (time_ms >= junction_ms) & (time_ms <= junction_ms + window)
        if tw_mask.sum() < 3:
            return 0.0
        tw_t = time_ms[tw_mask]
        tw_v = trace[tw_mask]
        tw_slope = np.polyfit(tw_t, tw_v, 1)[0]

        denom = 1.0 + st_slope * tw_slope
        if abs(denom) < 1e-12:
            return 90.0

        angle_rad = np.arctan(abs(tw_slope - st_slope) / abs(denom))
        angle_deg = float(np.degrees(angle_rad))
        return max(0.0, min(angle_deg, 180.0))
    except Exception:
        return 0.0


def compute_slope_progression(
    trace: np.ndarray,
    time_ms: np.ndarray,
    j_point_ms: float,
) -> List[float]:
    """Sample instantaneous slope (uV/ms) at four points after the J-point:
    J+0, J+40, J+80, J+120 ms.

    Source: Apr28 notebook ``q_psi_ai_for_ecg_Apr28.ipynb`` line 5493.
    """
    offsets = [0.0, 40.0, 80.0, 120.0]
    half_window = 10.0
    slopes: List[float] = []
    try:
        for off in offsets:
            t_center = j_point_ms + off
            idx_lo = int(np.argmin(np.abs(time_ms - (t_center - half_window))))
            idx_hi = int(np.argmin(np.abs(time_ms - (t_center + half_window))))
            if idx_lo == idx_hi or idx_lo >= len(trace) or idx_hi >= len(trace):
                slopes.append(0.0)
                continue
            dt = time_ms[idx_hi] - time_ms[idx_lo]
            if abs(dt) < 1e-6:
                slopes.append(0.0)
                continue
            slope_mv_per_ms = (trace[idx_hi] - trace[idx_lo]) / dt
            slopes.append(float(slope_mv_per_ms * 1000.0))
        return slopes
    except Exception:
        return [0.0] * 4


def calculate_st_integral(
    st_segment: np.ndarray,
    st_time: np.ndarray,
    baseline: float,
) -> float:
    """Calculate area between ST segment and baseline."""
    if len(st_segment) < 2:
        return 0.0
    st_relative = st_segment - baseline
    integral = _trapezoid(st_relative, st_time)
    return float(integral)


# ---------------------------------------------------------------------------
# Clinical interpretation
# ---------------------------------------------------------------------------

def interpret_st_changes(
    features: STSegmentFeatures,
    has_global_depression: bool = False,
    age: int = 50,
    sex: str = "Male",
) -> Dict[str, Any]:
    """Clinical interpretation using AHA/ACC/ESC guideline thresholds.

    Requires age and sex for precise V2/V3 evaluation.
    """
    interpretation: Dict[str, Any] = {
        "classification": [],
        "severity": "normal",
        "action_needed": False,
    }

    try:
        if features is None:
            return interpretation

        # --- 1. Define guideline thresholds ---
        # Default threshold for most leads is 0.1 mV (1 mm).
        # Uses ST_ELEV_OTHER module-level constant (HPO-tunable per C2.EP cleanup).
        elev_thresh = ST_ELEV_OTHER

        # Special rules for V2 and V3 — age/sex dependent (HPO-tunable constants).
        if features.lead_name in ["V2", "V3"]:
            if sex and "FEMALE" in sex.upper():
                elev_thresh = ST_ELEV_V2V3_WOMAN  # 1.5 mm for women
            else:
                # Male logic
                if age < 40:
                    elev_thresh = ST_ELEV_V2V3_MAN_YOUNG  # 2.5 mm for men < 40
                else:
                    elev_thresh = ST_ELEV_V2V3_MAN_OLD  # 2.0 mm for men >= 40

        # Context modifier: if global depression exists (reciprocal change),
        # we might accept lower thresholds or be stricter to avoid high take-off
        # artifacts.  Literature suggests maintaining specificity in V2/V3 even
        # with reciprocal changes.

        # --- 2. Evaluate elevation ---
        if features.j_plus_60_amplitude >= elev_thresh:

            # A. High take-off screen (benign early repolarisation)
            # Logic: STE < 35% of T-wave height + upsloping + concave
            is_high_takeoff = False
            if features.t_wave_amplitude > 0.25:  # Robust T-wave
                if features.st_t_ratio < 0.35 and features.st_shape == "upsloping":
                    is_high_takeoff = True

            if is_high_takeoff:
                interpretation["classification"].append(
                    f"High Take-off Pattern (Ratio {features.st_t_ratio:.2f})"
                )
                interpretation["severity"] = "benign variant"
            elif features.st_shape in ["convex", "horizontal"]:
                interpretation["classification"].append(
                    f"Significant ST Elevation (\u2265{elev_thresh * 10:.1f}mm)"
                )
                interpretation["severity"] = "critical"
                interpretation["action_needed"] = True
            else:
                interpretation["classification"].append("ST Elevation (Ischemic possibility)")
                interpretation["severity"] = "abnormal"

        # --- 3. Evaluate depression ---
        # Guideline: >= 0.05 mV (0.5 mm) in context of ischaemia, but 0.10 mV
        # is standard for specificity.
        # Uses ST_DEPR_OTHER module-level constant (HPO-tunable per C2.EP cleanup).
        depr_thresh = ST_DEPR_OTHER
        if features.j_plus_60_amplitude <= depr_thresh:
            if features.st_shape in ["horizontal", "downsloping"]:
                interpretation["classification"].append("Pathologic ST Depression")
                interpretation["severity"] = "abnormal"
                interpretation["action_needed"] = True
            elif features.st_shape == "upsloping":
                # Upsloping is often rate-related or benign unless very deep (> 1.5 mm)
                if features.j_plus_60_amplitude < -0.15:
                    interpretation["classification"].append("Upsloping ST Depression (Deep)")
                    interpretation["severity"] = "abnormal"
                else:
                    interpretation["classification"].append(
                        "Upsloping ST Depression (Non-specific)"
                    )

        if not interpretation["classification"]:
            interpretation["classification"].append("Normal ST Segment")

    except Exception as e:
        interpretation["classification"].append(f"Error: {str(e)}")

    return interpretation


# ---------------------------------------------------------------------------
# Per-lead comprehensive analysis
# ---------------------------------------------------------------------------

def analyze_st_segment_comprehensive(
    trace: np.ndarray,
    time_ms: np.ndarray,
    qrs_fits: List[Dict],
    t_wave_fits: List[Dict],
    p_wave_bounds: Optional[Tuple[float, float]] = None,
    lead_name: str = "Unknown",
    global_estimated_qt_ms: float = 180.0,
    age: int = 50,
    sex: str = "Male",
) -> STSegmentFeatures:
    """Comprehensive ST-segment analysis with QRS duration, T-wave polarity,
    and lead-specific threshold-normalized deviation scores."""

    j_point_ms, _ = find_j_point_precise(trace, time_ms, qrs_fits)
    baseline = measure_baseline(trace, time_ms, p_wave_bounds)

    # Offset logic
    if global_estimated_qt_ms <= 200:
        chosen_offset = 60
    else:
        chosen_offset = 80

    def get_amp(offset: float) -> float:
        idx = np.argmin(np.abs(time_ms - (j_point_ms + offset)))
        return float(trace[idx] - baseline) if 0 <= idx < len(trace) else 0.0

    # Get amplitudes for all offsets to satisfy STSegmentFeatures
    offsets = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    amps = {off: get_amp(off) for off in offsets}
    amp_60 = amps.get(60, 0.0)
    chosen_amp = get_amp(chosen_offset)

    st_end = j_point_ms + chosen_offset + 20
    st_mask = (time_ms >= j_point_ms) & (time_ms <= st_end)
    st_segment = trace[st_mask] - baseline
    st_time = time_ms[st_mask]

    st_slope = (
        (st_segment[-1] - st_segment[0]) / (st_time[-1] - st_time[0] + 1e-6)
        if len(st_segment) > 1
        else 0.0
    )
    st_shape, confidence, st_curvature = analyze_st_morphology(st_segment, st_time)
    st_int = calculate_st_integral(st_segment, st_time, 0)
    slope_uv = st_slope * 1000

    # J-point consistency
    j_amp = amps.get(0, 0.0)
    j100_amp = amps.get(100, 0.0)
    st_diff = j100_amp - j_amp
    if j_amp > 0.05 and abs(st_diff) < 0.05:
        st_consistency = "plateau_elevation"
    elif j_amp < 0.05 and j100_amp > 0.10:
        st_consistency = "rising_from_baseline"
    else:
        st_consistency = "indeterminate"

    # === NEW: Calculate QRS Duration ===
    if qrs_fits:
        qrs_start = min(f["center_ms"] - f["sigma_ms"] * 2 for f in qrs_fits)
        qrs_dur = j_point_ms - qrs_start
    else:
        qrs_dur = 80.0

    # === NEW: Calculate T-Wave Polarity ===
    t_pol = "flat"
    t_amp = 0.0
    t_wave_onset_ms = j_point_ms + global_estimated_qt_ms  # default estimate
    if t_wave_fits:
        dom_t = max(t_wave_fits, key=lambda x: abs(x.get("amp_mv", 0)))
        t_amp = dom_t.get("amp_mv", 0)
        if t_amp > 0.1:
            t_pol = "positive"
        elif t_amp < -0.1:
            t_pol = "negative"
        # Better T-wave onset estimate from the fit (Apr28 Item 4.5.3)
        t_comp = min(t_wave_fits, key=lambda x: x["center_ms"])
        t_wave_onset_ms = t_comp["center_ms"] - 2 * t_comp.get("sigma_ms", 20.0)

    # --- Apr28 Item 4.5.3: Cell 11 ST morphology additions ---
    junction_angle = compute_st_t_junction_angle(
        trace, time_ms, st_end, t_wave_onset_ms
    )
    slope_prog = compute_slope_progression(trace, time_ms, j_point_ms)

    # === Lead-specific thresholds (AHA/ACC/ESC) ===
    lead_elev_thr = get_lead_elevation_threshold(lead_name, age, sex)
    lead_depr_thr = get_lead_depression_threshold(lead_name)

    # Abnormality flags using lead-specific thresholds
    has_elev = chosen_amp >= lead_elev_thr
    has_depr = chosen_amp <= lead_depr_thr

    # Special check: in V1/V2, if QRS is wide, ignore mild elevation (LBBB context)
    if lead_name in ["V1", "V2"] and qrs_dur > 110 and has_elev and chosen_amp < 0.25:
        has_elev = False

    # Compute threshold-normalized deviation score
    norm_dev = compute_normalized_deviation(chosen_amp, lead_name, age, sex)

    # MORPHOLOGICAL FIX (parity with Apr28 notebook line 5697): signed ST-T
    # ratio with `abs(t_amp) > 0.02` gate preserves concordance information
    # (negative ratio = discordance). The prior gate `t_amp > 0.05 and amp_60 > 0`
    # clipped all ST-depression cases (negative amp_60) to 0 and discarded ~12
    # leads x ~99% of patients in the n=100 Apr28 paired-extraction panel.
    st_t_ratio = 0.0
    if abs(t_amp) > 0.02:
        st_t_ratio = amp_60 / t_amp  # signed: preserves concordance info

    is_abnormal = (
        has_elev
        or has_depr
        or abs(slope_uv) > ST_SLOPE_THR_UV_PER_MS
        or st_shape in ["scooped", "downsloping"]
        or abs(st_int) > 5.0
    )

    if has_elev:
        st_primary_pattern = "elevation"
    elif has_depr:
        st_primary_pattern = "depression"
    else:
        st_primary_pattern = "none"

    interp = f"ST Amp {chosen_amp * 10:.1f}mm"

    return STSegmentFeatures(
        j_point_ms=j_point_ms,
        j_point_amplitude=amps.get(0, 0.0),
        j_plus_20_amplitude=amps.get(20, 0.0),
        j_plus_40_amplitude=amps.get(40, 0.0),
        j_plus_60_amplitude=amps.get(60, 0.0),
        j_plus_80_amplitude=amps.get(80, 0.0),
        j_plus_100_amplitude=amps.get(100, 0.0),
        j_plus_120_amplitude=amps.get(120, 0.0),
        j_plus_140_amplitude=amps.get(140, 0.0),
        j_plus_160_amplitude=amps.get(160, 0.0),
        j_plus_180_amplitude=amps.get(180, 0.0),
        st_slope=st_slope,
        st_shape=st_shape,
        st_elevation=chosen_amp if chosen_amp > 0 else 0.0,
        st_depression=chosen_amp if chosen_amp < 0 else 0.0,
        st_integral=st_int,
        # New params
        qrs_duration_ms=qrs_dur,
        t_wave_polarity=t_pol,
        t_wave_amplitude=t_amp,
        st_t_ratio=st_t_ratio,
        # Lead-normalized deviation fields
        normalized_deviation=norm_dev,
        lead_elev_threshold_mv=lead_elev_thr,
        lead_depr_threshold_mv=lead_depr_thr,
        t_wave_onset=t_wave_onset_ms,
        clinical_interpretation=interp,
        is_abnormal=is_abnormal,
        confidence=confidence,
        estimated_qt_ms=global_estimated_qt_ms,
        st_primary_pattern=st_primary_pattern,
        st_consistency=st_consistency,
        lead_name=lead_name,
        # Apr28 Item 4.5.3 additions
        st_curvature=st_curvature,
        st_t_junction_angle=junction_angle,
        st_slope_progression=slope_prog,
    )


# ---------------------------------------------------------------------------
# ST vector axis estimation  -- Feb14
# ---------------------------------------------------------------------------

def estimate_st_axis(st_results: Dict[str, STSegmentFeatures]) -> float:
    """Estimate ST vector axis using Lead I and aVF deviations."""
    try:
        dev_i = st_results["I"].j_plus_60_amplitude if "I" in st_results else 0.0
        dev_avf = st_results["aVF"].j_plus_60_amplitude if "aVF" in st_results else 0.0

        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(dev_avf, dev_i))
        return angle
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Multi-lead entry point
# ---------------------------------------------------------------------------

def analyze_st_segments_all_leads(
    stack_sync: np.ndarray,
    time_ms: np.ndarray,
    lead_fits: Dict[str, Dict],
    leads: List[str],
    wave_bounds: Dict[str, Tuple[float, float]],
    age: int = 50,
    sex: str = "Male",
    avg_per_lead: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Analyse ST segments across all 12 leads.

    Parameters
    ----------
    stack_sync : (nBeats, 12, T) aligned beat stack
    time_ms    : signed time axis (T,)
    lead_fits  : {lead: {"avg": [Gaussian dicts], ...}}
    leads      : ordered lead name list
    wave_bounds: per-wave timing bounds (e.g. {"P": (start, end), ...})
    age        : patient age (for V2/V3 threshold)
    sex        : patient sex (for V2/V3 threshold)

    Returns
    -------
    dict with keys: lead_features, patterns, summary, computed_metrics,
    clinical_alert, estimated_qt_ms
    """
    if avg_per_lead is None:
        avg_per_lead = np.nanmean(stack_sync, axis=0)

    st_results: Dict[str, STSegmentFeatures] = {}

    # 1. Estimate QT
    global_estimated_qt = 180.0
    if "II" in lead_fits:
        qrs_fits = [f for f in lead_fits["II"]["avg"] if f["wave_type"] == "QRS"]
        t_fits = [f for f in lead_fits["II"]["avg"] if f["wave_type"] == "T"]
        if t_fits and qrs_fits:
            qrs_onset = min(g["center_ms"] - g["sigma_ms"] for g in qrs_fits)
            t_comp = min(t_fits, key=lambda x: x["center_ms"])
            t_onset = t_comp["center_ms"] - 2 * t_comp["sigma_ms"]
            global_estimated_qt = float(t_onset - qrs_onset)

    # 2. Analyse all leads
    for i, lead_name in enumerate(leads):
        if lead_name not in lead_fits:
            continue
        avg_trace = avg_per_lead[i, :]
        qrs_fits = [f for f in lead_fits[lead_name]["avg"] if f["wave_type"] == "QRS"]
        t_fits = [f for f in lead_fits[lead_name]["avg"] if f["wave_type"] == "T"]
        p_bounds = wave_bounds.get("P", None)

        st_features = analyze_st_segment_comprehensive(
            avg_trace,
            time_ms,
            qrs_fits,
            t_fits,
            p_bounds,
            lead_name,
            global_estimated_qt_ms=global_estimated_qt,
            age=age,
            sex=sex,
        )
        st_results[lead_name] = st_features

    # 3. Global context
    depression_count = sum(
        1
        for l in leads
        if l in st_results and st_results[l].st_depression < ST_DEPRESSION_THR_MV
    )
    has_global_depression = depression_count >= 2

    # 4. Patterns & summary
    patterns = detect_st_patterns(
        st_results,
        leads,
        estimated_qt_ms=global_estimated_qt,
        has_global_depression=has_global_depression,
        age=age,
        sex=sex,
    )
    summary = generate_st_summary(
        st_results,
        patterns,
        has_global_depression=has_global_depression,
        age=age,
        sex=sex,
    )

    # =========================================================================
    # 5. Generate computed metrics
    # =========================================================================
    computed_metrics: Dict[str, Any] = {}

    chest_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]

    # Extract ratios
    chest_ratios = [
        getattr(st_results[l], "st_t_ratio", 0.0)
        for l in chest_leads
        if l in st_results
    ]
    limb_ratios = [
        getattr(st_results[l], "st_t_ratio", 0.0)
        for l in limb_leads
        if l in st_results
    ]

    computed_metrics["ST_T_Ratio_Max_Chest"] = max(chest_ratios) if chest_ratios else 0.0
    computed_metrics["ST_T_Ratio_Max_Limb"] = max(limb_ratios) if limb_ratios else 0.0

    # Add discriminators
    if "discriminators" in summary:
        computed_metrics["ST_Balance_Score"] = summary["discriminators"].get("balance", 0.0)
        computed_metrics["ST_Axis_Deg"] = summary["discriminators"].get("axis", 0.0)
        computed_metrics["QRS_Wide_Count"] = (
            1.0 if summary["discriminators"].get("wide_qrs") else 0.0
        )

    def get_lead_val(lead: str, wave: str, param: str) -> float:
        """Helper to dig out T-wave angle from lead_fits."""
        try:
            waves = lead_fits[lead]["avg"]
            target = next((w for w in waves if w["wave_type"] == wave), None)
            return target.get(param, 0.0)
        except Exception:
            return 0.0

    # Calculate T-axis using Lead I and aVF (approximation)
    t_I_amp = get_lead_val("I", "T", "amp_mv")
    t_aVF_amp = get_lead_val("aVF", "T", "amp_mv")
    t_axis_deg = np.degrees(np.arctan2(t_aVF_amp, t_I_amp))

    # Calculate ST divergence
    st_axis = computed_metrics.get("ST_Axis_Deg", 0.0)

    # Angular difference logic (handles 359 vs 1 degree wrapping)
    diff = abs(st_axis - t_axis_deg) % 360
    if diff > 180:
        diff = 360 - diff

    computed_metrics["ST_T_Vector_Divergence"] = diff

    # === NEW: "Extension Index" (the final separator) ===
    # Combines ratio (morphology) + concordance (vector).
    # High index = ST extension; low index = ST drop down / normal.
    # Logic: if divergence is low (concordant), boost the ratio score.
    concordance_factor = 1.0
    if diff < 45:
        concordance_factor = 2.0   # Strongly concordant (extension characteristic)
    elif diff > 120:
        concordance_factor = 0.5   # Discordant (ischaemia / drop down characteristic)

    computed_metrics["ST_Extension_Index"] = (
        computed_metrics["ST_T_Ratio_Max_Chest"] * concordance_factor
    )

    # =========================================================================
    # Lead-normalized elevation / depression separation metrics
    # =========================================================================
    # These metrics express ST deviation on a threshold-relative scale so that
    # a patient with +0.15 mV in V2 (man >= 40, thr = 0.20) is scored differently
    # from +0.15 mV in V5 (thr = 0.10).
    #
    # Positive normalized_deviation -> elevation direction
    # Negative normalized_deviation -> depression direction
    # |value| >= 1.0 means the lead-specific clinical threshold is met

    norm_elevations: List[float] = []    # positive normalized scores
    norm_depressions: List[float] = []   # negative normalized scores
    per_lead_norm: Dict[str, float] = {}

    for ld in leads:
        if ld not in st_results:
            continue
        f = st_results[ld]
        nd = getattr(f, "normalized_deviation", 0.0)
        per_lead_norm[ld] = nd
        if nd > 0:
            norm_elevations.append(nd)
        elif nd < 0:
            norm_depressions.append(nd)

    # --- Aggregate normalized scores ---
    # Max normalized elevation across all leads (>= 1.0 means at least one lead
    # crosses its own clinical threshold)
    computed_metrics["Norm_Elev_Max"] = max(norm_elevations) if norm_elevations else 0.0
    # Max normalized depression (most negative; <= -1.0 means threshold crossed)
    computed_metrics["Norm_Depr_Max"] = min(norm_depressions) if norm_depressions else 0.0
    # Sum of normalized elevations (captures spatial extent)
    computed_metrics["Norm_Elev_Sum"] = sum(norm_elevations)
    # Sum of normalized depressions (captures spatial extent of depression)
    computed_metrics["Norm_Depr_Sum"] = sum(norm_depressions)
    # Count of leads that cross their own elevation threshold
    computed_metrics["Norm_Elev_Count"] = sum(1 for v in norm_elevations if v >= 1.0)
    # Count of leads that cross their own depression threshold
    computed_metrics["Norm_Depr_Count"] = sum(1 for v in norm_depressions if v <= -1.0)

    # --- The key separator: normalized polarity balance ---
    # Positive -> net elevation dominance (weighted by clinical significance)
    # Negative -> net depression dominance
    # Near zero -> mixed or insignificant
    norm_total_elev = sum(norm_elevations)
    norm_total_depr = sum(abs(v) for v in norm_depressions)
    norm_denom = max(norm_total_elev + norm_total_depr, 1e-6)
    # Range [-1, +1]:  +1 = pure elevation, -1 = pure depression
    computed_metrics["Norm_Polarity_Balance"] = (
        (norm_total_elev - norm_total_depr) / norm_denom
    )

    # --- Per-territory normalized averages ---
    territory_map = {
        "anterior": ["V1", "V2", "V3", "V4"],
        "inferior": ["II", "III", "aVF"],
        "lateral": ["I", "aVL", "V5", "V6"],
    }
    for ter_name, ter_leads in territory_map.items():
        ter_vals = [per_lead_norm[l] for l in ter_leads if l in per_lead_norm]
        if ter_vals:
            computed_metrics[f"Norm_ST_{ter_name}_mean"] = float(np.mean(ter_vals))
            computed_metrics[f"Norm_ST_{ter_name}_max"] = float(max(ter_vals, key=abs))
        else:
            computed_metrics[f"Norm_ST_{ter_name}_mean"] = 0.0
            computed_metrics[f"Norm_ST_{ter_name}_max"] = 0.0

    # Store the per-lead map for downstream use
    computed_metrics["per_lead_normalized_deviation"] = per_lead_norm

    # =========================================================================
    # Apr28 Item 4.5.3 — per-lead morphology container (Cell 11)
    # Verbatim mirror of notebook line 6274-6292: 16 keys per lead, sourced
    # entirely from STSegmentFeatures fields.
    # =========================================================================
    per_lead_morphology: Dict[str, Dict[str, Any]] = {}
    for ld in leads:
        if ld not in st_results:
            continue
        f = st_results[ld]
        per_lead_morphology[ld] = {
            "shape": f.st_shape,
            "curvature": f.st_curvature,
            "slope_uv_per_ms": f.st_slope * 1000.0,
            "st_t_junction_angle": f.st_t_junction_angle,
            "slope_progression": list(f.st_slope_progression),
            "st_t_ratio": f.st_t_ratio,
            "confidence": f.confidence,
            "normalized_deviation": f.normalized_deviation,
            # --- Phase 1 Brugada additions ---
            "j_point_amplitude": f.j_point_amplitude,
            "t_wave_amplitude": f.t_wave_amplitude,
            "t_wave_polarity": f.t_wave_polarity,
            "j_plus_60_amplitude": f.j_plus_60_amplitude,
            "j_plus_80_amplitude": f.j_plus_80_amplitude,
            "j_plus_100_amplitude": f.j_plus_100_amplitude,
            "j_plus_120_amplitude": f.j_plus_120_amplitude,
            "st_integral": f.st_integral,
        }
    computed_metrics["per_lead_morphology"] = per_lead_morphology

    # =========================================================================
    # Apr28 Item 4.5.2 — Cell 11 Brugada absorption (verbatim port of Apr28
    # lines 6240-6443). Computes:
    #   - QRS_ST_Concordance + QRS_Axis_Deg
    #   - ST_{anterior,inferior,lateral}_dominant_shape + mean_slope
    #   - brugada_pattern dict (from detect_brugada_pattern)
    #   - st_reciprocal_pattern, st_measurement_offset_ms
    #   - ST_Discrimination_Score, ST_Territory_Contrast,
    #     ST_Inferior_Morphology_Index
    #   - Reciprocal_Change_Index, ST_Abnormal_Lead_Count,
    #     ST_Abnormal_Territory_Count, ST_Diffuseness_Index
    #   - aVR_ST_Amplitude_mV / Shape / T_Ratio
    # =========================================================================

    # --- A) QRS-ST Concordance ---
    # Apr28 calls estimate_qrs_axis here (strict "QRS" filter); the value
    # gets overwritten by Cell 9E's compute_qrs_axis_features (full QRS
    # family) in build_final_record. Item 4.5.5 (2026-05-19) restored
    # exact Apr28 wire-in by switching from calculate_qrs_axis to
    # estimate_qrs_axis here.
    qrs_axis_deg = estimate_qrs_axis(lead_fits)
    st_axis_deg_val = estimate_st_axis(st_results)
    qrs_st_conc = compute_qrs_st_concordance(qrs_axis_deg, st_axis_deg_val)
    computed_metrics["QRS_ST_Concordance"] = qrs_st_conc
    computed_metrics["QRS_Axis_Deg"] = qrs_axis_deg

    # --- B) Per-territory dominant shape and mean slope ---
    for ter_name, ter_leads in territory_map.items():
        shapes_in_group: List[str] = []
        slopes_in_group: List[float] = []
        for ld in ter_leads:
            if ld in st_results:
                shapes_in_group.append(st_results[ld].st_shape)
                slopes_in_group.append(st_results[ld].st_slope * 1000.0)
        if shapes_in_group:
            shape_counts = Counter(shapes_in_group)
            computed_metrics[f"ST_{ter_name}_dominant_shape"] = (
                shape_counts.most_common(1)[0][0]
            )
        else:
            computed_metrics[f"ST_{ter_name}_dominant_shape"] = "indeterminate"
        if slopes_in_group:
            computed_metrics[f"ST_{ter_name}_mean_slope_uv_per_ms"] = float(
                np.mean(slopes_in_group)
            )
        else:
            computed_metrics[f"ST_{ter_name}_mean_slope_uv_per_ms"] = 0.0

    # --- C) Brugada pattern detection ---
    brugada_result = detect_brugada_pattern(st_results)
    computed_metrics["brugada_pattern"] = brugada_result

    # --- D) Reciprocal pattern detection (territory pairs) ---
    reciprocal_pairs_def = [
        ("anterior", ["V1", "V2", "V3", "V4"], "inferior", ["II", "III", "aVF"]),
        ("lateral", ["I", "aVL", "V5", "V6"], "inferior", ["II", "III", "aVF"]),
    ]
    reciprocal_pattern: Dict[str, Any] = {"detected": False, "pairs": []}
    for name_a, leads_a, name_b, leads_b in reciprocal_pairs_def:
        elev_a = sum(
            1
            for l in leads_a
            if l in st_results
            and st_results[l].st_elevation
            >= get_lead_elevation_threshold(l, age, sex)
        )
        depr_b = sum(
            1
            for l in leads_b
            if l in st_results
            and st_results[l].st_depression <= get_lead_depression_threshold(l)
        )
        elev_b = sum(
            1
            for l in leads_b
            if l in st_results
            and st_results[l].st_elevation
            >= get_lead_elevation_threshold(l, age, sex)
        )
        depr_a = sum(
            1
            for l in leads_a
            if l in st_results
            and st_results[l].st_depression <= get_lead_depression_threshold(l)
        )
        if elev_a >= 1 and depr_b >= 1:
            reciprocal_pattern["detected"] = True
            reciprocal_pattern["pairs"].append({
                "elevation_territory": name_a,
                "elevation_leads": [
                    l for l in leads_a
                    if l in st_results
                    and st_results[l].st_elevation
                    >= get_lead_elevation_threshold(l, age, sex)
                ],
                "depression_territory": name_b,
                "depression_leads": [
                    l for l in leads_b
                    if l in st_results
                    and st_results[l].st_depression
                    <= get_lead_depression_threshold(l)
                ],
            })
        if elev_b >= 1 and depr_a >= 1:
            reciprocal_pattern["detected"] = True
            reciprocal_pattern["pairs"].append({
                "elevation_territory": name_b,
                "elevation_leads": [
                    l for l in leads_b
                    if l in st_results
                    and st_results[l].st_elevation
                    >= get_lead_elevation_threshold(l, age, sex)
                ],
                "depression_territory": name_a,
                "depression_leads": [
                    l for l in leads_a
                    if l in st_results
                    and st_results[l].st_depression
                    <= get_lead_depression_threshold(l)
                ],
            })
    computed_metrics["st_reciprocal_pattern"] = reciprocal_pattern

    # --- E) Measurement offset for audit trail ---
    if global_estimated_qt <= 130:
        meas_offset = 80
    elif global_estimated_qt <= 180:
        meas_offset = 100
    elif global_estimated_qt <= 200:
        meas_offset = 140
    elif global_estimated_qt <= 220:
        meas_offset = 160
    elif global_estimated_qt <= 240:
        meas_offset = 180
    else:
        meas_offset = 190
    computed_metrics["st_measurement_offset_ms"] = meas_offset

    # =========================================================================
    # ROUND 2: Composite Discrimination Features
    # =========================================================================

    # --- a) ST_Discrimination_Score ---
    qrs_conc = computed_metrics.get("QRS_ST_Concordance", 0.0)
    inf_leads = ["II", "III", "aVF"]
    inf_angles: List[float] = []
    inf_curvatures: List[float] = []
    plm = computed_metrics.get("per_lead_morphology", {})
    for ld in inf_leads:
        if ld in plm:
            inf_angles.append(plm[ld].get("st_t_junction_angle", 0.0))
            inf_curvatures.append(plm[ld].get("curvature", 0.0))
    mean_inf_angle = float(np.mean(inf_angles)) if inf_angles else 0.0
    mean_inf_curv = float(np.mean(inf_curvatures)) if inf_curvatures else 0.0
    # angle_component: large angle (drop-down) -> negative score
    # curv_component: high curvature (drop-down) -> negative score
    # qrs_conc: positive (concordant) -> extension
    angle_component = -mean_inf_angle * 2.0
    curv_component = -mean_inf_curv * 3.0
    computed_metrics["ST_Discrimination_Score"] = round(
        0.5 * qrs_conc + 0.3 * angle_component + 0.2 * curv_component, 4
    )

    # --- b) ST_Territory_Contrast ---
    inf_dev = computed_metrics.get("Norm_ST_inferior_mean", 0.0)
    ant_dev = computed_metrics.get("Norm_ST_anterior_mean", 0.0)
    computed_metrics["ST_Territory_Contrast"] = round(inf_dev - ant_dev, 4)

    # --- c) ST_Inferior_Morphology_Index ---
    inf_slopes: List[float] = []
    for ld in inf_leads:
        if ld in plm:
            inf_slopes.append(plm[ld].get("slope_uv_per_ms", 0.0))
    mean_inf_slope = float(np.mean(inf_slopes)) if inf_slopes else 0.0
    conc_sign = 1.0 if qrs_conc >= 0 else -1.0
    computed_metrics["ST_Inferior_Morphology_Index"] = round(
        conc_sign * (mean_inf_slope - mean_inf_angle), 4
    )

    # =========================================================================
    # ROUND 3: ST-only discrimination features (Cell 11)
    # =========================================================================

    # --- d) Reciprocal Change Index (Brady 2003) ---
    ter_signs: Dict[str, float] = {}
    for ter_name, ter_leads in [
        ("inferior", ["II", "III", "aVF"]),
        ("anterior", ["V1", "V2", "V3", "V4"]),
        ("lateral", ["I", "aVL", "V5", "V6"]),
    ]:
        devs = [
            st_results[l].st_elevation + st_results[l].st_depression
            for l in ter_leads
            if l in st_results
        ]
        if devs:
            ter_signs[ter_name] = float(np.mean(devs))
    reciprocal_count = 0
    territory_names = list(ter_signs.keys())
    for i in range(len(territory_names)):
        for j in range(i + 1, len(territory_names)):
            if ter_signs[territory_names[i]] * ter_signs[territory_names[j]] < 0:
                reciprocal_count += 1
    computed_metrics["Reciprocal_Change_Index"] = round(reciprocal_count / 3.0, 4)

    # --- e) ST Diffuseness Index (Wang 2003 NEJM) ---
    n_abnormal = sum(
        1 for _l, f in st_results.items() if getattr(f, "is_abnormal", False)
    )
    n_territories_abnormal = 0
    for ter_name, ter_leads in [
        ("inferior", ["II", "III", "aVF"]),
        ("anterior", ["V1", "V2", "V3", "V4"]),
        ("lateral", ["I", "aVL", "V5", "V6"]),
    ]:
        if any(
            getattr(st_results.get(l), "is_abnormal", False)
            for l in ter_leads
            if l in st_results
        ):
            n_territories_abnormal += 1
    computed_metrics["ST_Abnormal_Lead_Count"] = n_abnormal
    computed_metrics["ST_Abnormal_Territory_Count"] = n_territories_abnormal
    computed_metrics["ST_Diffuseness_Index"] = round(
        n_abnormal / max(len(st_results), 1), 4
    )

    # --- f) aVR ST Deviation (Williamson 2006) ---
    if "aVR" in st_results:
        avr = st_results["aVR"]
        computed_metrics["aVR_ST_Amplitude_mV"] = round(
            avr.st_elevation + avr.st_depression, 4
        )
        computed_metrics["aVR_ST_Shape"] = avr.st_shape
        avr_st_t = getattr(avr, "st_t_ratio", 0.0)
        computed_metrics["aVR_ST_T_Ratio"] = round(avr_st_t, 4)

    if patterns is None:
        patterns = {}
    if summary is None:
        summary = {"primary_diagnosis": "Incomplete", "urgency": "normal"}

    return {
        "lead_features": st_results,
        "patterns": patterns,
        "summary": summary,
        "computed_metrics": computed_metrics,
        "clinical_alert": any(f.is_abnormal for f in st_results.values()),
        "estimated_qt_ms": global_estimated_qt,
    }


# ---------------------------------------------------------------------------
# Multi-lead pattern detection
# ---------------------------------------------------------------------------

def detect_st_patterns(
    st_results: Dict[str, STSegmentFeatures],
    leads: List[str],
    estimated_qt_ms: float,
    has_global_depression: bool = False,
    age: int = 50,
    sex: str = "Male",
) -> Dict[str, Any]:
    """Detect multi-lead ST patterns using lead-specific clinical thresholds."""
    patterns: Dict[str, bool] = {
        "anterior_stemi": False,
        "inferior_stemi": False,
        "lateral_stemi": False,
        "posterior_stemi": False,
        "diffuse_st_depression": False,
        "pericarditis_pattern": False,
        "lvh_strain_pattern": False,
    }

    if estimated_qt_ms <= 140:
        offset = 60
    elif estimated_qt_ms <= 180:
        offset = 80
    elif estimated_qt_ms <= 220:
        offset = 100
    else:
        offset = 120
    attr = f"j_plus_{offset}_amplitude"

    def amp(lead: str) -> float:
        f = st_results.get(lead)
        return getattr(f, attr, 0.0) if f else 0.0

    # Lead-specific threshold using AHA/ACC/ESC guidelines
    def get_stemi_thresh(lead: str) -> float:
        # Use the guideline-based function which accounts for age/sex in V2/V3
        base_thr = get_lead_elevation_threshold(lead, age, sex)
        # Context modifier: when global depression is present AND we are in V1/V2,
        # use the higher "high take-off" threshold for specificity
        if lead in ("V1", "V2") and has_global_depression:
            return max(base_thr, ST_ELEVATION_V1V2_HIGH)
        return base_thr

    def get_depr_thresh(lead: str) -> float:
        return get_lead_depression_threshold(lead)

    anterior = ["V1", "V2", "V3", "V4"]
    inferior = ["II", "III", "aVF"]
    lateral = ["I", "aVL", "V5", "V6"]

    if sum(1 for l in anterior if amp(l) >= get_stemi_thresh(l)) >= 2:
        patterns["anterior_stemi"] = True
    if sum(1 for l in inferior if amp(l) >= get_stemi_thresh(l)) >= 2:
        patterns["inferior_stemi"] = True
    if sum(1 for l in lateral if amp(l) >= get_stemi_thresh(l)) >= 2:
        patterns["lateral_stemi"] = True

    if sum(1 for l in ["V1", "V2", "V3"] if amp(l) <= get_depr_thresh(l)) >= 2:
        patterns["posterior_stemi"] = True
    if sum(1 for l in leads if amp(l) <= get_depr_thresh(l)) >= 6:
        patterns["diffuse_st_depression"] = True
    if (
        sum(
            1
            for l in leads
            if amp(l) > get_lead_elevation_threshold(l, age, sex) / 2
        )
        >= 8
    ):
        patterns["pericarditis_pattern"] = True
    if (
        sum(
            1
            for l in ["V5", "V6", "I", "aVL"]
            if l in st_results
            and amp(l) <= get_depr_thresh(l)
            and st_results[l].st_shape in ["downsloping", "scooped"]
        )
        >= 2
    ):
        patterns["lvh_strain_pattern"] = True

    return patterns


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_st_summary(
    st_results: Dict[str, STSegmentFeatures],
    patterns: Dict[str, Any],
    has_global_depression: bool = False,
    age: int = 50,
    sex: str = "Male",
) -> Dict[str, Any]:
    """Generate summary incorporating age/sex-specific logic.

    Tracks max_elevation and max_depression to prevent pipeline errors.
    """
    summary: Dict[str, Any] = {
        "abnormal_leads": [],
        "primary_diagnosis": "Normal ST segments",
        "urgency": "routine",
        "discriminators": {},
        "max_elevation": 0.0,
        "max_depression": 0.0,
    }

    # 1. Aggregate and interpret individual leads
    total_elev_load = 0.0
    total_depr_load = 0.0

    max_elev = 0.0
    max_depr = 0.0

    for lead, f in st_results.items():
        # Track maximums across ALL leads (regardless of abnormality status)
        if f.st_elevation > max_elev:
            max_elev = f.st_elevation
        if f.st_depression < max_depr:
            max_depr = f.st_depression

        # Pass age/sex to the interpreter
        f.interpretation_dict = interpret_st_changes(f, has_global_depression, age, sex)

        # Check if the result was abnormal based on new thresholds
        is_crit = f.interpretation_dict["severity"] in ["critical", "abnormal"]

        if is_crit:
            f.is_abnormal = True  # Force update flag based on new rules
            interp = ", ".join(f.interpretation_dict["classification"])
            f.clinical_interpretation = interp

            summary["abnormal_leads"].append(
                {
                    "lead": lead,
                    "finding": interp,
                    "st_t_ratio": getattr(f, "st_t_ratio", 0.0),
                    "consistency": getattr(f, "st_consistency", "indeterminate"),
                    "classification": f.interpretation_dict["classification"],
                }
            )

            # Accumulate loads for balance score
            if f.st_elevation > 0:
                total_elev_load += f.st_elevation
            if f.st_depression < 0:
                total_depr_load += abs(f.st_depression)

    # Set the max values in the summary
    summary["max_elevation"] = max_elev
    summary["max_depression"] = max_depr

    # 2. Calculate discriminators
    #    a) Raw mV balance (original)
    dev_balance = total_elev_load - total_depr_load
    summary["discriminators"]["balance"] = dev_balance

    #    b) Normalized balance (NEW: accounts for lead-specific thresholds)
    #       Sum of positive normalized deviations vs sum of |negative| ones
    norm_elev_total = 0.0
    norm_depr_total = 0.0
    norm_elev_leads: List[str] = []
    norm_depr_leads: List[str] = []
    max_norm_elev = 0.0
    max_norm_depr = 0.0
    for lead, f in st_results.items():
        nd = getattr(f, "normalized_deviation", 0.0)
        if nd > 0:
            norm_elev_total += nd
            norm_elev_leads.append(lead)
            max_norm_elev = max(max_norm_elev, nd)
        elif nd < 0:
            norm_depr_total += abs(nd)
            norm_depr_leads.append(lead)
            max_norm_depr = max(max_norm_depr, abs(nd))

    norm_balance_denom = max(norm_elev_total + norm_depr_total, 1e-6)
    # Range [-1, +1]: positive = elevation dominant, negative = depression dominant
    norm_polarity_balance = (norm_elev_total - norm_depr_total) / norm_balance_denom

    summary["discriminators"]["norm_balance"] = norm_polarity_balance
    summary["discriminators"]["norm_elev_total"] = norm_elev_total
    summary["discriminators"]["norm_depr_total"] = norm_depr_total
    summary["discriminators"]["norm_elev_count"] = len(norm_elev_leads)
    summary["discriminators"]["norm_depr_count"] = len(norm_depr_leads)
    summary["discriminators"]["max_norm_elev"] = max_norm_elev
    summary["discriminators"]["max_norm_depr"] = max_norm_depr

    # 3. Pattern recognition (contiguity check)
    contiguous_sets = [
        ["V1", "V2", "V3"],
        ["V2", "V3", "V4"],
        ["V3", "V4", "V5"],
        ["V4", "V5", "V6"],  # Anterior
        ["I", "aVL"],
        ["V5", "V6"],  # Lateral
        ["II", "III", "aVF"],  # Inferior
    ]

    has_contiguous_ste = False
    for group in contiguous_sets:
        # Count how many leads in this group have critical elevation
        count = sum(
            1
            for l in group
            if l in st_results
            and st_results[l].is_abnormal
            and "Elevation" in st_results[l].clinical_interpretation
            and "High Take-off" not in st_results[l].clinical_interpretation
        )
        if count >= 2:
            has_contiguous_ste = True
            break

    # 4. Final diagnosis logic
    #    Uses BOTH the raw mV balance AND the normalized balance.
    #    The normalized balance resolves the "V2 elevation + V1/V3 depression"
    #    ambiguity by weighting each lead's deviation against its own clinical
    #    threshold.  A patient with 0.18 mV in V2 (man >= 40: norm = 0.90) and
    #    -0.08 mV in V1 (norm = -1.60) will now show a *negative* normalized
    #    balance despite having a positive raw mV balance.

    if has_contiguous_ste:
        # Check for "extension" vs "ischaemia" using normalized balance
        # norm_polarity_balance < -0.25 means depression is dominant after
        # accounting for the higher V2/V3 thresholds
        if norm_polarity_balance < -0.25 or total_depr_load > (total_elev_load * 2.0):
            summary["primary_diagnosis"] = "ST Drop Down (Depression Dominant)"
            summary["urgency"] = "URGENT"
        else:
            summary["primary_diagnosis"] = "ST Extension (Acute Elevation Pattern)"
            summary["urgency"] = "EMERGENCY"

    elif norm_depr_total > 1.0 and norm_polarity_balance < -0.30:
        # Normalized depression is clinically significant across leads
        # even if raw mV values are small (e.g., multiple leads just above 0.05 mV)
        summary["primary_diagnosis"] = "ST Drop Down (Ischemic Depression)"
        summary["urgency"] = "URGENT"

    elif total_depr_load > 0.10 and dev_balance < -0.1:
        # Fallback to original raw mV logic
        summary["primary_diagnosis"] = "ST Drop Down (Ischemic Depression)"
        summary["urgency"] = "URGENT"

    elif max_norm_elev >= 1.0 and max_elev > 0.05:
        # At least one lead crosses its own elevation threshold
        if has_contiguous_ste:
            summary["primary_diagnosis"] = "ST Extension (Acute Elevation Pattern)"
            summary["urgency"] = "EMERGENCY"
        else:
            summary["primary_diagnosis"] = "ST Elevation (Non-contiguous / Benign)"
            summary["urgency"] = "abnormal"

    elif max_elev > 0.1:
        summary["primary_diagnosis"] = "ST Elevation (Non-contiguous / Benign)"
        summary["urgency"] = "abnormal"

    elif summary["abnormal_leads"]:
        # Fallback if we have abnormal leads but no major pattern
        summary["primary_diagnosis"] = "Non-specific ST changes"
        summary["urgency"] = "abnormal"

    return summary


# ---------------------------------------------------------------------------
# Plotting  (diagnostic / debugging)
# ---------------------------------------------------------------------------

def plot_st_segment_analysis(
    trace: np.ndarray,
    time_ms: np.ndarray,
    st_features: STSegmentFeatures,
    lead_name: str,
    estimated_qt_ms: float,
    save_path: Optional[str] = None,
) -> None:
    """Plot ST segment for a single lead.

    Diagnostic (black) point uses the same QT -> offset mapping as analysis.
    """
    import matplotlib.patches as patches  # noqa: F401 (available for callers)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(time_ms, trace, "b-", lw=1.5, label=f"Lead {lead_name}")

    j_idx = np.argmin(np.abs(time_ms - st_features.j_point_ms))
    ax1.plot(st_features.j_point_ms, trace[j_idx], "ro", markersize=8, label="J-point")

    # ---- same offset mapping as analysis ------------------------------------
    if estimated_qt_ms <= 130:
        chosen_offset = 80
    elif estimated_qt_ms <= 180:
        chosen_offset = 100
    elif estimated_qt_ms <= 200:
        chosen_offset = 140
    elif estimated_qt_ms <= 220:
        chosen_offset = 160
    elif estimated_qt_ms <= 240:
        chosen_offset = 180
    else:
        chosen_offset = 190

    chosen_amp = getattr(st_features, f"j_plus_{chosen_offset}_amplitude", 0.0)

    # ---- draw diagnostic point ----------------------------------------------
    t_chosen = st_features.j_point_ms + chosen_offset
    idx_chosen = np.argmin(np.abs(time_ms - t_chosen))
    ax1.plot(
        t_chosen,
        trace[idx_chosen],
        "o",
        mfc="none",
        mec="black",
        mew=2,
        ms=8,
        label=f"J+{chosen_offset}",
    )

    # ---- highlight ST window and baseline -----------------------------------
    st_start = st_features.j_point_ms
    st_end = st_features.j_point_ms + 250.0
    ax1.axvspan(st_start, st_end, alpha=0.2, color="yellow", label="ST segment")

    baseline_idx = np.argmin(np.abs(time_ms + 100))
    baseline = trace[baseline_idx] if 0 <= baseline_idx < len(trace) else 0.0
    ax1.axhline(baseline, color="gray", ls="--", alpha=0.5, label="Baseline")

    # ---- annotation box -----------------------------------------------------
    interp_text = (
        f"Shape: {st_features.st_shape}\n"
        f"Chosen: J+{chosen_offset} = {chosen_amp * 10:.1f} mm\n"
        f"Slope: {st_features.st_slope * 1000:.2f} \u00b5V/ms\n"
        f"{st_features.clinical_interpretation}"
    )
    ax1.text(
        0.02,
        0.98,
        interp_text,
        transform=ax1.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax1.set_ylabel("Amplitude (mV)")
    ax1.set_title(f"ST-Segment Analysis \u2014 Lead {lead_name}")
    ax1.legend(loc="upper right", fontsize="small")
    ax1.grid(True, alpha=0.3)

    # ---- zoomed-in view -----------------------------------------------------
    st_mask = (time_ms >= st_start) & (time_ms <= st_end)
    if np.any(st_mask):
        ax2.plot(time_ms[st_mask], trace[st_mask], "b-", lw=2)
        ax2.axhline(baseline, color="gray", ls="--", alpha=0.5)
        ax2.plot(
            t_chosen,
            trace[idx_chosen],
            "o",
            mfc="none",
            mec="black",
            mew=2,
            ms=8,
        )
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("ST Amplitude (mV)")
        ax2.set_title(f"ST-Segment Detail \u2014 J+{chosen_offset} ms diagnostic point")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   ST plot saved to: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Context-based ST info helper
# ---------------------------------------------------------------------------

def get_st_segment_info(
    ctx: Any,
    lead: str,
    *,
    j_point_method: str = "derivative",
) -> Dict[str, float]:
    """Per-patient, per-lead ST-segment landmarks (J-point, T-onset, duration).

    Args:
        ctx: FeatureContext with ``lead_fits``, ``time_ms``, ``stack_sync``,
            ``leads``.
        lead: Lead name (e.g. ``"V2"``).
        j_point_method: One of ``"derivative"``, ``"tangent"``, or any other
            value (falls back to ``S_center + 2*sigma``). See
            ``find_j_point_precise`` docstring. Defaults to ``"derivative"``
            to match the precedent in ``analyze_st_segment_comprehensive``.

    Returns:
        ``{"j_point": j_point_ms, "t_start": t_onset_ms, "duration": ms}``.
        Returns ``{nan, nan, nan}`` when required ctx fields are missing or
        the Gaussian fits don't contain QRS / T components for this lead —
        QPSI is an unbiased feature space, so missing inputs surface as
        NaN rather than hardcoded clinical placeholders. Downstream rules
        in ``global_features.py`` evaluate False on NaN comparisons, which
        preserves the "silently skip" behaviour they relied on under the
        former stub for missing data without baking in a fake value.

    Notes:
        - J-point detection bridges ``find_j_point_precise`` (line ~186):
          extracts ``avg_trace = nanmean(ctx.stack_sync[:, lead_idx, :])``,
          ``time_ms = ctx.time_ms``, and the ``wave_type == "QRS"``
          Gaussians from ``ctx.lead_fits[lead]["avg"]`` (same filter
          ``analyze_st_segment_comprehensive`` uses at line ~637).
        - T-onset detection delegates to ``_t_amp_and_duration`` in
          ``helpers.py`` which computes ``min(center_ms - 2*sigma_ms)``
          over the lead's T Gaussians — the canonical T-onset definition
          used elsewhere in qpsi.
        - Duration is ``t_start - j_point``; negative deltas (T fit lands
          before J-point, indicating a likely misfit) clamp to NaN rather
          than 0.

    History:
        C0.6 / Q-A6-1 stub (returning hardcoded {80, 200, 120}) replaced
        2026-05-18 under K3 (the private training repository IS the GT). Three previously-
        silent clinical rules in ``global_features.py`` (:860 athletes,
        :912 STEMI overlay, :963 metabolic) now fire on real measurements.
        FIDELITY_REPORT.md entry updated from MATCH(stub) → MATCH(real).
    """
    nan_result: Dict[str, float] = {
        "j_point": float("nan"),
        "t_start": float("nan"),
        "duration": float("nan"),
    }

    lead_fits = getattr(ctx, "lead_fits", None) or {}
    if lead not in lead_fits:
        return nan_result
    leads = getattr(ctx, "leads", None) or []
    if lead not in leads:
        return nan_result
    stack_sync = getattr(ctx, "stack_sync", None)
    time_ms = getattr(ctx, "time_ms", None)
    if stack_sync is None or time_ms is None:
        return nan_result

    lead_idx = leads.index(lead)
    if lead_idx >= stack_sync.shape[1]:
        return nan_result

    avg_data = lead_fits[lead].get("avg", [])
    qrs_fits = [f for f in avg_data if f.get("wave_type") == "QRS"]
    t_fits = [f for f in avg_data if f.get("wave_type") == "T"]
    if not qrs_fits or not t_fits:
        return nan_result

    # Prefer cached per-lead average from pipeline; fall through to a
    # single reduction when ctx didn't pre-populate it.
    cached_avg = getattr(ctx, "avg_per_lead", None)
    if cached_avg is not None and lead_idx < cached_avg.shape[0]:
        avg_trace = cached_avg[lead_idx, :]
    else:
        avg_trace = np.nanmean(stack_sync[:, lead_idx, :], axis=0)
    if not np.any(np.isfinite(avg_trace)):
        return nan_result

    j_point_ms, _ = find_j_point_precise(
        avg_trace, np.asarray(time_ms), qrs_fits, method=j_point_method
    )
    _, t_start_ms, _, _ = _t_amp_and_duration(t_fits)

    if t_start_ms is None or not np.isfinite(j_point_ms):
        return nan_result

    duration_ms = float(t_start_ms) - float(j_point_ms)
    if duration_ms < 0:
        # T-fit landed before J-point — likely Gaussian misfit on this lead.
        # Surface as NaN rather than negative-duration noise.
        return nan_result

    return {
        "j_point": float(j_point_ms),
        "t_start": float(t_start_ms),
        "duration": duration_ms,
    }
