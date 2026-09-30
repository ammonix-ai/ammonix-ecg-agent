"""
JSON writer -- output assembly for QPSI pipeline results.

Source: Cell 17 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (lines 426-494, 3089-3464)

Key functions:
    build_final_record(): Assembles the final JSON output structure
    _build_plane_ecg_data(): Per-plane ECG data with wave structure + SNR gating
    _build_computed_parameters(): Clinical intervals, RR stats, AF metrics
    compare_wave_variation(): Compare two wave dicts (ref vs current beat)
    summarize_variability(): Aggregate beat-to-beat variation stats

Dependencies (all Layer 0-2):
    constants: ensure_json_serializable, round_json_values, logger
    wave_classification: compute_clinical_intervals
    clinical_summaries: build_clinical_summaries_tierA
    wfdb_io: SNOMED_TO_TEXT
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from qpsi.constants import logger, ensure_json_serializable, round_json_values
from qpsi.wave_classification import compute_clinical_intervals
from qpsi.clinical_summaries import build_clinical_summaries_tierA
from qpsi.features.lvh_axis import (
    compute_lvh_voltage_features,
    compute_qrs_axis_features,
)
from qpsi.wfdb_io import SNOMED_TO_TEXT

# ---------------------------------------------------------------------------
# Wave-variation helpers (Cell 17, lines 426-494)
# ---------------------------------------------------------------------------

def _ang_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Absolute angular difference in degrees, handling wraparound."""
    if a is None or b is None:
        return None
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def _nested_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Return the value for the first key found in *d*, else *default*.

    Renamed from ``_g`` in Cell 17 to avoid collision with the identically-
    named helper in ``features/global_features.py``.
    """
    for k in keys:
        if k in d:
            return d[k]
    return default


def compare_wave_variation(ref_wave: dict, cur_wave: dict) -> dict:
    """Compare current wave to a reference wave (avg-beat),
    returning deltas robust to missing keys.
    """
    amp_ref = _nested_get(ref_wave, "amp_mv", "amplitude_mv")
    amp_cur = _nested_get(cur_wave, "amp_mv", "amplitude_mv")
    dur_ref = _nested_get(ref_wave, "duration_ms", "width_ms")
    dur_cur = _nested_get(cur_wave, "duration_ms", "width_ms")
    tpk_ref = _nested_get(ref_wave, "t_peak_ms", "peak_time_ms", "tp_ms")
    tpk_cur = _nested_get(cur_wave, "t_peak_ms", "peak_time_ms", "tp_ms")
    ax_ref = _nested_get(ref_wave, "axis_deg", "angle_deg")
    ax_cur = _nested_get(cur_wave, "axis_deg", "angle_deg")

    res: Dict[str, Any] = {
        "present": True,
        "amp_diff_mv": (
            None if (amp_ref is None or amp_cur is None)
            else float(amp_cur) - float(amp_ref)
        ),
        "duration_diff_ms": (
            None if (dur_ref is None or dur_cur is None)
            else float(dur_cur) - float(dur_ref)
        ),
        "peak_time_diff_ms": (
            None if (tpk_ref is None or tpk_cur is None)
            else float(tpk_cur) - float(tpk_ref)
        ),
        "axis_diff_deg": _ang_diff_deg(ax_cur, ax_ref),
        "polarity_flip": False,
        "notching_change": (
            bool(_nested_get(cur_wave, "notched", "notch", default=False))
            != bool(_nested_get(ref_wave, "notched", "notch", default=False))
        ),
    }

    if amp_ref is not None and amp_cur is not None:
        res["polarity_flip"] = float(amp_ref) * float(amp_cur) < 0

    area_ref = _nested_get(ref_wave, "area")
    area_cur = _nested_get(cur_wave, "area")
    res["auc_diff"] = (
        None if (area_ref is None or area_cur is None)
        else float(area_cur) - float(area_ref)
    )
    return res


def summarize_variability(diffs: List[dict]) -> dict:
    """Aggregate variability over a set of per-beat diffs.

    Returns compact stats (mean/sd and event fractions).
    """
    if not diffs:
        return {}

    def _nz(seq: List[Any]) -> np.ndarray:
        vals = [x for x in seq if x is not None]
        return np.array(vals, float) if vals else np.array([], float)

    amp = _nz([d.get("amp_diff_mv") for d in diffs])
    dur = _nz([d.get("duration_diff_ms") for d in diffs])
    axis = _nz([d.get("axis_diff_deg") for d in diffs])
    auc = _nz([d.get("auc_diff") for d in diffs])
    flips = sum(1 for d in diffs if d.get("polarity_flip"))

    return {
        "n_beats": len(diffs),
        "amp_diff_mean_mv": float(amp.mean()) if amp.size else None,
        "amp_diff_sd_mv": float(amp.std()) if amp.size else None,
        "duration_diff_sd_ms": float(dur.std()) if dur.size else None,
        "axis_diff_sd_deg": float(axis.std()) if axis.size else None,
        "auc_diff_sd": float(auc.std()) if auc.size else None,
        "polarity_flip_frac": flips / len(diffs),
    }


# ---------------------------------------------------------------------------
# JSON output helpers (Cell 17, lines 3331-3369)
# ---------------------------------------------------------------------------

def _safe_get(dct: Optional[dict], key: str, default: float = 0.0) -> Any:
    """Return *dct[key]* or *default* if key is missing / value is None."""
    if dct is None:
        return default
    val = dct.get(key, default)
    return default if val is None else val


def _collect(col_name: str, coll: List[dict]) -> List[Any]:
    """Collect *col_name* from all dicts in *coll* (only those that contain it)."""
    return [c[col_name] for c in coll if col_name in c]


def _build_wave_block(avg_wave: Optional[dict], vars_list: List[dict]) -> dict:
    """Compose the JSON sub-structure for one wave type (P / QRS / T)."""
    avg_wave = avg_wave or {}
    avg = {
        "start time (ms)": round(_safe_get(avg_wave, "start_time"), 2),
        "peak time (ms)": round(_safe_get(avg_wave, "peak_time"), 2),
        "end time (ms)": round(_safe_get(avg_wave, "end_time"), 2),
        "amplitude (mV)": round(_safe_get(avg_wave, "amplitude_mV"), 2),
        "angle (deg)": round(_safe_get(avg_wave, "angle_degrees"), 2),
    }
    wav_var = {
        "count": len(vars_list),
        "amplitude mean variation": round(
            float(np.mean(_collect("Amplitude_MV", vars_list)))
            if _collect("Amplitude_MV", vars_list) else 0.0,
            2,
        ),
        "peak time mean variation": round(
            float(np.mean(_collect("PeakTime Variation (ms)", vars_list)))
            if _collect("PeakTime Variation (ms)", vars_list) else 0.0,
            2,
        ),
        "angle mean variation": round(
            float(np.mean(_collect("Angle_deg", vars_list)))
            if _collect("Angle_deg", vars_list) else 0.0,
            2,
        ),
    }
    return {"average": avg, "variations": wav_var}


# ---------------------------------------------------------------------------
# Per-plane ECG data builder (Cell 17, lines 3089-3165)
# ---------------------------------------------------------------------------

def _build_plane_ecg_data(
    lumps: List[dict],
    p_vars: List[dict],
    t_vars: List[dict],
    extra_beats: List[dict],
    arrhythmic_beats: List[dict],
    rr_stats: dict,
    st_info: dict,
) -> Dict[str, Any]:
    """Build ECG data structure for one plane (limb or chest)."""

    p_wave = next((w for w in lumps if w["wave_type"] == "P"), None) or {}
    t_wave = next(
        (w for w in lumps if w["wave_type"] in ("T", "ST1", "T1", "ST")),
        None,
    ) or {}

    # -- SNR gating logic --------------------------------------------------
    p_amp_raw = float(p_wave.get("amplitude_mV", 0.0))
    if "p_info" in st_info and "snr_db" in st_info["p_info"]:
        p_snr = float(st_info["p_info"]["snr_db"])
        p_amp_gated = p_amp_raw if p_snr >= 1.5 else 0.0
        snr_display: Optional[float] = round(p_snr, 1)
    else:
        p_amp_gated = p_amp_raw
        snr_display = None

    # -- QRS components ----------------------------------------------------
    qrs_components: Dict[str, dict] = {}
    for name in ["Q", "R", "S"]:
        w = next((w for w in lumps if w["wave_type"] == name), None) or {}
        qrs_components[name] = {
            "amplitude mV": round(w.get("amplitude_mV", 0.0), 2),
            "duration ms": round(
                w.get("end_time", 0.0) - w.get("start_time", 0.0), 2
            ),
            "angle (\u00b0)": round(w.get("angle_degrees", 0.0), 2),
        }

    # -- P wave block (with SNR gating) ------------------------------------
    p_dict: Dict[str, Any] = {
        "average": {
            "start time (ms)": round(p_wave.get("start_time", 0.0), 2),
            "peak time (ms)": round(p_wave.get("peak_time", 0.0), 2),
            "end time (ms)": round(p_wave.get("end_time", 0.0), 2),
            "amplitude (mV)": round(p_amp_gated, 4),
            "angle (deg)": round(p_wave.get("angle_degrees", 0.0), 2),
        },
        "variations": {
            "count": len(p_vars),
            "amplitude mean variation": round(
                float(np.mean([v.get("Amplitude_MV", 0) for v in p_vars]))
                if p_vars else 0.0,
                2,
            ),
            "peak time mean variation": round(
                float(np.mean([v.get("PeakTime Variation (ms)", 0) for v in p_vars]))
                if p_vars else 0.0,
                2,
            ),
            "angle mean variation": round(
                float(np.mean([v.get("Angle_deg", 0) for v in p_vars]))
                if p_vars else 0.0,
                2,
            ),
        },
    }

    if snr_display is not None:
        p_dict["average"]["snr_db"] = snr_display

    # -- T wave block ------------------------------------------------------
    t_dict: Dict[str, Any] = {
        "average": {
            "start time (ms)": round(t_wave.get("start_time", 0.0), 2),
            "peak time (ms)": round(t_wave.get("peak_time", 0.0), 2),
            "end time (ms)": round(t_wave.get("end_time", 0.0), 2),
            "amplitude (mV)": round(t_wave.get("amplitude_mV", 0.0), 2),
            "angle (deg)": round(t_wave.get("angle_degrees", 0.0), 2),
        },
        "variations": {
            "count": len(t_vars),
            "amplitude mean variation": round(
                float(np.mean([v.get("Amplitude_MV", 0) for v in t_vars]))
                if t_vars else 0.0,
                2,
            ),
            "peak time mean variation": round(
                float(np.mean([v.get("PeakTime Variation (ms)", 0) for v in t_vars]))
                if t_vars else 0.0,
                2,
            ),
            "angle mean variation": round(
                float(np.mean([v.get("Angle_deg", 0) for v in t_vars]))
                if t_vars else 0.0,
                2,
            ),
        },
    }

    return {
        "waves": {
            "P": p_dict,
            "QRS": {"components": qrs_components},
            "T": t_dict,
        },
    }


# ---------------------------------------------------------------------------
# Computed parameters builder (Cell 17, lines 3168-3229)
# ---------------------------------------------------------------------------

def _build_computed_parameters(
    lumps_limb: List[dict],
    lumps_chest: List[dict],
    rr_stats: dict,
    extras_limb: dict,
    extras_chest: dict,
    extra_beats: List[dict],
    arrhythmic_beats: List[dict],
    total_beats: int,
    identical_beats: int,
    pacing_waves_total: int,
    computed_metrics: dict,
) -> Dict[str, Any]:
    """Build computed parameters section."""

    rr_mean = rr_stats.get("rr_mean_ms", 0.0)
    clin_limb = compute_clinical_intervals(lumps_limb, rr_ms=rr_mean)
    clin_chest = compute_clinical_intervals(lumps_chest, rr_ms=rr_mean)

    # QT: prefer limb plane; fall back to chest plane if limb has no T-wave
    qt_ms = clin_limb.get("QT_interval_ms", 0.0)
    if qt_ms <= 0:
        qt_ms = clin_chest.get("QT_interval_ms", 0.0)
    rr_ms = rr_stats.get("rr_mean_ms", 0.0)
    qtc_ms = 0.0
    if qt_ms > 0 and rr_ms > 0:
        qtc_ms = round(
            (qt_ms / 1000.0) / ((rr_ms / 1000.0) ** 0.5) * 1000.0, 1
        )

    params: Dict[str, Any] = {
        "PQ Interval ms": round(clin_limb.get("PQ_interval_ms", 0.0), 2),
        "QT Interval ms": round(qt_ms, 2),
        "QTc Interval ms (Bazett)": qtc_ms,
        "QRS Duration ms (limb)": round(clin_limb.get("QRS_interval_ms", 0.0), 2),
        "QRS Duration ms (chest)": round(clin_chest.get("QRS_interval_ms", 0.0), 2),
        "All RR Intervals std": round(rr_stats.get("rr_sd_ms_all", 0.0), 2),
        "Main RR Interval mean": round(rr_stats.get("rr_mean_ms", 0.0), 2),
        "Main RR Interval std": round(rr_stats.get("rr_sd_ms", 0.0), 2),
        "TotalBeats": total_beats,
        "IdenticalBeats": identical_beats,
        "PacingWavesDetected": pacing_waves_total,
        "Extras beats %": round(
            100.0 * len(extra_beats) / max(total_beats, 1), 1
        ),
        "Arrhythmic beats %": round(
            100.0 * len(arrhythmic_beats) / max(total_beats, 1), 1
        ),
        "Extra Beat QRS Duration ms": round(
            float(np.median([e["QRS Duration"] for e in extra_beats]))
            if extra_beats else 0.0,
            2,
        ),
        "ExtraBeat RR Duration ms": round(
            float(np.median([e["RR_ms"] for e in arrhythmic_beats]))
            if arrhythmic_beats else 0.0,
            2,
        ),
        "AF_Metrics_RR_CV": computed_metrics.get("AF_Metrics_RR_CV", 0.0),
        "AF_Metrics_RR_pNN50": computed_metrics.get("AF_Metrics_RR_pNN50", 0.0),
        "AF_Metrics_F_Wave_Coherence": computed_metrics.get(
            "AF_Metrics_F_Wave_Coherence", 0.0
        ),
        "AF_Metrics_F_Wave_Max_ACF": computed_metrics.get(
            "AF_Metrics_F_Wave_Max_ACF", 0.0
        ),
        "AF_Metrics_F_Wave_Freq_Hz": computed_metrics.get(
            "AF_Metrics_F_Wave_Freq_Hz", 0.0
        ),
        "AF_Metrics_Integer_Ratio_Prevalence": computed_metrics.get(
            "AF_Metrics_Integer_Ratio_Prevalence", 0.0
        ),
        "ST Amplitude (mV) (limb)": extras_limb.get("ST_amplitude_mV", 0.0),
        "ST Angle (\u00b0) (limb)": extras_limb.get("ST_angle", 0.0),
        "ST Amplitude (mV) (chest)": extras_chest.get("ST_amplitude_mV", 0.0),
        "ST Angle (\u00b0) (chest)": extras_chest.get("ST_angle", 0.0),
    }
    return params


# ---------------------------------------------------------------------------
# Patient demographics (Cell 17, lines 3231-3250)
# ---------------------------------------------------------------------------

def _build_patient_data(meta: dict) -> Dict[str, Any]:
    """Extract patient demographic data from record metadata."""
    age = meta.get("age")
    sex = meta.get("sex")
    if age is None or sex is None:
        for ln in meta.get("comments", []):
            txt = ln.lstrip("#").strip()
            if age is None and txt.startswith("Age:"):
                age = txt.split(":", 1)[1].strip()
            if sex is None and txt.startswith("Sex:"):
                sex = txt.split(":", 1)[1].strip()
    return {
        "Basic": {
            "Age": age if age is not None else "N/A",
            "Sex": sex if sex is not None else "N/A",
        },
    }


# ---------------------------------------------------------------------------
# Diagnosis code extraction (Cell 17, lines 3252-3271)
# ---------------------------------------------------------------------------

def _extract_diagnosis_codes(meta: dict) -> List[str]:
    """Extract and decode diagnosis codes from record metadata."""
    codes = meta.get("dx_codes", [])
    if not codes:
        for ln in meta.get("comments", []):
            if ln.lstrip("#").strip().upper().startswith("DX:"):
                codes = re.findall(r"\d{6,}", ln)
                break
    diag_text: List[str] = []
    for code in codes:
        text = SNOMED_TO_TEXT.get(code, code)
        diag_text.append(text.lower())
    return diag_text if diag_text else ["No diagnosis codes available"]


# ---------------------------------------------------------------------------
# Final record assembly (Cell 17, lines 3397-3464)
# ---------------------------------------------------------------------------

def build_final_record(
    *,
    ecg_data: Dict[str, Any],
    semantic_flags: Dict[str, Any],
    label_block: Dict[str, Any],
    multi_gaussian_parameters: Optional[Dict[str, Any]] = None,
    existing_semantic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the final JSON record.

    Parameters
    ----------
    ecg_data:
        Full ECG data dict (planes, computed parameters, etc.).
    semantic_flags:
        Semantic feature flags produced by the pipeline.
    label_block:
        Label / diagnosis block for this record.
    multi_gaussian_parameters:
        Optional multi-gaussian fit parameters to embed in ECG data.
    existing_semantic:
        Previously computed semantic block (used as fallback for MGP).

    Returns
    -------
    dict
        The final output record with keys ``"ECG data"``, ``"Label"``,
        ``"Semantic Features"``.
    """
    computed = (
        ecg_data.get("Computed Parameters", {})
        if isinstance(ecg_data, dict) else {}
    )
    poor_obj: Optional[Dict[str, Any]] = None
    if isinstance(semantic_flags, dict):
        g = semantic_flags.get("global", {})
        if isinstance(g, dict):
            poor_obj = g.get("poor_quality_ecg", None)

    clinical_summaries = build_clinical_summaries_tierA(
        semantic_flags, computed, poor_obj
    )

    quality_warning: Optional[str] = None
    if poor_obj and poor_obj.get("present"):
        issues = poor_obj.get("issues", [])
        issues_txt = f" Issues: {', '.join(issues)}." if issues else ""
        quality_warning = (
            "Poor ECG quality may degrade feature accuracy." + issues_txt
        )

    semantic: Dict[str, Any] = {
        "clinical_summaries": clinical_summaries,
        "semantic_flags": semantic_flags,
    }
    if quality_warning:
        semantic["quality_warning"] = quality_warning

    final_rec: Dict[str, Any] = {
        "ECG data": ecg_data.copy(),
        "Label": label_block,
        "Semantic Features": semantic,
    }

    # -- Multi-gaussian parameter promotion --------------------------------
    mgp = multi_gaussian_parameters
    if mgp is None and isinstance(existing_semantic, dict):
        mgp = existing_semantic.get("multi_gaussian_parameters")
    if mgp is None and "Semantic Features" in final_rec:
        mgp = final_rec["Semantic Features"].get("multi_gaussian_parameters")
    if mgp is not None:
        if "ECG data" not in final_rec or not isinstance(
            final_rec["ECG data"], dict
        ):
            final_rec["ECG data"] = {}
        final_rec["ECG data"]["multi lead parameters"] = mgp
        if "Semantic Features" in final_rec and isinstance(
            final_rec["Semantic Features"], dict
        ):
            final_rec["Semantic Features"].pop(
                "multi gaussian parameters", None
            )

    # -- Cell 9E axis + LVH voltage features (Apr28 Item 4.5.1) ----------
    if mgp:
        sex = (
            ((label_block or {}).get("Patient Data", {}) or {})
            .get("Basic", {})
            .get("Sex", "")
        )
        cp = final_rec["ECG data"].setdefault("Computed Parameters", {})
        try:
            cp.update(compute_qrs_axis_features(mgp))
            cp.update(compute_lvh_voltage_features(mgp, sex))
        except Exception as exc:
            logger.warning("axis/LVH wrapper skipped: %s", exc)

    return final_rec
