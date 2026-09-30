"""
Pipeline orchestrator -- 11-step ECG processing pipeline.

Source: Cell 17 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (lines 870-2010)

The run_pipeline() function takes raw 12-lead ECG data and produces a comprehensive
analysis result including wave decomposition, feature extraction, clinical summaries,
and serialized JSON output.

Pipeline steps:
    1. Preprocessing (filter, R-peaks, baseline correction)
    2. RR statistics (dual: "main" regular beats + "all" NN intervals)
    3. Beat padding
    4. Average-beat plane analysis (limb + chest via _analyse_plane)
    5. Per-lead Gaussian refits
    5.5. Energy trace analysis (AF vs APB -- Feb14)
    6. ST segment analysis
    6.5. J-point aware ST summary integration
    7. Per-beat analysis loop (variability, QWVA)
    7.5. FeatureContext creation + rhythm panorama
    7.6. Atrial f_wave detection + flutter metrics
    8. Semantic feature extraction (exclude ST) + post-processing
    9. Neural network predictions CLEARED (dormant)
    10. Output generation (build_final_record, JSONL write)
    11. Return comprehensive results

Dependencies: ALL lower-layer modules
"""
from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from qpsi.constants import (
    PipelineState,
    PLANE_IDX,
    FS_HZ,
    logger,
    ensure_json_serializable,
    P_amp_difference_threshold,
    P_time_difference_threshold,
    P_theta_difference_threshold,
    P_AMP_VARIABILITY_THRESHOLD_MV,
    T_amp_thr,
    T_dt_thr,
    T_dxy_thr,
)
from qpsi.preprocessing import (
    preprocess_ecg,
    detect_r_peaks,
    baseline_subtract_twoanchors,
    energy_trace,
)
from qpsi.segmentation import (
    segment_rr,
    synchronise_beats,
    pad_beats,
    robust_rr_from_rpeaks as _robust_rr_canonical,
)
from qpsi.beats import filter_beats_by_length, slice_beats_to_plane
from qpsi.plane_analysis import _analyse_plane, get_waveforms_from_lumps
from qpsi.lead_fitting import process_leads_with_proper_timing, analyse_energy_trace
from qpsi.lead_fitting_helpers import calculate_residuals_from_fits
from qpsi.wave_classification import (
    _ensure_wave_keys,
    compute_clinical_intervals,
    collect_p_variations,
    collect_t_variations,
)
from qpsi.features.st_deviation import (
    analyze_st_segments_all_leads,
    generate_st_summary,
)
from qpsi.features.context import FeatureContext
from qpsi.features.registry import extract_semantic_features, postprocess_semantic_flags
from qpsi.features.rhythm import (
    rhythm_panorama_from_ctx,
    rhythm_atrial_f_wave,
    rhythm_flutter_like,
    merge_panorama_into_semantics,
    analyse_complex_arrhythmias,
)
from qpsi.features.helpers import format_lead_list
from qpsi.qwva import (
    analyse_rat_variants,
    analyse_rvt_variants,
    summarize_qwva_results,
)
from qpsi.clinical_summaries import (
    build_clinical_summaries_tierA,
    generate_t_wave_clinical_summary,
    generate_p_wave_clinical_summary,
    generate_qrs_clinical_summary,
    generate_rhythm_clinical_summary,
    generate_global_clinical_summary,
)
from qpsi.json_writer import (
    build_final_record,
    _build_plane_ecg_data,
    _build_computed_parameters,
    _build_patient_data,
    _extract_diagnosis_codes,
    compare_wave_variation,
    summarize_variability,
)


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------

def _territory(leads_list: List[str]) -> str:
    """Map lead list to a territory name (anterior / inferior / lateral)."""
    if any(lead in leads_list for lead in ["V1", "V2", "V3", "V4"]):
        return "anterior"
    if any(lead in leads_list for lead in ["II", "III", "aVF"]):
        return "inferior"
    if any(lead in leads_list for lead in ["I", "aVL", "V5", "V6"]):
        return "lateral"
    return ""


def _count_p(lumps_list: List[List[Dict[str, Any]]]) -> int:
    """Count beats that contain at least one P-wave."""
    return sum(
        1 for lumps in lumps_list
        if any(w["wave_type"] == "P" for w in lumps)
    )


# ---------------------------------------------------------------------------
# H31 / Q-A6-6: robust_rr_from_rpeaks consolidated to qpsi.segmentation with
# kwargs (min_ms, max_ms, mad_trim). This wrapper locks Cell 13/17 kwargs
# (wide bounds [240, 3000], no MAD) and preserves the symbol so any external
# import of qpsi.pipeline.robust_rr_from_rpeaks keeps resolving.
# ---------------------------------------------------------------------------

def robust_rr_from_rpeaks(r_peaks: Any, fs: int) -> list:
    """Cell 13/17 wrapper — wide bounds [240, 3000] ms, no MAD trimming."""
    return _robust_rr_canonical(r_peaks, fs, min_ms=240.0, max_ms=3000.0, mad_trim=None)


# ---------------------------------------------------------------------------
# RR features (matches notebook Cell 17 inline _rr_features_nn — no trimming)
# ---------------------------------------------------------------------------

def _rr_features_nn(rr_ms_list: Any) -> Dict[str, Any]:
    """Return mean, SDNN, RMSSD, pNN50, CV for a list of RR (ms).

    This matches the notebook's inline implementation exactly:
    no percentile trimming, no outlier removal.
    """
    if not rr_ms_list or (hasattr(rr_ms_list, '__len__') and len(rr_ms_list) == 0):
        return {"mean_ms": 0.0, "sdnn_ms": 0.0, "rmssd_ms": 0.0, "pnn50": 0.0, "cv": 0.0}
    rr = np.asarray(rr_ms_list, dtype=float)
    diffs = np.diff(rr)
    sdnn = float(np.std(rr, ddof=0))
    rmssd = float(np.sqrt(np.mean(diffs**2))) if diffs.size else 0.0
    pnn50 = float((np.abs(diffs) >= 50.0).mean() * 100.0) if diffs.size else 0.0
    mean = float(rr.mean()) if rr.size else 0.0
    cv = float(sdnn / mean) if mean else 0.0
    return {"mean_ms": mean, "sdnn_ms": sdnn, "rmssd_ms": rmssd, "pnn50": pnn50, "cv": cv}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_ecg_12: np.ndarray,
    *,
    fs: int = 500,
    recording_id: str = "(Unnamed)",
    meta: Optional[Dict[str, Any]] = None,
    amplitude_threshold_unscaled: float = 0.04,
    plot_enabled: bool = True,
    enable_qwva: bool = True,
    output_jsonl_path: Optional[str] = None,
    serialize_variability: bool = False,
    offset_ms: int = 50,
    qpsi_config: Optional["QPSIConfig"] = None,
    pruning: str = "all",
) -> Dict[str, Any]:
    """Unified ECG analysis pipeline with integrated ST segment analysis.

    Writes JSON with: "ECG data", "Label", and simplified "Semantic Features".
    All plots controlled by *plot_enabled* parameter.

    Parameters
    ----------
    raw_ecg_12 : np.ndarray
        (12, N) array of raw ECG signals.
    fs : int
        Sampling rate in Hz (default 500).
    recording_id : str
        Human-readable ID for logging.
    meta : dict, optional
        Recording metadata (age, sex, lead_names, dx_codes, ...).
    amplitude_threshold_unscaled : float
        Relative amplitude threshold for wave fitting (default 0.04).
    plot_enabled : bool
        Enable/disable all diagnostic plots.
    enable_qwva : bool
        Enable QRS-Wave Variability Analysis (default True).
    output_jsonl_path : str, optional
        Path for JSONL output. Defaults to ``"q_psi_data.jsonl"``.
    serialize_variability : bool
        Include beat-to-beat variability detail in output (default False).
    offset_ms : int
        Timing window offset in ms (default 50).
    pruning : str
        Pruning level controlling which shortcuts are active:
        ``"none"`` — all fitting runs exhaustively (no shortcuts).
        ``"p_absence"`` — skip per-beat P-wave fitting when average P absent.
        ``"p_absence+qrs_clean"`` — + skip multi-Gaussian QRS when single is clean.
        ``"p_absence+qrs_clean+aicc"`` — + AICc early termination on K=2.
        ``"all"`` — all pruning active (production mode, default).

    Returns
    -------
    dict
        Comprehensive results dict with all intermediate products, or ``{}``
        if the recording has no usable beats.
    """

    # ===================================================================
    # 0. Initial Setup
    # ===================================================================
    pipeline_start = time.perf_counter()
    timings: Dict[str, float] = {}
    meta = meta or {}
    state = PipelineState(not_noisy=True, plot_enabled=plot_enabled)

    # QPSIConfig — toggleable feature flags (default: all OFF for backward compat)
    from qpsi.config import QPSIConfig
    if qpsi_config is None:
        qpsi_config = QPSIConfig()

    logger.info("=" * 70)
    logger.info("PROCESSING: %s", recording_id)
    logger.info("=" * 70)

    # ===================================================================
    # 1. Preprocessing
    # ===================================================================
    step_start = time.perf_counter()
    ecg_filt, not_noisy = preprocess_ecg(raw_ecg_12, fs=fs)
    state.not_noisy = not_noisy
    energy_sig = energy_trace(ecg_filt, fs=fs)
    r_peaks = detect_r_peaks(energy_sig, fs=fs)
    ecg_bs, _ = baseline_subtract_twoanchors(
        ecg_filt, r_peaks, fs=fs
    )
    timings["preprocessing"] = time.perf_counter() - step_start
    logger.info("[TIMING] Preprocessing completed in %.4f seconds", timings["preprocessing"])

    # Beat grid
    step_start = time.perf_counter()
    beats_all = synchronise_beats(segment_rr(ecg_bs, r_peaks, fs=fs), fs=fs, win_ms=60)
    beats_good = filter_beats_by_length(beats_all, tolerance_ms=30, fs=fs)
    timings["beat_sync"] = time.perf_counter() - step_start
    logger.info("[TIMING] Beat synchronization completed in %.4f seconds", timings["beat_sync"])

    if not beats_good:
        logger.warning("%s -- no usable beats after filtering; skipping record.", recording_id)
        return {}

    total_beats = len(beats_all)
    identical_beats = len(beats_good)

    # ===================================================================
    # 2. RR statistics (dual semantics: "main" regular beats + "all" NN)
    # ===================================================================
    step_start = time.perf_counter()

    # "main" = stats from regular-length beats only (legacy behavior)
    rr_good_ms: List[float] = [
        (b.ecg_12.shape[1] / fs) * 1000.0 for b in beats_good
    ]

    # "all/NN" = robust R-to-R intervals from raw R-peaks (HRV-style)
    rr_nn_ms = robust_rr_from_rpeaks(r_peaks, fs)

    _rr = _rr_features_nn(rr_nn_ms)

    rr_stats: Dict[str, float] = {
        # Backward-compatible "main" (regular beats only)
        "rr_mean_ms":   float(np.mean(rr_good_ms)) if rr_good_ms else 0.0,
        "rr_sd_ms":     float(np.std(rr_good_ms, ddof=0)) if rr_good_ms else 0.0,
        "beats_used":   identical_beats,
        "beats_all":    total_beats,

        # "All" from robust NN intervals (distinct from "main")
        "rr_mean_ms_all": _rr["mean_ms"],
        "rr_sd_ms_all":   _rr["sdnn_ms"],

        # Extra HRV metrics (from NN list)
        "rmssd_ms": _rr["rmssd_ms"],
        "pnn50":    _rr["pnn50"],
    }
    rr_mean_main = rr_stats["rr_mean_ms"]

    leads: List[str] = meta.get("lead_names", [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ])
    timings["rr_statistics"] = time.perf_counter() - step_start
    logger.info("[TIMING] RR statistics calculation completed in %.4f seconds", timings["rr_statistics"])

    # ===================================================================
    # 3. Beat Padding
    # ===================================================================
    step_start = time.perf_counter()
    stack_sync, time_ms = pad_beats(beats_good, fs=fs)
    # Cache per-lead synchronised-stack mean — reused by
    # process_leads_with_proper_timing, analyze_st_segments_all_leads,
    # calculate_residuals_from_fits, the energy-trace step, the ST plot
    # guard, the feature-context segment build, and the ctx-based J-point
    # T-onset duration helper in features/st_deviation.py.
    avg_per_lead = np.nanmean(stack_sync, axis=0)
    for i, beat in enumerate(beats_good):
        beat.ecg_sync = stack_sync[i]
    timings["beat_padding"] = time.perf_counter() - step_start
    logger.info("[TIMING] Beat padding completed in %.4f seconds", timings["beat_padding"])

    # ===================================================================
    # 4. Average-beat analysis (limb & chest)
    # ===================================================================
    step_start = time.perf_counter()
    limb_beats = slice_beats_to_plane(beats_good, "limb")
    chest_beats = slice_beats_to_plane(beats_good, "chest")

    lumps_limb, extras_limb_av = _analyse_plane(
        "limb", limb_beats, rr_stats, fs=fs,
        state=state,
        amplitude_threshold_unscaled=amplitude_threshold_unscaled,
        plot_enabled=plot_enabled, plot_title_prefix=recording_id,
        offset_ms=offset_ms,
    )
    lumps_chest, extras_chest_av = _analyse_plane(
        "chest", chest_beats, rr_stats, fs=fs,
        state=state,
        amplitude_threshold_unscaled=amplitude_threshold_unscaled,
        plot_enabled=plot_enabled, plot_title_prefix=recording_id,
        offset_ms=offset_ms,
    )
    timings["avg_beat_analysis"] = time.perf_counter() - step_start
    logger.info("[TIMING] Average-beat analysis (limb & chest) completed in %.4f seconds", timings["avg_beat_analysis"])

    # ===================================================================
    # 5. Per-lead Gaussian refits & segment bounds
    # ===================================================================
    step_start = time.perf_counter()
    lead_results = process_leads_with_proper_timing(
        recording_id=recording_id,
        beats_good=beats_good,
        lumps_limb=lumps_limb,
        leads=leads,
        fs=fs,
        plot_enabled=plot_enabled,
        time_ms=time_ms,
        stack_sync=stack_sync,
        rr_stats=rr_stats,
        offset_ms=offset_ms,
        pruning=pruning,
        avg_per_lead=avg_per_lead,
    )

    lead_fits = lead_results["lead_fits"]
    segment_bounds = lead_results["segment_bounds"]
    wave_bounds = lead_results.get(
        "wave_bounds",
        {"P": (-200, -60), "QRS": (-50, 80), "T": (120, 350)},
    )
    timings["lead_gaussian_refits"] = time.perf_counter() - step_start
    logger.info("[TIMING] Per-lead Gaussian refits & segment bounds completed in %.4f seconds", timings["lead_gaussian_refits"])

    # ===================================================================
    # 5.5 Energy trace analysis (AF vs APB discrimination -- Feb14)
    # ===================================================================
    step_start = time.perf_counter()

    computed: Dict[str, Any] = {}

    # Average beat (12 x N) — reused from pad_beats cache
    avg_beat_12 = avg_per_lead
    # Energy trace of the average beat (RMS across leads)
    energy_trace_avg = np.sqrt(np.mean(avg_beat_12 ** 2, axis=0))

    energy_results = analyse_energy_trace(
        energy_signal=energy_trace_avg,
        time_ms=time_ms,
        wave_bounds=wave_bounds,
        debug=plot_enabled,
    )

    energy_metrics = energy_results["metrics"]
    logger.debug(
        "    Energy Trace Analysis: %s (Integral: %.2f)",
        energy_metrics["discrimination_code"],
        energy_metrics["p_energy_integral"],
    )

    computed["Energy_P_Integral"] = energy_metrics["p_energy_integral"]
    computed["Energy_P_MaxAmp"] = energy_metrics["p_max_amp"]
    computed["AF_Likely_Energy_Check"] = (energy_metrics["discrimination_code"] == "AF_LIKELY")

    timings["energy_trace"] = time.perf_counter() - step_start
    logger.info("[TIMING] Energy trace analysis completed in %.4f seconds", timings["energy_trace"])

    # ===================================================================
    # 5.6 Complex arrhythmia detection (Apr28 Item 4.5.4 — Cell 13 TdP)
    # ===================================================================
    # Detects Short-Long-Short triggers, amplitude modulation, ectopic
    # patterns (bigeminy/trigeminy), and overall TdP risk. Emits 16
    # Complex_* keys into Computed Parameters. Designed to work even when
    # the underlying rhythm is AFib (uses beat-level energy morphology).
    # Skip-guard: needs >= 6 beats AND >= 5 RR intervals. Fail-soft on
    # exception (logs warning, leaves Complex_* keys absent).
    step_start = time.perf_counter()
    try:
        if stack_sync.shape[0] >= 6 and len(rr_good_ms) >= 5:
            complex_arrhythmia_result = analyse_complex_arrhythmias(
                stack_sync=stack_sync,
                time_ms=time_ms,
                rr_ms=np.asarray(rr_good_ms, dtype=float),
                debug=plot_enabled,
            )
            ca_metrics = complex_arrhythmia_result.get("computed_metrics", {})
            computed.update(ca_metrics)
        else:
            logger.debug(
                "Skipped complex arrhythmia analysis: n_beats=%d, rr_count=%d",
                stack_sync.shape[0], len(rr_good_ms),
            )
    except Exception as exc:
        logger.warning("Complex arrhythmia analysis failed: %s", exc)
    timings["complex_arrhythmia"] = time.perf_counter() - step_start
    logger.info(
        "[TIMING] Complex arrhythmia analysis completed in %.4f seconds",
        timings["complex_arrhythmia"],
    )

    # ===================================================================
    # 6. Comprehensive ST segment analysis
    # ===================================================================
    step_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("ST SEGMENT ANALYSIS")
    logger.info("=" * 60)

    # Extract demographics
    patient_age = 50  # Default
    patient_sex = "Male"
    if meta and meta.get("age") is not None:
        try:
            patient_age = int(meta["age"])
        except (ValueError, TypeError):
            pass
    if meta and meta.get("sex") is not None:
        patient_sex = str(meta["sex"])

    st_analysis = analyze_st_segments_all_leads(
        stack_sync=stack_sync,
        time_ms=time_ms,
        lead_fits=lead_fits,
        leads=leads,
        wave_bounds=wave_bounds,
        avg_per_lead=avg_per_lead,
    ) or {}

    # Handle missing/failed analysis
    if not st_analysis:
        logger.warning("ST segment analysis returned None/Empty, using defaults")
        st_analysis = {
            "lead_features": {},
            "summary": {
                "primary_diagnosis": "ST analysis unavailable",
                "urgency": "normal",
                "max_elevation": 0.0,
                "max_depression": 0.0,
                "abnormal_leads": [],
                "recommendations": [],
            },
            "patterns": {},
            "clinical_alert": False,
            "estimated_qt_ms": 0.0,
            "computed_metrics": {},
        }

    # Merge computed metrics from ST analysis
    st_metrics = st_analysis.get("computed_metrics", {})
    if st_metrics:
        logger.debug("   -> Merging %d ST metrics into Computed Parameters", len(st_metrics))
        computed.update(st_metrics)
    else:
        logger.debug("   -> No 'computed_metrics' found in ST analysis result")
        computed["ST_T_Ratio_Max_Chest"] = 0.0
        computed["ST_T_Ratio_Max_Limb"] = 0.0

    # Console summary
    summ = st_analysis.get("summary") or {}
    if st_analysis.get("clinical_alert"):
        logger.info("ST SEGMENT ABNORMALITIES DETECTED!")
        logger.info("Primary diagnosis: %s", summ.get("primary_diagnosis"))

    # Rank leads by |ST deviation|
    lead_feats = st_analysis.get("lead_features") or {}

    def _get_st_attr(attr: str, f: Any, default: Any = None) -> Any:
        return f.get(attr, default) if isinstance(f, dict) else getattr(f, attr, default)

    ranked = sorted(
        lead_feats.items(),
        key=lambda kv: (
            abs(_get_st_attr("st_elevation", kv[1], 0.0))
            if abs(_get_st_attr("st_elevation", kv[1], 0.0))
            >= abs(_get_st_attr("st_depression", kv[1], 0.0))
            else abs(_get_st_attr("st_depression", kv[1], 0.0))
        ),
        reverse=True,
    )[:10]

    if ranked:
        logger.debug("Top ST deviations (mV) by lead:")
        for ln, f in ranked:
            elev = _get_st_attr("st_elevation", f, 0.0)
            depr = _get_st_attr("st_depression", f, 0.0)
            dev = elev if abs(elev) >= abs(depr) else depr
            dev = float(dev or 0.0)
            jpt = _get_st_attr("j_point_ms", f, None)
            logger.debug(
                "  %s: %.3f mV%s", ln, dev,
                f" @ J+{int(jpt)} ms" if jpt is not None else "",
            )

    # Plot per-lead (average beat) -- guarded
    if plot_enabled and ranked:
        try:
            from qpsi.features.st_deviation import plot_st_segment_analysis
            lead_to_idx = {name: i for i, name in enumerate(leads)}
            for lead_name, fobj in ranked:
                li = lead_to_idx.get(lead_name)
                if li is None:
                    continue
                trace = avg_per_lead[li, :]
                plot_st_segment_analysis(
                    trace,
                    time_ms,
                    fobj,
                    lead_name,
                    estimated_qt_ms=st_analysis.get("estimated_qt_ms", 0.0),
                )
        except ImportError:
            pass

    residuals_by_lead = calculate_residuals_from_fits(
        stack_sync=stack_sync,
        time_ms=time_ms,
        lead_fits=lead_fits,
        leads=leads,
        avg_per_lead=avg_per_lead,
    )
    timings["st_analysis"] = time.perf_counter() - step_start
    logger.info("[TIMING] ST segment analysis completed in %.4f seconds", timings["st_analysis"])

    # ===================================================================
    # 6.5  J-point-aware summary integration
    # ===================================================================
    step_start = time.perf_counter()
    logger.info("Integrating J-point-aware ST summary into semantics...")

    patterns = st_analysis.get("patterns", {})
    summary = generate_st_summary(st_analysis.get("lead_features", {}), patterns)

    # Integrate into semantic_flags
    semantic_flags: Dict[str, Any] = {}
    semantic_flags.setdefault("ST", {})

    if "elevation" in summary["primary_diagnosis"].lower():
        semantic_flags["ST"]["st_elevation"] = {
            "present": True,
            "description": summary["primary_diagnosis"],
            "affected_leads": [x["lead"] for x in summary["abnormal_leads"]],
            "max_deviation_mv": round(summary["max_elevation"], 3),
            "urgency": summary["urgency"],
            "consistency_notes": [
                f"{x['lead']}: {x.get('consistency', 'unknown')}"
                for x in summary["abnormal_leads"]
            ],
        }
    elif "depression" in summary["primary_diagnosis"].lower():
        semantic_flags["ST"]["st_depression"] = {
            "present": True,
            "description": summary["primary_diagnosis"],
            "affected_leads": [x["lead"] for x in summary["abnormal_leads"]],
            "max_deviation_mv": round(abs(summary["max_depression"]), 3),
            "urgency": summary["urgency"],
            "consistency_notes": [
                f"{x['lead']}: {x.get('consistency', 'unknown')}"
                for x in summary["abnormal_leads"]
            ],
        }
    else:
        semantic_flags["ST"]["st_normal"] = {
            "present": False,
            "description": summary["primary_diagnosis"],
            "urgency": summary["urgency"],
        }

    # Add to clinical summaries (Tier A)
    st_text: str = ""
    if "elevation" in summary["primary_diagnosis"].lower():
        if any(
            "plateau" in note
            for note in semantic_flags["ST"].get("st_elevation", {}).get("consistency_notes", [])
        ):
            st_text = f"ST-elevation ({summary['primary_diagnosis']}); J-point plateau consistent across leads"
        else:
            st_text = f"ST-elevation ({summary['primary_diagnosis']}); J-point rise from baseline (benign variant)"
    elif "depression" in summary["primary_diagnosis"].lower():
        st_text = f"ST-depression ({summary['primary_diagnosis']})"
    else:
        st_text = summary["primary_diagnosis"]

    semantic_flags.setdefault("clinical_summaries", {})
    semantic_flags["clinical_summaries"]["ST_analysis"] = (
        f"{st_text}; max {summary['max_elevation']:.2f} mV"
        if summary["max_elevation"] > 0
        else f"{st_text}; max depression {abs(summary['max_depression']):.2f} mV"
    )
    timings["jpoint_integration"] = time.perf_counter() - step_start
    logger.info("[TIMING] J-point aware ST summary integration completed in %.4f seconds", timings["jpoint_integration"])

    # ===================================================================
    # 7. Per-beat analysis
    # ===================================================================
    step_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("PER-BEAT ANALYSIS")
    logger.info("=" * 60)

    arrhythmic_beats: List[Dict[str, Any]] = []
    extra_beats: List[Dict[str, Any]] = []
    p_vars_limb: List[Dict[str, Any]] = []
    t_vars_limb: List[Dict[str, Any]] = []
    p_vars_chest: List[Dict[str, Any]] = []
    t_vars_chest: List[Dict[str, Any]] = []

    ref_p_limb = next((w for w in lumps_limb if w["wave_type"] == "P"), None)
    ref_p_chest = next((w for w in lumps_chest if w["wave_type"] == "P"), None)
    _T_TYPES = {"T", "ST1", "T1", "ST"}
    ref_t_limb = next((w for w in lumps_limb if w["wave_type"] in _T_TYPES), None)
    ref_t_chest = next((w for w in lumps_chest if w["wave_type"] in _T_TYPES), None)

    # PRUNING: Skip per-beat P-wave variability analysis when the average-beat
    # P-wave amplitude is below the noise floor of standard ECG acquisition
    # (typical noise ~0.01-0.05 mV). Beat-to-beat variability of a sub-threshold
    # P-wave is dominated by noise rather than true morphological variation, so
    # computing it adds cost without diagnostic value. The P-wave reference is
    # kept for presence/absence reporting but variability analysis is skipped.
    # Threshold lives in qpsi.constants.P_AMP_VARIABILITY_THRESHOLD_MV.
    skip_p_variability_limb = (
        ref_p_limb is None
        or abs(ref_p_limb.get("amplitude_mV", ref_p_limb.get("amp_mv", 0.0))) < P_AMP_VARIABILITY_THRESHOLD_MV
    )
    skip_p_variability_chest = (
        ref_p_chest is None
        or abs(ref_p_chest.get("amplitude_mV", ref_p_chest.get("amp_mv", 0.0))) < P_AMP_VARIABILITY_THRESHOLD_MV
    )
    if skip_p_variability_limb:
        logger.debug("PRUNE: skipping per-beat P variability (limb) — P amp below threshold")
    if skip_p_variability_chest:
        logger.debug("PRUNE: skipping per-beat P variability (chest) — P amp below threshold")

    beatwise_limb_lumps: List[List[Dict[str, Any]]] = []
    beatwise_chest_lumps: List[List[Dict[str, Any]]] = []
    all_rat_events_limb: List[Dict[str, Any]] = []
    all_rvt_events_limb: List[Dict[str, Any]] = []
    all_rat_events_chest: List[Dict[str, Any]] = []
    all_rvt_events_chest: List[Dict[str, Any]] = []

    # Per-beat plane analysis — try full Rust rayon parallelism first.
    # Falls back to sequential Python if Rust fails.
    _used_rust_beats = False
    try:
        import qpsi_native as _qn
        if not hasattr(_qn, "analyse_beats_full_rust"):
            raise ImportError("qpsi_native missing Rust plane analysis")
        from qpsi.constants import LIMB_MAT, CHEST_MAT
        from qpsi.plane_fitting import fit_scalar_gain as _fsg

        # Build input arrays for Rust
        _n_beats = len(beats_all)
        _n_samples = beats_all[0].time_ms.shape[0] if _n_beats > 0 else 0
        _seg6_list = []
        _time_list = []
        _rr_list = []
        _sample_counts = []
        for beat in beats_all:
            _seg6_list.append(beat.ecg_12.ravel())
            _time_list.append(beat.time_ms)
            _rr_list.append(beat.ecg_12.shape[1] / fs * 1_000.0)
            _sample_counts.append(beat.time_ms.shape[0])

        _seg6_flat = np.concatenate(_seg6_list).astype(np.float64)
        _time_flat = np.concatenate(_time_list).astype(np.float64)
        _rr_arr = np.array(_rr_list, dtype=np.float64)
        _sc_arr = np.array(_sample_counts, dtype=np.float64)

        # Get cached gains
        _g_limb = state.plane_cache.get("limb", 1.0)
        _g_chest = state.plane_cache.get("chest", 1.0)

        _e2_limb = np.ascontiguousarray(LIMB_MAT, dtype=np.float64).ravel()
        _e2_chest = np.ascontiguousarray(CHEST_MAT, dtype=np.float64).ravel()

        _rust_results = _qn.analyse_beats_full_rust(
            np.ascontiguousarray(_seg6_flat),
            np.ascontiguousarray(_time_flat),
            _n_beats, np.ascontiguousarray(_sc_arr),
            _e2_limb, _e2_chest,
            float(_g_limb), float(_g_chest),
            state.not_noisy,
            amplitude_threshold_unscaled,
            np.ascontiguousarray(_rr_arr),
            float(offset_ms),
            3,  # n_extra
        )

        # Parse Rust results into per-beat lumps
        _limb_lumps_by_beat = [[] for _ in range(_n_beats)]
        _chest_lumps_by_beat = [[] for _ in range(_n_beats)]
        for d in _rust_results:
            bi = int(d["beat_idx"])
            lump = {
                "wave_type": str(d["wave_type"]),
                "start_time": float(d["start_time"]),
                "peak_time": float(d["peak_time"]),
                "end_time": float(d["end_time"]),
                "amplitude_mV": float(d["amplitude_mV"]),
                "angle": float(d["angle"]),
                "angle_degrees": float(d["angle_degrees"]),
                "center_ms": float(d["center_ms"]),
                "sigma_ms": float(d["sigma_ms"]),
            }
            if str(d["plane"]) == "limb":
                _limb_lumps_by_beat[bi].append(lump)
            else:
                _chest_lumps_by_beat[bi].append(lump)

        # Build beat_futures equivalent
        beat_futures = []
        for idx in range(_n_beats):
            rr_ms_i = _rr_list[idx]
            lumps_xy = _limb_lumps_by_beat[idx]
            lumps_xz = _chest_lumps_by_beat[idx]
            # info dicts (simplified — wave-based metrics)
            info_xy = {"QRS_interval_ms": 0.0}
            for lump in lumps_xy:
                if "R" in lump.get("wave_type", ""):
                    info_xy["QRS_interval_ms"] = lump.get("end_time", 0) - lump.get("start_time", 0)
            info_xz = {}
            beat_futures.append((lumps_xy, info_xy, lumps_xz, info_xz, rr_ms_i))

        _used_rust_beats = True
        logger.debug("Per-beat analysis: Rust rayon (%d beats x 2 planes)", _n_beats)
    except Exception as e:
        logger.debug("Rust per-beat analysis failed, falling back to Python: %s", e)

    if not _used_rust_beats:
        beat_futures = []
        for idx, beat in enumerate(beats_all):
            rr_ms_i = beat.ecg_12.shape[1] / fs * 1_000.0
            seg_xy = beat.ecg_12[PLANE_IDX["limb"]]
            seg_xz = beat.ecg_12[PLANE_IDX["chest"]]
            lumps_xy_loc, info_xy = _analyse_plane(
                "limb",
                [{"ecg_segment": seg_xy, "time_ms": beat.time_ms}],
                {"rr_mean_ms": rr_ms_i},
                fs=fs, state=state,
                amplitude_threshold_unscaled=amplitude_threshold_unscaled,
                individual_traces=True, plot_enabled=False, offset_ms=offset_ms,
            )
            lumps_xz_loc, info_xz = _analyse_plane(
                "chest",
                [{"ecg_segment": seg_xz, "time_ms": beat.time_ms}],
                {"rr_mean_ms": rr_ms_i},
                fs=fs, state=state,
                amplitude_threshold_unscaled=amplitude_threshold_unscaled,
                individual_traces=True, plot_enabled=False, offset_ms=offset_ms,
            )
            beat_futures.append((lumps_xy_loc, info_xy, lumps_xz_loc, info_xz, rr_ms_i))

    for idx, (lumps_xy_loc, info_xy, lumps_xz_loc, info_xz, rr_ms_i) in enumerate(beat_futures):
        clin_xy_loc = info_xy
        beatwise_limb_lumps.append(lumps_xy_loc)
        beatwise_chest_lumps.append(lumps_xz_loc)

        # P/T variability capture (compare_wave_variation)
        # PRUNING: P-wave variability skipped when average P amplitude is
        # below threshold — noise-dominated variation has no clinical value.
        if not skip_p_variability_limb and "P" in {w["wave_type"] for w in lumps_xy_loc}:
            p_vars_limb.append(
                compare_wave_variation(
                    ref_p_limb,
                    next(w for w in lumps_xy_loc if w["wave_type"] == "P"),
                )
            )
        if not skip_p_variability_chest and "P" in {w["wave_type"] for w in lumps_xz_loc}:
            p_vars_chest.append(
                compare_wave_variation(
                    ref_p_chest,
                    next(w for w in lumps_xz_loc if w["wave_type"] == "P"),
                )
            )
        if ref_t_limb and _T_TYPES & {w["wave_type"] for w in lumps_xy_loc}:
            t_vars_limb.append(
                compare_wave_variation(
                    ref_t_limb,
                    next(w for w in lumps_xy_loc if w["wave_type"] in _T_TYPES),
                )
            )
        if ref_t_chest and _T_TYPES & {w["wave_type"] for w in lumps_xz_loc}:
            t_vars_chest.append(
                compare_wave_variation(
                    ref_t_chest,
                    next(w for w in lumps_xz_loc if w["wave_type"] in _T_TYPES),
                )
            )

        # Arrhythmia flag
        if rr_mean_main and abs(rr_ms_i - rr_mean_main) / rr_mean_main > 0.2:
            arrhythmic_beats.append({"Interval": idx, "RR_ms": round(rr_ms_i, 2)})

        # Extra-beat detection (local QRS wider than average QRS threshold)
        T_qrs_thr = 20  # local override (constants.py has 50)
        if clin_xy_loc.get("QRS_interval_ms", 0.0) - extras_limb_av.get("QRS_interval_ms", 0.0) > T_qrs_thr:
            extra_beats.append({"Interval": idx, "QRS Duration": round(clin_xy_loc["QRS_interval_ms"], 2)})

        # -- P-wave variations (limb) --
        # PRUNING: skip when P amplitude below threshold (noise-dominated).
        if not skip_p_variability_limb:
            collect_p_variations(
                dst_list=p_vars_limb,
                ref_wave=ref_p_limb,
                loc_lumps=lumps_xy_loc,
                idx=idx,
                amp_thr=P_amp_difference_threshold,
                dt_thr=P_time_difference_threshold,
                dxy_thr=P_theta_difference_threshold,
            )
        # -- P-wave variations (chest) --
        # PRUNING: skip when P amplitude below threshold (noise-dominated).
        if not skip_p_variability_chest:
            collect_p_variations(
                dst_list=p_vars_chest,
                ref_wave=ref_p_chest,
                loc_lumps=lumps_xz_loc,
                idx=idx,
                amp_thr=P_amp_difference_threshold,
                dt_thr=P_time_difference_threshold,
                dxy_thr=P_theta_difference_threshold,
            )
        # -- T-wave variations (limb) --
        if ref_t_limb:
            collect_t_variations(
                dst_list=t_vars_limb,
                ref_wave=ref_t_limb,
                loc_lumps=lumps_xy_loc,
                idx=idx,
                amp_thr=T_amp_thr,
                dt_thr=T_dt_thr,
                dxy_thr=T_dxy_thr,
            )
        # -- T-wave variations (chest) --
        if ref_t_chest:
            collect_t_variations(
                dst_list=t_vars_chest,
                ref_wave=ref_t_chest,
                loc_lumps=lumps_xz_loc,
                idx=idx,
                amp_thr=T_amp_thr,
                dt_thr=T_dt_thr,
                dxy_thr=T_dxy_thr,
            )

        # -- QWVA: collect beatwise RAT/RVT events --
        if enable_qwva:
            rat_limb = analyse_rat_variants(idx, ref_p_limb, lumps_xy_loc, fs, rr_ms_i)
            rvt_limb = analyse_rvt_variants(idx, ref_t_limb, lumps_xy_loc, fs, rr_ms_i)
            rat_chest = analyse_rat_variants(idx, ref_p_chest, lumps_xz_loc, fs, rr_ms_i)
            rvt_chest = analyse_rvt_variants(idx, ref_t_chest, lumps_xz_loc, fs, rr_ms_i)

            all_rat_events_limb.extend(rat_limb)
            all_rvt_events_limb.extend(rvt_limb)
            all_rat_events_chest.extend(rat_chest)
            all_rvt_events_chest.extend(rvt_chest)

    pacing_waves_total = int(
        extras_limb_av.get("pacing_detected", False) or extras_chest_av.get("pacing_detected", False)
    )
    p_var_stats: Dict[str, Any] = {
        "limb":  summarize_variability(p_vars_limb),
        "chest": summarize_variability(p_vars_chest),
    }
    t_var_stats: Dict[str, Any] = {
        "limb":  summarize_variability(t_vars_limb),
        "chest": summarize_variability(t_vars_chest),
    }
    timings["per_beat_analysis"] = time.perf_counter() - step_start
    logger.info("[TIMING] Per-beat analysis completed in %.4f seconds (%d beats)", timings["per_beat_analysis"], len(beats_all))

    # ===================================================================
    # 7.5 Create FeatureContext with raw ECG data
    # ===================================================================
    step_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("CREATING FEATURE CONTEXT")
    logger.info("=" * 60)

    rr_intervals_list: List[float] = [
        b.ecg_12.shape[1] / fs * 1000.0 for b in beats_good
    ] if beats_good else []

    beat_plane: Dict[str, Dict[str, List[Any]]] = {
        "limb": {
            "T": get_waveforms_from_lumps(beatwise_limb_lumps, "T"),
            "P": get_waveforms_from_lumps(beatwise_limb_lumps, "P"),
            "QRS": get_waveforms_from_lumps(beatwise_limb_lumps, "QRS"),
        },
        "chest": {
            "T": get_waveforms_from_lumps(beatwise_chest_lumps, "T"),
            "P": get_waveforms_from_lumps(beatwise_chest_lumps, "P"),
            "QRS": get_waveforms_from_lumps(beatwise_chest_lumps, "QRS"),
        },
    }

    # Segments dictionary with raw ECG data organized by lead
    segments: Dict[str, Any] = {}

    # Method 1: Use stack_sync (averaged/synchronized beats)
    segments["averaged"] = {"by_lead": {}}
    for i, lead_name in enumerate(leads):
        if i < stack_sync.shape[1]:
            segments["averaged"]["by_lead"][lead_name] = {
                "raw": avg_per_lead[i, :],
                "signal": stack_sync[:, i, :],
                "time_ms": time_ms,
            }

    # Method 2: Also store raw ECG data directly
    segments["raw"] = {"by_lead": {}}
    for i, lead_name in enumerate(leads):
        if i < raw_ecg_12.shape[0]:
            segments["raw"]["by_lead"][lead_name] = {
                "raw": raw_ecg_12[i, :],
                "signal": raw_ecg_12[i, :],
            }

    # Create the comprehensive FeatureContext
    feature_ctx = FeatureContext(
        avg_plane={
            "limb":  _ensure_wave_keys(lumps_limb),
            "chest": _ensure_wave_keys(lumps_chest),
        },
        beat_plane=beat_plane,
        leads=leads,
        fs=fs,
        segment_bounds=segment_bounds,
        segments=segments,
        rr_intervals=rr_intervals_list,
        lead_fits=lead_fits,
        residuals_by_lead=residuals_by_lead,
        r_peaks=r_peaks,
    )

    feature_ctx.stack_sync = stack_sync
    feature_ctx.avg_per_lead = avg_per_lead
    feature_ctx.raw_ecg_12 = raw_ecg_12
    feature_ctx.time_ms = time_ms

    # --- Rhythm Panorama (10 s global view) ---
    panorama = rhythm_panorama_from_ctx(feature_ctx, plot_enabled=plot_enabled)
    merge_panorama_into_semantics(semantic_flags, panorama)
    timings["feature_context"] = time.perf_counter() - step_start
    logger.info("[TIMING] Feature context creation completed in %.4f seconds", timings["feature_context"])

    # ===================================================================
    # 7.6 Atrial f_wave detection & flutter metrics
    # ===================================================================
    step_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("ATRIAL ARRHYTHMIA METRICS")
    logger.info("=" * 60)

    f_wave_result: Dict[str, Any] = {}
    flutter_result: Dict[str, Any] = {}

    try:
        # 1. Run F-Wave ACF
        f_wave_result = rhythm_atrial_f_wave(feature_ctx, debug=plot_enabled)
        f_metrics = f_wave_result.get("metrics", {})

        # 2. Run Flutter RR Analysis
        flutter_result = rhythm_flutter_like(feature_ctx)
        fl_metrics = flutter_result.get("metrics", {})

        # 3. Consolidate Metrics
        computed["AF_Metrics_F_Wave_Coherence"] = f_metrics.get("median_coherence", 0.0)
        computed["AF_Metrics_F_Wave_Max_ACF"] = f_metrics.get("max_acf_height", 0.0)

        # Only include frequency if valid
        if f_wave_result.get("atrial_f_wave", {}).get("present"):
            info = f_wave_result["atrial_f_wave"]
            computed["AF_Metrics_F_Wave_Freq_Hz"] = info.get("f_wave_freq_hz", 0.0)
            logger.info("ATRIAL f_wave: %s", info.get("description"))
        else:
            computed["AF_Metrics_F_Wave_Freq_Hz"] = 0.0

        computed["AF_Metrics_Integer_Ratio_Prevalence"] = fl_metrics.get("integer_ratio_prevalence", 0.0)

    except Exception as e:
        logger.warning("Atrial metrics failed: %s", e)
        if plot_enabled:
            traceback.print_exc()

    timings["atrial_metrics"] = time.perf_counter() - step_start
    logger.info("[TIMING] Atrial metrics completed in %.4f seconds", timings["atrial_metrics"])

    # ===================================================================
    # 8. Semantic feature extraction (including ST features, f_wave)
    # ===================================================================
    step_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("SEMANTIC FEATURE EXTRACTION")
    logger.info("=" * 60)

    # 8.1 Base extraction -- EXCLUDE ST features (we add them from comprehensive analysis)
    try:
        semantic_flags = extract_semantic_features(
            feature_ctx,
            record_json={},
            exclude_domains={"ST"},
            debug=plot_enabled,
        )
        semantic_flags = dict(semantic_flags or {})
    except Exception as e:
        msg = str(e)
        if "Number of labels" in msg or "valid values are 2" in msg:
            logger.warning(
                "%s: skipping semantic extraction (labeling issue: %s)",
                recording_id, msg,
            )
            semantic_flags = {}
        else:
            raise

    # 8.2 Add f_wave (if present) into rhythm domain
    if f_wave_result and f_wave_result.get("atrial_f_wave", {}).get("present"):
        semantic_flags.setdefault("rhythm", {})["atrial_f_wave"] = f_wave_result["atrial_f_wave"]

    # --- integrate QWVA summaries ---
    if enable_qwva:
        qwva_summary = summarize_qwva_results(
            all_rat_events_limb, all_rvt_events_limb,
            all_rat_events_chest, all_rvt_events_chest,
            rr_nn_ms,
        )

        semantic_flags.setdefault("waves", {}).setdefault("P", {})
        semantic_flags.setdefault("waves", {}).setdefault("T", {})

        semantic_flags["waves"]["P"]["RAT_clusters_limb"] = qwva_summary["RAT_clusters_limb"]
        semantic_flags["waves"]["T"]["RVT_clusters_limb"] = qwva_summary["RVT_clusters_limb"]
        semantic_flags["waves"]["T"]["PRD_metrics_limb"] = qwva_summary["PRD_metrics_limb"]

        semantic_flags["waves"]["P"]["RAT_clusters_chest"] = qwva_summary["RAT_clusters_chest"]
        semantic_flags["waves"]["T"]["RVT_clusters_chest"] = qwva_summary["RVT_clusters_chest"]
        semantic_flags["waves"]["T"]["PRD_metrics_chest"] = qwva_summary["PRD_metrics_chest"]

        # optional numeric flattening
        computed["rat_count_limb"] = len(all_rat_events_limb)
        computed["rat_count_chest"] = len(all_rat_events_chest)
        # NOTE: rvt_prd_power is a Welch periodogram over ~10-15 beat-to-beat T-wave
        # variations. With so few samples the spectral estimate has high variance.
        # Machine-epsilon differences in per-beat T-wave fits (Rust vs Python solver)
        # can cause 20-90% swings in this metric. This is inherent instability of
        # the estimator, not a bug in either implementation. Verified 2026-04-01.
        computed["rvt_prd_power_limb"] = qwva_summary["PRD_metrics_limb"]["prd_power"]
        computed["rvt_prd_power_chest"] = qwva_summary["PRD_metrics_chest"]["prd_power"]

        # P-wave existence ratio (fraction of beats with P-wave)
        n_beats_total = max(1, len(beats_all))
        computed["AF_Metrics_P_Fraction_Limb"] = _count_p(beatwise_limb_lumps) / n_beats_total
        computed["AF_Metrics_P_Fraction_Chest"] = _count_p(beatwise_chest_lumps) / n_beats_total

    # 8.3 Build ST semantic features -- morphology only (no diagnostic phrasing)
    st_semantic_features: Dict[str, Any] = {}
    leads_elev: List[str] = []
    leads_depr: List[str] = []
    max_elev = 0.0
    max_depr = 0.0

    lead_features = st_analysis.get("lead_features") or {}

    for lead, f in lead_features.items():
        if not getattr(f, "is_abnormal", False):
            continue

        elev = getattr(f, "st_elevation", 0.0)
        depr = getattr(f, "st_depression", 0.0)
        dev = elev if abs(elev) >= abs(depr) else depr
        dev = float(dev or 0.0)

        if dev >= 0.0:
            leads_elev.append(lead)
            max_elev = max(max_elev, abs(dev))
            logger.debug("%s: elevation %.3f mV -> current max_elev %.3f", lead, dev, max_elev)
        else:
            leads_depr.append(lead)
            max_depr = max(max_depr, abs(dev))
            logger.debug("%s: depression %.3f mV -> current max_depr %.3f", lead, dev, max_depr)

    if leads_elev:
        ter = _territory(leads_elev)
        summary_dict = (st_analysis or {}).get("summary") or {}
        urgency = summary_dict.get("urgency", "abnormal")

        st_semantic_features["st_elevation"] = {
            "present": True,
            "description": (
                f"ST elevation in {format_lead_list(leads_elev)}"
                + (f" ({ter} territory)" if ter else "")
            ),
            "affected_leads": leads_elev,
            "max_deviation_mv": round(max_elev, 3),
            "urgency": urgency,
        }

    if leads_depr:
        ter = _territory(leads_depr)
        summary_dict = (st_analysis or {}).get("summary") or {}
        urgency = summary_dict.get("urgency", "abnormal") if isinstance(summary_dict, dict) else "abnormal"

        st_semantic_features["st_depression"] = {
            "present": True,
            "description": (
                f"ST depression in {format_lead_list(leads_depr)}"
                + (f" ({ter} territory)" if ter else "")
            ),
            "affected_leads": leads_depr,
            "max_deviation_mv": round(max_depr, 3),
            "urgency": urgency,
        }

    semantic_flags["ST"] = st_semantic_features

    # =========================================================================
    # Apr28 Item 4.5.2 — Cell-11 Brugada composite scoring (verbatim port of
    # Apr28 notebook lines 17346-17694). Adds discrimination/composite features
    # to semantic_flags["ST"] AND computed[]. Final Brugada_Likelihood_Score
    # lands in both places (dual emission).
    # =========================================================================

    # --- Round 2: Composite discrimination features into semantic_flags ---
    st_metrics = st_analysis.get("computed_metrics", {})
    if "ST_Discrimination_Score" in st_metrics:
        semantic_flags["ST"]["st_discrimination_score"] = st_metrics["ST_Discrimination_Score"]
        semantic_flags["ST"]["st_territory_contrast"] = st_metrics.get("ST_Territory_Contrast", 0.0)
        semantic_flags["ST"]["st_inferior_morphology_index"] = st_metrics.get(
            "ST_Inferior_Morphology_Index", 0.0
        )

    # Inferior territory summary into semantic_flags
    inf_leads = ["II", "III", "aVF"]
    plm = st_metrics.get("per_lead_morphology", {})
    inf_summary: Dict[str, Any] = {}
    for ld in inf_leads:
        if ld in plm:
            inf_summary[ld] = {
                "shape": plm[ld].get("shape", "indeterminate"),
                "slope_uv_per_ms": plm[ld].get("slope_uv_per_ms", 0.0),
                "curvature": plm[ld].get("curvature", 0.0),
                "st_t_junction_angle": plm[ld].get("st_t_junction_angle", 0.0),
            }
    semantic_flags["ST"]["inferior_summary"] = inf_summary

    # =========================================================================
    # ROUND 3: Cross-domain discrimination features (T-wave + QRS Gaussian)
    # =========================================================================

    def _get_gaussian_component(lead_name: str, component_name: str):
        """Extract a specific Gaussian component from lead_fits."""
        if not isinstance(lead_fits, dict) or lead_name not in lead_fits:
            return None
        lf = lead_fits[lead_name]
        if isinstance(lf, dict):
            fits = lf.get("avg") or lf.get("fits")
            if isinstance(fits, dict):
                gaussians = fits.get("gaussians", [])
            elif isinstance(fits, list):
                gaussians = fits
            else:
                return None
        else:
            return None
        for g in gaussians:
            if isinstance(g, dict) and g.get("component") == component_name:
                return g
        return None

    # --- a) R-Wave Amplitude Inferior Ratio ---
    inf_r_amps: List[float] = []
    all_r_amps: List[float] = []
    for ld in leads:
        r_comp = _get_gaussian_component(ld, "R")
        if r_comp and "amp_mv" in r_comp:
            r_amp = abs(float(r_comp["amp_mv"]))
            all_r_amps.append(r_amp)
            if ld in ["II", "III", "aVF"]:
                inf_r_amps.append(r_amp)
    if inf_r_amps and all_r_amps:
        mean_inf_r = float(np.mean(inf_r_amps))
        mean_all_r = float(np.mean(all_r_amps))
        computed["R_Wave_Inferior_Mean_mV"] = round(mean_inf_r, 4)
        computed["R_Wave_Inferior_Ratio"] = round(mean_inf_r / max(mean_all_r, 0.001), 4)
    else:
        computed["R_Wave_Inferior_Mean_mV"] = 0.0
        computed["R_Wave_Inferior_Ratio"] = 0.0

    # --- b) T-Wave Width Index (precordial leads) ---
    precordial_t_sigmas: List[float] = []
    for ld in ["V2", "V3", "V4", "V5"]:
        t1_comp = _get_gaussian_component(ld, "T1")
        if t1_comp and "sigma_ms" in t1_comp:
            precordial_t_sigmas.append(float(t1_comp["sigma_ms"]))
    if precordial_t_sigmas:
        computed["T_Wave_Width_Precordial_Mean_ms"] = round(float(np.mean(precordial_t_sigmas)), 2)
    else:
        computed["T_Wave_Width_Precordial_Mean_ms"] = 0.0

    # --- c) T-Wave Inversion Count (inferior leads) ---
    inf_t_inversions = 0
    inf_t_count = 0
    for ld in ["II", "III", "aVF"]:
        t1_comp = _get_gaussian_component(ld, "T1")
        if t1_comp and "amp_mv" in t1_comp:
            inf_t_count += 1
            if float(t1_comp["amp_mv"]) < 0:
                inf_t_inversions += 1
    computed["T_Wave_Inversion_Count_Inferior"] = inf_t_inversions
    computed["T_Wave_Inversion_Fraction_Inferior"] = round(inf_t_inversions / max(inf_t_count, 1), 4)

    # --- d) T-Wave Amplitude in aVF ---
    t1_avf = _get_gaussian_component("aVF", "T1")
    computed["T_Wave_Amplitude_aVF_mV"] = (
        round(float(t1_avf["amp_mv"]), 4)
        if t1_avf and "amp_mv" in t1_avf
        else 0.0
    )

    # --- e) ST/T Ratio Lateral Mean ---
    lat_st_t_ratios: List[float] = []
    lead_feats_r3 = st_analysis.get("lead_features") or {}
    for ld in ["I", "aVL", "V5", "V6"]:
        if ld in lead_feats_r3:
            ratio = getattr(lead_feats_r3[ld], "st_t_ratio", 0.0)
            lat_st_t_ratios.append(float(ratio))
    computed["ST_T_Ratio_Lateral_Mean"] = (
        round(float(np.mean(lat_st_t_ratios)), 4) if lat_st_t_ratios else 0.0
    )

    # --- f) Updated ST_Discrimination_Score_v3 ---
    qrs_conc = st_metrics.get("QRS_ST_Concordance", 0.0)
    r_ratio = computed.get("R_Wave_Inferior_Ratio", 0.0)
    t_width = computed.get("T_Wave_Width_Precordial_Mean_ms", 0.0)
    t_inv_frac = computed.get("T_Wave_Inversion_Fraction_Inferior", 0.0)
    st_t_lat = computed.get("ST_T_Ratio_Lateral_Mean", 0.0)
    recip_idx = st_metrics.get("Reciprocal_Change_Index", 0.0)
    diffuse_idx = st_metrics.get("ST_Diffuseness_Index", 0.0)

    c_concordance = qrs_conc
    c_r_wave = -(r_ratio - 1.0) * 2.0
    c_t_width = (t_width - 42.0) / 10.0
    c_t_inv = -(t_inv_frac - 0.5) * 2.0
    c_st_t_lat = (st_t_lat + 0.1) * 3.0
    c_reciprocal = recip_idx * 2.0
    c_diffuse = -(diffuse_idx - 0.5) * 2.0
    disc_v3 = (
        0.25 * c_concordance
        + 0.20 * c_r_wave
        + 0.15 * c_t_width
        + 0.10 * c_t_inv
        + 0.10 * c_st_t_lat
        + 0.10 * c_reciprocal
        + 0.10 * c_diffuse
    )
    computed["ST_Discrimination_Score_v3"] = round(disc_v3, 4)

    semantic_flags["ST"]["st_discrimination_score_v3"] = round(disc_v3, 4)
    semantic_flags["ST"]["r_wave_inferior_ratio"] = computed["R_Wave_Inferior_Ratio"]
    semantic_flags["ST"]["t_wave_width_precordial_mean"] = computed["T_Wave_Width_Precordial_Mean_ms"]
    semantic_flags["ST"]["t_wave_inversion_inferior"] = computed["T_Wave_Inversion_Fraction_Inferior"]
    semantic_flags["ST"]["t_wave_amplitude_avf"] = computed["T_Wave_Amplitude_aVF_mV"]
    semantic_flags["ST"]["st_t_ratio_lateral_mean"] = computed["ST_T_Ratio_Lateral_Mean"]
    semantic_flags["ST"]["reciprocal_change_index"] = recip_idx
    semantic_flags["ST"]["st_diffuseness_index"] = st_metrics.get("ST_Diffuseness_Index", 0.0)

    # =========================================================================
    # ROUND 4: Brugada detection features (V1-V3)
    # =========================================================================

    # 4a. Right precordial summary mirrors inferior_summary pattern
    right_precordial_leads = ["V1", "V2", "V3"]
    rp_summary: Dict[str, Any] = {}
    for ld in right_precordial_leads:
        if ld in plm:
            rp_summary[ld] = {
                "shape": plm[ld].get("shape", "indeterminate"),
                "slope_uv_per_ms": plm[ld].get("slope_uv_per_ms", 0.0),
                "curvature": plm[ld].get("curvature", 0.0),
                "st_t_junction_angle": plm[ld].get("st_t_junction_angle", 0.0),
                "slope_progression": plm[ld].get("slope_progression", [0.0, 0.0, 0.0, 0.0]),
                "st_t_ratio": plm[ld].get("st_t_ratio", 0.0),
                "normalized_deviation": plm[ld].get("normalized_deviation", 0.0),
                "j_point_amplitude": plm[ld].get("j_point_amplitude", 0.0),
                "t_wave_amplitude": plm[ld].get("t_wave_amplitude", 0.0),
                "t_wave_polarity": plm[ld].get("t_wave_polarity", "positive"),
            }
    semantic_flags["ST"]["right_precordial_summary"] = rp_summary

    # 4b. Derived Brugada-specific scalar features
    v2_morph = plm.get("V2", {})
    v2_sp = v2_morph.get("slope_progression", [0.0, 0.0, 0.0, 0.0])
    v2_slope_reversal = v2_sp[0] - v2_sp[1] if len(v2_sp) >= 2 else 0.0
    semantic_flags["ST"]["v2_slope_reversal"] = round(v2_slope_reversal, 4)

    v2_slope_j120 = v2_sp[3] if len(v2_sp) >= 4 else 0.0
    semantic_flags["ST"]["v2_slope_j120"] = round(v2_slope_j120, 4)

    v1_morph = plm.get("V1", {})
    v1_sp = v1_morph.get("slope_progression", [0.0, 0.0, 0.0, 0.0])
    v1_slope_j80 = v1_sp[2] if len(v1_sp) >= 3 else 0.0
    semantic_flags["ST"]["v1_slope_j80"] = round(v1_slope_j80, 4)

    v1_peak_then_descent = (
        1.0 if (len(v1_sp) >= 2 and v1_sp[0] > 0.3 and v1_sp[1] < -0.3) else 0.0
    )
    v2_peak_then_descent = (
        1.0 if (len(v2_sp) >= 2 and v2_sp[0] > 0.3 and v2_sp[1] < -0.3) else 0.0
    )
    semantic_flags["ST"]["v1_peak_then_descent"] = v1_peak_then_descent
    semantic_flags["ST"]["v2_peak_then_descent"] = v2_peak_then_descent

    # 4c. Route Brugada detection result from Cell 11
    brugada_info = st_metrics.get("brugada_pattern", {})
    semantic_flags["ST"]["brugada_pattern"] = {
        "detected": brugada_info.get("detected", False),
        "type": brugada_info.get("type", "none"),
        "affected_leads": brugada_info.get("affected_leads", []),
        "coved_leads": brugada_info.get("coved_leads", []),
        "saddleback_leads": brugada_info.get("saddleback_leads", []),
        "per_lead_scores": brugada_info.get("per_lead_scores", {}),
    }

    # 4d. Brugada likelihood composite score (Phase 1 rebuild)
    # --- Existing slope extraction ---
    v2_sj120 = v2_sp[3] if len(v2_sp) >= 4 else 0.0
    v2_sj40 = v2_sp[1] if len(v2_sp) >= 2 else 0.0
    v2_slope_overall = v2_morph.get("slope_uv_per_ms", 0.0)
    v1_sj80 = v1_sp[2] if len(v1_sp) >= 3 else 0.0
    v1_sj40 = v1_sp[1] if len(v1_sp) >= 2 else 0.0

    # T-wave negativity in V1/V2 (backward compat)
    lead_feats_b = st_analysis.get("lead_features") or {}
    t_neg_score = 0.0
    for ld_b in ["V1", "V2"]:
        if ld_b in lead_feats_b:
            if getattr(lead_feats_b[ld_b], "t_wave_polarity", "positive") == "negative":
                t_neg_score += 0.5

    # --- NEW BRUGADA FEATURES (Phase 1) ---
    right_leads = ["V1", "V2", "V3"]

    # 1. J-point elevation features
    j_amps_v1v3 = [plm.get(ld, {}).get("j_point_amplitude", 0.0) for ld in right_leads]
    j_elev_max_v1v3 = max(j_amps_v1v3) if j_amps_v1v3 else 0.0
    j_elev_mean_v1v3 = sum(j_amps_v1v3) / len(j_amps_v1v3) if j_amps_v1v3 else 0.0
    j_exceeds_2mm = 1.0 if any(a >= 0.2 for a in j_amps_v1v3) else 0.0

    # 2. T-wave features
    t_amps_v1v3 = [plm.get(ld, {}).get("t_wave_amplitude", 0.0) for ld in right_leads]
    t_pols_v1v3 = [plm.get(ld, {}).get("t_wave_polarity", "positive") for ld in right_leads]
    t_amp_mean_v1v3 = sum(t_amps_v1v3) / len(t_amps_v1v3) if t_amps_v1v3 else 0.0
    t_inv_count_v1v3 = sum(1 for p in t_pols_v1v3 if p == "negative")
    t_inv_concordance = 1.0 if t_inv_count_v1v3 == 3 else 0.0

    # 3. Shape concordance
    shapes_v1v3 = [plm.get(ld, {}).get("shape", "") for ld in right_leads]
    coved_count_v1v3 = sum(1 for s in shapes_v1v3 if s in ("convex", "downsloping"))
    coved_concordance = coved_count_v1v3 / 3.0

    # 4. V3 slope features
    v3_morph = plm.get("V3", {})
    v3_sp = v3_morph.get("slope_progression", [0.0, 0.0, 0.0, 0.0])
    v3_slope_j120 = v3_sp[3] if len(v3_sp) > 3 else 0.0
    v3_slope_j40 = v3_sp[1] if len(v3_sp) > 1 else 0.0

    # 5. ST descent integral (V1-V3 mean) using j_plus amplitudes
    st_integrals: List[float] = []
    for ld in right_leads:
        ld_morph = plm.get(ld, {})
        j0 = ld_morph.get("j_point_amplitude", 0.0)
        j60 = ld_morph.get("j_plus_60_amplitude", 0.0)
        j80 = ld_morph.get("j_plus_80_amplitude", 0.0)
        j100 = ld_morph.get("j_plus_100_amplitude", 0.0)
        j120 = ld_morph.get("j_plus_120_amplitude", 0.0)
        integral = (
            (j0 + j60) / 2 * 60.0
            + (j60 + j80) / 2 * 20.0
            + (j80 + j100) / 2 * 20.0
            + (j100 + j120) / 2 * 20.0
        )
        st_integrals.append(integral)
    st_descent_integral_mean = sum(st_integrals) / len(st_integrals) if st_integrals else 0.0

    # 6. Right-left precordial contrast
    left_leads = ["V4", "V5", "V6"]
    j_amps_left = [plm.get(ld, {}).get("j_point_amplitude", 0.0) for ld in left_leads]
    j_mean_right = j_elev_mean_v1v3
    j_mean_left = sum(j_amps_left) / len(j_amps_left) if j_amps_left else 0.0
    rl_precordial_contrast = j_mean_right - j_mean_left

    # 7. ST-T ratio aggregate
    st_t_ratios_v1v3 = [plm.get(ld, {}).get("st_t_ratio", 0.0) for ld in right_leads]
    st_t_ratio_mean_v1v3 = (
        sum(st_t_ratios_v1v3) / len(st_t_ratios_v1v3) if st_t_ratios_v1v3 else 0.0
    )

    # 8. Type 1 triad score (best lead)
    type1_triad_score = 0.0
    for ld in right_leads:
        ld_morph = plm.get(ld, {})
        criteria_met = 0
        if ld_morph.get("j_point_amplitude", 0.0) >= 0.2:
            criteria_met += 1
        if ld_morph.get("shape", "") in ("convex", "downsloping"):
            criteria_met += 1
        if ld_morph.get("t_wave_polarity", "positive") == "negative":
            criteria_met += 1
        type1_triad_score = max(type1_triad_score, criteria_met)

    # Store new features in semantic_flags
    semantic_flags["ST"]["j_elev_max_v1v3"] = round(j_elev_max_v1v3, 4)
    semantic_flags["ST"]["j_elev_mean_v1v3"] = round(j_elev_mean_v1v3, 4)
    semantic_flags["ST"]["j_exceeds_2mm"] = j_exceeds_2mm
    semantic_flags["ST"]["t_amp_mean_v1v3"] = round(t_amp_mean_v1v3, 4)
    semantic_flags["ST"]["t_inv_count_v1v3"] = t_inv_count_v1v3
    semantic_flags["ST"]["t_inv_concordance"] = t_inv_concordance
    semantic_flags["ST"]["coved_count_v1v3"] = coved_count_v1v3
    semantic_flags["ST"]["coved_concordance"] = coved_concordance
    semantic_flags["ST"]["v3_slope_j120"] = round(v3_slope_j120, 4)
    semantic_flags["ST"]["v3_slope_j40"] = round(v3_slope_j40, 4)
    semantic_flags["ST"]["st_descent_integral_mean"] = round(st_descent_integral_mean, 4)
    semantic_flags["ST"]["rl_precordial_contrast"] = round(rl_precordial_contrast, 4)
    semantic_flags["ST"]["st_t_ratio_mean_v1v3"] = round(st_t_ratio_mean_v1v3, 4)
    semantic_flags["ST"]["type1_triad_score"] = type1_triad_score

    # --- REBUILT BRUGADA COMPOSITE SCORE ---
    # Normalize components to [0, 1] where higher = more Brugada-like
    c_v2_j120 = max(0.0, min(1.0, -(v2_sj120 - 2.0) / 3.0))
    c_v2_j40 = max(0.0, min(1.0, -(v2_sj40 - 0.0) / 2.0))
    c_v2_slope = max(0.0, min(1.0, -(v2_slope_overall - 0.0) / 2.0))
    c_v1_j80 = max(0.0, min(1.0, -(v1_sj80 - 0.0) / 2.0))
    c_v1_j40 = max(0.0, min(1.0, -(v1_sj40 - 0.0) / 2.0))
    c_j_elev = max(0.0, min(1.0, j_elev_max_v1v3 / 0.3))
    c_t_inv = t_inv_count_v1v3 / 3.0
    c_t_depth = max(0.0, min(1.0, -t_amp_mean_v1v3 / 0.3))
    c_coved = coved_concordance

    brugada_score = (
        0.15 * c_v2_j120
        + 0.10 * c_v2_j40
        + 0.08 * c_v2_slope
        + 0.08 * c_v1_j80
        + 0.07 * c_v1_j40
        + 0.20 * c_j_elev
        + 0.10 * c_t_inv
        + 0.07 * c_t_depth
        + 0.05 * v2_peak_then_descent
        + 0.05 * c_coved
        + 0.05 * max(0.0, min(1.0, type1_triad_score / 3.0))
    )
    # Total weights: 0.15+0.10+0.08+0.08+0.07+0.20+0.10+0.07+0.05+0.05+0.05 = 1.00

    semantic_flags["ST"]["brugada_likelihood_score"] = round(brugada_score, 4)
    computed["Brugada_Likelihood_Score"] = round(brugada_score, 4)
    # Mirror likelihood_score INTO the brugada_pattern dict so downstream LLM
    # consumers can read it alongside detected/type/affected_leads. Matches
    # deployed Apr28 GT JSONL structure (computed["brugada_pattern"]
    # ["likelihood_score"]) which is added post-cell-emission in the deployed
    # pipeline but not visible in the notebook source itself.
    if isinstance(computed.get("brugada_pattern"), dict):
        computed["brugada_pattern"]["likelihood_score"] = round(brugada_score, 4)
    if isinstance(semantic_flags["ST"].get("brugada_pattern"), dict):
        semantic_flags["ST"]["brugada_pattern"]["likelihood_score"] = round(brugada_score, 4)

    # --- Build J-point-aware ST summary text for later integration ---
    try:
        elev_feat = (st_semantic_features or {}).get("st_elevation") or {}
        depr_feat = (st_semantic_features or {}).get("st_depression") or {}

        parts: List[str] = []
        if elev_feat.get("present"):
            desc = elev_feat.get("description", "")
            mv = elev_feat.get("max_deviation_mv")
            if mv is not None:
                desc = f"{desc}; max {mv:.2f} mV"
            parts.append(desc)

        if depr_feat.get("present"):
            desc = depr_feat.get("description", "")
            mv = depr_feat.get("max_deviation_mv")
            if mv is not None:
                desc = f"{desc}; max {mv:.2f} mV"
            parts.append(desc)

        if not parts:
            st_summary_text = "ST segment: normal morphology"
        else:
            st_summary_text = "; ".join(p for p in parts if p)

        semantic_flags.setdefault("clinical_summaries", {})
        semantic_flags["clinical_summaries"]["ST_analysis"] = st_summary_text

    except Exception as e:
        logger.warning("Could not build ST summary text: %s", e)

    if st_semantic_features:
        elev_f = st_semantic_features.get("st_elevation", {})
        depr_f = st_semantic_features.get("st_depression", {})
        parts2: List[str] = []
        if elev_f.get("present"):
            parts2.append(elev_f.get("description", ""))
            parts2.append(f"max {elev_f.get('max_deviation_mv', 0):.2f} mV")
        if depr_f.get("present"):
            parts2.append(depr_f.get("description", ""))
            parts2.append(f"max {depr_f.get('max_deviation_mv', 0):.2f} mV")
        if not parts2:
            st_summary_text2 = "ST segment: normal morphology"
        else:
            st_summary_text2 = "; ".join(p for p in parts2 if p)
        semantic_flags.setdefault("clinical_summaries", {})["ST_analysis"] = st_summary_text2

        # 8.4 P-wave enrichment from fitter outputs (optional; only fills gaps)
        pfit = getattr(feature_ctx, "pfit_summary", None)
        if isinstance(pfit, dict):
            P = semantic_flags.setdefault("P", {})

            if pfit.get("PR_prolonged") and "p_wave_first_degree_av_block" not in P:
                P["p_wave_first_degree_av_block"] = {"present": True, "description": "PR prolonged"}
            if pfit.get("PR_short") and "p_wave_short_pr_syndrome" not in P:
                P["p_wave_short_pr_syndrome"] = {"present": True, "description": "short PR"}
            if pfit.get("P_notched") and "p_wave_notching" not in P:
                P["p_wave_notching"] = {"present": True, "affected_leads": pfit.get("leads_notched", [])}
            if (pfit.get("flutter_like") or pfit.get("f_waves")) and "p_wave_atrial_f_wave" not in P:
                P["p_wave_atrial_f_wave"] = {"present": True, "description": "atrial f-waves"}
            if pfit.get("p_absent_or_unstable") and "p_wave_atrial_fibrillation" not in P:
                P["p_wave_atrial_fibrillation"] = {"present": True, "description": "no stable discrete P"}
            if isinstance(pfit.get("P_axis_deg"), (int, float)) and "p_wave_axis_deviation" not in P:
                deg = float(pfit["P_axis_deg"])
                P["p_wave_axis_deviation"] = {"present": True, "description": f"P axis {deg:.0f} deg"}

    # 8.4a STEMI territory features (toggle-gated, default OFF)
    if qpsi_config.compute_stemi_territories:
        try:
            from qpsi.features.stemi_territories import extract_stemi_territory_features
            stemi_feats = extract_stemi_territory_features(semantic_flags)
            semantic_flags["stemi_territories"] = stemi_feats
            logger.info("[QPSI] STEMI territory features: %d extracted", len(stemi_feats))
        except Exception as e:
            logger.warning("[QPSI] STEMI territory extraction failed: %s", e)

    # 8.4b Template distance features (toggle-gated, default OFF)
    if qpsi_config.compute_template_distances:
        try:
            from ammonix.templates.waveform_templates import TemplateSet, extract_gaussian_params
            from qpsi.features.template_distances import extract_template_distance_features
            tpl_path = qpsi_config.resolve_template_path()
            if tpl_path:
                template_set = TemplateSet.load(tpl_path)
                # Build a flat features dict from computed + semantic for param extraction
                _flat_for_tpl = dict(computed)
                for _domain, _dfeats in semantic_flags.items():
                    if isinstance(_dfeats, dict):
                        for _k, _v in _dfeats.items():
                            if isinstance(_v, (int, float)):
                                _flat_for_tpl[f"Sem_{_domain}_{_k}"] = float(_v)
                patient_params = extract_gaussian_params(_flat_for_tpl)
                template_dist_feats = extract_template_distance_features(patient_params, template_set)
                semantic_flags["template_distances"] = template_dist_feats
                logger.info("[QPSI] Template distance features: %d extracted", len(template_dist_feats))
            else:
                logger.debug("[QPSI] Template distances enabled but waveform_templates.json not found")
        except Exception as e:
            logger.warning("[QPSI] Template distance extraction failed: %s", e)

    # 8.5 Post-process
    semantic_flags = postprocess_semantic_flags(semantic_flags, feature_ctx)

    # RR variability metrics for AF
    computed["AF_Metrics_RR_CV"] = rr_stats.get("rr_sd_ms_all", 0.0) / max(1.0, rr_stats.get("rr_mean_ms_all", 1.0))
    computed["AF_Metrics_RR_pNN50"] = rr_stats.get("pnn50", 0.0)

    timings["semantic_extraction"] = time.perf_counter() - step_start
    logger.info("[TIMING] Semantic feature extraction completed in %.4f seconds", timings["semantic_extraction"])

    # ===================================================================
    # 9. Neural network predictions — AF re-activated, OPT-IN (Item 4.3)
    # ===================================================================
    # Per K3 (the private training repository IS the GT), the AF F_waveNet call previously
    # cleared in Feb14_Adele was wired back into the platform.
    #
    # OPT-IN gate (2026-05-18): QPSI colleague signalled that they stopped
    # using neural nets. Default behaviour matches their direction (key not
    # produced) so the production feature space stays consistent with the
    # notebook. Set ``QPSI_AF_NEURAL_ENABLE=1`` (or "true"/"yes"/"on") to
    # opt in — the wire-in path is preserved for any future evaluation,
    # A/B test, or reactivation per the unbiased-feature-space "toggles OK"
    # principle ([[feedback-qpsi-unbiased-measurements]]).
    #
    # Model + Platt calibration paths are env-var-overridable; defaults
    # point at ``WaveMedix/models/``. Graceful-degrade chain when ENABLED:
    # torch missing → debug log + skip key; either file missing → debug
    # log + skip; prediction exception → warning + skip. No sentinel 0.0
    # — downstream sees feature as absent rather than baked-in zero.
    #
    # Other 3 historical CNN models (flutter, SinTach, T-wave) remain
    # unwired because the Apr28 notebook also dropped them.
    import os
    nn_probs: Dict[str, Any] = {}
    _af_neural_enabled = os.environ.get(
        "QPSI_AF_NEURAL_ENABLE", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if _af_neural_enabled:
        _wavemedix_root = Path(__file__).resolve().parent.parent  # qpsi/ → WaveMedix/
        _af_model_path = Path(
            os.environ.get("QPSI_AF_MODEL_PATH",
                           str(_wavemedix_root / "models" / "best_model_AF.pth"))
        )
        _af_calib_path = Path(
            os.environ.get("QPSI_AF_CALIB_PATH",
                           str(_wavemedix_root / "models" / "calib_AF.json"))
        )
        if _af_model_path.exists() and _af_calib_path.exists():
            try:
                from qpsi.nn_models import _af_probability
                _af_prob_value = _af_probability(
                    leads, raw_ecg_12, fs,
                    model_path=_af_model_path,
                    calib_path=_af_calib_path,
                )
                nn_probs["af_neural"] = _af_prob_value
                semantic_flags["af_neural_prob"] = _af_prob_value
            except ImportError as _af_exc:
                logger.debug("AF neural probability skipped (torch missing): %s", _af_exc)
            except Exception as _af_exc:
                logger.warning("AF neural probability failed: %s", _af_exc)
        else:
            logger.debug(
                "AF neural model not on disk (model=%s, calib=%s); skipping",
                _af_model_path, _af_calib_path,
            )

    # ===================================================================
    # 10. Generate output
    # ===================================================================
    step_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("GENERATING OUTPUT")
    logger.info("=" * 60)

    # Process domains and create summaries
    clinical_summaries: Dict[str, str] = {}
    feature_counts: Dict[str, int] = {}
    all_detected_features: Dict[str, List[str]] = {}
    critical_findings: List[str] = []

    domains = ["T", "P", "QRS", "ST", "rhythm", "global"]
    domain_names: Dict[str, str] = {
        "T": "T-WAVE",
        "P": "P-WAVE",
        "QRS": "QRS COMPLEX",
        "ST": "ST SEGMENT",
        "rhythm": "RHYTHM",
        "global": "GLOBAL",
    }

    # --- preserve J-point-aware ST summary if already built earlier ---
    existing_st_summary: Optional[str] = None
    if "clinical_summaries" in semantic_flags and "ST_analysis" in semantic_flags["clinical_summaries"]:
        existing_st_summary = semantic_flags["clinical_summaries"]["ST_analysis"]

    for domain in domains:
        feats = semantic_flags.get(domain) or {}
        if not feats:
            continue

        present_features = [
            name for name, fd in feats.items()
            if isinstance(fd, dict) and fd.get("present")
        ]
        feature_counts[f"{domain}_count"] = len(present_features)
        if present_features:
            all_detected_features[domain] = present_features

        # Generate domain-specific summary
        if domain == "ST":
            if present_features:
                st_descriptions: List[str] = []
                urgency = "normal"

                for feat_name, feat_data in feats.items():
                    if isinstance(feat_data, dict) and feat_data.get("present"):
                        st_descriptions.append(feat_data.get("description", ""))
                        feat_urgency = feat_data.get("urgency", "normal")
                        if feat_urgency in ["EMERGENCY", "URGENT"]:
                            urgency = feat_urgency
                        elif feat_urgency == "abnormal" and urgency == "normal":
                            urgency = "abnormal"

                st_summary_obj = st_analysis.get("summary", {})
                primary_diag = st_summary_obj.get("primary_diagnosis", "")

                if primary_diag and primary_diag != "Normal ST segments":
                    st_text_out = f"{primary_diag}. {' '.join(st_descriptions)}"
                elif st_descriptions:
                    st_text_out = " ".join(st_descriptions)
                else:
                    st_text_out = "ST segment analysis: abnormalities detected"

                if urgency in ["EMERGENCY", "URGENT"]:
                    st_text_out = f"{urgency}: {st_text_out}"
                    critical_findings.append(f"ST SEGMENT -- {st_text_out}")
            else:
                st_text_out = "ST segment analysis: no significant deviation"

            # Preserve earlier J-point summary if present
            if existing_st_summary:
                clinical_summaries["ST_analysis"] = existing_st_summary
            else:
                clinical_summaries["ST_analysis"] = st_text_out

        elif domain == "rhythm":
            summary_rhythm = generate_rhythm_clinical_summary(feats)
            clinical_summaries["rhythm_analysis"] = summary_rhythm

        else:
            if domain == "T":
                summary_dom = generate_t_wave_clinical_summary(feats)
            elif domain == "P":
                summary_dom = generate_p_wave_clinical_summary(feats)
            elif domain == "QRS":
                summary_dom = generate_qrs_clinical_summary(feats)
            elif domain == "global":
                summary_dom = generate_global_clinical_summary(feats)
            else:
                if present_features:
                    summary_dom = (
                        f"{domain_names.get(domain, domain)} abnormalities detected: "
                        f"{', '.join(present_features)}"
                    )
                else:
                    summary_dom = f"{domain_names.get(domain, domain)} analysis: normal"
            clinical_summaries[f"{domain}_analysis"] = summary_dom

        # Collect critical findings
        for feat_name, fd in feats.items():
            if not (isinstance(fd, dict) and fd.get("present")):
                continue
            desc = fd.get("description", "")
            urg = fd.get("urgency", "")
            if urg in ["EMERGENCY", "URGENT"] or any(
                word in desc.upper()
                for word in ["CRITICAL", "EMERGENCY", "WARNING", "HIGH RISK", "URGENT", "STEMI", "VT", "VF", "f_wave"]
            ):
                if "ST SEGMENT" not in desc:
                    critical_findings.append(f"{domain_names.get(domain, domain)} -- {feat_name}: {desc}")

    # Build multi-Gaussian parameters summary
    multi_gauss: Dict[str, Any] = {}
    if isinstance(lead_fits, dict):
        for lead, fits in lead_fits.items():
            if not isinstance(fits, dict):
                continue
            avg = fits.get("avg") or []
            # Sort by (wave_type priority, component) to match GT ordering:
            # T first (T, T1, T2), then P (P, P1, P2), then QRS (Q, R, S, R2)
            _wt_order = {"T": 0, "P": 1, "QRS": 2}
            avg_sorted = sorted(avg, key=lambda c: (
                _wt_order.get(c.get("wave_type", ""), 3),
                c.get("component", ""),
            ))
            byb = fits.get("by_beat") or []
            try:
                n_beats = len(byb)
            except Exception:
                n_beats = 0
            multi_gauss[lead] = {
                "avg_gaussians": avg_sorted,
                "num_beats": n_beats,
            }

    # Build SIMPLIFIED semantic_features
    semantic_features_simplified: Dict[str, Any] = {
        "clinical_summaries": clinical_summaries,
        "semantic_flags": {
            domain: {
                fname: fdata
                for fname, fdata in (semantic_flags.get(domain, {}) or {}).items()
                if isinstance(fdata, dict) and fdata.get("present")
            }
            for domain in domains
            if domain in semantic_flags
        },
        "multi_gaussian_parameters": multi_gauss,
    }

    # Build ECG data category
    ecg_data_category: Dict[str, Any] = {
        "Limb": _build_plane_ecg_data(
            lumps_limb, p_vars_limb, t_vars_limb,
            extra_beats, arrhythmic_beats, rr_stats,
            extras_limb_av,
        ),
        "Chest": _build_plane_ecg_data(
            lumps_chest, p_vars_chest, t_vars_chest,
            extra_beats, arrhythmic_beats, rr_stats,
            extras_chest_av,
        ),
        "Computed Parameters": _build_computed_parameters(
            lumps_limb, lumps_chest, rr_stats, extras_limb_av,
            extras_chest_av, extra_beats, arrhythmic_beats,
            total_beats, identical_beats, pacing_waves_total,
            computed,
        ),
    }

    # Merge any new computed metrics (e.g. from QWVA or f_wave)
    if computed:
        ecg_data_category["Computed Parameters"].update(computed)

    # Also add explicit RR variability metrics for AF
    ecg_data_category["Computed Parameters"]["AF_Metrics_RR_CV"] = (
        rr_stats.get("rr_sd_ms_all", 0.0) / max(1.0, rr_stats.get("rr_mean_ms_all", 1.0))
    )
    ecg_data_category["Computed Parameters"]["AF_Metrics_RR_pNN50"] = rr_stats.get("pnn50", 0.0)

    # Build Label category
    label_category: Dict[str, Any] = {
        "recording id": recording_id,
        "Patient Data": _build_patient_data(meta),
        "Diagnosis": _extract_diagnosis_codes(meta),
    }

    # --- assemble final JSON ---
    # Inject J-point-aware ST text if it exists
    if "clinical_summaries" in semantic_flags and "ST_analysis" in semantic_flags["clinical_summaries"]:
        jpoint_summary = semantic_flags["clinical_summaries"]["ST_analysis"]
        if jpoint_summary and "ST_analysis" in clinical_summaries:
            # merge both summaries if distinct
            if jpoint_summary.strip() != clinical_summaries["ST_analysis"].strip():
                clinical_summaries["ST_analysis"] = (
                    f"{clinical_summaries['ST_analysis']} | J-point: {jpoint_summary}"
                )
        elif jpoint_summary:
            clinical_summaries["ST_analysis"] = jpoint_summary

    # ensure merged ST summary appears in semantic_flags for consistency
    semantic_flags.setdefault("clinical_summaries", {})
    semantic_flags["clinical_summaries"]["ST_analysis"] = clinical_summaries.get("ST_analysis", "")

    # ============================================================
    # Direct-measurement parallel feature track (Phase 2 §4.6)
    # ============================================================
    # When ``qpsi_config.compute_direct_measurements`` is True, compute the
    # qpsi_direct no-Gaussian primitives over all beats × all leads ×
    # {P, QRS, T} regions, plus per-beat plane primitives (limb + chest planes
    # via Einthoven/Wilson matrices), then aggregate to robust per-recording
    # statistics including multi-period alternation indices (P-wave alternans,
    # trigeminy, quadrigeminy) and composite morphology-alternans score per
    # (lead/plane, region). Adds 4 net-new top-level keys: ``direct_features``
    # (flat), ``direct_aggregated`` (nested per-lead), ``plane_direct_features``
    # (flat plane), and ``plane_direct_aggregated`` (nested per-plane).
    #
    # Toggle defaults OFF for backward compatibility; opt-in via
    # ``QPSIConfig(compute_direct_measurements=True)``. When OFF the 4 keys
    # are still added to ``final_json`` but as empty dicts, so consumers can
    # depend on the keys existing.
    #
    # The try/except is broad (``BaseException``) because the direct
    # measurement code path goes through scipy + numpy primitives that can
    # raise unusual exceptions on degraded recordings; a failure here must
    # not abort the legacy Gaussian pipeline.
    direct_flat: Dict[str, float] = {}
    plane_flat: Dict[str, float] = {}
    agg_direct: Dict[str, Any] = {}
    agg_plane: Dict[str, Any] = {}
    if qpsi_config.compute_direct_measurements:
        try:
            from qpsi.direct_measurements import (
                measure_all_beats,
                aggregate_recording,
                aggregate_plane_recording,
                flatten_features,
            )
            from qpsi.constants import LIMB_MAT, CHEST_MAT
            per_beat_direct, per_beat_plane = measure_all_beats(
                beats_all, leads, fs=fs,
                limb_mat=LIMB_MAT, chest_mat=CHEST_MAT,
            )
            agg_direct = aggregate_recording(per_beat_direct, leads)
            agg_plane = aggregate_plane_recording(per_beat_plane)
            direct_flat = flatten_features(agg_direct, prefix="ml_direct")
            plane_flat = flatten_features(agg_plane, prefix="plane_direct")
        except BaseException as _e_direct:
            logger.warning("direct_measurements failed: %s", _e_direct)

    final_json = build_final_record(
        ecg_data=ecg_data_category,
        semantic_flags=semantic_flags,
        label_block=label_category,
        multi_gaussian_parameters=multi_gauss,
    )
    # Attach direct-measurement tracks (empty dicts when toggle OFF, so
    # consumers can always depend on the keys existing).
    #   * "direct_features"        — per-lead flat dict (xgb-consumable)
    #   * "direct_aggregated"      — per-lead nested (lead → region → scalar → stats)
    #   * "plane_direct_features"  — per-plane flat dict (xgb-consumable)
    #   * "plane_direct_aggregated" — per-plane nested (limb/chest → region → scalar → stats)
    final_json["direct_features"] = direct_flat
    final_json["direct_aggregated"] = agg_direct
    final_json["plane_direct_features"] = plane_flat
    final_json["plane_direct_aggregated"] = agg_plane

    # --- write JSONL (only when an explicit path is provided) ---
    if output_jsonl_path is not None:
        output_path = Path(output_jsonl_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(ensure_json_serializable(final_json), ensure_ascii=False) + "\n")
        logger.info("Structured JSON appended -> %s", output_path)
    timings["output_generation"] = time.perf_counter() - step_start
    logger.info("[TIMING] Output generation completed in %.4f seconds", timings["output_generation"])

    # ===================================================================
    # 11. Return comprehensive results
    # ===================================================================
    results: Dict[str, Any] = {
        "recording_id": recording_id,
        "lumps_limb": lumps_limb,
        "lumps_chest": lumps_chest,
        "rr_stats": rr_stats,
        "lead_fits": lead_fits,
        "st_analysis": st_analysis,
        "f_wave_analysis": f_wave_result,
        "st_info_limb": extras_limb_av,
        "st_info_chest": extras_chest_av,
        "semantic_features": semantic_features_simplified,
        "nn_probs": nn_probs,
        "arrhythmic_beats": arrhythmic_beats,
        "extra_beats": extra_beats,
        "total_beats": total_beats,
        "identical_beats": identical_beats,
        "variability": {"P": p_var_stats, "T": t_var_stats},
        "_final_json": final_json,
        "timings": timings,
    }

    # Final timing summary
    timings["total"] = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info("[TIMING SUMMARY] Total pipeline execution time: %.4f seconds", timings["total"])
    for step_name, step_secs in timings.items():
        if step_name != "total":
            logger.info("  %-25s %8.4f s  (%5.1f%%)", step_name, step_secs, 100.0 * step_secs / max(timings["total"], 1e-9))
    logger.info("=" * 70)

    logger.info("Analysis complete for %s", recording_id)
    logger.info("  Total beats: %d", total_beats)
    logger.info("  Identical beats: %d", identical_beats)

    if f_wave_result and f_wave_result.get("atrial_f_wave", {}).get("present"):
        logger.info("  ATRIAL f_wave DETECTED")

    if st_analysis and "summary" in st_analysis:
        logger.info(
            "  ST segment status: %s",
            st_analysis["summary"].get("primary_diagnosis", "Unknown"),
        )

        if st_analysis["summary"].get("urgency") in ["EMERGENCY", "URGENT"]:
            logger.info("=" * 70)
            logger.info("URGENT CLINICAL ALERT: %s", st_analysis["summary"]["primary_diagnosis"])
            logger.info("Recommendations:")
            for rec in st_analysis["summary"].get("recommendations", []):
                logger.info("  - %s", rec)
            logger.info("=" * 70)

    logger.debug("rr_good_ms: %s", rr_good_ms)
    return results
