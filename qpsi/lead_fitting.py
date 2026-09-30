"""
Lead-wise wave fitting — per-lead P/QRS/T decomposition and energy trace analysis.

Three main functions:
  * ``analyse_energy_trace`` — NEW Feb14: P-wave energy analysis for AF vs APB.
  * ``fit_lead_waves_enhanced`` — fit P/QRS/T on average + per-beat traces.
  * ``process_leads_with_proper_timing`` — top-level entry used by the pipeline.

Source: Cell 16 (lines 859-966, 1824-2240) of q_psi_ai_for_ecg_Feb14_Adele.ipynb.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qpsi.wave_classification import _blank_bounds
from qpsi.wave_fitters import (
    build_t_tail_residual_from_prior_t_components,
    fit_p_wave_components,
    fit_t_wave_components,
    _fit_qrs_unified,
    p_center_consensus,
)
from qpsi.lead_fitting_helpers import (
    _extract_t_wave,
    create_wave_bounds_from_lumps_enhanced,
    estimate_pr_interval,
    _fit_multi_gaussian,
)
from qpsi.segmentation import pad_beats
from qpsi.constants import logger

# Rust parallel beat fitting (optional)
try:
    import qpsi_native as _qpsi_native
    if not hasattr(_qpsi_native, "fit_all_leads_beats_parallel"):
        raise ImportError("qpsi_native found but missing Rust functions")
    _HAS_RUST_PARALLEL = True
except ImportError:
    _HAS_RUST_PARALLEL = False


# ---------------------------------------------------------------------------
# 1)  Energy trace analysis (NEW Feb14)
# ---------------------------------------------------------------------------

def analyse_energy_trace(
    energy_signal: np.ndarray,
    time_ms: np.ndarray,
    wave_bounds: Dict[str, Tuple[float, float]],
    *,
    min_p_energy_snr: float = -6.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """Analyse energy trace for P-wave activity (AF vs APB discrimination).

    Fits P-wave components in the energy domain and applies physiological
    filters (amplitude cap, T-tail shape, noise floor) to discriminate
    genuine P-wave activity from noise/artifacts.

    Parameters
    ----------
    energy_signal : 1-D energy envelope.
    time_ms : corresponding time axis in ms.
    wave_bounds : ``{"P": (start, end), "QRS": (start, end), ...}``.
    min_p_energy_snr : minimum SNR for P-wave detection in dB.
    debug : print diagnostics.

    Returns
    -------
    Dict with ``"lumps"`` (list of labeled P-energy components) and
    ``"metrics"`` (integral, max amp, discrimination code).
    """
    # 1. Determine QRS Max Energy (reference standard)
    q0, q1 = wave_bounds.get("QRS", (-50.0, 80.0))
    qrs_mask = (time_ms >= q0) & (time_ms <= q1)

    if np.any(qrs_mask):
        qrs_max_energy = float(np.max(energy_signal[qrs_mask]))
    else:
        qrs_max_energy = 1.0

    # Dynamic physiological cap: P should not exceed 25% of QRS energy
    p_amp_cap = 0.25 * qrs_max_energy

    # 2. Setup P-window
    p0, p1 = wave_bounds.get("P", (-200.0, -50.0))
    p_mask = (time_ms >= p0) & (time_ms <= p1)

    # Fit P-wave components in energy domain
    p_comps, p_info = fit_p_wave_components(
        time_ms[p_mask],
        energy_signal[p_mask],
        sigma_bounds_ms=(6.0, 30.0),
        min_separation_ms=15.0,
        max_separation_ms=150.0,
        allow_biphasic=False,
        allow_two=True,
        min_p_snr_db=min_p_energy_snr,
        prefer_early=True,
        af_flutter_probe=False,
    )

    p_energy_integral = 0.0
    p_max_amp = 0.0
    labeled_comps: List[Dict[str, Any]] = []

    # Estimate baseline noise in the P-window
    local_noise_floor = (
        float(np.percentile(energy_signal[p_mask], 10))
        if np.any(p_mask)
        else 0.0
    )

    # Relaxed detection threshold (rely on shape and cap to filter noise)
    detection_threshold = max(0.008, 1.2 * local_noise_floor)

    if p_comps:
        for c in p_comps:
            A = float(c.get("amp_mv", 0.0))
            s = float(c.get("sigma_ms", 1.0))

            # Filter 1: Physiological cap (rejects artifacts / QRS bleed)
            if A > p_amp_cap:
                continue

            # Filter 2: T-wave / noise shape (rejects T-tails)
            if s > 25.0 and A < (0.05 * qrs_max_energy):
                continue

            # Filter 3: Minimum detection (rejects baseline noise)
            if s < 3.0 or abs(A) < detection_threshold:
                continue

            area = abs(A) * s * 2.5066  # sqrt(2*pi)
            p_energy_integral += area
            p_max_amp = max(p_max_amp, abs(A))

            c_out = c.copy()
            c_out["wave_type"] = "P_Energy"
            c_out["area"] = area
            labeled_comps.append(c_out)

    # 5. Discrimination logic
    has_energy = p_energy_integral > 0.5
    has_peak = p_max_amp > detection_threshold

    if not has_energy or not has_peak:
        discrimination = "AF_LIKELY"
    else:
        discrimination = "P_ACTIVITY_PRESENT"

    return {
        "lumps": labeled_comps,
        "metrics": {
            "p_energy_integral": float(p_energy_integral),
            "p_max_amp": float(p_max_amp),
            "p_component_count": len(labeled_comps),
            "discrimination_code": discrimination,
            "snr_db": float(p_info.get("snr_db", 0.0)),
        },
    }


# ---------------------------------------------------------------------------
# 2)  Per-lead wave fitting (average + per-beat)
# ---------------------------------------------------------------------------

def fit_lead_waves_enhanced(
    beat_stack: np.ndarray,
    time_ms: np.ndarray,
    wave_bounds: Dict[str, Tuple[float, float]],
    lead_name: str = "II",
    debug_png: Optional[Path] = None,
    fs: float = 500.0,
    rr_mean_ms: Optional[float] = None,
    offset_ms: float = 50.0,
    pruning: str = "all",
) -> Dict[str, Any]:
    """Fit P/QRS/T on the average trace and each beat individually.

    Fitting order: T first (to build P residual for long-PR cases),
    then P (with T-tail residual hint), then QRS (unified fitter).

    Parameters
    ----------
    beat_stack : ``(n_beats, n_samples)`` array for one lead.
    time_ms : shared time axis in ms (0 = R-peak).
    wave_bounds : ``{"P": (lo, hi), "QRS": (lo, hi), "T": (lo, hi)}``.
    lead_name : lead identifier for diagnostics.
    debug_png : if not None, save QC plot.
    fs : sampling frequency in Hz.
    rr_mean_ms : mean RR interval in ms (for T-tail projection).
    offset_ms : timing offset in ms (Feb14 addition).

    Returns
    -------
    Dict with ``"avg"`` (list of Gaussian dicts), ``"by_beat"``
    (list of lists), and ``"p_info"`` dict.
    """
    n_beats, n_samples = beat_stack.shape
    avg_trace = np.nanmean(beat_stack, axis=0)

    out: Dict[str, Any] = {
        "avg": [],
        "by_beat": [[] for _ in range(n_beats)],
    }

    p0, p1 = wave_bounds["P"]
    q0, q1 = wave_bounds["QRS"]
    t0, t1 = wave_bounds["T"]

    # ===================================================================
    # STEP 1: Fit the average trace
    # ===================================================================

    # --- T wave (fit FIRST) ---
    t_mask = (time_ms >= t0) & (time_ms <= t1)
    try:
        compsT_avg, _shapeT_avg = fit_t_wave_components(
            time_ms[t_mask], avg_trace[t_mask],
            sigma_bounds_ms=(30.0, 140.0),
            min_separation_ms=40.0, max_separation_ms=180.0,
            aic_delta_two=4.0, allow_biphasic=True, allow_two=True,
        )
        for c in compsT_avg:
            out["avg"].append(c)
    except Exception as e:
        logger.warning("T wave fitting failed for %s: %s", lead_name, e)
        compsT_avg = []

    # --- Build T-tail residual projected into P window ---
    p_mask = (time_ms >= p0) & (time_ms <= p1)
    shift_ms = -float(rr_mean_ms or 0.0)
    residual_hint_mv_avg = None
    if compsT_avg:
        try:
            residual_hint_mv_avg = build_t_tail_residual_from_prior_t_components(
                time_ms[p_mask],
                prev_t_components=compsT_avg,
                shift_ms=shift_ms,
                gating="right-soft",
                gate_width_ms=12.0,
            )
        except Exception as e:
            logger.warning("T-tail residual failed for %s: %s", lead_name, e)
            residual_hint_mv_avg = None

    # --- P wave on average ---
    try:
        compsP_avg, infoP_avg = fit_p_wave_components(
            time_ms[p_mask], avg_trace[p_mask],
            residual_hint_mv=residual_hint_mv_avg,
            sigma_bounds_ms=(8.0, 60.0),
            min_separation_ms=18.0, max_separation_ms=140.0,
            aic_delta_two=4.0, allow_biphasic=True, allow_two=True,
        )
        # Filter out P-waves that are likely previous T-wave
        limit_ms = -0.5 * (rr_mean_ms if rr_mean_ms else 800.0) + offset_ms
        compsP_avg = [c for c in compsP_avg if c["center_ms"] >= limit_ms]

        out["avg"].extend(compsP_avg)
        out["p_info"] = dict(infoP_avg)
    except Exception as e:
        logger.warning("P wave fitting failed for %s: %s", lead_name, e)
        out["p_info"] = {"p_present": False, "snr_db": 0.0}

    # --- QRS on average (unified fitter) ---
    qrs_mask = (time_ms >= q0) & (time_ms <= q1)
    if np.any(qrs_mask):
        qrs_trace_avg = avg_trace[qrs_mask]
        qrs_time = time_ms[qrs_mask]
        r_idx = int(np.argmax(np.abs(qrs_trace_avg)))
        r_peak_approx = float(qrs_time[r_idx])
    else:
        r_peak_approx = None
        qrs_time = time_ms  # fallback for variable reference

    qrs_fit_result = None
    try:
        qrs_fit = _fit_qrs_unified(
            avg_trace, time_ms, (q0, q1),
            tR_ms=r_peak_approx, fs=fs,
            min_qrs_ms=60.0, max_qrs_ms=250.0, padding_ms=10.0,
            force_narrow=False, sigma_bounds_ms=(6.0, 60.0),
            max_components=4, q_s_threshold_rel=0.03,
            rprime_threshold_rel=0.05, min_sep_ms=8.0,
        )

        if getattr(qrs_fit, "params", None):
            r_mu = float(qrs_fit.r_peak_time)
            params_list = list(qrs_fit.params)
            idx_R = int(np.argmin(
                [abs(mu - r_mu) for (_, mu, _) in params_list]
            ))
            sign_R = np.sign(params_list[idx_R][0]) or 1.0

            for A, mu, s in params_list:
                if abs(mu - r_mu) < 6.0:
                    label = "R"
                elif mu < r_mu:
                    label = "Q" if np.sign(A) != sign_R else "R2"
                else:
                    label = "S" if np.sign(A) != sign_R else "R2"

                out["avg"].append({
                    "wave_type": "QRS",
                    "amp_mv": float(A),
                    "center_ms": float(mu),
                    "sigma_ms": float(s),
                    "component": label,
                })

            qrs_fit_result = qrs_fit
        else:
            raise ValueError("Unified QRS fit returned no params.")

    except Exception as e:
        logger.warning("Unified QRS fitting failed for %s: %s", lead_name, e)
        parsQRS = _fit_multi_gaussian(avg_trace, time_ms, q0, q1, n_comp=2)
        for i, (A, mu, s) in enumerate(np.reshape(parsQRS, (-1, 3))):
            label = "R" if i == 0 else ("S" if mu > 0 else "Q")
            out["avg"].append({
                "wave_type": "QRS",
                "amp_mv": float(A),
                "center_ms": float(mu),
                "sigma_ms": float(s),
                "component": label,
            })
        qrs_fit_result = None

    # ===================================================================
    # PRUNING — Speed vs Completeness Trade-off
    # ===================================================================
    # Pruning controls how much per-beat fitting work is skipped based on
    # the average-beat result. It is a SPEED optimization only — it does
    # not improve fit quality. With pruning disabled (pruning="none"), every
    # beat gets full P+T+QRS fitting regardless of the average. This is
    # slower but guarantees all per-beat features (alternans, notching, etc.)
    # are computed.
    #
    # Modes:
    #   "none"  — No pruning. Full per-beat fitting for P, T, and QRS (4 components).
    #             Slowest but most complete. Required for p_wave_alternans/notching.
    #             ~2.5s/patient.
    #
    #   "all"   — QRS-clean pruning only. If avg QRS is a clean single Gaussian,
    #             per-beat QRS uses max_components=1 instead of 4 (~30% QRS speedup).
    #             Per-beat P fitting is NOT skipped (F1 fix, 2026-03-31).
    #             ~2.0s/patient.
    #
    #   "p_absence" — Skip per-beat P fitting when avg P is absent. Saves ~1s/patient
    #             for AF/flutter records, but loses p_wave_alternans/notching features.
    #
    #   "p_absence+qrs_clean" — Both P-absence and QRS-clean pruning.
    #
    #   "p_absence+qrs_clean+aicc" — All above plus AICc early termination.
    #
    # Recommendation: use "none" for production extraction (feature completeness),
    # use "all" for interactive/preview (balanced), use "p_absence+qrs_clean"
    # for maximum speed on large batches where P-wave features are not needed.
    # ===================================================================

    # ===================================================================
    # PRUNING: QRS clean-fit detection
    # ===================================================================
    # If the average-beat QRS was well-explained by a single Gaussian (1
    # component, low normalized residual), per-beat QRS fitting can use a
    # narrower search (max_components=1) instead of probing up to 4 lobes.
    # Clinical justification: a clean, monophasic R-wave (common in limb
    # leads and left precordial leads) is morphologically stable beat-to-beat.
    # Multi-component search on each beat wastes ~70% of QRS fitting time
    # when only one lobe is present. The residual threshold (< 0.15 relative
    # to peak) ensures we only prune truly clean QRS, not fragmented ones.
    _QRS_CLEAN_RESIDUAL_THRESHOLD = 0.15
    _prune_qrs = pruning in ("all", "p_absence+qrs_clean", "p_absence+qrs_clean+aicc")
    avg_qrs_is_clean = False
    avg_qrs_n_components = 0
    if _prune_qrs and qrs_fit_result is not None and getattr(qrs_fit_result, "params", None):
        avg_qrs_n_components = len(qrs_fit_result.params)
        if avg_qrs_n_components == 1 and getattr(qrs_fit_result, "yfit", None) is not None:
            fit_mask = getattr(qrs_fit_result, "mask", None)
            if fit_mask is not None and np.any(fit_mask):
                residual = np.sqrt(np.mean((avg_trace[fit_mask] - qrs_fit_result.yfit[fit_mask]) ** 2))
                peak_amp = max(abs(avg_trace[fit_mask].max()), abs(avg_trace[fit_mask].min()), 1e-9)
                avg_qrs_is_clean = (residual / peak_amp) < _QRS_CLEAN_RESIDUAL_THRESHOLD
    if avg_qrs_is_clean:
        logger.debug("PRUNE: QRS clean single-Gaussian — using max_components=1 for per-beat fits")

    # PRUNING: If P-wave was not detected on the average beat (p_present=False),
    # skip P-wave fitting on individual beats entirely. Clinical justification:
    # if the averaged signal (which has the best SNR due to beat averaging) shows
    # no P-wave, individual noisier beats will not reveal one either. This avoids
    # K=0/1/2 model selection on every beat's P-window when the patient has atrial
    # fibrillation or otherwise absent P-waves.
    # F1 fix: "all" no longer suppresses per-beat P-fitting — use explicit
    # "p_absence" modes if P-pruning is desired.  This restores p_wave_alternans,
    # p_wave_notching and other per-beat P features for all records.
    _prune_p = pruning in ("p_absence", "p_absence+qrs_clean", "p_absence+qrs_clean+aicc")
    avg_p_present = out.get("p_info", {}).get("p_present", False)
    # When pruning is disabled, always attempt per-beat P fitting
    if not _prune_p:
        avg_p_present = True
    if not avg_p_present:
        logger.debug("PRUNE: P-wave absent on average — skipping per-beat P fitting for %s", lead_name)

    # ===================================================================
    # STEP 2: Fit each beat individually (for variability analysis)
    # ===================================================================

    # Try Rust parallel T+P fitting for all beats at once
    _used_rust_parallel = False
    if _HAS_RUST_PARALLEL and n_beats >= 2:
        try:
            _stack_c = np.ascontiguousarray(beat_stack, dtype=np.float64)
            _time_c = np.ascontiguousarray(time_ms, dtype=np.float64)
            _stack_flat = _stack_c.ravel()
            _p_flag = np.array([1.0 if avg_p_present else 0.0], dtype=np.float64)
            # Expand p_flag to n_leads=1 (single lead mode)
            _rr = float(rr_mean_ms) if rr_mean_ms else 800.0

            rust_comps = _qpsi_native.fit_all_leads_beats_parallel(
                _stack_flat, _time_c,
                n_beats, 1,  # n_leads=1 (single lead mode)
                float(t0), float(t1),  # T window
                float(p0), float(p1),  # P window
                _p_flag,
                _rr, float(offset_ms),
            )
            # Populate by_beat from Rust results (T and P only)
            for comp_dict in rust_comps:
                bi = int(comp_dict["beat_idx"])
                out["by_beat"][bi].append({
                    "wave_type": str(comp_dict["wave_type"]),
                    "amp_mv": float(comp_dict["amp_mv"]),
                    "center_ms": float(comp_dict["center_ms"]),
                    "sigma_ms": float(comp_dict["sigma_ms"]),
                    "component": str(comp_dict["component"]),
                })
            # QRS fitting still done per-beat in Python (complex fitter)
            for beat_idx in range(n_beats):
                beat_trace = beat_stack[beat_idx, :]
                try:
                    if np.any(qrs_mask):
                        qrs_trace_beat = beat_trace[qrs_mask]
                        r_idx_beat = int(np.argmax(np.abs(qrs_trace_beat)))
                        r_peak_approx_beat = float(qrs_time[r_idx_beat])
                    else:
                        r_peak_approx_beat = None
                    _beat_max_components = 1 if avg_qrs_is_clean else 4
                    qrs_fit_beat = _fit_qrs_unified(
                        beat_trace, time_ms, (q0, q1),
                        tR_ms=r_peak_approx_beat, fs=fs,
                        min_qrs_ms=60.0, max_qrs_ms=250.0, padding_ms=10.0,
                        force_narrow=False, sigma_bounds_ms=(6.0, 60.0),
                        max_components=_beat_max_components, q_s_threshold_rel=0.03,
                        rprime_threshold_rel=0.05, min_sep_ms=8.0,
                    )
                    if getattr(qrs_fit_beat, "params", None):
                        r_mu_beat = float(qrs_fit_beat.r_peak_time)
                        params_list = list(qrs_fit_beat.params)
                        idx_R_beat = int(np.argmin(
                            [abs(mu - r_mu_beat) for (_, mu, _) in params_list]
                        ))
                        sign_R_beat = np.sign(params_list[idx_R_beat][0]) or 1.0
                        for A, mu, s in params_list:
                            if abs(mu - r_mu_beat) < 6.0:
                                label = "R"
                            elif mu < r_mu_beat:
                                label = "Q" if np.sign(A) != sign_R_beat else "R2"
                            else:
                                label = "S" if np.sign(A) != sign_R_beat else "R2"
                            out["by_beat"][beat_idx].append({
                                "wave_type": "QRS", "amp_mv": float(A),
                                "center_ms": float(mu), "sigma_ms": float(s),
                                "component": label,
                            })
                except Exception:
                    try:
                        parsQRS_beat = _fit_multi_gaussian(beat_trace, time_ms, q0, q1, n_comp=2)
                        if parsQRS_beat is not None and parsQRS_beat.size > 0:
                            for i, (A, mu, s) in enumerate(np.reshape(parsQRS_beat, (-1, 3))):
                                label = "R" if i == 0 else ("S" if mu > 0 else "Q")
                                out["by_beat"][beat_idx].append({
                                    "wave_type": "QRS", "amp_mv": float(A),
                                    "center_ms": float(mu), "sigma_ms": float(s),
                                    "component": label,
                                })
                    except Exception as e:
                        # H47 / Q-A6-11: best-effort beat-level QRS fallback;
                        # if both unified + multi-gaussian fail, leave the
                        # beat's component list empty rather than crashing
                        # the pipeline. Logged at debug to keep failure rate
                        # observable without spamming WARN/INFO.
                        logger.debug(
                            "Rust-path QRS fallback failed for %s beat %d: %s",
                            lead_name, beat_idx, e,
                        )
            _used_rust_parallel = True
        except Exception as e:
            logger.debug("Rust parallel beat fitting failed, falling back to Python: %s", e)

    if not _used_rust_parallel:
        for beat_idx in range(n_beats):
            beat_trace = beat_stack[beat_idx, :]
            beat_components: List[Dict[str, Any]] = []

            try:
                compsT_beat, _ = fit_t_wave_components(
                    time_ms[t_mask], beat_trace[t_mask],
                    sigma_bounds_ms=(30.0, 140.0),
                    min_separation_ms=40.0, max_separation_ms=180.0,
                    aic_delta_two=4.0, allow_biphasic=True, allow_two=True,
                )
                beat_components.extend(compsT_beat)
            except Exception as e:
                # H47 / Q-A6-11: per-beat T-wave fitting is best-effort —
                # leave T components empty for this beat rather than crash.
                logger.debug(
                    "Per-beat T-wave fitting failed for %s beat %d: %s",
                    lead_name, beat_idx, e,
                )

            # PRUNING: Skip per-beat P-wave fitting when average P absent.
            if avg_p_present:
                try:
                    residual_hint_mv_beat = None
                    if beat_components:
                        residual_hint_mv_beat = (
                            build_t_tail_residual_from_prior_t_components(
                                time_ms[p_mask],
                                prev_t_components=[
                                    c for c in beat_components
                                    if c.get("wave_type") == "T"
                                ],
                                shift_ms=shift_ms,
                                gating="right-soft",
                                gate_width_ms=12.0,
                            )
                        )

                    compsP_beat, _ = fit_p_wave_components(
                        time_ms[p_mask], beat_trace[p_mask],
                        residual_hint_mv=residual_hint_mv_beat,
                        sigma_bounds_ms=(8.0, 60.0),
                        min_separation_ms=18.0, max_separation_ms=140.0,
                        aic_delta_two=4.0, allow_biphasic=True, allow_two=True,
                    )
                    limit_ms = -0.5 * (rr_mean_ms if rr_mean_ms else 800.0) + offset_ms
                    compsP_beat = [c for c in compsP_beat if c["center_ms"] >= limit_ms]
                    beat_components.extend(compsP_beat)
                except Exception as e:
                    # H47 / Q-A6-11: per-beat P-wave fitting is best-effort.
                    logger.debug(
                        "Per-beat P-wave fitting failed for %s beat %d: %s",
                        lead_name, beat_idx, e,
                    )

            try:
                if np.any(qrs_mask):
                    qrs_trace_beat = beat_trace[qrs_mask]
                    r_idx_beat = int(np.argmax(np.abs(qrs_trace_beat)))
                    r_peak_approx_beat = float(qrs_time[r_idx_beat])
                else:
                    r_peak_approx_beat = None

                # PRUNING: Use reduced max_components when avg QRS is clean
                # single-Gaussian. A clean monophasic R avoids probing Q/S/R'
                # lobes that cannot exist in a morphologically simple beat.
                _beat_max_components = 1 if avg_qrs_is_clean else 4
                qrs_fit_beat = _fit_qrs_unified(
                    beat_trace, time_ms, (q0, q1),
                    tR_ms=r_peak_approx_beat, fs=fs,
                    min_qrs_ms=60.0, max_qrs_ms=250.0, padding_ms=10.0,
                    force_narrow=False, sigma_bounds_ms=(6.0, 60.0),
                    max_components=_beat_max_components, q_s_threshold_rel=0.03,
                    rprime_threshold_rel=0.05, min_sep_ms=8.0,
                )

                if getattr(qrs_fit_beat, "params", None):
                    r_mu_beat = float(qrs_fit_beat.r_peak_time)
                    params_list = list(qrs_fit_beat.params)
                    idx_R_beat = int(np.argmin(
                        [abs(mu - r_mu_beat) for (_, mu, _) in params_list]
                    ))
                    sign_R_beat = np.sign(params_list[idx_R_beat][0]) or 1.0

                    for A, mu, s in params_list:
                        if abs(mu - r_mu_beat) < 6.0:
                            label = "R"
                        elif mu < r_mu_beat:
                            label = "Q" if np.sign(A) != sign_R_beat else "R2"
                        else:
                            label = "S" if np.sign(A) != sign_R_beat else "R2"

                        beat_components.append({
                            "wave_type": "QRS",
                            "amp_mv": float(A),
                            "center_ms": float(mu),
                            "sigma_ms": float(s),
                            "component": label,
                        })
                else:
                    raise ValueError("QRS fit returned no params")

            except Exception:
                try:
                    parsQRS_beat = _fit_multi_gaussian(
                        beat_trace, time_ms, q0, q1, n_comp=2,
                    )
                    if parsQRS_beat is not None and parsQRS_beat.size > 0:
                        for i, (A, mu, s) in enumerate(
                            np.reshape(parsQRS_beat, (-1, 3))
                        ):
                            label = "R" if i == 0 else ("S" if mu > 0 else "Q")
                            beat_components.append({
                                "wave_type": "QRS",
                                "amp_mv": float(A),
                                "center_ms": float(mu),
                                "sigma_ms": float(s),
                                "component": label,
                            })
                except Exception as e:
                    # H47 / Q-A6-11: per-beat QRS fallback is best-effort —
                    # both unified and multi-gaussian failed; leave QRS
                    # components empty for this beat. Same fallback shape as
                    # the Rust-parallel path above.
                    logger.debug(
                        "Python-path QRS fallback failed for %s beat %d: %s",
                        lead_name, beat_idx, e,
                    )

            out["by_beat"][beat_idx] = beat_components

    # ===================================================================
    # STEP 3: Optional QC plot (average only)
    # ===================================================================

    if debug_png is not None and qrs_fit_result is not None:
        try:
            from qpsi.plotting import plot_improved_fit
            plot_improved_fit(
                avg_trace, time_ms, wave_bounds, out["avg"],
                qrs_fit_result, lead_name, debug_png,
            )
        except Exception as e:
            # H47 / Q-A6-11: debug PNG is diagnostic-only (caller-requested).
            # Plotting backend issues should not fail extraction.
            logger.debug("Debug plot generation failed for %s: %s", lead_name, e)

    return out


# ---------------------------------------------------------------------------
# 3)  Top-level entry used by the pipeline
# ---------------------------------------------------------------------------

def process_leads_with_proper_timing(
    recording_id: str,
    beats_good: List[Any],
    lumps_limb: List[Dict],
    leads: List[str],
    fs: int,
    plot_enabled: bool = True,
    time_ms: Optional[np.ndarray] = None,
    stack_sync: Optional[np.ndarray] = None,
    rr_stats: Optional[Dict[str, float]] = None,
    offset_ms: float = 50.0,
    pruning: str = "all",
    avg_per_lead: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build windows, fit per-lead on the synchronised stack, return results.

    Adds a gentle second pass to nudge weak/edge P fits using cross-lead
    consensus (non-destructive).

    Parameters
    ----------
    recording_id : patient / recording identifier.
    beats_good : filtered Beat objects.
    lumps_limb : aggregated lumps from limb-plane analysis.
    leads : ordered lead names (length 12 typically).
    fs : sampling frequency.
    plot_enabled : save per-lead QC plots.
    time_ms : pre-computed time axis (if None, computed via ``pad_beats``).
    stack_sync : pre-computed synchronised stack (if None, computed).
    rr_stats : dict with ``"rr_mean_ms"`` key.
    offset_ms : timing offset in ms (Feb14).

    Returns
    -------
    Dict with ``lead_fits``, ``avg_lead``, ``segment_bounds``,
    ``time_ms``, ``wave_bounds``.
    """
    # If caller didn't pass stack/time, compute from beats_good
    if stack_sync is None or time_ms is None:
        stack_sync, time_ms = pad_beats(beats_good, fs=fs)

    assert stack_sync.ndim == 3 and stack_sync.shape[1] == len(leads), (
        f"stack_sync must be (n_beats, {len(leads)}, n_samples); "
        f"got {stack_sync.shape}"
    )

    # Cache per-lead synchronised-stack mean (shape (n_leads, n_samples)).
    # Pipeline.run_pipeline passes this in to avoid recomputing; external
    # callers fall through to a single reduction here.
    if avg_per_lead is None:
        avg_per_lead = np.nanmean(stack_sync, axis=0)

    logger.info(
        "Lead processing: stack %s, time %.1f..%.1f ms (%d samples)",
        stack_sync.shape, time_ms[0], time_ms[-1], len(time_ms),
    )

    # RR for T positioning
    rr_mean_ms = (rr_stats or {}).get("rr_mean_ms", 800.0)
    wave_bounds = create_wave_bounds_from_lumps_enhanced(
        lumps_limb, rr_mean_ms=rr_mean_ms, offset_ms=offset_ms,
    )

    lead_idx_map = {name: i for i, name in enumerate(leads)}

    # ---------- First pass: fit all leads ----------
    # Try Rust parallel avg-trace P+T fitting across all 12 leads at once.
    # Falls back to sequential Python if Rust unavailable.
    lead_fits: Dict[str, Dict[str, Any]] = {}
    _used_rust_avg = False
    try:
        import qpsi_native as _qn
        if hasattr(_qn, "fit_avg_traces_all_leads"):
            n_beats, n_leads_stack, n_samples = stack_sync.shape
            # Flat array of avg traces per lead — sourced from cache
            avg_traces = avg_per_lead[[lead_idx_map[ld] for ld in leads], :]
            avg_flat = np.ascontiguousarray(avg_traces.ravel(), dtype=np.float64)
            time_c = np.ascontiguousarray(time_ms, dtype=np.float64)

            p0, p1 = wave_bounds["P"]
            t0, t1 = wave_bounds["T"]

            q0_b, q1_b = wave_bounds.get("QRS", (-40.0, 80.0))
            rust_results = _qn.fit_avg_traces_all_leads(
                avg_flat, time_c,
                len(leads), n_samples,
                float(p0), float(p1),
                float(t0), float(t1),
                float(rr_mean_ms),
                float(offset_ms),
                float(q0_b), float(q1_b),
                float(fs),
            )

            # Parse Rust results and do QRS fitting in Python (fast)
            for li, lead_name in enumerate(leads):
                rd = rust_results[li]
                lead_stack = stack_sync[:, lead_idx_map[lead_name], :]
                avg_trace = np.nanmean(lead_stack, axis=0)

                # Start building the per-lead result
                out = {"avg": [], "by_beat": [[] for _ in range(n_beats)]}

                # Add T components from Rust (labels T1, T2, etc.)
                for c in rd["t_components"]:
                    out["avg"].append({
                        "wave_type": "T",
                        "amp_mv": float(c["amp_mv"]),
                        "center_ms": float(c["center_ms"]),
                        "sigma_ms": float(c["sigma_ms"]),
                        "component": str(c["component"]),
                    })

                # Add P components from Rust (labels P1, P2, etc.)
                for c in rd["p_components"]:
                    out["avg"].append({
                        "wave_type": "P",
                        "amp_mv": float(c["amp_mv"]),
                        "center_ms": float(c["center_ms"]),
                        "sigma_ms": float(c["sigma_ms"]),
                        "component": str(c["component"]),
                    })

                out["p_info"] = {
                    "present": bool(rd["p_present"]),
                    "p_present": bool(rd["p_present"]),
                    "snr_db": float(rd["p_snr_db"]),
                    "notched": bool(rd["p_notched"]),
                    "p_edge_clipped": bool(rd.get("p_edge_left", False)),
                    "p_overlaps_t_tail": bool(rd.get("p_edge_right", False)),
                }

                # QRS from Rust — uses Rust yfit/mask directly (no Python reconstruction)
                from types import SimpleNamespace
                qrs_fit_result = None
                q0, q1 = wave_bounds["QRS"]
                qrs_mask = (time_ms >= q0) & (time_ms <= q1)
                qrs_from_rust = rd.get("qrs_components", [])
                if qrs_from_rust:
                    _rust_qrs_params = []
                    _r_peak_t = 0.0
                    # Find R component for polarity reference (same as Python line 297-300)
                    for c in qrs_from_rust:
                        if str(c["component"]) == "R":
                            _r_peak_t = float(c["center_ms"])
                            break
                    # Determine R polarity from the component closest to R peak
                    _sign_R = 1.0
                    if _r_peak_t != 0.0:
                        _r_comp = min(qrs_from_rust,
                                      key=lambda c: abs(float(c["center_ms"]) - _r_peak_t))
                        _sign_R = float(np.sign(float(_r_comp["amp_mv"]))) or 1.0

                    for c in qrs_from_rust:
                        A = float(c["amp_mv"])
                        mu = float(c["center_ms"])
                        s = float(c["sigma_ms"])
                        _rust_qrs_params.append((A, mu, s))
                        # Apply Python's EXACT relabeling (lead_fitting.py:302-308)
                        # NOT Rust's geometric labels — Python uses polarity-based labels
                        if abs(mu - _r_peak_t) < 6.0:
                            label = "R"
                        elif mu < _r_peak_t:
                            label = "Q" if np.sign(A) != _sign_R else "R2"
                        else:
                            label = "S" if np.sign(A) != _sign_R else "R2"
                        out["avg"].append({
                            "wave_type": "QRS",
                            "amp_mv": A,
                            "center_ms": mu,
                            "sigma_ms": s,
                            "component": label,
                        })
                    # Use yfit and mask directly from Rust
                    if _rust_qrs_params:
                        _yfit_arr = np.asarray(rd.get("qrs_yfit", []), dtype=float)
                        _mask_arr = np.asarray(rd.get("qrs_mask", []), dtype=bool)
                        if len(_yfit_arr) == len(time_ms) and len(_mask_arr) == len(time_ms):
                            qrs_fit_result = SimpleNamespace(
                                params=_rust_qrs_params,
                                yfit=_yfit_arr,
                                r_peak_time=_r_peak_t,
                                mask=_mask_arr,
                            )
                if qrs_fit_result is None:
                    # Fallback: Python QRS
                    if np.any(qrs_mask):
                        qrs_trace_avg = avg_trace[qrs_mask]
                        qrs_time = time_ms[qrs_mask]
                        r_idx = int(np.argmax(np.abs(qrs_trace_avg)))
                        r_peak_approx = float(qrs_time[r_idx])
                        try:
                            qrs_fit = _fit_qrs_unified(
                                avg_trace, time_ms, (q0, q1),
                                tR_ms=r_peak_approx, fs=fs,
                                min_qrs_ms=60.0, max_qrs_ms=250.0, padding_ms=10.0,
                                force_narrow=False, sigma_bounds_ms=(6.0, 60.0),
                                max_components=4, q_s_threshold_rel=0.03,
                                rprime_threshold_rel=0.05, min_sep_ms=8.0,
                            )
                            if getattr(qrs_fit, "params", None):
                                r_mu = float(qrs_fit.r_peak_time)
                                params_list = list(qrs_fit.params)
                                idx_R = int(np.argmin(
                                    [abs(mu - r_mu) for (_, mu, _) in params_list]
                                ))
                                sign_R_py = np.sign(params_list[idx_R][0]) or 1.0
                                for A, mu, s in params_list:
                                    if abs(mu - r_mu) < 6.0:
                                        label = "R"
                                    elif mu < r_mu:
                                        label = "Q" if np.sign(A) != sign_R_py else "R2"
                                    else:
                                        label = "S" if np.sign(A) != sign_R_py else "R2"
                                    out["avg"].append({
                                        "wave_type": "QRS",
                                        "amp_mv": float(A),
                                        "center_ms": float(mu),
                                        "sigma_ms": float(s),
                                        "component": label,
                                    })
                                qrs_fit_result = qrs_fit
                        except Exception as e:
                            logger.warning("QRS fitting failed for %s: %s", lead_name, e)

                # Per-beat fitting (use existing Rust parallel or Python)
                _prune_p = pruning in ("p_absence", "p_absence+qrs_clean", "p_absence+qrs_clean+aicc")
                avg_p_present = out.get("p_info", {}).get("p_present", False)
                if not _prune_p:
                    avg_p_present = True

                if _HAS_RUST_PARALLEL and n_beats >= 2:
                    try:
                        _stack_c = np.ascontiguousarray(lead_stack, dtype=np.float64)
                        _time_c = np.ascontiguousarray(time_ms, dtype=np.float64)
                        _p_flag = np.array([1.0 if avg_p_present else 0.0], dtype=np.float64)
                        _rr = float(rr_mean_ms) if rr_mean_ms else 800.0
                        t0_w, t1_w = wave_bounds["T"]
                        p0_w, p1_w = wave_bounds["P"]

                        rust_comps = _qn.fit_all_leads_beats_parallel(
                            _stack_c.ravel(), _time_c,
                            n_beats, 1,
                            float(t0_w), float(t1_w),
                            float(p0_w), float(p1_w),
                            _p_flag, _rr, float(offset_ms),
                        )
                        for comp_dict in rust_comps:
                            bi = int(comp_dict["beat_idx"])
                            out["by_beat"][bi].append({
                                "wave_type": str(comp_dict["wave_type"]),
                                "amp_mv": float(comp_dict["amp_mv"]),
                                "center_ms": float(comp_dict["center_ms"]),
                                "sigma_ms": float(comp_dict["sigma_ms"]),
                                "component": str(comp_dict["component"]),
                            })
                    except Exception as e:
                        # H47 / Q-A6-11: Rust avg-trace P+T beat fitting can
                        # fail (rust binding error, dimension mismatch); fall
                        # through to per-beat QRS below.
                        logger.debug(
                            "Rust avg-trace P+T fitting failed for %s, "
                            "falling back to per-beat QRS: %s",
                            lead_name, e,
                        )

                # Per-beat QRS (still Python — unified fitter)
                q0, q1 = wave_bounds["QRS"]
                qrs_mask = (time_ms >= q0) & (time_ms <= q1)
                avg_qrs_is_clean = False
                if qrs_fit_result and getattr(qrs_fit_result, "params", None):
                    if len(qrs_fit_result.params) == 1 and getattr(qrs_fit_result, "yfit", None) is not None:
                        fit_mask = getattr(qrs_fit_result, "mask", None)
                        if fit_mask is not None and np.any(fit_mask):
                            residual = np.sqrt(np.mean((avg_trace[fit_mask] - qrs_fit_result.yfit[fit_mask]) ** 2))
                            peak_amp = max(abs(avg_trace[fit_mask].max()), abs(avg_trace[fit_mask].min()), 1e-9)
                            avg_qrs_is_clean = (residual / peak_amp) < 0.15

                for beat_idx in range(n_beats):
                    beat_trace = lead_stack[beat_idx, :]
                    try:
                        if np.any(qrs_mask):
                            qrs_trace_beat = beat_trace[qrs_mask]
                            r_idx_beat = int(np.argmax(np.abs(qrs_trace_beat)))
                            r_peak_approx_beat = float(time_ms[qrs_mask][r_idx_beat])
                        else:
                            r_peak_approx_beat = None
                        _beat_max_components = 1 if avg_qrs_is_clean else 4
                        qrs_fit_beat = _fit_qrs_unified(
                            beat_trace, time_ms, (q0, q1),
                            tR_ms=r_peak_approx_beat, fs=fs,
                            min_qrs_ms=60.0, max_qrs_ms=250.0, padding_ms=10.0,
                            force_narrow=False, sigma_bounds_ms=(6.0, 60.0),
                            max_components=_beat_max_components, q_s_threshold_rel=0.03,
                            rprime_threshold_rel=0.05, min_sep_ms=8.0,
                        )
                        if getattr(qrs_fit_beat, "params", None):
                            r_mu_beat = float(qrs_fit_beat.r_peak_time)
                            for A, mu, s in qrs_fit_beat.params:
                                if abs(mu - r_mu_beat) < 6.0:
                                    label = "R"
                                elif mu < r_mu_beat:
                                    label = "Q" if np.sign(A) != np.sign(list(qrs_fit_beat.params)[0][0]) else "R2"
                                else:
                                    label = "S" if np.sign(A) != np.sign(list(qrs_fit_beat.params)[0][0]) else "R2"
                                out["by_beat"][beat_idx].append({
                                    "wave_type": "QRS", "amp_mv": float(A),
                                    "center_ms": float(mu), "sigma_ms": float(s),
                                    "component": label,
                                })
                    except Exception as e:
                        # H47 / Q-A6-11: per-beat QRS in the Rust avg-trace
                        # fallback path is best-effort. Same shape as the
                        # other per-beat QRS fallbacks.
                        logger.debug(
                            "Per-beat QRS (rust avg-trace fallback) failed "
                            "for %s beat %d: %s",
                            lead_name, beat_idx, e,
                        )

                lead_fits[lead_name] = out

            _used_rust_avg = True
            logger.debug("Avg-trace fitting: Rust rayon (%d leads, P+T)", len(leads))
    except Exception as e:
        logger.debug("Rust avg-trace fitting failed, falling back to Python: %s", e)

    if not _used_rust_avg:
        for lead_name in leads:
            li = lead_idx_map[lead_name]
            lead_stack = stack_sync[:, li, :]
            lead_fits[lead_name] = fit_lead_waves_enhanced(
                lead_stack,
                time_ms,
                wave_bounds,
                lead_name=lead_name,
                debug_png=(
                    Path(f"debug/{recording_id}_{lead_name}.png")
                    if plot_enabled
                    else None
                ),
                fs=fs,
                rr_mean_ms=rr_mean_ms,
                offset_ms=offset_ms,
                pruning=pruning,
            )

    # ---------- Second pass: cross-lead P-center consensus ----------
    p_fit_results_by_lead: Dict[str, tuple] = {}
    for ld, fd in lead_fits.items():
        compsP = [c for c in fd["avg"] if c.get("wave_type") == "P"]
        infoP = fd.get("p_info", {"p_present": bool(compsP), "snr_db": 0.0})
        p_fit_results_by_lead[ld] = (compsP, infoP)

    center_hint_ms, center_hint_sigma_ms = p_center_consensus(
        p_fit_results_by_lead,
        prefer_early=True, min_snr_db=-3.0,
        spread_guard_ms=80.0, default_sigma_ms=25.0,
    )

    # Re-fit weak/edge leads with the gentle prior — parallelised
    # via ThreadPool because fit_p_wave_components is dominated by
    # GIL-releasing Rust LM kernels. The fits are independent per
    # lead (each writes only its own ``fd`` dict) and the mutation
    # step runs in the main thread in submission order, so the
    # result is bit-deterministic regardless of completion order.
    if center_hint_ms is not None:
        p0, p1 = wave_bounds["P"]
        p_mask = (time_ms >= p0) & (time_ms <= p1)
        time_ms_p = time_ms[p_mask]
        limit_ms = -0.5 * (rr_mean_ms if rr_mean_ms else 800.0) + offset_ms

        refit_work: List[Tuple[str, Dict[str, Any], np.ndarray, Optional[np.ndarray]]] = []
        for ld, fd in lead_fits.items():
            compsP = [c for c in fd["avg"] if c.get("wave_type") == "P"]
            infoP = fd.get(
                "p_info", {"p_present": bool(compsP), "snr_db": 0.0}
            )
            needs_help = (
                (not infoP.get("p_present", False))
                or (float(infoP.get("snr_db", -999.0)) < -2.0)
                or bool(infoP.get("p_edge_clipped", False))
            )
            if not needs_help:
                continue

            li = lead_idx_map[ld]
            avg_trace_p = avg_per_lead[li, p_mask]

            compsT = [c for c in fd["avg"] if c.get("wave_type") == "T"]
            residual_hint = (
                build_t_tail_residual_from_prior_t_components(
                    time_ms_p, compsT,
                    shift_ms=-float(rr_mean_ms or 0.0),
                    gating="right-soft", gate_width_ms=12.0,
                )
                if compsT
                else None
            )
            refit_work.append((ld, fd, avg_trace_p, residual_hint))

        if refit_work:
            def _refit_one(
                avg_trace_p: np.ndarray,
                residual_hint: Optional[np.ndarray],
            ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
                return fit_p_wave_components(
                    time_ms_p, avg_trace_p,
                    residual_hint_mv=residual_hint,
                    center_hint_ms=center_hint_ms,
                    center_hint_sigma_ms=center_hint_sigma_ms,
                    sigma_bounds_ms=(8.0, 60.0),
                    min_separation_ms=18.0, max_separation_ms=140.0,
                    aic_delta_two=4.0, allow_biphasic=True, allow_two=True,
                )

            # Pool only when we have >=2 refits — for a single refit
            # the pool creation overhead on Windows (~5-10ms) exceeds
            # any GIL-release benefit. Most clean ECGs trigger 0-1
            # refits per record; AF / low-SNR records drive higher
            # counts and benefit from the parallel path.
            if len(refit_work) >= 2:
                with ThreadPoolExecutor(max_workers=len(refit_work)) as ex:
                    fit_results = list(
                        ex.map(
                            lambda job: _refit_one(job[2], job[3]),
                            refit_work,
                        )
                    )
            else:
                fit_results = [_refit_one(job[2], job[3]) for job in refit_work]

            for (ld, fd, _, _), (comps_refit, info_refit) in zip(
                refit_work, fit_results
            ):
                # Filter previous-T contamination
                comps_refit = [
                    c for c in comps_refit if c["center_ms"] >= limit_ms
                ]
                fd["avg"] = [c for c in fd["avg"] if c.get("wave_type") != "P"]
                fd["avg"].extend(comps_refit)
                fd["p_info"] = dict(info_refit)

    # ---------- Aggregate ----------
    avg_lead, seg_bounds = _extract_t_wave(lead_fits)
    segment_bounds = {
        "T": seg_bounds["T"],
        "U": _blank_bounds(leads),
        "P_next": _blank_bounds(leads),
        "ST": _blank_bounds(leads),
        "P": seg_bounds.get("P", _blank_bounds(leads)),
    }

    return {
        "lead_fits": lead_fits,
        "avg_lead": avg_lead,
        "segment_bounds": segment_bounds,
        "time_ms": time_ms,
        "wave_bounds": wave_bounds,
    }
