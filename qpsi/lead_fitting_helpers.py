"""
Lead-level fitting helper functions for P-wave detection, residual
calculation, and multi-Gaussian fitting.

Source: Cell 16 lines ~1275-1824 (helpers only, not the main fitter) +
        Cell 10 ``estimate_pr_interval`` (cross-layer fix section 3.5).

External dependencies: numpy, scipy (signal, optimize, ndimage).
Internal: imports ``_blank_bounds`` from ``qpsi.wave_classification``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from qpsi.wave_classification import _blank_bounds

# Numba-accelerated multi-Gaussian model (optional)
try:
    import numba
    import math as _math

    @numba.njit(cache=True, fastmath=True)
    def _multi_gauss_func_numba(t, params_arr, n_comp):
        """Numba-accelerated sum of Gaussians + baseline."""
        n_pts = t.shape[0]
        baseline = params_arr[3 * n_comp]
        out = np.empty(n_pts)
        for i in range(n_pts):
            out[i] = baseline
        for c in range(n_comp):
            A = params_arr[3 * c]
            mu = params_arr[3 * c + 1]
            s = params_arr[3 * c + 2] + 1e-9
            inv_s = 1.0 / s
            for i in range(n_pts):
                z = (t[i] - mu) * inv_s
                out[i] += A * _math.exp(-0.5 * z * z)
        return out

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

logger = logging.getLogger("qpsi")


# ---------------------------------------------------------------------------
# T-wave / segment-bounds extraction from lead_fits
# ---------------------------------------------------------------------------

def _extract_t_wave(
    lead_fits: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Optional[np.ndarray]]], Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]]:
    """Build avg_lead waveforms and segment bounds from fitted Gaussians.

    Returns:
        ``(avg_lead, seg_bounds)`` where *avg_lead* maps
        ``lead -> {"T": ..., "P": ..., "QRS": ...}`` and *seg_bounds*
        maps ``"T"/"P" -> {lead: (start_ms, end_ms)}``.
    """
    avg_lead: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    seg_bounds: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {
        "T": {},
        "P": {},
    }

    for ld, fd in lead_fits.items():
        avg_lead[ld] = {"T": None, "P": None, "QRS": None}

        def _recon(wname: str) -> np.ndarray:
            params: List[float] = []
            for f in fd.get("avg", []):
                if f["wave_type"] == wname:
                    params += [f["amp_mv"], f["center_ms"], f["sigma_ms"]]
            return np.array(params)

        for wname in ("T", "P"):
            pars = _recon(wname)
            if pars.size:
                comps = np.reshape(pars, (-1, 3))
                start = float(np.min(comps[:, 1] - 2 * comps[:, 2]))
                end = float(np.max(comps[:, 1] + 2 * comps[:, 2]))
                seg_bounds[wname][ld] = (start, end)
            else:
                seg_bounds[wname][ld] = (None, None)

    return avg_lead, seg_bounds


# ---------------------------------------------------------------------------
# PR-interval estimation (from Cell 10, cross-layer fix section 3.5)
# ---------------------------------------------------------------------------

def estimate_pr_interval(
    trace: np.ndarray,
    time_ms: np.ndarray,
    qrs_bounds: Tuple[float, float],
    rr_mean_ms: float = 800.0,
) -> Tuple[float, float]:
    """Estimate the expected PR-interval range (min, max) in milliseconds.

    Uses heart-rate-based priors and signal-range heuristics to widen the
    range when first-degree AV block is plausible.
    """
    normal_pr = (120.0, 200.0)
    if rr_mean_ms < 600:
        base_pr = (100.0, 180.0)
    elif rr_mean_ms > 1200:
        base_pr = (140.0, 300.0)
    else:
        base_pr = normal_pr

    qrs_start_ms = qrs_bounds[0]
    search_start = qrs_start_ms - 600.0
    search_end = qrs_start_ms - 40.0

    mask = (time_ms >= search_start) & (time_ms <= search_end)
    if not np.any(mask):
        return base_pr

    pre_qrs_trace = trace[mask]
    baseline_std = np.std(pre_qrs_trace)
    signal_range = np.ptp(pre_qrs_trace)

    if signal_range > 2 * baseline_std:
        max_pr = min(500.0, base_pr[1] * 2.0)
        min_pr = base_pr[0]
    else:
        min_pr, max_pr = base_pr

    return (min_pr, max_pr)


# ---------------------------------------------------------------------------
# Adaptive energy envelope (multi-method)
# ---------------------------------------------------------------------------

def adaptive_energy_envelope(
    trace: np.ndarray,
    time_ms: np.ndarray,
    *,
    win_ms: float = 18.0,
    method: str = "hybrid",
) -> np.ndarray:
    """Multi-method energy envelope optimised for P-wave detection.

    Args:
        trace: ECG signal.
        time_ms: Time vector in milliseconds.
        win_ms: Smoothing window size.
        method: ``"gradient"``, ``"variance"``, ``"hybrid"``, or
            ``"morphological"``.
    """
    t = np.asarray(time_ms, float)
    y = np.asarray(trace, float)

    if t.size < 5:
        return np.zeros_like(t)

    dt = float(max(np.median(np.diff(t)), 1e-3))

    if method == "gradient":
        dydt = np.gradient(y, t, edge_order=2)
        energy = np.abs(dydt)

    elif method == "variance":
        w_samples = max(3, int(round(win_ms / dt)) | 1)
        half_w = w_samples // 2
        energy = np.zeros_like(y)
        for i in range(len(y)):
            start = max(0, i - half_w)
            end = min(len(y), i + half_w + 1)
            window = y[start:end]
            energy[i] = np.var(window) if len(window) > 1 else 0

    elif method == "morphological":
        w_samples = max(3, int(round(win_ms / dt)) | 1)
        from scipy.ndimage import grey_dilation, grey_erosion
        dilated = grey_dilation(y, size=w_samples)
        eroded = grey_erosion(y, size=w_samples)
        energy = dilated - eroded

    else:  # hybrid
        dydt = np.gradient(y, t, edge_order=2)
        grad_energy = np.abs(dydt)

        w_samples = max(3, int(round(win_ms / dt)) | 1)
        half_w = w_samples // 2
        var_energy = np.zeros_like(y)
        for i in range(len(y)):
            start = max(0, i - half_w)
            end = min(len(y), i + half_w + 1)
            window = y[start:end]
            var_energy[i] = np.var(window) if len(window) > 1 else 0

        # Normalise both components
        grad_norm = (grad_energy - np.median(grad_energy)) / (np.std(grad_energy) + 1e-9)
        var_norm = (var_energy - np.median(var_energy)) / (np.std(var_energy) + 1e-9)

        # Weight variance more heavily for P-wave detection
        energy = 0.3 * grad_norm + 0.7 * var_norm

    # Smooth the energy signal
    w = max(5, int(round(win_ms / dt)) | 1)
    if w < len(energy):
        energy_smooth = savgol_filter(energy, w, min(2, w - 1), mode="interp")
    else:
        energy_smooth = energy

    return energy_smooth


# ---------------------------------------------------------------------------
# P-wave candidate detection & selection
# ---------------------------------------------------------------------------

def detect_p_wave_candidates(
    trace: np.ndarray,
    time_ms: np.ndarray,
    qrs_bounds: Tuple[float, float],
    pr_range: Tuple[float, float],
    energy_methods: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Detect P-wave candidates using multiple energy methods and template
    correlation.

    Returns:
        List of candidate dicts with ``start_ms``, ``end_ms``,
        ``center_ms``, ``confidence``, ``method``, ``duration_ms``.
    """
    if energy_methods is None:
        energy_methods = ["hybrid", "variance", "morphological"]

    qrs_start_ms = qrs_bounds[0]
    min_pr_ms, max_pr_ms = pr_range

    # Define search window
    search_start = qrs_start_ms - max_pr_ms - 50.0  # buffer
    search_end = qrs_start_ms - 40.0  # stay away from QRS

    mask = (time_ms >= search_start) & (time_ms <= search_end)
    if not np.any(mask):
        return []

    search_time = time_ms[mask]
    search_trace = trace[mask]

    candidates: List[Dict[str, Any]] = []

    # --- Method 1: multiple energy envelope approaches ---
    for method in energy_methods:
        energy = adaptive_energy_envelope(
            search_trace, search_time, win_ms=15.0, method=method,
        )

        # Adaptive thresholding
        energy_median = np.median(energy)
        energy_mad = np.median(np.abs(energy - energy_median))
        threshold = energy_median + 1.5 * energy_mad

        dt_search = np.median(np.diff(search_time))
        peaks, properties = find_peaks(
            energy,
            height=threshold,
            distance=int(30.0 / dt_search),
            width=2,
            prominence=0.5 * energy_mad,
        )

        for i, peak_idx in enumerate(peaks):
            peak_time = search_time[peak_idx]

            left_bases = properties.get("left_bases", [max(0, peak_idx - 10)])
            right_bases = properties.get("right_bases", [min(len(search_time) - 1, peak_idx + 10)])

            if i < len(left_bases) and i < len(right_bases):
                start_ms = search_time[left_bases[i]]
                end_ms = search_time[right_bases[i]]

                duration_ms = end_ms - start_ms
                if 40.0 <= duration_ms <= 150.0:
                    confidence = (
                        float(properties["peak_heights"][i])
                        if "peak_heights" in properties
                        else 0.5
                    )
                    candidates.append({
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "center_ms": peak_time,
                        "confidence": confidence,
                        "method": method,
                        "duration_ms": duration_ms,
                    })

    # --- Method 2: template correlation for known P-wave shapes ---
    dt = np.median(np.diff(search_time))
    gaussian_widths = [20.0, 35.0, 50.0]

    for width_ms in gaussian_widths:
        sigma_samples = width_ms / dt
        template_len = int(6 * sigma_samples)
        if template_len > 0 and template_len < len(search_trace):
            template_t = np.arange(template_len) - template_len // 2
            template = np.exp(-0.5 * (template_t / sigma_samples) ** 2)

            correlation = np.correlate(search_trace, template, mode="valid")
            if len(correlation) > 0:
                corr_threshold = np.mean(correlation) + 2 * np.std(correlation)
                corr_peaks, _ = find_peaks(correlation, height=corr_threshold)

                for peak_idx in corr_peaks:
                    actual_idx = peak_idx + len(template) // 2
                    if actual_idx < len(search_time):
                        center_time = search_time[actual_idx]
                        half_width = width_ms / 2

                        candidates.append({
                            "start_ms": center_time - half_width,
                            "end_ms": center_time + half_width,
                            "center_ms": center_time,
                            "confidence": float(correlation[peak_idx]),
                            "method": f"template_{width_ms}ms",
                            "duration_ms": width_ms,
                        })

    return candidates


def select_best_p_candidate(
    candidates: List[Dict[str, Any]],
    pr_range: Tuple[float, float],
    qrs_start_ms: float,
) -> Optional[Dict[str, Any]]:
    """Select the best P-wave candidate based on timing, confidence, and
    morphology.
    """
    if not candidates:
        return None

    min_pr_ms, max_pr_ms = pr_range

    scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
    for candidate in candidates:
        score = 0.0
        center_ms = candidate["center_ms"]
        pr_interval = qrs_start_ms - center_ms

        # PR interval scoring
        if min_pr_ms <= pr_interval <= max_pr_ms:
            score += 2.0
        elif pr_interval > max_pr_ms and pr_interval <= 500.0:
            score += 1.0  # acceptable for AV blocks
        else:
            score -= 1.0

        # Confidence scoring
        score += np.clip(candidate["confidence"], 0, 2.0)

        # Duration scoring
        duration = candidate["duration_ms"]
        if 60.0 <= duration <= 120.0:
            score += 1.0
        elif 40.0 <= duration <= 150.0:
            score += 0.5

        # Method scoring (prefer more robust methods)
        method_weights = {
            "hybrid": 1.5,
            "variance": 1.2,
            "morphological": 1.0,
            "gradient": 0.8,
        }
        cand_method = candidate.get("method", "")
        for key, weight in method_weights.items():
            if key in cand_method:
                score += weight
                break

        scored_candidates.append((score, candidate))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return scored_candidates[0][1]


# ---------------------------------------------------------------------------
# Enhanced P-window detection (combines estimate + detect + select)
# ---------------------------------------------------------------------------

def enhanced_p_window_detection(
    trace: np.ndarray,
    time_ms: np.ndarray,
    qrs_bounds: Tuple[float, float],
    rr_mean_ms: float = 800.0,
) -> Optional[Tuple[float, float]]:
    """Enhanced P-wave window detection for long PR intervals and AV blocks.

    Returns:
        ``(p_start_ms, p_end_ms)`` or ``None`` if no reliable P-wave detected.
    """
    pr_range = estimate_pr_interval(trace, time_ms, qrs_bounds, rr_mean_ms)
    candidates = detect_p_wave_candidates(trace, time_ms, qrs_bounds, pr_range)
    best_candidate = select_best_p_candidate(candidates, pr_range, qrs_bounds[0])

    if best_candidate is None:
        return None

    # Safety margins
    p_start = best_candidate["start_ms"] - 10.0
    p_end = best_candidate["end_ms"] + 10.0

    # Ensure bounds are within reasonable limits
    p_start = max(p_start, time_ms[0], qrs_bounds[0] - 600.0)
    p_end = min(p_end, time_ms[-1], qrs_bounds[0] - 30.0)

    return (p_start, p_end) if p_start < p_end else None


# ---------------------------------------------------------------------------
# Enhanced wave-bounds builder from lumps
# ---------------------------------------------------------------------------

def create_wave_bounds_from_lumps_enhanced(
    lumps: List[Dict],
    rr_mean_ms: float = 800.0,
    trace: Optional[np.ndarray] = None,
    time_ms: Optional[np.ndarray] = None,
    *,
    use_enhanced_p_detection: bool = True,
    offset_ms: float = 50.0,
) -> Dict[str, Tuple[float, float]]:
    """Enhanced bounds creation with adaptive P-wave detection for
    pathological cases.  Now with robust error handling for noisy signals.

    Args:
        lumps: List of lump dicts (from initial Gaussian decomposition).
        rr_mean_ms: Mean RR interval in milliseconds.
        trace: Optional 1-D ECG trace for enhanced P detection.
        time_ms: Optional time axis matching *trace*.
        use_enhanced_p_detection: Try adaptive P-wave detection.
        offset_ms: Buffer subtracted from the RR-based P-start limit.
    """
    # Constants
    EARLY_P_MS = -400.0  # extended for long PR intervals
    QRS_DEFAULT = (-50.0, 80.0)
    QRS_MIN_SAFE = (-40.0, 100.0)
    QRS_HARD_MIN = -80.0
    QRS_HARD_MAX = 140.0

    # Dynamic p_start to avoid previous T-wave
    p_limit_ms = -0.5 * rr_mean_ms + offset_ms
    p_start = max(EARLY_P_MS, p_limit_ms)
    p_end = -offset_ms  # notebook uses -offset_ms (typically -50.0), not hardcoded -60

    # T wave bounds based on heart rate
    if rr_mean_ms < 600:
        t_start, t_end = 120.0, 280.0
    elif rr_mean_ms < 1000:
        t_start, t_end = 150.0, 350.0
    else:
        t_start, t_end = 180.0, 400.0

    # Check if lumps is valid
    if not lumps or not isinstance(lumps, list):
        logger.warning("Invalid or empty lumps, using default bounds")
        return {
            "P": (p_start, p_end),
            "QRS": QRS_DEFAULT,
            "T": (t_start, t_end),
        }

    # Validate lumps
    try:
        lump_centers = [
            l.get("center_ms", l.get("peak_time", 0.0)) for l in lumps
        ]
        lumps_have_valid_timing = any(abs(c) > 1e-6 for c in lump_centers)
    except Exception as exc:
        logger.warning("Error processing lumps: %s", exc)
        lumps_have_valid_timing = False

    # QRS bounds from lumps
    qrs_start, qrs_end = QRS_DEFAULT
    try:
        qrs_lumps = [l for l in lumps if l.get("wave_type") == "QRS"]
        if qrs_lumps:
            qrs_times: List[float] = []
            for lump in qrs_lumps:
                c = float(lump.get("center_ms", lump.get("peak_time", 0.0)))
                s = float(lump.get("sigma_ms", 15.0))
                qrs_times.extend([c - 1.5 * s, c + 1.5 * s])
            if qrs_times:
                qrs_start = max(QRS_HARD_MIN, min(qrs_times) - 20.0)
                qrs_end = min(QRS_HARD_MAX, max(qrs_times) + 20.0)
                qrs_start = min(qrs_start, QRS_MIN_SAFE[0])
                qrs_end = max(qrs_end, QRS_MIN_SAFE[1])
    except Exception as exc:
        logger.warning("Error processing QRS bounds: %s", exc)
        qrs_start, qrs_end = QRS_DEFAULT

    # Enhanced P-wave detection
    if (
        use_enhanced_p_detection
        and trace is not None
        and time_ms is not None
        and len(time_ms) == len(trace)
    ):
        try:
            qrs_bounds = (qrs_start, qrs_end)
            p_bounds_enhanced = enhanced_p_window_detection(
                trace, time_ms, qrs_bounds, rr_mean_ms,
            )

            if p_bounds_enhanced is not None:
                p_start_enhanced, p_end_enhanced = p_bounds_enhanced
                logger.debug(
                    "Enhanced P detection: [%.1f, %.1f] ms",
                    p_start_enhanced, p_end_enhanced,
                )
                # Use enhanced bounds, but respect physiological limits
                p_start = max(p_start, p_start_enhanced)
                p_end = min(-30.0, p_end_enhanced)
        except Exception as exc:
            logger.warning("Enhanced P detection failed: %s", exc)
            # Keep default P bounds

    # Fallback: P bounds from lumps if enhanced detection didn't work
    if lumps_have_valid_timing:
        try:
            p_lump = next((l for l in lumps if l.get("wave_type") == "P"), None)
            if p_lump:
                c = p_lump.get("center_ms", p_lump.get("peak_time", -120.0))
                s = float(p_lump.get("sigma_ms", 30.0))
                if abs(c) > 1e-6:
                    # Only update if we don't have enhanced detection results
                    if not use_enhanced_p_detection or trace is None:
                        p_start = max(p_limit_ms, max(EARLY_P_MS - 20.0, c - 2.5 * s))
                        p_end = min(-40.0, c + 2.5 * s)
        except Exception as exc:
            logger.warning("P lump processing failed: %s", exc)

    # T bounds from lumps
    if lumps_have_valid_timing:
        try:
            t_lump = next((l for l in lumps if l.get("wave_type") == "T"), None)
            if t_lump:
                c = float(t_lump.get("center_ms", t_lump.get("peak_time", 250.0)))
                s = float(t_lump.get("sigma_ms", 60.0))
                if abs(c) > 1e-6:
                    t_start = max(100.0, c - 2.5 * s)
                    t_end = min(500.0, c + 2.5 * s)
        except Exception as exc:
            logger.warning("T lump processing failed: %s", exc)

    wave_bounds: Dict[str, Tuple[float, float]] = {
        "P": (p_start, p_end),
        "QRS": (qrs_start, qrs_end),
        "T": (t_start, t_end),
    }

    # Final validation
    for k, (a, b) in list(wave_bounds.items()):
        try:
            if not (a < b):
                logger.warning("Invalid bounds for %s: [%s, %s], using defaults", k, a, b)
                if k == "P":
                    # Parity with Apr28 notebook line 14723 fallback: use -60.0
                    # explicitly so both error branches share the same default.
                    wave_bounds[k] = (p_start, -60.0)
                elif k == "QRS":
                    wave_bounds[k] = QRS_DEFAULT
                elif k == "T":
                    wave_bounds[k] = (150.0, 350.0)
        except Exception as exc:
            logger.warning("Validation failed for %s: %s", k, exc)
            if k == "P":
                wave_bounds[k] = (p_start, -60.0)
            elif k == "QRS":
                wave_bounds[k] = QRS_DEFAULT
            elif k == "T":
                wave_bounds[k] = (150.0, 350.0)

    logger.debug(
        "Enhanced wave bounds: P=[%.1f, %.1f]  QRS=[%.1f, %.1f]  T=[%.1f, %.1f]",
        wave_bounds["P"][0], wave_bounds["P"][1],
        wave_bounds["QRS"][0], wave_bounds["QRS"][1],
        wave_bounds["T"][0], wave_bounds["T"][1],
    )

    return wave_bounds


# ---------------------------------------------------------------------------
# Residual calculation from fits
# ---------------------------------------------------------------------------

def calculate_residuals_from_fits(
    stack_sync: np.ndarray,
    time_ms: np.ndarray,
    lead_fits: Dict,
    leads: List[str],
    avg_per_lead: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Calculate residuals by subtracting fitted Gaussians from original
    signals.

    ``avg_per_lead`` is the (n_leads, n_samples) cached per-lead mean of
    ``stack_sync`` along the beat axis. Pipeline callers pass it in;
    external callers fall through to a single reduction.

    Returns:
        Dictionary mapping lead names to their residual signals.
    """
    if avg_per_lead is None:
        avg_per_lead = np.nanmean(stack_sync, axis=0)
    residuals_by_lead: Dict[str, np.ndarray] = {}
    for lead_idx, lead_name in enumerate(leads):
        avg_trace = avg_per_lead[lead_idx, :]
        total_fit = np.zeros_like(time_ms)
        if lead_name in lead_fits:
            fits = lead_fits[lead_name].get("avg", [])
            for fit in fits:
                amp = fit.get("amp_mv", 0)
                center = fit.get("center_ms", 0)
                sigma = fit.get("sigma_ms", 1)
                gaussian = amp * np.exp(-0.5 * ((time_ms - center) / sigma) ** 2)
                total_fit += gaussian
        residual = avg_trace - total_fit
        residuals_by_lead[lead_name] = residual
    return residuals_by_lead


# ---------------------------------------------------------------------------
# Windowed RMSE and R-squared helpers
# ---------------------------------------------------------------------------

def _window_rmse(
    t: np.ndarray,
    resid: np.ndarray,
    bounds: Tuple[float, float],
) -> float:
    """RMSE of *resid* within the time window defined by *bounds*."""
    lo, hi = bounds
    m = (t >= lo) & (t <= hi)
    return float(np.sqrt(np.mean(resid[m] ** 2))) if np.any(m) else np.nan


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination (R-squared)."""
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Multi-Gaussian fitting (legacy fallback adapter)
# ---------------------------------------------------------------------------

def _multi_gauss_func(t: np.ndarray, *params: float) -> np.ndarray:
    """Sum of Gaussians with shared baseline.

    ``params = [A1, mu1, s1, A2, mu2, s2, ..., baseline]``
    """
    if _HAS_NUMBA:
        n_comp = (len(params) - 1) // 3
        return _multi_gauss_func_numba(
            np.ascontiguousarray(t), np.asarray(params, dtype=float), n_comp
        )
    baseline = params[-1]
    n = (len(params) - 1) // 3
    out = np.zeros_like(t, dtype=float) + baseline
    for i in range(n):
        A, mu, s = params[3 * i : 3 * i + 3]
        out += A * np.exp(-0.5 * ((t - mu) / (s + 1e-9)) ** 2)
    return out


def _fit_multi_gaussian(
    signal_mv: np.ndarray,
    time_ms: np.ndarray,
    win_lo_ms: float,
    win_hi_ms: float,
    n_comp: int = 2,
) -> np.ndarray:
    """Simple multi-Gaussian fitter (legacy fallback).

    Returns:
        ``np.ndarray`` of shape ``(n_fitted, 3)`` with columns
        ``[amplitude, center_ms, sigma_ms]``.
    """
    m = (time_ms >= win_lo_ms) & (time_ms <= win_hi_ms)
    t_win = np.asarray(time_ms)[m]
    y_win = np.asarray(signal_mv)[m]

    if t_win.size < 5:
        return np.zeros((0, 3))

    A0 = float(np.max(y_win) - np.min(y_win))
    mu0 = float(t_win[np.argmax(y_win)])
    s0 = float(0.1 * (t_win[-1] - t_win[0]))

    p0: List[float] = []
    for i in range(n_comp):
        frac = (i + 1) / (n_comp + 1)
        p0 += [A0, t_win[0] + frac * (t_win[-1] - t_win[0]), s0]
    p0 += [float(np.median(y_win))]  # baseline

    try:
        popt, _ = curve_fit(
            lambda t, *p: _multi_gauss_func(t, *p),
            t_win,
            y_win,
            p0=p0,
            maxfev=8000,
        )
        n = (len(popt) - 1) // 3
        pars: List[List[float]] = []
        for i in range(n):
            pars.append([popt[3 * i], popt[3 * i + 1], abs(popt[3 * i + 2])])
        return np.array(pars, float)
    except Exception as exc:
        logger.warning("_fit_multi_gaussian failed: %s", exc)
        return np.array([[A0, mu0, s0]], float)
