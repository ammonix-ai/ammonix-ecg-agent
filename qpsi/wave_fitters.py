"""
Generic 1D Gaussian wave component fitting with Huber LS + AICc model selection.

This is the lead-wise fitting engine (distinct from the 2D plane Gaussian
fitting in gaussian_fitting.py). Fits P, T, and QRS wave components
independently on each lead using model selection among K=0/1/2 Gaussians.

Source: Cell 16 lines ~23-1270 (core engine functions only)
No internal dependencies. External: numpy, scipy (optimize, signal, special).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter, welch
from scipy.special import erf

from qpsi.compat import trapezoid as _trapezoid

logger = logging.getLogger("qpsi")

# ---------------------------------------------------------------------------
# Rust-accelerated solver (optional — fallback to scipy if unavailable)
# ---------------------------------------------------------------------------
try:
    import qpsi_native as _qpsi_native
    if not hasattr(_qpsi_native, "fit_k1_lm"):
        raise ImportError("qpsi_native found but missing Rust functions")
    _HAS_RUST_SOLVER = True
except ImportError:
    _HAS_RUST_SOLVER = False

# ---------------------------------------------------------------------------
# Numba-accelerated kernels (optional — fallback to NumPy if unavailable)
# ---------------------------------------------------------------------------
try:
    import numba

    @numba.njit(cache=True, fastmath=True)
    def _gauss_numba(x, A, mu, s):
        """Numba-accelerated single Gaussian."""
        n = x.shape[0]
        out = np.empty(n)
        inv_s = 1.0 / (s + 1e-9)
        for i in range(n):
            z = (x[i] - mu) * inv_s
            out[i] = A * np.exp(-0.5 * z * z)
        return out

    @numba.njit(cache=True, fastmath=True)
    def _gauss_k2_residual_numba(x, y, p, center_hint_ms, center_hint_sigma_ms, has_hint):
        """Numba-accelerated K=2 residual for wave fitting."""
        A1, mu1, s1, A2, sep, s2, b0 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        mu2 = mu1 + sep
        n = x.shape[0]
        n_out = n + (1 if has_hint else 0)
        out = np.empty(n_out)
        inv_s1 = 1.0 / (s1 + 1e-9)
        inv_s2 = 1.0 / (s2 + 1e-9)
        for i in range(n):
            z1 = (x[i] - mu1) * inv_s1
            z2 = (x[i] - mu2) * inv_s2
            g1 = A1 * np.exp(-0.5 * z1 * z1)
            g2 = A2 * np.exp(-0.5 * z2 * z2)
            out[i] = y[i] - (g1 + g2 + b0)
        if has_hint:
            out[n] = (mu1 - center_hint_ms) / max(center_hint_sigma_ms, 1e-6)
        return out

    @numba.njit(cache=True, fastmath=True)
    def _gauss_k1_residual_numba(x, y, p, center_hint_ms, center_hint_sigma_ms, has_hint):
        """Numba-accelerated K=1 residual for wave fitting."""
        A1, mu1, s1, b0 = p[0], p[1], p[2], p[3]
        n = x.shape[0]
        n_out = n + (1 if has_hint else 0)
        out = np.empty(n_out)
        inv_s = 1.0 / (s1 + 1e-9)
        for i in range(n):
            z = (x[i] - mu1) * inv_s
            out[i] = y[i] - (A1 * np.exp(-0.5 * z * z) + b0)
        if has_hint:
            out[n] = (mu1 - center_hint_ms) / max(center_hint_sigma_ms, 1e-6)
        return out

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# =============================== small utils ===============================

def _energy_envelope(
    trace: np.ndarray,
    time_ms: np.ndarray,
    *,
    win_ms: float = 18.0,
) -> np.ndarray:
    """Short-time energy envelope via RMS of |dy/dt| with Savitzky-Golay smoothing.

    Args:
        trace: 1-D signal amplitude array (mV).
        time_ms: Corresponding time axis in milliseconds.
        win_ms: Smoothing window width in milliseconds.

    Returns:
        Robust-normalised energy envelope (same length as *trace*).
    """
    t = np.asarray(time_ms, float)
    y = np.asarray(trace, float)
    if t.size < 5:
        return np.zeros_like(t)

    dt = float(max(np.median(np.diff(t)), 1e-3))
    dydt = np.gradient(y, t, edge_order=2)
    e = np.abs(dydt)

    # smooth with Savitzky-Golay ~win_ms
    w = max(5, int(round(win_ms / dt)) | 1)
    e_s = savgol_filter(e, w, 2, mode="interp")
    # normalize (robust)
    med = np.median(e_s)
    iqr = np.percentile(e_s, 75) - np.percentile(e_s, 25)
    denom = max(iqr, 1e-9)
    return (e_s - med) / denom


def weighted_quantile(
    values: np.ndarray,
    quantiles: float | np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float | np.ndarray:
    """Compute weighted quantile(s) via linear interpolation on the weighted CDF.

    Args:
        values: 1-D array of data values.
        quantiles: Scalar or array of quantile levels in [0, 1].
        sample_weight: Optional non-negative weights (same length as *values*).
            Negative or non-finite weights are treated as zero.

    Returns:
        Interpolated quantile value(s).  Scalar when *quantiles* is scalar,
        array otherwise.
    """
    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if sample_weight is None:
        sample_weight = np.ones_like(values, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        sample_weight = np.where(
            np.isfinite(sample_weight) & (sample_weight >= 0),
            sample_weight,
            0.0,
        )
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cdf = np.cumsum(sample_weight)
    if cdf[-1] == 0:
        cdf = np.linspace(0, 1, len(values))
    else:
        cdf = cdf / cdf[-1]
    out = np.interp(quantiles, cdf, values)
    return out if out.size > 1 else out.item()


def _robust_mad(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Median and MAD-based sigma estimate (1.4826 * MAD).

    Args:
        values: 1-D data array.
        weights: Optional weights for weighted median computation.

    Returns:
        ``(median, 1.4826 * MAD)`` tuple.
    """
    values = np.asarray(values, dtype=float)
    if weights is None:
        med = np.median(values)
        mad = np.median(np.abs(values - med))
    else:
        med = weighted_quantile(values, 0.5, weights)
        mad = weighted_quantile(np.abs(values - med), 0.5, weights)
    return med, 1.4826 * mad


def _robust_scale(z: np.ndarray) -> float:
    """MAD-based robust sigma estimate.

    Args:
        z: 1-D data array.

    Returns:
        Robust scale (1.4826 * MAD).
    """
    z = np.asarray(z, float)
    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-12
    return 1.4826 * mad


def _clip_inside(
    v: float,
    lo: float,
    hi: float,
    frac: float = 1e-6,
) -> float:
    """Clip *v* to ``[lo + frac*span, hi - frac*span]``."""
    span = hi - lo
    return float(np.clip(v, lo + frac * span, hi - frac * span))


def _gauss(x: np.ndarray, A: float, mu: float, s: float) -> np.ndarray:
    """Single Gaussian evaluated at *x*.

    ``A * exp(-0.5 * ((x - mu) / s)^2)``
    """
    z = (x - mu) / (s + 1e-9)
    return A * np.exp(-0.5 * z * z)


def _aicc(rss: float, k: int, n: int) -> float:
    """Corrected Akaike Information Criterion (AICc).

    Args:
        rss: Residual sum of squares.
        k: Number of model parameters.
        n: Number of data points.

    Returns:
        AICc value (lower is better).
    """
    n = int(max(n, 1))
    return n * np.log(rss / n + 1e-12) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)


# ------------------------------ P-center helpers ------------------------------

def _choose_p_center_from_comps(
    comps: List[Dict[str, Any]],
    prefer_early: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """Pick the best P-wave centre from a list of fitted components.

    For a single component, returns its centre and absolute amplitude.
    For two components, returns the earlier (if *prefer_early*) or
    largest-amplitude one.
    """
    if not comps:
        return None, None
    if len(comps) == 1:
        c = comps[0]
        return float(c.get("center_ms", np.nan)), float(abs(c.get("amp_mv", 0.0)))
    if prefer_early:
        c = min(comps, key=lambda d: d.get("center_ms", np.inf))
    else:
        c = max(comps, key=lambda d: abs(d.get("amp_mv", 0.0)))
    return float(c.get("center_ms", np.nan)), float(abs(c.get("amp_mv", 0.0)))


def p_center_consensus(
    fit_results_by_lead: Any,
    *,
    prefer_early: bool = True,
    min_snr_db: float = -3.0,
    iqr_k: float = 1.5,
    spread_guard_ms: float = 80.0,
    default_sigma_ms: float = 25.0,
    max_weight_ratio: float = 8.0,
    return_debug: bool = False,
) -> Tuple[Optional[float], float] | Tuple[Optional[float], float, Dict[str, Any]]:
    """Compute a cross-lead consensus P-wave centre from per-lead fit results.

    Collects P-centre estimates from all leads that report ``p_present=True``
    above *min_snr_db*, then computes a robust (weighted-median) consensus
    with IQR outlier rejection and a spread guard.

    Args:
        fit_results_by_lead: Dict or list of per-lead fit results.  Each entry
            should be ``(components, info)`` or a dict with ``'components'``
            and ``'info'`` keys.  The ``info`` dict must contain
            ``'p_present'`` (bool) and ``'snr_db'`` (float).
        prefer_early: Passed to :func:`_choose_p_center_from_comps`.
        min_snr_db: Minimum SNR (dB) for a lead to participate.
        iqr_k: IQR multiplier for outlier rejection.
        spread_guard_ms: Maximum allowed spread (ms) among kept centres;
            consensus is rejected if exceeded.
        default_sigma_ms: Returned sigma when no consensus is possible.
        max_weight_ratio: Cap on the ratio between largest and smallest
            SNR-derived weights to avoid a single lead dominating.
        return_debug: If *True*, return a third element with debug details.

    Returns:
        ``(center_ms, sigma_ms)`` or ``(center_ms, sigma_ms, debug_dict)``
        when *return_debug* is True.  *center_ms* is ``None`` when no
        consensus could be formed.
    """
    mus: List[float] = []
    wts: List[float] = []
    used: List[Tuple[Any, float, float]] = []
    dropped: List[Tuple[Any, str]] = []

    def _iter_entries(obj: Any):  # noqa: ANN202
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, tuple) and len(v) == 2:
                    comps, info = v
                    yield k, comps, info
                elif isinstance(v, dict):
                    comps = v.get("components", v.get("comps", []))
                    info = v.get("info", {})
                    yield k, comps, info
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                if isinstance(v, tuple) and len(v) == 2:
                    comps, info = v
                    lead = info.get("lead") if isinstance(info, dict) else None
                    yield lead or f"lead{i}", comps, info
                elif isinstance(v, dict):
                    comps = v.get("components", v.get("comps", []))
                    info = v.get("info", {})
                    lead = v.get("lead", f"lead{i}")
                    yield lead, comps, info

    for lead, comps, info in _iter_entries(fit_results_by_lead):
        info = info or {}
        if not info.get("p_present", False):
            dropped.append((lead, "absent"))
            continue
        snr = float(info.get("snr_db", -999.0))
        if not np.isfinite(snr) or snr < min_snr_db:
            dropped.append((lead, f"snr<{min_snr_db}"))
            continue
        mu, _ = _choose_p_center_from_comps(comps, prefer_early=prefer_early)
        if mu is None or not np.isfinite(mu):
            dropped.append((lead, "mu_nan"))
            continue
        w = max(10.0 ** (0.05 * snr), 1e-3)  # soften SNR weighting
        mus.append(float(mu))
        wts.append(float(w))
        used.append((lead, mu, snr))

    mus_arr = np.asarray(mus, float)
    wts_arr = np.asarray(wts, float)
    if mus_arr.size == 0:
        out = (None, float(default_sigma_ms))
        if return_debug:
            return (*out, {"used": used, "dropped": dropped, "stage": "no-data"})
        return out

    if np.any(wts_arr > 0):
        wmin = np.min(wts_arr[wts_arr > 0])
        wmax = np.max(wts_arr)
        if wmax / max(wmin, 1e-12) > max_weight_ratio:
            scale = (max_weight_ratio * wmin) / wmax
            wts_arr = np.where(wts_arr == wmax, wmax * scale, wts_arr)

    q1, q2, q3 = weighted_quantile(mus_arr, [0.25, 0.5, 0.75], wts_arr)
    iqr = float(q3 - q1)
    lo = q1 - iqr_k * iqr
    hi = q3 + iqr_k * iqr
    keep = (mus_arr >= lo) & (mus_arr <= hi)
    mus_k = mus_arr[keep]
    wts_k = wts_arr[keep]

    if mus_k.size == 0 or (mus_k.size == 1 and mus_arr.size > 2):
        out = (None, float(default_sigma_ms))
        if return_debug:
            return (*out, {"used": used, "dropped": dropped + [("consensus", "empty-after-outlier")], "stage": "after-outlier"})
        return out

    center = float(weighted_quantile(mus_k, 0.5, wts_k))
    _, sigma_mad = _robust_mad(mus_k, wts_k)
    sigma = float(np.clip(sigma_mad, 12.0, max(default_sigma_ms, 50.0)))

    if (np.max(mus_k) - np.min(mus_k)) > spread_guard_ms:
        out = (None, float(default_sigma_ms))
        if return_debug:
            return (*out, {"used": used, "dropped": dropped + [("consensus", "spread-guard")], "stage": "spread-guard",
                           "spread": float(np.max(mus_k) - np.min(mus_k))})
        return out

    if return_debug:
        dbg: Dict[str, Any] = {
            "used": used, "dropped": dropped,
            "q1": float(q1), "median": float(q2), "q3": float(q3),
            "lo": float(lo), "hi": float(hi),
            "center": center, "sigma_mad": float(sigma_mad),
            "n_kept": int(mus_k.size), "n_total": int(mus_arr.size),
        }
        return center, sigma, dbg
    return center, sigma


# ------------------------------ T-tail residual builder ------------------------------

def build_t_tail_residual_from_prior_t_components(
    t_ms: np.ndarray,
    prev_t_components: Optional[List[Dict[str, Any]]],
    *,
    shift_ms: float = 0.0,
    gating: str = "right-soft",
    gate_width_ms: float = 12.0,
    y_for_scale_mv: Optional[np.ndarray] = None,
    scale_mode: str = "none",
    scale_bounds: Tuple[float, float] = (0.25, 2.0),
    scale_region_ms: Optional[Tuple[float, float]] = None,
    include_components: Tuple[str, ...] = ("T", "T1", "T2", "Tneg", "Tpos"),
    return_debug: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """Predict the prior beat's T-tail inside the current P-window.

    Reconstructs the tail of prior T-wave components (shifted into the
    current beat's coordinate frame) for use as a residual hint when
    fitting the P-wave.  Use ONLY for seeding (pass as
    ``residual_hint_mv``); do **not** subtract in the final fit.

    Args:
        t_ms: Time axis of the current P-window (ms).
        prev_t_components: List of component dicts from the prior beat's
            T-wave fit, each containing ``'wave_type'``, ``'component'``,
            ``'amp_mv'``, ``'center_ms'``, ``'sigma_ms'``.
        shift_ms: Time shift to align prior T centres into the current
            window's coordinate system (effective mu = center_ms + shift_ms).
        gating: Gating mode for the reconstructed tail:
            ``'none'`` (full Gaussian), ``'right'`` (hard right-half gate),
            or ``'right-soft'`` (soft erf-based gate).
        gate_width_ms: Transition width for ``'right-soft'`` gating (ms).
        y_for_scale_mv: Optional observed signal for OLS rescaling of the
            reconstructed tail.
        scale_mode: ``'none'`` or ``'ols'`` for optional amplitude rescaling.
        scale_bounds: ``(min, max)`` clamp for the OLS scale factor.
        scale_region_ms: ``(lo, hi)`` region (ms) for scale estimation;
            defaults to the earliest quarter of the window.
        include_components: Component names to include from *prev_t_components*.
        return_debug: If *True*, return ``(residual, debug_dict)``.

    Returns:
        Predicted T-tail residual array (same shape as *t_ms*), or
        ``(residual, debug_dict)`` when *return_debug* is True.
    """
    x = np.asarray(t_ms, float)
    n = int(x.size)
    base = np.zeros(n, dtype=float)
    comps_used: List[Tuple[float, float, float]] = []

    if prev_t_components is None:
        out = base
        return (out, {"alpha": 1.0, "comps_used": 0, "note": "no-prev-T"}) if return_debug else out

    for c in prev_t_components:
        A = float(c.get("amp_mv", 0.0))
        mu = float(c.get("center_ms", 0.0)) + float(shift_ms)
        s = float(c.get("sigma_ms", 20.0))
        wt = c.get("wave_type", None)
        comp = c.get("component", None)
        is_t = (wt == "T") or (comp in include_components)
        if not is_t:
            continue

        g = _gauss(x, A, mu, s)
        if gating == "right":
            g = g * (x >= mu)
        elif gating == "right-soft":
            step = 0.5 * (1.0 + erf((x - mu) / max(gate_width_ms, 1e-6)))
            g = g * step

        base += g
        comps_used.append((A, mu, s))

    alpha = 1.0
    if y_for_scale_mv is not None and scale_mode != "none":
        y = np.asarray(y_for_scale_mv, float)
        if scale_region_ms is None:
            tmin, tmax = float(x.min()), float(x.max())
            lo = tmin
            hi = tmin + max(40.0, 0.25 * (tmax - tmin))
        else:
            lo, hi = map(float, scale_region_ms)
        idx = (x >= lo) & (x <= hi)
        denom = float(np.dot(base[idx], base[idx])) + 1e-12
        if denom > 0:
            alpha = float(np.dot(y[idx], base[idx]) / denom)
            alpha = float(np.clip(alpha, scale_bounds[0], scale_bounds[1]))

    residual = alpha * base
    if return_debug:
        dbg: Dict[str, Any] = {
            "alpha": alpha, "comps_used": len(comps_used),
            "gating": gating, "gate_width_ms": gate_width_ms,
            "scale_region": scale_region_ms, "shift_ms": shift_ms,
        }
        return residual, dbg
    return residual


# ====================== wave-agnostic core (no P/T words) ===================

def fit_wave_components_core(
    t_ms: np.ndarray,
    y_mv: np.ndarray,
    *,
    # common modeling knobs
    sigma_bounds_ms: Tuple[float, float] = (8.0, 60.0),
    min_separation_ms: float = 18.0,
    max_separation_ms: float = 140.0,
    allow_biphasic: bool = True,
    allow_two: bool = True,
    aic_delta_two: float = 4.0,
    allow_baseline_k0: bool = True,
    # seeding + priors
    residual_hint_mv: np.ndarray | None = None,
    center_hint_ms: float | None = None,
    center_hint_sigma_ms: float = 25.0,
    prefer_left_bias: bool = True,
    multi_seed: bool = True,
    max_templates: int = 5,
    # acceptance
    min_peak_snr_db: float = -3.0,
    # AICc early termination (pruning)
    aicc_early_term: bool = False,  # disabled by default to match notebook GT
    aicc_early_term_delta: float = 20.0,
    # meta
    debug: bool = False,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """Generic K=0/1/2 Gaussian fitter with Huber least-squares and AICc selection.

    Chooses among K=0 (baseline-only), K=1 (single Gaussian + offset), and
    K=2 (two Gaussians + offset) models.  Multi-seed strategy with matched
    filter templates, peak detection, and optional edge-bias.

    Args:
        t_ms: Time axis in milliseconds.
        y_mv: Signal amplitude in millivolts.
        sigma_bounds_ms: (min, max) allowed Gaussian sigma in ms.
        min_separation_ms: Minimum centre separation for K=2 model.
        max_separation_ms: Maximum centre separation for K=2 model.
        allow_biphasic: Allow opposite-polarity components in K=2.
        allow_two: Allow K=2 model at all.
        aic_delta_two: AICc penalty added to K=2 (must beat K=1 by this).
        allow_baseline_k0: Include K=0 (constant) as a competitor.
        residual_hint_mv: Optional residual signal for cleaner seeding.
        center_hint_ms: Gentle prior on component centre location.
        center_hint_sigma_ms: Width of centre prior (in ms).
        prefer_left_bias: Tie-break toward left (earlier) components.
        multi_seed: Use multiple seed strategies (peaks + matched filter).
        max_templates: Number of matched-filter sigma templates.
        min_peak_snr_db: Minimum peak SNR (dB) for acceptance vs K=0.
        debug: Print diagnostic messages.

    Returns:
        Tuple of ``(components, info)`` where *components* is a list of dicts
        ``{'amp_mv', 'center_ms', 'sigma_ms'}`` and *info* is a dict with
        keys ``'present', 'model', 'snr_db', 'seed_method', 'edge_left',
        'edge_right', 'notched', ...``.
    """
    x = np.asarray(t_ms, float)
    y = np.asarray(y_mv, float)
    n = int(x.size)
    if n < 8 or np.ptp(x) <= 0:
        return [], {"present": False, "model": "NA", "snr_db": 0.0}

    x_lo, x_hi = float(x.min()), float(x.max())
    lo_sig, hi_sig = sigma_bounds_ms
    step_ms = float(abs(np.median(np.diff(x))) or 1.0)

    # --- light smoothing & detrend (for seeds only) ---
    k_sg = max(5, (n // 15) * 2 + 1)
    y_s = savgol_filter(y, k_sg, 2, mode="interp")
    A_lin = np.vstack([x - x.mean(), np.ones(n)]).T
    beta = np.linalg.lstsq(A_lin, y_s, rcond=None)[0]
    y_ds = y_s - (A_lin @ beta)

    # residual-aware seed signal
    if residual_hint_mv is not None and len(residual_hint_mv) == n:
        y_seed_base = y - np.asarray(residual_hint_mv, float)
    else:
        y_seed_base = y

    # high-pass flavor for seeding (trim slow curvature)
    k_hp = max(7, ((n // 8) * 2 + 1))
    y_hp = savgol_filter(y_seed_base, k_hp, 2, mode="interp")
    y_seed = y_seed_base - y_hp

    # amplitude bounds
    abs_y_ds = np.abs(y_ds)
    Amax = float(max(
        1.25 * np.max(abs_y_ds) if abs_y_ds.size else 0.0,
        0.5 * np.ptp(y),
        5.0 * np.median(abs_y_ds) if abs_y_ds.size else 0.0,
        1e-3,
    ))
    Bmax = float(max(0.25 * np.ptp(y), 1e-3))

    # noise, SNR
    noise_sigma = max(_robust_scale(y_seed), 1e-6)

    def snr_db(amp: float) -> float:
        return 20.0 * np.log10(max(abs(amp), 1e-12) / noise_sigma)

    # ---------------------------- seeds ------------------------------------
    seeds_mu: List[float] = []
    seed_methods: List[str] = []

    # 1) abs-peak seeds
    if multi_seed:
        dist = max(2, int(min_separation_ms / max(step_ms, 1e-9)))
        prom = max(0.02 * np.ptp(y_seed), 0.001)
        pk_idx, _ = find_peaks(np.abs(y_seed), distance=dist, prominence=prom)
        if pk_idx.size > 0:
            idx = pk_idx[np.argsort(np.abs(y_seed[pk_idx]))[-3:]]
            for i in np.sort(idx):
                seeds_mu.append(float(x[int(i)]))
                seed_methods.append("peaks")
        else:
            i0 = int(np.argmax(np.abs(y_seed)))
            seeds_mu.append(float(x[i0]))
            seed_methods.append("peaks-fallback")
    else:
        i0 = int(np.argmax(np.abs(y_seed)))
        seeds_mu.append(float(x[i0]))
        seed_methods.append("peaks")

    # 2) matched filter (multi-sigma)
    if multi_seed and max_templates >= 1:
        sigmas = np.linspace(lo_sig, hi_sig, int(max(2, max_templates)))
        stride = max(1, int(8.0 / step_ms))  # ~ every ~8 ms
        mu_grid = x[::stride]
        for s in sigmas:
            scores = []
            for mu in mu_grid:
                tmpl = np.exp(-0.5 * ((x - mu) / (s + 1e-9)) ** 2)
                num = float(np.dot(y_seed, tmpl))
                den = float(np.dot(tmpl, tmpl)) + 1e-12
                score = (num * num) / den
                scores.append(score)
            scores_arr = np.asarray(scores)
            j = int(np.argmax(scores_arr))
            seeds_mu.append(float(mu_grid[j]))
            seed_methods.append(f"matched(s={s:.1f})")

        # edge bias (earliest or latest quartile)
        if len(mu_grid) > 0:
            q_idx = int(0.25 * len(mu_grid))
            if prefer_left_bias:
                jrange = range(0, max(1, q_idx))
                tag = "matched-left-bias"
            else:
                jrange = range(max(0, len(mu_grid) - q_idx), len(mu_grid))
                tag = "matched-right-bias"
            if jrange:
                scores_all = np.zeros(len(mu_grid))
                s0 = float(np.median(sigmas))
                for k_idx, mu in enumerate(mu_grid):
                    tmpl = np.exp(-0.5 * ((x - mu) / (s0 + 1e-9)) ** 2)
                    num = float(np.dot(y_seed, tmpl))
                    den = float(np.dot(tmpl, tmpl)) + 1e-12
                    scores_all[k_idx] = (num * num) / den
                j = int(np.argmax(scores_all[list(jrange)]))
                mu_bias = float(mu_grid[list(jrange)][j])
                seeds_mu.append(mu_bias)
                seed_methods.append(tag)

    # deduplicate seeds ~12 ms
    seeds: List[Tuple[float, str]] = []
    for mu, mname in zip(seeds_mu, seed_methods):
        if not any(abs(mu - s[0]) <= 12.0 for s in seeds):
            seeds.append((float(np.clip(mu, x_lo, x_hi)), mname))
    if not seeds:
        seeds = [(float(x[int(np.argmax(np.abs(y_seed)))]), "fallback")]

    # ---------------------- K=0: baseline competitor -----------------------
    aic0 = float("inf")
    b0_hat = 0.0
    if allow_baseline_k0:
        b0_hat = _clip_inside(float(np.mean(y)), -Bmax, Bmax)
        rss0 = float(np.sum((y - b0_hat) ** 2))
        aic0 = _aicc(rss0, k=1, n=n)

    # -------------------- fit around each seed: K=1/K=2 --------------------
    best: Dict[str, Any] = {
        "aic_eff": float("inf"),
        "model": "K0",
        "comps": [],
        "seed": None,
    }

    for mu_seed, method in seeds:
        i_near = int(np.argmin(np.abs(x - mu_seed)))
        A1_0 = _clip_inside(float(y_ds[i_near]), -Amax, Amax)
        s1_0 = _clip_inside(0.6 * (lo_sig + hi_sig), lo_sig, hi_sig)
        mu1_0 = _clip_inside(mu_seed, x_lo, x_hi)
        b0_0 = _clip_inside(0.0, -Bmax, Bmax)

        # gentle mu prior
        def prior_resid(mu: float) -> np.ndarray:
            if center_hint_ms is None:
                return np.array([], dtype=float)
            return np.array(
                [(mu - float(center_hint_ms)) / max(center_hint_sigma_ms, 1e-6)],
                dtype=float,
            )

        # K=1
        _has_hint = center_hint_ms is not None
        _hint_ms = float(center_hint_ms) if _has_hint else 0.0
        _hint_sigma = float(center_hint_sigma_ms)
        if _HAS_NUMBA:
            def r1(p: np.ndarray) -> np.ndarray:
                return _gauss_k1_residual_numba(x, y, p, _hint_ms, _hint_sigma, _has_hint)
        else:
            def r1(p: np.ndarray) -> np.ndarray:
                A1, mu1, s1, b0 = p
                return np.hstack([
                    y - (_gauss(x, A1, mu1, s1) + b0),
                    prior_resid(mu1),
                ])

        bnds1 = (
            [-Amax, x_lo, lo_sig, -Bmax],
            [Amax, x_hi, hi_sig, Bmax],
        )
        if _HAS_RUST_SOLVER:
            _rs1 = _qpsi_native.fit_k1_lm(
                x, y,
                np.array([A1_0, mu1_0, s1_0, b0_0]),
                np.array(bnds1[0]), np.array(bnds1[1]),
                _hint_ms, _hint_sigma, _has_hint,
            )
            A1, mu1, s1, b01 = float(_rs1[0]), float(_rs1[1]), float(_rs1[2]), float(_rs1[3])
        else:
            ls1 = least_squares(
                r1, x0=[A1_0, mu1_0, s1_0, b0_0],
                bounds=bnds1, loss="huber", f_scale=1.5,
            )
            A1, mu1, s1, b01 = ls1.x
        y1 = _gauss(x, A1, mu1, s1) + b01
        rss1 = float(np.sum((y - y1) ** 2))
        aic1 = _aicc(rss1, k=4, n=n)
        cand1: Dict[str, Any] = {
            "aic": aic1,
            "aic_eff": aic1,
            "model": "K1",
            "seed": (mu_seed, method),
            "comps": [{"amp_mv": float(A1), "center_ms": float(mu1), "sigma_ms": float(s1)}],
            "amp_abs": float(abs(A1)),
        }

        # K=2
        # PRUNING: Early termination — if the K=0 baseline model beats K=1 by
        # a large AICc margin (>10), the signal contains no discernible wave
        # component above noise. Fitting K=2 (7 parameters) is futile because
        # adding more Gaussians to a noise-dominated window cannot improve the
        # AICc enough to overcome the parameter penalty. This is clinically
        # justified: a featureless P or T window means no discrete wave is
        # present, and fitting multiple Gaussians to noise wastes compute
        # without improving diagnostic accuracy.
        skip_k2_for_k0_dominance = (
            aicc_early_term
            and allow_baseline_k0
            and (aic0 + aicc_early_term_delta < aic1)
        )
        if skip_k2_for_k0_dominance:
            logger.debug(
                "PRUNE: AICc early term — K=0 wins by %.1f (threshold %.1f)",
                aic1 - aic0, aicc_early_term_delta,
            )
        cand2: Dict[str, Any] | None = None
        if allow_two and not skip_k2_for_k0_dominance:
            A2_0_guess = float(y_ds[i_near])
            if not allow_biphasic:
                A2_0_guess = abs(A2_0_guess) * np.sign(A1_0 if A1_0 != 0 else 1.0)
            A2_0 = _clip_inside(A2_0_guess, -Amax, Amax)
            sep0 = _clip_inside(
                float(min(max(min_separation_ms, 0.5 * (hi_sig + lo_sig)), max_separation_ms)),
                min_separation_ms,
                max_separation_ms,
            )
            s2_0 = _clip_inside(s1_0, lo_sig, hi_sig)

            if _HAS_NUMBA:
                def r2(p: np.ndarray) -> np.ndarray:
                    return _gauss_k2_residual_numba(x, y, p, _hint_ms, _hint_sigma, _has_hint)
            else:
                def r2(p: np.ndarray) -> np.ndarray:
                    A1_, mu1_, s1_, A2_, sep_, s2_, b0_ = p
                    mu2_ = mu1_ + sep_
                    resid = y - (_gauss(x, A1_, mu1_, s1_) + _gauss(x, A2_, mu2_, s2_) + b0_)
                    return np.hstack([resid, prior_resid(mu1_)])

            bnds2 = (
                [-Amax, x_lo, lo_sig, -Amax, min_separation_ms, lo_sig, -Bmax],
                [Amax, x_hi, hi_sig, Amax, max_separation_ms, hi_sig, Bmax],
            )
            if _HAS_RUST_SOLVER:
                _rs2 = _qpsi_native.fit_k2_lm(
                    x, y,
                    np.array([A1_0, mu1_0, s1_0, A2_0, sep0, s2_0, b0_0]),
                    np.array(bnds2[0]), np.array(bnds2[1]),
                    _hint_ms, _hint_sigma, _has_hint,
                )
            else:
                ls2 = least_squares(
                    r2, x0=[A1_0, mu1_0, s1_0, A2_0, sep0, s2_0, b0_0],
                    bounds=bnds2, loss="huber", f_scale=1.5,
                )
            if _HAS_RUST_SOLVER:
                A1_, mu1_, s1_, A2_, sep_, s2_, b02 = (
                    float(_rs2[0]), float(_rs2[1]), float(_rs2[2]),
                    float(_rs2[3]), float(_rs2[4]), float(_rs2[5]), float(_rs2[6]),
                )
            else:
                A1_, mu1_, s1_, A2_, sep_, s2_, b02 = ls2.x
            mu2_ = mu1_ + sep_
            y2 = _gauss(x, A1_, mu1_, s1_) + _gauss(x, A2_, mu2_, s2_) + b02
            rss2 = float(np.sum((y - y2) ** 2))
            aic2 = _aicc(rss2, k=7, n=n)
            if mu2_ < mu1_:
                (A1_, mu1_, s1_), (A2_, mu2_, s2_) = (A2_, mu2_, s2_), (A1_, mu1_, s1_)
            cand2 = {
                "aic": aic2,
                "aic_eff": aic2 + float(aic_delta_two),
                "model": "K2",
                "seed": (mu_seed, method),
                "comps": [
                    {"amp_mv": float(A1_), "center_ms": float(mu1_), "sigma_ms": float(s1_)},
                    {"amp_mv": float(A2_), "center_ms": float(mu2_), "sigma_ms": float(s2_)},
                ],
                "amp_abs": float(max(abs(A1_), abs(A2_))),
            }

        # keep best by aic_eff; tie-bias toward left/right as set
        for c in (cand1, cand2):
            if c is None:
                continue
            better = (c["aic_eff"] < best["aic_eff"]) or (
                abs(c["aic_eff"] - best["aic_eff"]) <= 0.75
                and c["comps"]
                and best["comps"]
                and (
                    (prefer_left_bias and c["comps"][0]["center_ms"] < best["comps"][0]["center_ms"])
                    or (
                        (not prefer_left_bias)
                        and c["comps"][0]["center_ms"] > best["comps"][0]["center_ms"]
                    )
                )
            )
            if better:
                best = c

    # ---------------------- choose vs K=0 + SNR gating ----------------------
    chosen: Dict[str, Any] = {"model": "K0", "comps": [], "seed": None, "aic": aic0}
    if best["comps"]:
        peak = best["amp_abs"]
        ok_aic = (not allow_baseline_k0) or (best["aic"] + 1.0 < aic0)
        ok_snr = 20.0 * np.log10(max(peak, 1e-12) / noise_sigma) >= min_peak_snr_db
        if ok_aic and ok_snr:
            chosen = best

    # --------------------------- notch detection ----------------------------
    notch: Dict[str, Any] = {"notched": False}
    if chosen["model"] == "K2":
        c1, c2 = chosen["comps"]
        A1_n, mu1_n, s1_n = c1["amp_mv"], c1["center_ms"], c1["sigma_ms"]
        A2_n, mu2_n, s2_n = c2["amp_mv"], c2["center_ms"], c2["sigma_ms"]
        same_pol = (np.sign(A1_n) == np.sign(A2_n)) and (np.sign(A1_n) != 0)
        if same_pol:
            grid = np.linspace(mu1_n, mu2_n, 200)
            yfit = _gauss(grid, A1_n, mu1_n, s1_n) + _gauss(grid, A2_n, mu2_n, s2_n)
            valley = float(np.min(np.abs(yfit)))
            minpk = max(1e-9, min(abs(A1_n), abs(A2_n)))
            drop = (minpk - valley) / minpk
            if (drop >= 0.20) and ((mu2_n - mu1_n) >= 20.0):
                notch = {
                    "notched": True,
                    "interpeak_ms": float(mu2_n - mu1_n),
                    "valley_drop_frac": float(drop),
                }

    # ------------------------------ info -----------------------------------
    present = chosen["model"] != "K0"
    peak_amp = max([abs(c["amp_mv"]) for c in chosen["comps"]], default=0.0)
    mu_edge = chosen["comps"][0]["center_ms"] if present else None
    s_edge = chosen["comps"][0]["sigma_ms"] if present else None
    edge_left = bool(present and (mu_edge - x_lo) <= max(12.0, 0.8 * s_edge))
    edge_right = bool(present and (x_hi - mu_edge) <= max(16.0, 0.8 * s_edge))

    info: Dict[str, Any] = dict(notch)
    info.update({
        "present": present,
        "model": chosen["model"],
        "snr_db": float(snr_db(peak_amp)),
        "seed_method": (chosen.get("seed") or (None, None))[1],
        "edge_left": edge_left,
        "edge_right": edge_right,
        "noise_sigma": float(noise_sigma),
        "x_lo": x_lo,
        "x_hi": x_hi,
    })

    if debug:
        if present:
            msg = f"[core:{info['model']}] " + ", ".join(
                f"A={c['amp_mv']:.3f}, mu={c['center_ms']:.1f}, sigma={c['sigma_ms']:.1f}"
                for c in chosen["comps"]
            ) + f" | SNR={info['snr_db']:.1f} dB"
        else:
            msg = "[core:K0] baseline-only chosen"
        print(msg)

    return chosen["comps"], info


# ============================= P wrapper (clean) ============================

def fit_p_wave_components(
    t_ms: np.ndarray,
    y_mv: np.ndarray,
    *,
    sigma_bounds_ms: Tuple[float, float] = (8.0, 60.0),
    min_separation_ms: float = 18.0,
    max_separation_ms: float = 140.0,
    aic_delta_two: float = 4.0,
    allow_biphasic: bool = True,
    allow_two: bool = True,
    debug: bool = False,
    # P-only knobs
    residual_hint_mv: np.ndarray | None = None,
    center_hint_ms: float | None = None,
    center_hint_sigma_ms: float = 25.0,
    prefer_early: bool = True,
    multi_seed: bool = True,
    max_templates: int = 5,
    min_p_snr_db: float = -3.0,
    af_flutter_probe: bool = True,
    flutter_freq_hz_range: Tuple[float, float] = (3.5, 8.5),
    fibrillatory_band_hz: Tuple[float, float] = (7.0, 15.0),
    lead_name: str | None = None,
    beat_index: int | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fit P-wave Gaussian components with optional AF/flutter spectral probe.

    Wraps :func:`fit_wave_components_core` with P-wave-specific defaults and
    post-processing (component labelling, spectral flutter/fibrillation probe
    when no discrete P is found).

    Args:
        t_ms: Time axis (ms) for the P-wave window.
        y_mv: Signal amplitude (mV) in the P-wave window.
        sigma_bounds_ms: Allowed Gaussian sigma range (ms).
        min_separation_ms: Minimum centre separation for K=2.
        max_separation_ms: Maximum centre separation for K=2.
        aic_delta_two: AICc penalty for K=2 model.
        allow_biphasic: Allow opposite-polarity components.
        allow_two: Allow K=2 model.
        debug: Print diagnostics.
        residual_hint_mv: Optional residual for seeding (e.g. prior T-tail).
        center_hint_ms: Prior on P-wave centre.
        center_hint_sigma_ms: Width of centre prior.
        prefer_early: Tie-break toward earlier P-wave components.
        multi_seed: Use multiple seed strategies.
        max_templates: Number of matched-filter templates.
        min_p_snr_db: Minimum SNR (dB) for P-wave acceptance.
        af_flutter_probe: Run spectral analysis when no P detected.
        flutter_freq_hz_range: Flutter frequency band (Hz).
        fibrillatory_band_hz: Fibrillation frequency band (Hz).
        lead_name: Optional lead identifier for info dict.
        beat_index: Optional beat index for info dict.

    Returns:
        Tuple of ``(components, info)`` where components carry
        ``wave_type='P'`` and ``component`` labels (P, P1, P2).
    """
    comps_core, info_core = fit_wave_components_core(
        t_ms, y_mv,
        sigma_bounds_ms=sigma_bounds_ms,
        min_separation_ms=min_separation_ms,
        max_separation_ms=max_separation_ms,
        allow_biphasic=allow_biphasic,
        allow_two=allow_two,
        aic_delta_two=aic_delta_two,
        allow_baseline_k0=True,
        residual_hint_mv=residual_hint_mv,
        center_hint_ms=center_hint_ms,
        center_hint_sigma_ms=center_hint_sigma_ms,
        prefer_left_bias=bool(prefer_early),
        multi_seed=multi_seed,
        max_templates=max_templates,
        min_peak_snr_db=min_p_snr_db,
        debug=debug,
    )

    # label + wave_type
    comps: List[Dict[str, Any]] = []
    if len(comps_core) == 1:
        c = comps_core[0].copy()
        c.update({"wave_type": "P", "component": "P"})
        comps = [c]
    elif len(comps_core) == 2:
        c1, c2 = comps_core
        comps = [
            dict(c1, wave_type="P", component="P1"),
            dict(c2, wave_type="P", component="P2"),
        ]

    # P-specific info + flags
    info: Dict[str, Any] = {
        "p_present": bool(info_core["present"]),
        "model": info_core["model"],
        "snr_db": float(info_core["snr_db"]),
        "seed_method": info_core.get("seed_method"),
        "lead": lead_name,
        "beat_index": beat_index,
        "p_notched": bool(info_core.get("notched", False)),
        "p_edge_clipped": bool(info_core.get("edge_left", False)),
        "p_overlaps_t_tail": bool(info_core.get("edge_left", False)),
    }

    # Spectral probe if no discrete P chosen
    info["p_flutter_like"] = False
    info["p_fibrillatory_like"] = False
    if af_flutter_probe and (not info["p_present"]):
        x = np.asarray(t_ms, float)
        y = np.asarray(y_mv, float)
        n = int(x.size)
        if n >= 16:
            k_hp = max(7, ((n // 8) * 2 + 1))
            y_seed_base = (
                y - np.asarray(residual_hint_mv, float)
                if (residual_hint_mv is not None and len(residual_hint_mv) == n)
                else y
            )
            y_hp = savgol_filter(y_seed_base, k_hp, 2, mode="interp")
            y_seed = y_seed_base - y_hp

            dt = float(max(abs(np.median(np.diff(x))) or 1.0, 1e-3))
            fs = 1000.0 / dt
            f, Pxx = welch(y_seed, fs=fs, nperseg=min(256, n), scaling="spectrum")
            total = float(_trapezoid(Pxx, f)) + 1e-12

            def band_energy(lo: float, hi: float) -> float:
                m = (f >= lo) & (f <= hi)
                return float(_trapezoid(Pxx[m], f[m])) if np.any(m) else 0.0

            flutter_e = band_energy(*flutter_freq_hz_range)
            fibrill_e = band_energy(*fibrillatory_band_hz)
            flutter_ratio = flutter_e / total
            fibrill_ratio = fibrill_e / total

            info["spectral"] = {
                "fs_hz": fs,
                "flutter_ratio": flutter_ratio,
                "fibrillatory_ratio": fibrill_ratio,
            }
            info["p_flutter_like"] = flutter_ratio >= 0.20 and flutter_ratio > fibrill_ratio
            info["p_fibrillatory_like"] = fibrill_ratio >= 0.20 and fibrill_ratio >= flutter_ratio

    if debug:
        if info["p_present"]:
            txt = ", ".join(
                f"{c['component']}: A={c['amp_mv']:.3f}, mu={c['center_ms']:.1f}, sigma={c['sigma_ms']:.1f}"
                for c in comps
            )
            print(f"[P-fit:{info['model']}] {txt} | SNR={info['snr_db']:.1f} dB, edge={info['p_edge_clipped']}")
        else:
            sp = info.get("spectral", {})
            smsg = (
                f" | flutter_ratio={sp.get('flutter_ratio', 0):.2f}, fibrill_ratio={sp.get('fibrillatory_ratio', 0):.2f}"
                if sp
                else ""
            )
            print(f"[P-fit:K0] no discrete P chosen{smsg}")

    return comps, info


# ============================= T wrapper (clean) ============================

def fit_t_wave_components(
    t_ms: np.ndarray,
    y_mv: np.ndarray,
    *,
    sigma_bounds_ms: Tuple[float, float] = (30.0, 140.0),
    min_separation_ms: float = 40.0,
    max_separation_ms: float = 180.0,
    aic_delta_two: float = 4.0,
    allow_biphasic: bool = True,
    allow_two: bool = True,
    debug: bool = False,
    center_hint_ms: float | None = None,
    center_hint_sigma_ms: float = 35.0,
    prefer_late: bool = True,
    multi_seed: bool = True,
    max_templates: int = 5,
    min_t_snr_db: float = -4.0,
    allow_k0: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fit T-wave Gaussian components.

    Wraps :func:`fit_wave_components_core` with T-wave-specific defaults
    (wider sigma, right-edge bias, biphasic detection).

    Args:
        t_ms: Time axis (ms) for the T-wave window.
        y_mv: Signal amplitude (mV) in the T-wave window.
        sigma_bounds_ms: Allowed Gaussian sigma range (ms).
        min_separation_ms: Minimum centre separation for K=2.
        max_separation_ms: Maximum centre separation for K=2.
        aic_delta_two: AICc penalty for K=2.
        allow_biphasic: Allow opposite-polarity T components.
        allow_two: Allow K=2 model.
        debug: Print diagnostics.
        center_hint_ms: Prior on T-wave centre.
        center_hint_sigma_ms: Width of centre prior.
        prefer_late: Tie-break toward later (right) T components.
        multi_seed: Use multiple seed strategies.
        max_templates: Number of matched-filter templates.
        min_t_snr_db: Minimum SNR (dB) for T-wave acceptance.
        allow_k0: Include K=0 baseline as competitor.

    Returns:
        Tuple of ``(components, shape_info)`` where components carry
        ``wave_type='T'`` labels and *shape_info* includes notch and
        biphasic flags.
    """
    comps_core, info_core = fit_wave_components_core(
        t_ms, y_mv,
        sigma_bounds_ms=sigma_bounds_ms,
        min_separation_ms=min_separation_ms,
        max_separation_ms=max_separation_ms,
        allow_biphasic=allow_biphasic,
        allow_two=allow_two,
        aic_delta_two=aic_delta_two,
        allow_baseline_k0=bool(allow_k0),
        residual_hint_mv=None,
        center_hint_ms=center_hint_ms,
        center_hint_sigma_ms=center_hint_sigma_ms,
        prefer_left_bias=not bool(prefer_late),
        multi_seed=multi_seed,
        max_templates=max_templates,
        min_peak_snr_db=min_t_snr_db,
        debug=debug,
    )

    comps: List[Dict[str, Any]] = []
    if len(comps_core) == 1:
        c = comps_core[0].copy()
        c.update({"wave_type": "T", "component": "T"})
        comps = [c]
    elif len(comps_core) == 2:
        c1, c2 = comps_core
        comps = [
            dict(c1, wave_type="T", component="T1"),
            dict(c2, wave_type="T", component="T2"),
        ]

    shape: Dict[str, Any] = {"t_notched": bool(info_core.get("notched", False)), "biphasic": False}
    if len(comps) == 2:
        same_pol = (
            (np.sign(comps[0]["amp_mv"]) == np.sign(comps[1]["amp_mv"]))
            and (np.sign(comps[0]["amp_mv"]) != 0)
        )
        if same_pol:
            sep = float(comps[1]["center_ms"] - comps[0]["center_ms"])
            shape.update({
                "t_notched": True,
                "interpeak_ms": sep,
                "valley_drop_frac": (
                    float(info_core.get("valley_drop_frac", np.nan))
                    if info_core.get("notched")
                    else None
                ),
            })
        else:
            shape["biphasic"] = True

    if debug:
        if comps:
            txt = ", ".join(
                f"{c['component']}: A={c['amp_mv']:.3f}, mu={c['center_ms']:.1f}, sigma={c['sigma_ms']:.1f}"
                for c in comps
            )
            print(f"[T-fit:{info_core['model']}] {txt}")
        else:
            print("[T-fit:K0] no discrete T chosen")

    return comps, shape


# =============================== QRS unified ===============================

def _fit_qrs_unified(
    trace: np.ndarray,
    time_ms: np.ndarray,
    qrs_window: Tuple[float, float],
    tR_ms: float | None = None,
    fs: float = 500.0,
    *,
    # ---- duration control (single source of truth) --------------------------
    min_qrs_ms: float = 60.0,
    max_qrs_ms: float = 180.0,
    padding_ms: float = 10.0,
    # ---- morphology / width control -----------------------------------------
    force_narrow: bool = False,
    sigma_bounds_ms: Tuple[float, float] = (6.0, 60.0),
    narrow_sigma_R_ms: float = 12.0,
    narrow_sigma_QS_ms: float = 10.0,
    max_components: int = 4,
    q_s_threshold_rel: float = 0.03,
    rprime_threshold_rel: float = 0.05,
    min_sep_ms: float = 8.0,
) -> SimpleNamespace:
    """Unified QRS fitter with automatic boundary detection and component seeding.

    Detects practical QRS boundaries using gradient analysis, seeds Q/R/S
    (and optional R') components from the signal, then solves for amplitudes
    via constrained least squares with fixed centres and sigmas.

    Args:
        trace: Full-beat signal amplitude (mV).
        time_ms: Corresponding time axis (ms).
        qrs_window: ``(start_ms, end_ms)`` approximate QRS window.
        tR_ms: Optional R-peak time hint (ms); auto-detected if *None*.
        fs: Sampling frequency (Hz) used as fallback for dt.
        min_qrs_ms: Minimum allowed QRS duration (ms).
        max_qrs_ms: Maximum allowed QRS duration (ms).
        padding_ms: Extra padding each side for the fit mask.
        force_narrow: Use fixed narrow sigmas for all lobes.
        sigma_bounds_ms: Allowed sigma range when not forcing narrow.
        narrow_sigma_R_ms: Fixed R-lobe sigma when *force_narrow*.
        narrow_sigma_QS_ms: Fixed Q/S-lobe sigma when *force_narrow*.
        max_components: Maximum number of QRS lobes (3 = Q/R/S, 4 = +R').
        q_s_threshold_rel: Q/S amplitude must exceed this fraction of |R|.
        rprime_threshold_rel: R' amplitude must exceed this fraction of |R|.
        min_sep_ms: Minimum separation between neighbouring lobes (ms).

    Returns:
        ``SimpleNamespace`` with attributes ``params`` (list of
        ``(amplitude, centre_ms, sigma_ms)`` tuples), ``yfit`` (fitted
        waveform), ``mask`` (boolean QRS region), ``r_peak_time`` (ms),
        ``bounds`` ``(left_ms, right_ms)``, and ``settings`` dict.
    """
    # ---------- inputs ----------
    t = np.asarray(time_ms, float)
    y = np.asarray(trace, float)
    q0, q1 = float(qrs_window[0]), float(qrs_window[1])
    if q1 < q0:
        q0, q1 = q1, q0

    # ---------- helper: index window ----------
    def idx_window(lo: float, hi: float) -> np.ndarray:
        return np.where((t >= lo) & (t <= hi))[0]

    # ---------- R-peak estimate ----------
    m0 = idx_window(q0, q1)
    if m0.size == 0:
        m0 = np.arange(t.size)
        q0, q1 = float(t[0]), float(t[-1])

    if tR_ms is None or not np.isfinite(tR_ms):
        r_idx0 = m0[np.argmax(np.abs(y[m0]))]
        tR = float(t[r_idx0])
    else:
        r_idx0 = int(np.argmin(np.abs(t - tR_ms)))
        tR = float(t[r_idx0])

    # refine R peak within +/-8 ms
    def refine_peak(center_idx: int, span_ms: float = 8.0) -> int:
        lo = t[center_idx] - span_ms
        hi = t[center_idx] + span_ms
        ii = idx_window(lo, hi)
        if ii.size:
            sgn = np.sign(y[center_idx]) or 1.0
            k = ii[np.argmax(sgn * y[ii])]
            return int(k)
        return int(center_idx)

    r_idx = refine_peak(r_idx0, 8.0)
    tR = float(t[r_idx])
    aR_obs = float(y[r_idx])
    sgnR = np.sign(aR_obs) or 1.0

    # ---------- boundary detection ----------
    dt_ms = float(np.median(np.diff(t))) if t.size > 1 else (1000.0 / fs)

    def gauss_smooth(x: np.ndarray, sigma_ms_smooth: float = 8.0) -> np.ndarray:
        if dt_ms <= 0:
            return x
        sig_samp = max(1, int(round(sigma_ms_smooth / dt_ms)))
        k = max(3, int(6 * sig_samp) | 1)
        xpad = np.pad(x, (k, k), mode='edge')
        grid = np.arange(-k, k + 1, 1.0)
        g = np.exp(-0.5 * (grid / sig_samp) ** 2)
        g /= g.sum()
        z = np.convolve(xpad, g, mode='same')[k:-k]
        return z

    a = np.abs(gauss_smooth(np.gradient(y, t, edge_order=2), 8.0))
    thr = 0.18 * np.max(a[m0]) if np.any(a[m0] > 0) else 0.0

    def idx_window_bool(lo: float, hi: float) -> np.ndarray:
        return (t >= lo) & (t <= hi)

    def walk_boundary(start_idx: int, step: int) -> float:
        i = int(start_idx)
        quiet = 0
        while 0 <= i < t.size:
            if a[i] < thr:
                quiet += 1
                if quiet >= 3:
                    return float(t[i])
            else:
                quiet = 0
            i += step
        return float(t[i - step])

    left_ms = walk_boundary(r_idx, -1)
    right_ms = walk_boundary(r_idx, +1)

    left_ms = max(left_ms, q0)
    right_ms = min(right_ms, q1)
    half = 0.5 * (right_ms - left_ms)
    half = np.clip(half, min_qrs_ms / 2.0, max_qrs_ms / 2.0)
    left_ms = max(q0, tR - half)
    right_ms = min(q1, tR + half)

    m_fit = np.where(idx_window_bool(left_ms - padding_ms, right_ms + padding_ms))[0]
    if m_fit.size == 0:
        m_fit = idx_window(left_ms - 40.0, right_ms + 40.0)

    # ---------- seed components ----------
    comps: List[Tuple[str, float, float | None, float]] = []

    def pick_extreme(
        lo: float, hi: float, polarity_sign: float,
    ) -> Tuple[float, float] | None:
        ii = idx_window(lo, hi)
        if ii.size == 0:
            return None
        k = ii[np.argmax(polarity_sign * y[ii])]
        return (float(t[k]), float(y[k]))

    mu_R, A_R0 = float(t[r_idx]), float(y[r_idx])
    comps.append(("R", mu_R, (12.0 if force_narrow else None), sgnR))

    cand = pick_extreme(left_ms, mu_R - min_sep_ms, -sgnR)
    if cand is not None and abs(cand[1]) >= q_s_threshold_rel * abs(A_R0):
        comps.append(("Q", cand[0], (10.0 if force_narrow else None), -sgnR))

    cand = pick_extreme(mu_R + min_sep_ms, right_ms, -sgnR)
    if cand is not None and abs(cand[1]) >= q_s_threshold_rel * abs(A_R0):
        comps.append(("S", cand[0], (10.0 if force_narrow else None), -sgnR))

    if max_components >= 4:
        cand = pick_extreme(mu_R + 1.5 * min_sep_ms, right_ms, sgnR)
        if cand is not None and abs(cand[1]) >= rprime_threshold_rel * abs(A_R0):
            comps.append(("R2", cand[0], (12.0 if force_narrow else None), sgnR))

    comps.sort(key=lambda c: c[1])
    comps = comps[:max_components]

    # ---------- estimate sigma if not forced narrow ----------
    def half_width_sigma(mu_ms_hw: float, sign_hint: float) -> float:
        ii = idx_window(left_ms, right_ms)
        if ii.size < 3:
            return 12.0
        idx_mu = int(np.argmin(np.abs(t[ii] - mu_ms_hw)))
        idx_mu = ii[idx_mu]
        ypeak = sign_hint * y[idx_mu]
        if ypeak <= 0:
            return 12.0
        h = 0.5 * ypeak
        j = idx_mu
        while j > ii[0] and sign_hint * y[j] > h:
            j -= 1
        tL = t[j]
        j = idx_mu
        while j < ii[-1] and sign_hint * y[j] > h:
            j += 1
        tRr = t[j]
        fwhm = max(4.0, float(tRr - tL))
        return float(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))))

    new_comps: List[Tuple[str, float, float, float]] = []
    for name, mu, sigma_fixed, sign_hint in comps:
        if sigma_fixed is None:
            s = np.clip(half_width_sigma(mu, sign_hint), sigma_bounds_ms[0], sigma_bounds_ms[1])
        else:
            s = float(sigma_fixed)
        new_comps.append((name, float(mu), float(s), float(sign_hint)))
    comps_final = new_comps

    for i in range(1, len(comps_final)):
        prev = comps_final[i - 1]
        cur = comps_final[i]
        if cur[1] - prev[1] < min_sep_ms:
            shift = 0.5 * (min_sep_ms - (cur[1] - prev[1]))
            comps_final[i - 1] = (prev[0], prev[1] - shift, prev[2], prev[3])
            comps_final[i] = (cur[0], cur[1] + shift, cur[2], cur[3])

    tm = t[m_fit]
    Ym = y[m_fit]

    def gauss_basis(mu: float, s: float, tt: np.ndarray) -> np.ndarray:
        z = (tt - mu) / (s + 1e-9)
        return np.exp(-0.5 * z * z)

    Phi: List[np.ndarray] = []
    sign_hints: List[float] = []
    for name, mu, s, sgn_hint in comps_final:
        Phi.append(gauss_basis(mu, s, tm))
        sign_hints.append(sgn_hint)
    Phi_mat = np.stack(Phi, axis=1) if len(Phi) else np.zeros((tm.size, 0))

    keep = np.ones(len(comps_final), dtype=bool)
    A_vec = np.zeros(len(comps_final), float)
    for _ in range(3):
        if not np.any(keep):
            break
        Ak, *_ = np.linalg.lstsq(Phi_mat[:, keep], Ym, rcond=None)
        A_vec[keep] = Ak
        bad = np.array([sign_hints[i] * A_vec[i] < 0 for i in range(len(comps_final))])
        if not np.any(bad & keep):
            break
        keep = keep & ~bad

    yfit = np.zeros_like(t, dtype=float)
    if len(comps_final):
        Gfull = [gauss_basis(mu, s, t) for (_, mu, s, _) in comps_final]
        ysum = np.zeros_like(t, float)
        for j, Gj in enumerate(Gfull):
            ysum += (A_vec[j] if np.isfinite(A_vec[j]) else 0.0) * Gj
        mask_full = (t >= (left_ms - padding_ms)) & (t <= (right_ms + padding_ms))
        yfit[mask_full] = ysum[mask_full]

    params = [
        (float(A_vec[i]), float(comps_final[i][1]), float(comps_final[i][2]))
        for i in range(len(comps_final))
    ]

    return SimpleNamespace(
        params=params,
        yfit=yfit,
        mask=(t >= left_ms) & (t <= right_ms),
        r_peak_time=float(tR),
        bounds=(float(left_ms), float(right_ms)),
        settings=dict(
            min_qrs_ms=min_qrs_ms, max_qrs_ms=max_qrs_ms,
            force_narrow=force_narrow, sigma_bounds_ms=sigma_bounds_ms,
            max_components=max_components, q_s_threshold_rel=q_s_threshold_rel,
            rprime_threshold_rel=rprime_threshold_rel, padding_ms=padding_ms,
        ),
    )
