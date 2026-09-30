"""2D Gaussian decomposition engine.

Models ECG wave components as sums of 2D Gaussians and fits via nonlinear
least squares. Zero internal deps, pure math. C++ priority #1.
Source: Cell 6.
"""
from __future__ import annotations

import logging
import math
from typing import List

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

logger = logging.getLogger("qpsi")

# ---------------------------------------------------------------------------
# Numba-accelerated kernels (optional)
# ---------------------------------------------------------------------------
try:
    import numba

    @numba.njit(cache=True, fastmath=True)
    def _sum_gaussians_2d_numba(t, params):
        """Numba-accelerated 2D Gaussian synthesis."""
        n_pts = t.shape[0]
        n_waves = len(params) // 4
        out = np.zeros((2, n_pts))
        for w in range(n_waves):
            t0 = params[4 * w]
            sigma = max(params[4 * w + 1], 1.0)
            amp = params[4 * w + 2]
            alpha = params[4 * w + 3]
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            inv_s = 1.0 / sigma
            for i in range(n_pts):
                z = (t[i] - t0) * inv_s
                env = amp * math.exp(-0.5 * z * z)
                out[0, i] += env * cos_a
                out[1, i] += env * sin_a
        return out

    @numba.njit(cache=True, fastmath=True)
    def _sum_gaussians_1d_numba(t, params, n_waves):
        """Numba-accelerated 1D Gaussian sum (for fit_gaussians_1d residual)."""
        n_pts = t.shape[0]
        out = np.zeros(n_pts)
        for w in range(n_waves):
            t0 = params[3 * w]
            s = max(params[3 * w + 1], 1.0)
            A = params[3 * w + 2]
            inv_s = 1.0 / s
            for i in range(n_pts):
                z = (t[i] - t0) * inv_s
                out[i] += A * math.exp(-0.5 * z * z)
        return out

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Rust-accelerated solver (optional)
try:
    import qpsi_native as _qpsi_native
    # Verify it's the real compiled extension, not a namespace package
    if not hasattr(_qpsi_native, "fit_1d_magnitude_lm"):
        raise ImportError(f"qpsi_native found but missing fit_1d_magnitude_lm. Available: {[x for x in dir(_qpsi_native) if not x.startswith('_')]}")
    _HAS_RUST = True
    logger.info("Rust QPSI Gaussian solver: ENABLED")
except ImportError as e:
    _HAS_RUST = False
    logger.warning("Rust QPSI Gaussian solver: DISABLED (%s)", e)


# =============================================================================
# 0.  Core synthesiser
# =============================================================================
def sum_gaussians_2d(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Synthesise a 2D vector trace from a sum of oriented Gaussians.

    Each Gaussian component is parameterised by 4 values packed consecutively
    in *params*: ``[t0, sigma, amp, alpha]``.

    Args:
        t: 1D time array of shape ``(N,)``.
        params: Flat parameter vector of length ``4 * n_waves``.

    Returns:
        2D array of shape ``(2, N)`` -- the synthesised X/Y vector trace.
    """
    if _HAS_NUMBA:
        return _sum_gaussians_2d_numba(np.ascontiguousarray(t), np.ascontiguousarray(params))
    n = len(params) // 4
    out = np.zeros((2, len(t)), dtype=float)
    for i in range(n):
        t0, sigma, amp, alpha = params[4 * i : 4 * i + 4]
        sigma = max(float(sigma), 1.0)
        env = float(amp) * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        out[0] += env * math.cos(alpha)
        out[1] += env * math.sin(alpha)
    return out


# =============================================================================
# 1-A.  1D multi-Gaussian fitter (magnitude only)
# =============================================================================
def fit_gaussians_1d(
    t: np.ndarray,
    mag: np.ndarray,
    n_waves: int,
    *,
    sigma_max: float = 80.0,
) -> np.ndarray:
    """Fit *n_waves* Gaussians to a 1D magnitude envelope.

    Uses ``scipy.optimize.least_squares`` with trust-region reflective (TRF)
    bounds to decompose the magnitude signal into a sum of Gaussian bumps.

    Args:
        t: 1D time array of shape ``(N,)``.
        mag: 1D magnitude array of shape ``(N,)`` (typically ``|v|``).
        n_waves: Number of Gaussian components to fit.
        sigma_max: Upper bound on Gaussian width (samples/ms).

    Returns:
        Flat parameter array ``[t0, sigma, A] * n_waves`` of length
        ``3 * n_waves``.  Returns zeros on degenerate inputs.
    """

    # -- guard 1: degenerate time window -----------------------------------
    if t.size < 3 or t[0] == t[-1]:
        return np.zeros(3 * n_waves, dtype=float)

    # -- guard 2: flat-line magnitude window -------------------------------
    if np.all(mag == 0):
        return np.zeros(3 * n_waves, dtype=float)

    # ---------- initial peak guess (unchanged) ----------------------------
    min_dist = max(3, len(t) // (2 * n_waves))
    peak_idx, _ = find_peaks(mag, distance=min_dist)
    if peak_idx.size == 0:
        peak_idx = np.array([np.argmax(mag)])

    dt = np.median(np.diff(t))
    sigma0 = min(0.45 * (len(t) // (2 * n_waves)) * dt, 0.9 * sigma_max)

    g: List[float] = []
    for idx in np.sort(peak_idx)[:n_waves]:
        g += [float(t[idx]), sigma0, float(mag[idx])]
    while len(g) < 3 * n_waves:
        g += g[-3:]

    init = np.asarray(g[: 3 * n_waves], dtype=float)

    # ---------- bounds ----------------------------------------------------
    t_min, t_max = float(t[0]), float(t[-1])
    amp_max = 5.0 * np.max(mag)

    if amp_max == 0:  # guard 3: still flat
        return np.zeros(3 * n_waves, dtype=float)

    lb: List[float] = []
    ub: List[float] = []
    for _ in range(n_waves):
        lb += [t_min, 1.0, 0.0]
        ub += [t_max, sigma_max, amp_max]
    lb_arr, ub_arr = np.asarray(lb), np.asarray(ub)
    init = np.clip(init, lb_arr + 1e-6, ub_arr - 1e-6)

    # ---------- residual & optimisation -----------------------------------
    _n_w = n_waves
    _t_c = np.ascontiguousarray(t)
    if _HAS_NUMBA:
        def residual(p: np.ndarray) -> np.ndarray:
            return _sum_gaussians_1d_numba(_t_c, p, _n_w) - mag
    else:
        def residual(p: np.ndarray) -> np.ndarray:
            n = len(p) // 3
            env = np.zeros_like(t)
            for i in range(n):
                t0, s, A = p[3 * i : 3 * i + 3]
                env += A * np.exp(-0.5 * ((t - t0) / max(s, 1)) ** 2)
            return env - mag

    if _HAS_RUST:
        res_x = _qpsi_native.fit_1d_magnitude_lm(
            np.ascontiguousarray(t, dtype=np.float64),
            np.ascontiguousarray(mag, dtype=np.float64),
            np.ascontiguousarray(init, dtype=np.float64),
            np.ascontiguousarray(lb_arr, dtype=np.float64),
            np.ascontiguousarray(ub_arr, dtype=np.float64),
            n_waves,
        )
        return np.asarray(res_x, dtype=float)
    else:
        res = least_squares(
            residual, init, bounds=(lb_arr, ub_arr), method="trf", max_nfev=3000
        )
        if not res.success:
            logger.warning("fit_gaussians_1d: %s", res.message)
        return res.x


# =============================================================================
# 1-B.  Deterministic initialiser for the 2D fit
# =============================================================================
def _initial_guess_2d(
    t: np.ndarray,
    v2: np.ndarray,
    n_waves: int,
    sigma_max: float,
) -> np.ndarray:
    """Deterministic guess via 1D magnitude fit + local direction.

    Returns a flat array ``[t0, sigma, A, alpha] * n_waves``.
    """

    # -- guard: if the slice has only ONE distinct sample, give up early ----
    if t.size < 2 or t[0] == t[-1]:
        return np.zeros(4 * n_waves, dtype=float)

    # ----------------------------------------------------------------------
    mag = np.linalg.norm(v2, axis=0)
    p1d = fit_gaussians_1d(t, mag, n_waves, sigma_max=sigma_max)

    angle = np.unwrap(np.arctan2(v2[1], v2[0]))

    g2d: list[float] = []
    for i in range(n_waves):
        t0, sigma, amp = p1d[3 * i : 3 * i + 3]
        idx = int(np.argmin(np.abs(t - t0)))
        alpha = float(angle[idx])
        g2d += [t0, sigma, amp, alpha]

    # pad if 1D fit produced fewer peaks than requested --------------------
    while len(g2d) < 4 * n_waves:
        g2d += g2d[-4:]

    return np.asarray(g2d[: 4 * n_waves], dtype=float)


# =============================================================================
# 2.  Full 2D fitter (deterministic by default)
# =============================================================================
def fit_gaussians_2d(
    t: np.ndarray,
    v2: np.ndarray,
    n_waves: int,
    *,
    angle_lambda: float = 0.1,
    amplitude_lambda: float = 0.2,
    sigma_max: float = 80.0,
    n_restarts: int = 0,
    jitter_scale: float = 0.5,
) -> np.ndarray:
    """Least-squares fit of exactly *n_waves* oriented 2D Gaussian components.

    Fits a sum of 2D Gaussians to the vector trace *v2* with penalty terms
    that discourage angle deviation and amplitude overshoot relative to the
    local signal.

    Args:
        t: 1D time array of shape ``(N,)``.
        v2: 2D vector trace of shape ``(2, N)``.
        n_waves: Number of Gaussian components to fit.
        angle_lambda: Weight of the angle-deviation penalty.
        amplitude_lambda: Weight of the amplitude-overshoot penalty.
        sigma_max: Upper bound on Gaussian width.
        n_restarts: Number of random-jitter restarts (0 = deterministic).
        jitter_scale: Scale of jitter relative to bound range.

    Returns:
        Flat parameter array ``[t0, sigma, A, alpha] * n_waves`` of length
        ``4 * n_waves``.
    """
    t_min, t_max = float(t[0]), float(t[-1])
    init = _initial_guess_2d(t, v2, n_waves, sigma_max)

    # bounds ---------------------------------------------------------------
    mag = np.linalg.norm(v2, axis=0)
    amp_max = max(5.0 * np.max(mag), 1e-3)
    lb: List[float] = []
    ub: List[float] = []
    for _ in range(n_waves):
        lb += [t_min, 1.0, 0.0, -math.pi]
        ub += [t_max, sigma_max, amp_max, math.pi]
    lb_arr = np.asarray(lb, float)
    ub_arr = np.asarray(ub, float)
    eps = 1e-6

    # --- keep the deterministic guess strictly inside bounds ---------------
    init = np.clip(init, lb_arr + eps, ub_arr - eps)

    # residual & penalties -------------------------------------------------
    def residual(p: np.ndarray) -> np.ndarray:
        base = (v2 - sum_gaussians_2d(t, p)).ravel()

        pen_ang: List[float] = []
        pen_amp: List[float] = []
        for i in range(n_waves):
            t0, _, A, alpha = p[4 * i : 4 * i + 4]
            idx = int(np.argmin(np.abs(t - t0)))
            alpha_ref = np.arctan2(v2[1, idx], v2[0, idx])
            local_mag = mag[idx]
            pen_ang.append(local_mag * (alpha - alpha_ref))
            pen_amp.append(max(A - local_mag, 0.0))

        return np.concatenate(
            [
                base,
                angle_lambda * np.asarray(pen_ang),
                amplitude_lambda * np.asarray(pen_amp),
            ]
        )

    # optional jittered restarts for tough cases ---------------------------
    best_init, best_chi2 = init, np.square(residual(init)).sum()
    for _ in range(n_restarts):
        trial = init + np.random.normal(0.0, jitter_scale, init.size) * (
            ub_arr - lb_arr
        )
        trial = np.clip(trial, lb_arr + eps, ub_arr - eps)
        chi2 = np.square(residual(trial)).sum()
        if chi2 < best_chi2:
            best_init, best_chi2 = trial, chi2

    # final optimisation ---------------------------------------------------
    if _HAS_RUST:
        v2_c = np.ascontiguousarray(v2, dtype=np.float64)
        res_x = _qpsi_native.fit_2d_lm(
            np.ascontiguousarray(t, dtype=np.float64),
            v2_c,
            np.ascontiguousarray(best_init, dtype=np.float64),
            np.ascontiguousarray(lb_arr, dtype=np.float64),
            np.ascontiguousarray(ub_arr, dtype=np.float64),
            n_waves,
            angle_lambda,
            amplitude_lambda,
        )
        return np.asarray(res_x, dtype=float)
    else:
        res = least_squares(
            residual, best_init, bounds=(lb_arr, ub_arr), method="trf", max_nfev=4000
        )
        if not res.success:
            logger.warning("fit_gaussians_2d: %s", res.message)
        return res.x


# =============================================================================
# 3.  Hybrid fitter (explicit bounds version)
# =============================================================================
def fit_gaussians_hybrid(
    t: np.ndarray,
    v2: np.ndarray,
    n_waves_2d: int,
    n_extra: int,
    amp_thr: float,
    *,
    t_low: float,
    t_high: float,
    sigma_max: float = 80.0,
    sigma_min: float = 1.0,
    amp_max_factor: float = 5.0,
    refine_sigmas: bool = False,
    angle_lambda: float = 0.1,
    amplitude_lambda: float = 0.2,
    n_restarts: int = 0,
    jitter_scale: float = 0.5,
    q_zone: tuple[float, float] = (-60.0, 0.0),
    q_angle_thresh_deg: float = 40.0,
) -> np.ndarray:
    """Two-stage hybrid Gaussian fitter with residual peak detection.

    Stage 1 fits *n_waves_2d* oriented 2D Gaussians via
    :func:`fit_gaussians_2d`.  Stage 2 detects residual peaks above
    *amp_thr* and refines amplitudes (and optionally sigmas) via a 1D
    magnitude-only optimisation.  A safety-net injects a Q-candidate when
    a strong angle swing is detected in *q_zone* but no residual peak
    was found there.

    Args:
        t: 1D time array of shape ``(N,)``.
        v2: 2D vector trace of shape ``(2, N)``.
        n_waves_2d: Number of Gaussians for the initial 2D fit.
        n_extra: Maximum number of extra residual-peak Gaussians.
        amp_thr: Amplitude threshold for residual peak detection.
        t_low: Lower time bound for extra-peak centres.
        t_high: Upper time bound for extra-peak centres.
        sigma_max: Upper bound on Gaussian width.
        sigma_min: Lower bound on Gaussian width.
        amp_max_factor: Factor applied to max magnitude for amplitude bound.
        refine_sigmas: If True, also optimise sigma of seeded waves in stage 2.
        angle_lambda: Weight of the angle-deviation penalty (stage 1).
        amplitude_lambda: Weight of the amplitude-overshoot penalty (stage 1).
        n_restarts: Number of random-jitter restarts for stage 1 (0 = deterministic).
        jitter_scale: Scale of jitter relative to bound range.
        q_zone: ``(t_lo, t_hi)`` time window where a Q-wave may appear.
        q_angle_thresh_deg: Angle-swing threshold (degrees) to force a Q-candidate.

    Returns:
        Flat parameter array ``[t0, sigma, A, alpha] * (n_waves_2d + n_sel)``
        where *n_sel* is the number of extra residual peaks selected.
        Returns ``np.empty(0, float)`` on failure.
    """
    # -- guards ------------------------------------------------------------
    if t.size < 3 or t_low >= t_high:
        return np.empty(0, float)

    # -- Stage-1 : deterministic 2D fit ------------------------------------
    p2d = fit_gaussians_2d(
        t,
        v2,
        n_waves_2d,
        sigma_max=sigma_max,
        angle_lambda=angle_lambda,
        amplitude_lambda=amplitude_lambda,
        n_restarts=n_restarts,
        jitter_scale=jitter_scale,
    )
    if p2d.size == 0:
        return p2d

    t0_seed = [float(p2d[4 * i]) for i in range(n_waves_2d)]
    s_seed = [float(p2d[4 * i + 1]) for i in range(n_waves_2d)]

    # -- Stage-2 : residual on **vector** rather than |v| ------------------
    mag = np.linalg.norm(v2, axis=0)
    fit_vec = sum_gaussians_2d(t, p2d)
    vec_res = v2 - fit_vec
    resid = np.linalg.norm(vec_res, axis=0)

    # -- Detect residual peaks above threshold -----------------------------
    peaks, props = find_peaks(resid, height=amp_thr)
    if peaks.size:
        order = np.argsort(props["peak_heights"])[::-1]
        peaks = peaks[order][:n_extra]
    n_sel = int(peaks.size)

    # -- Safety-net: force one Q-candidate if strong angle-swing exists ----
    t_q_lo, t_q_hi = q_zone
    idx_q = (t >= t_q_lo) & (t <= t_q_hi)
    angle = np.unwrap(np.arctan2(v2[1], v2[0]))
    if idx_q.any() and np.ptp(angle[idx_q]) > np.deg2rad(q_angle_thresh_deg):
        has_q_peak = np.any((t[peaks] >= t_q_lo) & (t[peaks] <= t_q_hi))
        if not has_q_peak:
            cand_idx = np.argmax(resid * idx_q)
            peaks = np.append(peaks, cand_idx)
            n_sel = int(peaks.size)
            if n_sel > n_extra:
                order = np.argsort(resid[peaks])[::-1][:n_extra]
                peaks = peaks[order]
                n_sel = n_extra
            logger.debug(
                "Q-slot injected at t=%.1f ms (resid=%.4f)",
                t[cand_idx],
                resid[cand_idx],
            )

    # -- Build optimisation vector & bounds --------------------------------
    x0: List[float] = []
    lb: List[float] = []
    ub: List[float] = []
    max_amp = amp_max_factor * max(1e-6, mag.max())

    # 4a) seeded waves: optimise amplitude (+sigma if requested)
    for i in range(n_waves_2d):
        A0 = float(p2d[4 * i + 2])
        x0.append(A0)
        lb.append(0.0)
        ub.append(max_amp)
        if refine_sigmas:
            s0 = s_seed[i]
            x0.append(s0)
            lb.append(sigma_min)
            ub.append(sigma_max)

    # 4b) extra residual waves: t0, sigma, A -- with explicit time bounds
    if n_sel:
        dt = np.median(np.diff(t)) if t.size > 1 else 1.0
        sigma_guess = min(
            0.45 * (len(t) // (2 * (n_waves_2d + n_sel))) * dt, 0.9 * sigma_max
        )
        for idx in peaks:
            t0_guess = float(t[idx])
            A0 = float(resid[idx])
            x0 += [t0_guess, sigma_guess, A0]
            lb += [t_low, sigma_min, 0.0]
            ub += [t_high, sigma_max, max_amp]

    x0_arr = np.asarray(x0, float)
    lb_arr = np.asarray(lb, float)
    ub_arr = np.asarray(ub, float)
    x0_arr = np.clip(x0_arr, lb_arr + 1e-6, ub_arr - 1e-6)

    # -- Residual for 1D optimisation --------------------------------------
    def _resid_1d(p: np.ndarray) -> np.ndarray:
        """Residual for 1D optimisation (amplitude-only or amplitude+sigma)."""
        env = np.zeros_like(t)
        ptr = 0

        # -------- seeded waves --------------------------------------------
        for i in range(n_waves_2d):
            A = p[ptr]
            ptr += 1
            if refine_sigmas:
                s_ = p[ptr]
                ptr += 1
            else:
                s_ = s_seed[i]
            env += A * np.exp(-0.5 * ((t - t0_seed[i]) / max(s_, 1.0)) ** 2)

        # -------- extra residual waves ------------------------------------
        for _ in range(n_sel):
            t0_e, s_e, A_e = p[ptr : ptr + 3]
            ptr += 3
            env += A_e * np.exp(-0.5 * ((t - t0_e) / max(s_e, 1.0)) ** 2)

        return env - mag

    # -- Optimise ----------------------------------------------------------
    if _HAS_RUST:
        p_opt = _qpsi_native.fit_1d_hybrid_lm(
            np.ascontiguousarray(t, dtype=np.float64),
            np.ascontiguousarray(mag, dtype=np.float64),
            np.ascontiguousarray(x0_arr, dtype=np.float64),
            np.ascontiguousarray(lb_arr, dtype=np.float64),
            np.ascontiguousarray(ub_arr, dtype=np.float64),
            np.asarray(t0_seed, dtype=np.float64),
            np.asarray(s_seed, dtype=np.float64),
            n_waves_2d, n_sel, refine_sigmas,
        )
        p_opt = np.asarray(p_opt)
    else:
        res = least_squares(
            _resid_1d, x0_arr, bounds=(lb_arr, ub_arr), method="trf", max_nfev=3000
        )
        if not res.success:
            logger.debug("hybrid 1D refinement: %s", res.message)
        p_opt = res.x

    # -- Assemble final parameter array ------------------------------------
    out: List[float] = []
    ptr = 0

    for i in range(n_waves_2d):
        A_i = p_opt[ptr]
        ptr += 1
        if refine_sigmas:
            s_i = p_opt[ptr]
            ptr += 1
        else:
            s_i = s_seed[i]
        idx_c = int(np.argmin(np.abs(t - t0_seed[i])))
        alpha_i = float(np.arctan2(v2[1, idx_c], v2[0, idx_c]))
        out += [t0_seed[i], s_i, A_i, alpha_i]

    for _ in range(n_sel):
        t0_e, s_e, A_e = p_opt[ptr : ptr + 3]
        ptr += 3
        idx_c = int(np.argmin(np.abs(t - t0_e)))
        alpha_e = float(np.arctan2(v2[1, idx_c], v2[0, idx_c]))
        out += [t0_e, s_e, A_e, alpha_e]

    return np.asarray(out, float)


# =============================================================================
# 4.  Hybrid sub-range wrapper with explicit bounds
# =============================================================================
def fit_gaussians_hybrid_subrange(
    t: np.ndarray,
    v2: np.ndarray,
    t_min: float,
    t_max: float,
    *,
    n_waves: int,
    n_extra: int,
    amp_thr: float,
    sigma_max: float,
    angle_lambda: float,
    amplitude_lambda: float,
    refine_sigmas: bool = False,
    n_restarts: int = 0,
    jitter_scale: float = 0.5,
    amp_min: float = 0.0,
) -> np.ndarray:
    """Sub-range wrapper for the hybrid fitter.

    Clips *t* and *v2* to the ``[t_min, t_max]`` window, checks the
    amplitude threshold, and delegates to :func:`fit_gaussians_hybrid`.

    Args:
        t: 1D time array of shape ``(N,)``.
        v2: 2D vector trace of shape ``(2, N)``.
        t_min: Lower time bound of the sub-range.
        t_max: Upper time bound of the sub-range.
        n_waves: Number of Gaussians for the initial 2D fit.
        n_extra: Maximum number of extra residual-peak Gaussians.
        amp_thr: Amplitude threshold for residual peak detection.
        sigma_max: Upper bound on Gaussian width.
        angle_lambda: Weight of the angle-deviation penalty.
        amplitude_lambda: Weight of the amplitude-overshoot penalty.
        refine_sigmas: If True, also optimise sigma of seeded waves.
        n_restarts: Number of random-jitter restarts (0 = deterministic).
        jitter_scale: Scale of jitter relative to bound range.
        amp_min: Minimum peak amplitude to proceed with fitting.

    Returns:
        Flat parameter array ``[t0, sigma, A, alpha] * n_total``.
        Returns ``np.empty(0, float)`` on failure or skip.
    """
    if t_min >= t_max:
        logger.debug("hybrid skip: t_min>=t_max (%.1f >= %.1f)", t_min, t_max)
        return np.empty(0, float)

    idx = (t >= t_min) & (t <= t_max)
    if not np.any(idx):
        logger.debug("hybrid skip: no points in [%.1f,%.1f]", t_min, t_max)
        return np.empty(0, float)

    t_sub, v2_sub = t[idx], v2[:, idx]
    if np.max(np.linalg.norm(v2_sub, axis=0)) < amp_min:
        logger.debug("hybrid skip: below amp_min=%.3f", amp_min)
        return np.empty(0, float)

    try:
        return fit_gaussians_hybrid(
            t_sub,
            v2_sub,
            n_waves_2d=n_waves,
            n_extra=n_extra,
            amp_thr=amp_thr,
            t_low=t_min,
            t_high=t_max,
            sigma_max=sigma_max,
            refine_sigmas=refine_sigmas,
            angle_lambda=angle_lambda,
            amplitude_lambda=amplitude_lambda,
            n_restarts=n_restarts,
            jitter_scale=jitter_scale,
        )
    except ValueError as e:
        msg = str(e)
        if "Initial guess is outside" in msg or "lower bound must be strictly less" in msg:
            logger.warning(
                "fit_gaussians_hybrid_subrange failed in [%.1f,%.1f]: %s",
                t_min,
                t_max,
                e,
            )
            return np.empty(0, float)
        raise


# =============================================================================
# 5.  2D sub-range wrapper
# =============================================================================
def fit_gaussians_2d_subrange(
    t: np.ndarray,
    v2: np.ndarray,
    t_min: float,
    t_max: float,
    *,
    n_waves: int,
    sigma_max: float = 80.0,
    amp_min: float = 0.0,
    angle_lambda: float,
    amplitude_lambda: float,
    **kwargs: float,
) -> np.ndarray:
    """Sub-range wrapper for the 2D fitter.

    Clips *t* and *v2* to the ``[t_min, t_max]`` window, checks the
    amplitude threshold, and delegates to :func:`fit_gaussians_2d`.

    Args:
        t: 1D time array of shape ``(N,)``.
        v2: 2D vector trace of shape ``(2, N)``.
        t_min: Lower time bound of the sub-range.
        t_max: Upper time bound of the sub-range.
        n_waves: Number of Gaussian components to fit.
        sigma_max: Upper bound on Gaussian width.
        amp_min: Minimum peak amplitude to proceed with fitting.
        angle_lambda: Weight of the angle-deviation penalty.
        amplitude_lambda: Weight of the amplitude-overshoot penalty.
        **kwargs: Forwarded to :func:`fit_gaussians_2d`.

    Returns:
        Flat parameter array ``[t0, sigma, A, alpha] * n_waves``.
        Returns ``np.empty(0, float)`` on failure or skip.
    """
    if t_min >= t_max:
        logger.warning("2D skip: t_min>=t_max (%.1f >= %.1f)", t_min, t_max)
        return np.empty(0, float)

    idx = (t >= t_min) & (t <= t_max)
    if not np.any(idx):
        logger.debug("2D skip: no points in [%.1f,%.1f]", t_min, t_max)
        return np.empty(0, float)

    t_sub = t[idx]
    v2_sub = v2[:, idx]

    if np.max(np.linalg.norm(v2_sub, axis=0)) < amp_min:
        logger.debug("2D skip: below amp_min=%.3f", amp_min)
        return np.empty(0, float)

    try:
        return fit_gaussians_2d(
            t_sub,
            v2_sub,
            n_waves,
            sigma_max=sigma_max,
            angle_lambda=angle_lambda,
            amplitude_lambda=amplitude_lambda,
            **kwargs,
        )
    except ValueError as e:
        msg = str(e)
        if "Initial guess is outside" in msg or "lower bound must be strictly less" in msg:
            logger.warning(
                "fit_gaussians_2d_subrange failed in [%.1f,%.1f]: %s",
                t_min,
                t_max,
                e,
            )
            return np.empty(0, float)
        raise


# =============================================================================
# 6.  Residual helper -- subtract fitted Gaussians from a 2D trace
# =============================================================================
def subtract_fitted_2d(
    t: np.ndarray, v2: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Subtract fitted Gaussian components from a 2D vector trace.

    Args:
        t: 1D time array of shape ``(N,)``.
        v2: 2D vector trace of shape ``(2, N)``.
        params: Flat parameter vector from a prior fit.

    Returns:
        Residual 2D array of shape ``(2, N)``.  If *params* is empty
        the original trace is returned (copied).
    """
    return v2 - sum_gaussians_2d(t, params) if params.size else v2.copy()
