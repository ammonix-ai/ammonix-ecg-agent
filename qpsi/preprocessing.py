"""
ECG preprocessing -- filtering, baseline correction, R-peak detection, noise detection.

Source: Cell 2 of q_psi_ai_for_ecg_Feb14_Adele.ipynb
Depends on: qpsi.constants (FS_HZ, MAX_NOISE_LEVEL, BASELINE_FLUCT_MAX)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt

from qpsi.constants import BASELINE_FLUCT_MAX, FS_HZ, MAX_NOISE_LEVEL

logger = logging.getLogger("qpsi")

# --------------------------------------------------------------------------- #
# Butterworth helper                                                          #
# --------------------------------------------------------------------------- #
# The notebook defines _butter_edge twice; V2 (line 209) shadows V1 at
# runtime.  This canonical version uses the modern scipy ``fs=`` parameter
# and accepts ``order`` as a parameter (default 4).


def _butter_edge(
    sig: np.ndarray,
    freq_hz: float,
    btype: str,
    fs: int,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter (forward-backward via SOS).

    Parameters
    ----------
    sig : np.ndarray
        1-D signal to filter.
    freq_hz : float
        Corner frequency in Hz.
    btype : str
        ``"lowpass"``, ``"highpass"``, ``"bandpass"``, or ``"bandstop"``.
    fs : int
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal (same shape as *sig*).
    """
    sos = butter(order, freq_hz, btype=btype, fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


# Internal helper: return the SOS coefficients without filtering.
# Used where the caller needs the SOS array directly (e.g. detect_r_peaks).

def _butter_sos(
    freq_hz: float,
    btype: str,
    fs: int,
    order: int = 4,
) -> np.ndarray:
    """Return Butterworth SOS coefficients (no filtering).

    Parameters
    ----------
    freq_hz : float
        Corner frequency in Hz.
    btype : str
        Filter type (``"lowpass"``, ``"highpass"``, etc.).
    fs : int
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Second-order sections array suitable for :func:`sosfiltfilt`.
    """
    return butter(order, freq_hz, btype=btype, fs=fs, output="sos")


# --------------------------------------------------------------------------- #
# Noise detection                                                             #
# --------------------------------------------------------------------------- #


def is_lead_noisy(
    sig: np.ndarray,
    fs: int,
    hf_band: Tuple[float, float] = (48.0, 200.0),
) -> bool:
    """Detect whether a single-lead signal is noisy.

    Uses a simple HF-power / total-power ratio.  If the fraction exceeds
    :data:`~qpsi.constants.MAX_NOISE_LEVEL` the lead is considered noisy.

    Parameters
    ----------
    sig : np.ndarray
        1-D signal (one lead).
    fs : int
        Sampling frequency in Hz.
    hf_band : tuple of float, optional
        (low, high) frequency band defining "high frequency" (default
        ``(48, 200)``).

    Returns
    -------
    bool
        ``True`` if the lead is noisy.
    """
    sos = butter(4, hf_band, btype="band", fs=fs, output="sos")
    hf = sosfiltfilt(sos, sig)
    p_total = np.mean(sig ** 2) + 1e-12
    p_hf = np.mean(hf ** 2)
    return bool((p_hf / p_total) > MAX_NOISE_LEVEL)


# --------------------------------------------------------------------------- #
# Main preprocessing pipeline                                                 #
# --------------------------------------------------------------------------- #


def preprocess_ecg(
    ecg_signal: np.ndarray,
    *,
    fs: int = FS_HZ,
    window_ms: int = 600,
) -> Tuple[np.ndarray, bool]:
    """Filter and baseline-correct a multi-lead ECG.

    Pipeline:
        1. 4-pole 0.1 Hz high-pass
        2. Per-lead noise check
        3. 2-pole low-pass (200 Hz if clean, 35 Hz if noisy)
        4. Running median baseline removal

    Parameters
    ----------
    ecg_signal : np.ndarray
        Shape ``(n_leads, n_samples)``.
    fs : int, optional
        Sampling frequency (default :data:`~qpsi.constants.FS_HZ`).
    window_ms : int, optional
        Median filter window in milliseconds (default 600).

    Returns
    -------
    tuple of (np.ndarray, bool)
        ``(ecg_filtered, not_noisy)`` where *ecg_filtered* has the same
        shape as the input and *not_noisy* is ``True`` when **all** leads
        passed the noise check.
    """
    sos_hp = _butter_sos(0.1, "highpass", fs=fs, order=4)
    x = sosfiltfilt(sos_hp, ecg_signal, axis=1)

    # Per-lead noise check
    good_leads = [not is_lead_noisy(x[k], fs) for k in range(x.shape[0])]
    not_noisy = all(good_leads)

    if not_noisy:
        sos_lp = _butter_sos(200.0, "lowpass", fs=fs, order=2)
    else:
        sos_lp = _butter_sos(35.0, "lowpass", fs=fs, order=2)
        logger.warning("Noisy signal detected -- low-pass clipped to 35 Hz")

    x = sosfiltfilt(sos_lp, x, axis=1)

    # Running median baseline removal — per-row 1-D median filter is
    # ~60× faster than scipy.signal.medfilt(x, (1, win)) or the 2-D form
    # of ndimage.median_filter, while remaining bit-identical (verified
    # against medfilt's zero-pad semantics via mode='constant', cval=0.0).
    win = int(round(window_ms * fs / 1_000))
    win |= 1  # force odd length
    baseline = np.stack([
        median_filter(row, size=win, mode="constant", cval=0.0)
        for row in x
    ])

    logger.debug(
        "preprocess_ecg: median kernel = %d samples  |  not_noisy=%s",
        win,
        not_noisy,
    )
    return x - baseline, not_noisy


# --------------------------------------------------------------------------- #
# Two-anchor baseline subtraction (core)                                      #
# --------------------------------------------------------------------------- #


def _baseline_twoanchors_core(
    ecg_filt: np.ndarray,
    r_peaks: np.ndarray,
    *,
    fs: int = FS_HZ,
    preQ_offset: int = -40,
    postT_offset: int = 50,
    interp_kind: str = "pchip",
    num_bins: int = 50,
    skip_preQ_ms: int = 70,
    skip_postT_ms: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-anchor baseline subtraction including first and last R-peak.

    For each R-peak, two anchor points are computed:
        * **pre-Q** anchor: a short window before QRS onset
        * **mid-RR** anchor: halfway to the next (or previous) R-peak

    The baseline is estimated per lead by interpolating through these
    anchors with PCHIP (or another 1-D interpolant).

    Parameters
    ----------
    ecg_filt : np.ndarray
        Shape ``(n_leads, n_samples)`` -- filtered ECG.
    r_peaks : np.ndarray
        R-peak sample indices.
    fs : int, optional
        Sampling frequency (default :data:`~qpsi.constants.FS_HZ`).
    preQ_offset : int, optional
        Offset in ms before R for the pre-Q anchor (default -40).
    postT_offset : int, optional
        Offset in ms after mid-RR for the post-T anchor (default 50).
    interp_kind : str, optional
        Interpolation method -- ``"pchip"`` or a :func:`scipy.interpolate.interp1d`
        kind (default ``"pchip"``).
    num_bins : int, optional
        Number of histogram bins for the modal-value estimator (default 50).
    skip_preQ_ms : int, optional
        Half-window in ms around the pre-Q anchor for mode estimation
        (default 70).
    skip_postT_ms : int, optional
        Half-window in ms around the post-T anchor for mode estimation
        (default 100).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(ecg_bs, bs_curve)`` -- baseline-subtracted ECG and the
        estimated baseline curve, both shape ``(n_leads, n_samples)``.
    """
    n_leads, n_samples = ecg_filt.shape
    bs_curve = np.zeros_like(ecg_filt)
    ecg_bs = np.zeros_like(ecg_filt)

    # Helper: find the modal value in a window around *centre*
    def _mode_around(vec: np.ndarray, centre: int, win_ms: int) -> Optional[float]:
        win = int(round(win_ms * fs / 1_000))
        left, right = max(0, centre - win), min(n_samples, centre + win + 1)
        chunk = vec[left:right]
        if chunk.size < 7:
            return None
        lo, hi = float(chunk.min()), float(chunk.max())
        if lo == hi:
            return lo
        hist, edges = np.histogram(chunk, bins=num_bins, range=(lo, hi))
        idx = int(np.argmax(hist))
        return 0.5 * (edges[idx] + edges[idx + 1])

    # Gather anchors per lead
    anchors_x: List[List[int]] = [[] for _ in range(n_leads)]
    anchors_y: List[List[float]] = [[] for _ in range(n_leads)]

    n_peaks = len(r_peaks)
    for i, rp in enumerate(r_peaks):
        r_curr = int(rp)
        # pre-Q anchor (just before QRS)
        a_preQ: Optional[int] = r_curr + int(preQ_offset * fs / 1_000)

        # choose neighbor for mid-RR anchor: next if exists, else previous
        if i < n_peaks - 1:
            r_neigh: Optional[int] = int(r_peaks[i + 1])
        elif i > 0:
            r_neigh = int(r_peaks[i - 1])
        else:
            # single R-peak: skip mid anchor
            r_neigh = None

        if r_neigh is not None:
            mid_pt = r_curr + (r_neigh - r_curr) // 2
            a_mid: Optional[int] = mid_pt + int(postT_offset * fs / 1_000)
        else:
            a_mid = None

        # validate anchor indices
        if not (0 <= a_preQ < n_samples):
            a_preQ = None
        if a_mid is not None and not (0 <= a_mid < n_samples):
            a_mid = None

        # collect per-lead modes
        for ld in range(n_leads):
            if a_preQ is not None:
                m1 = _mode_around(ecg_filt[ld], a_preQ, skip_preQ_ms)
                if m1 is not None:
                    anchors_x[ld].append(a_preQ)
                    anchors_y[ld].append(m1)
            if a_mid is not None:
                m2 = _mode_around(ecg_filt[ld], a_mid, skip_postT_ms)
                if m2 is not None:
                    anchors_x[ld].append(a_mid)
                    anchors_y[ld].append(m2)

    # Interpolate baseline per lead
    x_all = np.arange(n_samples, dtype=float)
    for ld in range(n_leads):
        xs = np.asarray(anchors_x[ld], dtype=float)
        ys = np.asarray(anchors_y[ld], dtype=float)

        # explicit boundary anchors for stable extrapolation
        xs = np.concatenate(([0.0], xs, [n_samples - 1.0]))
        ys = np.concatenate(([0.0], ys, [0.0]))

        # sort and remove duplicates (ensures strictly increasing x)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        xs_u, idx_u = np.unique(xs, return_index=True)
        xs, ys = xs_u, ys[idx_u]

        # build interpolator
        if interp_kind.lower() == "pchip":
            fn = PchipInterpolator(xs, ys, extrapolate=True)
        else:
            fn = interp1d(xs, ys, kind=interp_kind, fill_value="extrapolate")

        base = fn(x_all)
        bs_curve[ld] = base
        ecg_bs[ld] = ecg_filt[ld] - base

    return ecg_bs, bs_curve


# --------------------------------------------------------------------------- #
# Two-anchor baseline subtraction (wrapper with auto-skip)                    #
# --------------------------------------------------------------------------- #


def baseline_subtract_twoanchors(
    ecg_filt: np.ndarray,
    r_peaks: np.ndarray,
    *,
    fs: int = FS_HZ,
    preQ_offset: int = -40,
    postT_offset: int = 50,
    interp_kind: str = "pchip",
    num_bins: int = 50,
    skip_preQ_ms: int = 70,
    skip_postT_ms: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-anchor baseline subtraction with auto-skip.

    If the resulting baseline exceeds
    :data:`~qpsi.constants.BASELINE_FLUCT_MAX` (95th percentile ratio)
    of the trace amplitude on *any* lead, the input is returned unchanged.

    Parameters
    ----------
    ecg_filt : np.ndarray
        Shape ``(n_leads, n_samples)`` -- filtered ECG.
    r_peaks : np.ndarray
        R-peak sample indices.
    fs : int, optional
        Sampling frequency (default :data:`~qpsi.constants.FS_HZ`).
    preQ_offset : int, optional
        Offset in ms before R for the pre-Q anchor (default -40).
    postT_offset : int, optional
        Offset in ms after mid-RR for the post-T anchor (default 50).
    interp_kind : str, optional
        Interpolation method (default ``"pchip"``).
    num_bins : int, optional
        Histogram bins for modal-value estimator (default 50).
    skip_preQ_ms : int, optional
        Half-window in ms for pre-Q mode estimation (default 70).
    skip_postT_ms : int, optional
        Half-window in ms for post-T mode estimation (default 100).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(ecg_bs, bs_curve)`` -- baseline-subtracted ECG and baseline
        curve (or the original signal and a zero curve if the baseline
        was too large).
    """
    ecg_bs, bs_curve = _baseline_twoanchors_core(
        ecg_filt,
        r_peaks,
        fs=fs,
        preQ_offset=preQ_offset,
        postT_offset=postT_offset,
        interp_kind=interp_kind,
        num_bins=num_bins,
        skip_preQ_ms=skip_preQ_ms,
        skip_postT_ms=skip_postT_ms,
    )

    # Decide whether to keep or discard the subtracted result
    too_big = False
    for ld in range(ecg_filt.shape[0]):
        amp_trace = np.percentile(np.abs(ecg_filt[ld]), 95)
        amp_base = np.percentile(np.abs(bs_curve[ld]), 95)
        if amp_base > BASELINE_FLUCT_MAX * amp_trace:
            too_big = True
            logger.warning(
                "Lead %d: baseline too large (%.3f mV vs %.3f mV) -- "
                "keeping un-subtracted signal.",
                ld,
                amp_base,
                amp_trace,
            )
            break

    if too_big:
        return ecg_filt, np.zeros_like(ecg_filt)
    else:
        return ecg_bs, bs_curve


# --------------------------------------------------------------------------- #
# R-peak detection                                                            #
# --------------------------------------------------------------------------- #


def detect_r_peaks(
    energy_signal: np.ndarray,
    *,
    fs: int,
    top_percent: float = 66.0,
    min_rr_ms: int = 200,
    slope_win_ms: Tuple[int, int] = (50, 120),
    min_spike_width_ms: float = 11.0,
) -> np.ndarray:
    """Energy + slope R-peak detector with pacing-spike rejection.

    Candidates whose spike width is <= *min_spike_width_ms* are discarded
    as pacing artefacts.  Spike width is the time between the strongest
    up-slope and strongest down-slope within a window around each
    candidate.

    Parameters
    ----------
    energy_signal : np.ndarray
        1-D energy envelope signal.
    fs : int
        Sampling frequency in Hz.
    top_percent : float, optional
        Steepness gate -- keep candidates in the top *top_percent* %
        of steepness (default 66.0).
    min_rr_ms : int, optional
        Minimum R-R spacing in ms for final pruning (default 200).
    slope_win_ms : tuple of (int, int), optional
        ``(short, long)`` slope windows in ms (default ``(50, 120)``).
        The short window is used for width estimation.
    min_spike_width_ms : float, optional
        Spikes narrower than this (ms) are rejected as pacing
        (default 11.0).

    Returns
    -------
    np.ndarray
        Sorted array of R-peak sample indices (dtype ``int``).
    """
    # Envelope via low-pass (order=2 matches notebook Cell 2's shadowed
    # _butter_edge which defaults to order=2, NOT the original order=4)
    sos_lp = _butter_sos(200.0, "lowpass", fs=fs, order=2)
    env = sosfiltfilt(sos_lp, energy_signal)

    # Raw amplitude candidates
    amp_thr = 0.3 * np.percentile(env, 90)
    pk_raw, _ = find_peaks(env, height=amp_thr, distance=1)
    if pk_raw.size == 0:
        logger.warning("No amplitude candidates > %.3f", amp_thr)
        return pk_raw.astype(int)

    # Derivative / steepness / width per candidate
    dv = np.empty_like(env)
    dv[1:-1] = (env[2:] - env[:-2]) * (fs / 2)
    dv[[0, -1]] = dv[[1, -2]]

    win_s, win_l = (int(w * fs / 1000) for w in slope_win_ms)
    steep: List[float] = []
    width_ms_list: List[float] = []

    for p in pk_raw:
        sval, w_ms = 0.0, np.inf

        # use the *short* window for width-estimate
        win = win_s
        l = dv[max(0, p - win) : p]
        r = dv[p : min(len(dv), p + win)]

        if l.size and r.size:
            ps_i = int(np.argmax(l))
            ns_i = int(np.argmin(r))
            ps, ns = l[ps_i], r[ns_i]

            if ps > 0 and ns < 0:
                sval = float(np.sqrt(ps * -ns))
                left_idx = max(0, p - win) + ps_i
                right_idx = p + ns_i
                w_ms = (right_idx - left_idx) / fs * 1_000.0

        steep.append(sval)
        width_ms_list.append(w_ms)

    steep_arr = np.asarray(steep)
    width_ms_arr = np.asarray(width_ms_list)

    # Pacing-spike suppression (width gate)
    pacing_mask = width_ms_arr <= min_spike_width_ms
    pk = pk_raw[~pacing_mask]
    width_ms_arr = width_ms_arr[~pacing_mask]
    steep_arr = steep_arr[~pacing_mask]

    if pk.size == 0:
        logger.warning(
            "All %d candidates look like pacing spikes; returning empty array.",
            pk_raw.size,
        )
        return np.empty(0, int)

    # Steepness value gate (top_percent)
    keep_thr = (1.0 - top_percent / 100.0) * steep_arr.max()
    pk = pk[steep_arr >= keep_thr]

    # Right-to-left distance pruning
    dist_samples = int(min_rr_ms * fs / 1000)
    pk_sorted = np.sort(pk)
    accepted: List[int] = []
    last_keep: Optional[int] = None
    for p in pk_sorted[::-1]:  # walk backwards
        if last_keep is None or (last_keep - p) >= dist_samples:
            accepted.append(int(p))
            last_keep = int(p)
    fin_pk = np.sort(np.array(accepted))

    # Guarantee >= 3 peaks
    if fin_pk.size < 3:
        top3 = np.argsort(env[pk_raw])[-3:]
        fin_pk = np.sort(pk_raw[top3])

    logger.debug(
        "detect_r_peaks: kept %d (width<=%.1f ms tagged pacing, steep gate >=%.3f)",
        fin_pk.size,
        min_spike_width_ms,
        keep_thr,
    )
    return fin_pk.astype(int)


# --------------------------------------------------------------------------- #
# Energy trace computation                                                    #
# --------------------------------------------------------------------------- #

_DEFAULT_LEAD_ORDER: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def energy_trace(
    ecg12: np.ndarray,
    fs: int = FS_HZ,
    lead_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute the root-sum-of-squares energy envelope across leads.

    Parameters
    ----------
    ecg12 : np.ndarray
        Shape ``(n_leads, n_samples)`` (or 1-D for a single lead).
    fs : int, optional
        Sampling frequency in Hz (default :data:`~qpsi.constants.FS_HZ`).
    lead_names : list of str or None, optional
        Lead names corresponding to rows of *ecg12*.  Used only for
        validation / logging.  Defaults to the standard 12-lead order.

    Returns
    -------
    np.ndarray
        1-D energy envelope of length *n_samples*.
    """
    x = np.asarray(ecg12, dtype=float)
    if x.ndim == 1:
        x = x[None, :]

    if lead_names is None:
        lead_names = _DEFAULT_LEAD_ORDER[: x.shape[0]]

    if len(lead_names) != x.shape[0]:
        lead_names = _DEFAULT_LEAD_ORDER[: x.shape[0]]

    energy = np.sqrt(np.sum(x ** 2, axis=0))
    return energy


# --------------------------------------------------------------------------- #
# Angle utilities                                                             #
# --------------------------------------------------------------------------- #


def convert_angles_to_degrees(lumps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of *lumps* with angles converted to degrees.

    Each lump dict may store ``"angle"`` as a scalar or a length-1
    NumPy array.  A new key ``"angle_degrees"`` is added with the
    value in degrees.

    Parameters
    ----------
    lumps : list of dict
        Lump descriptors (wave classification output).

    Returns
    -------
    list of dict
        Independent copies with ``"angle_degrees"`` added.

    Raises
    ------
    ValueError
        If any ``"angle"`` value is an array with more than one element.
    """
    out: List[Dict[str, Any]] = []
    for w in lumps:
        d = dict(w)  # independent copy

        ang = w.get("angle")
        if isinstance(ang, np.ndarray):
            if ang.size != 1:
                raise ValueError(
                    f"convert_angles_to_degrees: expected scalar or "
                    f"length-1 array, got shape {ang.shape}"
                )
            ang = float(np.squeeze(ang))
        else:
            ang = float(ang)

        d["angle_degrees"] = round(math.degrees(ang), 2)
        out.append(d)
    return out


def angle_diff_rad(a: float, b: float) -> float:
    """Return the absolute smallest angular distance between *a* and *b*.

    Both inputs are in **radians**.  The result is in ``[0, pi]``.

    Parameters
    ----------
    a : float
        First angle (radians).
    b : float
        Second angle (radians).

    Returns
    -------
    float
        Absolute angular distance in radians.
    """
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


# --------------------------------------------------------------------------- #
# 2-D vector low-pass filter                                                  #
# --------------------------------------------------------------------------- #


def lowpass_vec2d(
    v2: np.ndarray,
    fs: float,
    cutoff_hz: float = 25.0,
    order: int = 4,
) -> np.ndarray:
    """Low-pass a 2-D vector trace so amplitude and direction are smoothed.

    Applies a Butterworth low-pass to the complex representation
    ``x + j*y`` for consistent smoothing of both magnitude and angle.

    Parameters
    ----------
    v2 : np.ndarray
        Shape ``(2, N)`` -- row 0 is x, row 1 is y.
    fs : float
        Sampling frequency in Hz.
    cutoff_hz : float, optional
        -3 dB corner frequency (default 25.0).
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Shape ``(2, N)`` -- the low-passed vector trace.

    Raises
    ------
    ValueError
        If *v2* does not have shape ``(2, N)``.
    """
    if v2.shape[0] != 2:
        raise ValueError("v2 must be 2xN")

    # complex representation  x + j*y  (shape N,)
    z = v2[0] + 1j * v2[1]

    # zero-phase IIR low-pass (transfer function form)
    b, a = butter(order, cutoff_hz / (0.5 * fs), btype="low")
    z_lp = filtfilt(b, a, z)

    return np.vstack((z_lp.real, z_lp.imag))
