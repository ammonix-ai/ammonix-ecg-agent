"""
Direct waveform-shape measurements — the no-Gaussian alternative.

For each (lead, wave-region, beat) triple, computes a vector of
continuous, baseline-aware primitives without any Gaussian fit.

Wave regions (three only, no sub-component naming):
    P-window:    [-0.5*RR + offset, -PAD_HEAD_R_MS]
    QRS-window:  [-OFFSET_MS, +QRS_FRAC_MS]
    T-window:    [+QRS_FRAC_MS, +0.5*RR]

Per (lead, region, beat) we emit:
    pos_peak_amp_mv, pos_peak_time_ms       — signed positive extremum
    neg_peak_amp_mv, neg_peak_time_ms       — signed negative extremum
    peak_to_peak_mv                         — baseline-invariant amplitude
    centroid_ms                             — first moment of |x|
    duration_ms_rms                         — 4× second-moment width
    duration_ms_fwhm                        — half-amplitude width
    max_abs_derivative_mv_per_ms            — baseline-invariant slope
    rms_amplitude_mv                        — RMS of detrended trace
    snr_db                                  — peak vs edge noise

Per recording, after aggregation across all beats:
    median, MAD, IQR for each scalar — plus alternation_index.

No fitting, no model selection, no AICc bifurcations.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.signal import savgol_filter

# Optional Rust fast path for the post-smoothing extrema loop. Bound
# at module load so the per-call hot path is a cheap attribute check.
# Rust kernel returns prominence-filtered extrema in time-insertion
# order; this module's ``find_extrema`` does the final descending-
# prominence sort via numpy.argsort to preserve the existing
# Apr28-GT tie-break contract (see ``qpsi_native::extrema`` module
# doc for the brief).
try:
    import qpsi_native as _qpsi_native
    _RUST_FIND_EXTREMA = getattr(
        _qpsi_native, "find_extrema_post_smooth_native", None
    )
    _RUST_MEASURE_WINDOW_CORE = getattr(
        _qpsi_native, "measure_window_core_native", None
    )
    _RUST_MEASURE_WINDOW_BATCH = getattr(
        _qpsi_native, "measure_window_batch_native", None
    )
    _RUST_DERIVED_MORPHOLOGY_BATCH = getattr(
        _qpsi_native, "derived_morphology_batch_native", None
    )
except ImportError:  # pragma: no cover - qpsi_native ships in prod
    _qpsi_native = None
    _RUST_FIND_EXTREMA = None
    _RUST_MEASURE_WINDOW_CORE = None
    _RUST_MEASURE_WINDOW_BATCH = None
    _RUST_DERIVED_MORPHOLOGY_BATCH = None


# ---------------------------------------------------------------------------
# Extrema-list primitives — captures saddles, notches, secondary peaks
# ---------------------------------------------------------------------------

K_EXTREMA = 4   # number of extrema to emit per (lead, region) — fixed-arity


def find_extrema(x: np.ndarray, t_ms: np.ndarray,
                 *,
                 smooth_ms: float = 10.0,
                 prominence_frac: float = 0.05,
                 ) -> list[dict[str, float]]:
    """Find local extrema (zero-crossings of the first derivative).

    Returns a list of dicts ordered by descending **prominence**, where
    prominence = depth of the extremum relative to the lower of the two
    flanking extrema (or window edges).

    Each dict carries:
        time_ms   : sub-sample time of the extremum
        amp_mv    : sub-sample amplitude (in the detrended signal)
        kind      : +1 for local max, -1 for local min
        prominence: |amp − max(higher_flanking_amp, edge_amp)|

    Window-edge artefacts are intrinsically rejected: a monotonic edge has
    no zero-crossing of dx/dt inside the window, so it never registers as
    an extremum. Saddles, notches, and secondary peaks ARE captured because
    they DO produce derivative zero-crossings.
    """
    n = len(x)
    if n < 7:
        return []

    # Local detrend (same as measure_window)
    edge = max(3, int(0.10 * n))
    edge_mean = np.concatenate([x[:edge], x[-edge:]]).mean()
    x_d = x - edge_mean

    # Light smoothing — Savitzky-Golay window in samples
    fs_implied = 1000.0 / max(t_ms[1] - t_ms[0], 1e-6)
    win = max(5, int(round(smooth_ms * fs_implied / 1000.0)) | 1)  # odd
    if win >= n:
        win = max(5, (n // 2) * 2 - 1)
    poly = min(2, win - 1)
    try:
        x_s = savgol_filter(x_d, win, poly)
    except Exception:
        x_s = x_d.copy()

    # ---- Optional Rust fast path ----
    # qpsi_native covers steps 3–7 of this function (gradient +
    # sign-scan + parabolic refine + prominence + threshold filter)
    # in compiled code. Returns extrema in INSERTION ORDER (time-
    # ascending); we apply ``np.argsort(-prom)`` here so the final
    # descending-prominence ordering matches the canonical reference
    # exactly — including the numpy-quicksort tie behaviour Apr28
    # GT was generated against.
    if _RUST_FIND_EXTREMA is not None:
        arr = _RUST_FIND_EXTREMA(
            np.ascontiguousarray(x_d, dtype=np.float64),
            np.ascontiguousarray(x_s, dtype=np.float64),
            np.ascontiguousarray(t_ms, dtype=np.float64),
            float(prominence_frac),
        )
        if arr.shape[0] == 0:
            return []
        order = np.argsort(-arr[:, 3])
        return [
            {"time_ms": float(arr[i, 0]),
             "amp_mv": float(arr[i, 1]),
             "kind": int(arr[i, 2]),
             "prominence": float(arr[i, 3])}
            for i in order
        ]

    # ---- Python fallback (steps 3–8) ----
    # First derivative via central differences (unitless w.r.t. x; we only
    # care about sign changes here)
    dxdt = np.gradient(x_s, t_ms)

    # Sign-change indices in the derivative
    signs = np.sign(dxdt)
    signs[signs == 0] = 1  # treat exact zeros as positive
    sign_changes = np.where(np.diff(signs) != 0)[0]
    # sign_changes[i] is the index just before the change; the extremum
    # sits between samples sign_changes[i] and sign_changes[i]+1.

    if len(sign_changes) == 0:
        return []

    # Sub-sample refinement — parabolic fit through 3 samples around each
    # extremum
    extrema_raw: list[tuple[float, float, int]] = []
    for ix in sign_changes:
        i = ix + 1   # the actual local extremum sample
        if i < 1 or i > n - 2:
            continue
        a, b, c = x_s[i - 1], x_s[i], x_s[i + 1]
        denom = (a - 2 * b + c)
        if abs(denom) > 1e-12:
            offset = 0.5 * (a - c) / denom
            t_e = float(t_ms[i] + offset * (t_ms[1] - t_ms[0]))
            amp_e = float(b - 0.25 * (a - c) * offset)
        else:
            t_e = float(t_ms[i]); amp_e = float(b)
        kind = +1 if (signs[ix] > 0 and signs[ix + 1] <= 0) else -1
        extrema_raw.append((t_e, amp_e, kind))

    if not extrema_raw:
        return []

    # Compute prominence for each extremum (depth relative to flanking
    # extrema or window edges). For a max: prominence = amp - max(left_min,
    # right_min); for a min: |amp - min(left_max, right_max)|. If there's
    # no flanking extremum on a side, use the window edge.
    amps = np.array([e[1] for e in extrema_raw])
    times = np.array([e[0] for e in extrema_raw])
    kinds = np.array([e[2] for e in extrema_raw])

    proms = np.zeros(len(extrema_raw))
    for i, (te, ae, ke) in enumerate(extrema_raw):
        # Look at flanking extrema of opposite kind
        left_flank = -np.inf if ke > 0 else +np.inf
        for j in range(i - 1, -1, -1):
            if kinds[j] == -ke:
                left_flank = amps[j]; break
        else:
            left_flank = float(x_d[0])
        right_flank = -np.inf if ke > 0 else +np.inf
        for j in range(i + 1, len(extrema_raw)):
            if kinds[j] == -ke:
                right_flank = amps[j]; break
        else:
            right_flank = float(x_d[-1])
        if ke > 0:
            ref = max(left_flank, right_flank)
            proms[i] = max(ae - ref, 0.0)
        else:
            ref = min(left_flank, right_flank)
            proms[i] = max(ref - ae, 0.0)

    # Prominence threshold — fraction of window peak-to-peak
    p2p = float(x_d.max() - x_d.min())
    thr = max(prominence_frac * p2p, 1e-6)
    keep = proms >= thr
    extrema_raw = [extrema_raw[i] for i in range(len(extrema_raw)) if keep[i]]
    proms = proms[keep]

    if not extrema_raw:
        return []

    # Sort by descending prominence
    order = np.argsort(-proms)
    return [
        {"time_ms": extrema_raw[i][0],
         "amp_mv": extrema_raw[i][1],
         "kind": extrema_raw[i][2],
         "prominence": float(proms[i])}
        for i in order
    ]


def derived_morphology(extrema: list[dict[str, float]],
                       window_size_ms: float) -> dict[str, float]:
    """Derived shape descriptors from the extrema list. All baseline-invariant
    or relative; all continuous; no thresholds applied here."""
    if not extrema:
        return {
            "n_significant_extrema": 0,
            "slope_reversal_count": 0,
            "biphasic_ratio": 0.0,
            "notch_depth_mv": 0.0,
            "notch_position_frac": 0.0,
            "saddle_depth_relative": 0.0,
            "monotonic_fraction": 1.0,
        }
    # Sort chronologically for shape-tracing
    chrono = sorted(extrema, key=lambda e: e["time_ms"])
    amps = np.array([e["amp_mv"] for e in chrono])
    kinds = np.array([e["kind"] for e in chrono])

    n_extr = len(chrono)
    n_reversals = max(n_extr - 1, 0)

    # Biphasic ratio: smaller / larger of the dominant max and dominant min
    dom_max = max((e["amp_mv"] for e in extrema if e["kind"] > 0), default=0.0)
    dom_min = min((e["amp_mv"] for e in extrema if e["kind"] < 0), default=0.0)
    if abs(dom_max) > 1e-9 and abs(dom_min) > 1e-9:
        biphasic_ratio = min(abs(dom_max), abs(dom_min)) / max(abs(dom_max), abs(dom_min))
    else:
        biphasic_ratio = 0.0

    # Notch: two same-sign maxima with a same-sign min between them (or the
    # opposite for negative-going waves). Look chronologically.
    notch_depth = 0.0
    notch_pos = 0.0
    # Try positive notch: max-min-max with all three positive amplitude (or
    # at least the two peaks above zero and the trough between them shallower)
    for i in range(1, n_extr - 1):
        if kinds[i - 1] > 0 and kinds[i] < 0 and kinds[i + 1] > 0:
            peak1 = amps[i - 1]; valley = amps[i]; peak2 = amps[i + 1]
            depth = min(peak1, peak2) - valley
            if depth > notch_depth:
                notch_depth = depth
                notch_pos = ((chrono[i]["time_ms"] - chrono[i - 1]["time_ms"]) /
                             max(chrono[i + 1]["time_ms"] - chrono[i - 1]["time_ms"], 1e-6))
        elif kinds[i - 1] < 0 and kinds[i] > 0 and kinds[i + 1] < 0:
            # Inverted notch (in negative-going wave)
            v1 = amps[i - 1]; bump = amps[i]; v2 = amps[i + 1]
            depth = bump - max(v1, v2)
            if depth > notch_depth:
                notch_depth = depth
                notch_pos = ((chrono[i]["time_ms"] - chrono[i - 1]["time_ms"]) /
                             max(chrono[i + 1]["time_ms"] - chrono[i - 1]["time_ms"], 1e-6))

    # Saddle: a local extremum on the *rising* side of the dominant peak
    # whose amplitude is between the window-edge baseline and the dominant
    # peak (i.e. a shoulder, not a counter-polarity bump). Captures Brugada
    # Type 2 saddleback and similar.
    if n_extr >= 2:
        dom_idx = int(np.argmax(np.abs(amps)))
        dom_amp = amps[dom_idx]; dom_t = chrono[dom_idx]["time_ms"]
        saddle_rel = 0.0
        for i, e in enumerate(chrono):
            if i == dom_idx:
                continue
            if e["time_ms"] >= dom_t:
                continue                     # only earlier shoulders
            if np.sign(e["amp_mv"]) != np.sign(dom_amp):
                continue                     # same-sign only
            if abs(e["amp_mv"]) >= abs(dom_amp):
                continue                     # not the dominant
            rel = abs(e["amp_mv"]) / max(abs(dom_amp), 1e-6)
            if rel > saddle_rel:
                saddle_rel = rel
    else:
        saddle_rel = 0.0

    # Monotonic fraction — coarse: fraction of inter-extrema time intervals
    # that span the window (i.e. if very few extrema, signal is mostly
    # monotonic)
    if n_extr <= 1:
        monotonic_fraction = 1.0
    else:
        total_span = chrono[-1]["time_ms"] - chrono[0]["time_ms"]
        monotonic_fraction = max(0.0, 1.0 - total_span / max(window_size_ms, 1e-6))

    return {
        "n_significant_extrema": n_extr,
        "slope_reversal_count": n_reversals,
        "biphasic_ratio": float(biphasic_ratio),
        "notch_depth_mv": float(notch_depth),
        "notch_position_frac": float(notch_pos),
        "saddle_depth_relative": float(saddle_rel),
        "monotonic_fraction": float(monotonic_fraction),
    }


# ---------------------------------------------------------------------------
# Single-window measurement
# ---------------------------------------------------------------------------

def measure_window(x: np.ndarray, t_ms: np.ndarray) -> dict[str, float]:
    """Direct measurement of one (lead, region, beat) signal slice.

    x      : 1-D waveform inside the wave window
    t_ms   : signed time axis (0 ms = R-peak)
    """
    n = len(x)
    if n < 5:
        return _empty_window()

    # Local detrend — subtract mean of window edges
    edge = max(3, int(0.10 * n))
    edge_mean = np.concatenate([x[:edge], x[-edge:]]).mean()
    x_d = x - edge_mean

    # ---- Optional Rust fast path (the 11 base scalars + extrema list) ----
    # qpsi_native.measure_window_core_native covers the 11 base scalars
    # (pos/neg peaks, p2p, centroid, durations, max-abs-deriv, rms, snr)
    # AND the prominence-filtered extrema list in compiled code. Savgol
    # stays on the scipy side (same constraint as Item #1); the kernel
    # returns extrema unsorted by prominence — we apply np.argsort here
    # to preserve the numpy-quicksort tie behaviour Apr28 GT uses.
    if _RUST_MEASURE_WINDOW_CORE is not None:
        # Pre-compute the smoothed signal once for the extrema sub-step;
        # we mirror the same win/poly logic find_extrema uses internally
        # so the Rust kernel sees the exact same x_s scipy produces.
        fs_implied = 1000.0 / max(t_ms[1] - t_ms[0], 1e-6)
        win = max(5, int(round(8.0 * fs_implied / 1000.0)) | 1)
        if win >= n:
            win = max(5, (n // 2) * 2 - 1)
        poly = min(2, win - 1)
        try:
            x_s = savgol_filter(x_d, win, poly)
        except Exception:
            x_s = x_d.copy()
        scalars, ext_arr = _RUST_MEASURE_WINDOW_CORE(
            np.ascontiguousarray(x_d, dtype=np.float64),
            np.ascontiguousarray(x_s, dtype=np.float64),
            np.ascontiguousarray(t_ms, dtype=np.float64),
            0.03,
        )
        if ext_arr.shape[0] == 0:
            extrema = []
        else:
            order = np.argsort(-ext_arr[:, 3])
            extrema = [
                {"time_ms": float(ext_arr[i, 0]),
                 "amp_mv": float(ext_arr[i, 1]),
                 "kind": int(ext_arr[i, 2]),
                 "prominence": float(ext_arr[i, 3])}
                for i in order
            ]
        morph = derived_morphology(
            extrema, window_size_ms=float(t_ms[-1] - t_ms[0])
        )
        out = {
            "pos_peak_amp_mv": float(scalars[1]),
            "pos_peak_time_ms": float(scalars[0]),
            "neg_peak_amp_mv": float(scalars[3]),
            "neg_peak_time_ms": float(scalars[2]),
            "peak_to_peak_mv": float(scalars[4]),
            "centroid_ms": float(scalars[5]),
            "duration_ms_rms": float(scalars[6]),
            "duration_ms_fwhm": float(scalars[7]),
            "max_abs_deriv_mv_per_ms": float(scalars[8]),
            "rms_amplitude_mv": float(scalars[9]),
            "snr_db": float(scalars[10]),
        }
        for k in range(K_EXTREMA):
            e = extrema[k] if k < len(extrema) else None
            out[f"extr_{k+1}_t_ms"] = float(e["time_ms"]) if e else 0.0
            out[f"extr_{k+1}_amp_mv"] = float(e["amp_mv"]) if e else 0.0
            out[f"extr_{k+1}_kind"] = float(e["kind"]) if e else 0.0
            out[f"extr_{k+1}_prom"] = float(e["prominence"]) if e else 0.0
        out.update(morph)
        out["_extrema_list"] = extrema
        return out

    # ---- Python fallback ----
    # Signed extrema
    i_pos = int(np.argmax(x_d))
    i_neg = int(np.argmin(x_d))

    # Sub-sample peak refinement via parabolic interpolation (3-point)
    def _refine(i: int) -> tuple[float, float]:
        if 0 < i < n - 1:
            a, b, c = x_d[i - 1], x_d[i], x_d[i + 1]
            denom = (a - 2 * b + c)
            if abs(denom) > 1e-12:
                offset = 0.5 * (a - c) / denom
                t_peak = float(t_ms[i] + offset * (t_ms[1] - t_ms[0]))
                amp = float(b - 0.25 * (a - c) * offset)
                return t_peak, amp
        return float(t_ms[i]), float(x_d[i])

    pos_t, pos_a = _refine(i_pos)
    neg_t, neg_a = _refine(i_neg)

    # Peak-to-peak (baseline-invariant)
    p2p = pos_a - neg_a

    # Centroid and second moment of |x|
    abs_x = np.abs(x_d)
    norm = abs_x.sum() + 1e-12
    centroid_ms = float((t_ms * abs_x).sum() / norm)
    sigma_t_ms = float(np.sqrt(((t_ms - centroid_ms) ** 2 * abs_x).sum() / norm))
    duration_ms_rms = 4.0 * sigma_t_ms

    # Half-amplitude width (FWHM) — relative to dominant peak
    dom_a = pos_a if abs(pos_a) >= abs(neg_a) else neg_a
    if abs(dom_a) > 1e-9:
        thr = 0.5 * abs(dom_a)
        above = np.abs(x_d) >= thr
        if above.any():
            idxs = np.where(above)[0]
            duration_ms_fwhm = float(t_ms[idxs[-1]] - t_ms[idxs[0]])
        else:
            duration_ms_fwhm = 0.0
    else:
        duration_ms_fwhm = 0.0

    # Max absolute derivative (baseline-invariant)
    if n >= 3:
        dxdt = np.gradient(x_d, t_ms)
        max_abs_deriv = float(np.max(np.abs(dxdt)))
    else:
        max_abs_deriv = 0.0

    # RMS amplitude after detrend
    rms_amp = float(np.sqrt((x_d ** 2).mean()))

    # SNR — peak amplitude vs edge std
    edge_std = max(float(np.std(x_d[:edge])), float(np.std(x_d[-edge:])))
    snr_db = 20 * float(np.log10(max(abs(dom_a), 1e-9) / max(edge_std, 1e-9)))

    # Extrema list — captures saddles, notches, secondary peaks.
    # prominence_frac = 0.03 (3% of window peak-to-peak) is sensitive enough
    # to catch the Brugada saddleback shoulder (~5-8% of T amplitude) but
    # high enough to suppress sample-noise jitter.
    extrema = find_extrema(x, t_ms, smooth_ms=8.0, prominence_frac=0.03)
    morph = derived_morphology(extrema, window_size_ms=float(t_ms[-1] - t_ms[0]))

    out = {
        "pos_peak_amp_mv": pos_a,
        "pos_peak_time_ms": pos_t,
        "neg_peak_amp_mv": neg_a,
        "neg_peak_time_ms": neg_t,
        "peak_to_peak_mv": p2p,
        "centroid_ms": centroid_ms,
        "duration_ms_rms": duration_ms_rms,
        "duration_ms_fwhm": duration_ms_fwhm,
        "max_abs_deriv_mv_per_ms": max_abs_deriv,
        "rms_amplitude_mv": rms_amp,
        "snr_db": snr_db,
    }
    # Fixed-arity extrema list — pad with zeros if fewer than K_EXTREMA
    for k in range(K_EXTREMA):
        e = extrema[k] if k < len(extrema) else None
        out[f"extr_{k+1}_t_ms"]   = float(e["time_ms"]) if e else 0.0
        out[f"extr_{k+1}_amp_mv"] = float(e["amp_mv"])  if e else 0.0
        out[f"extr_{k+1}_kind"]   = float(e["kind"])    if e else 0.0
        out[f"extr_{k+1}_prom"]   = float(e["prominence"]) if e else 0.0
    out.update(morph)
    # Stash a non-flattened copy of extrema for visualisation; never goes
    # to the flat feature schema (the dot-prefixed key is filtered there).
    out["_extrema_list"] = extrema
    return out


def _empty_window() -> dict[str, float]:
    base = {k: 0.0 for k in (
        "pos_peak_amp_mv", "pos_peak_time_ms",
        "neg_peak_amp_mv", "neg_peak_time_ms",
        "peak_to_peak_mv", "centroid_ms",
        "duration_ms_rms", "duration_ms_fwhm",
        "max_abs_deriv_mv_per_ms", "rms_amplitude_mv", "snr_db",
    )}
    for k in range(K_EXTREMA):
        base[f"extr_{k+1}_t_ms"]   = 0.0
        base[f"extr_{k+1}_amp_mv"] = 0.0
        base[f"extr_{k+1}_kind"]   = 0.0
        base[f"extr_{k+1}_prom"]   = 0.0
    base.update({
        "n_significant_extrema": 0,
        "slope_reversal_count": 0,
        "biphasic_ratio": 0.0,
        "notch_depth_mv": 0.0,
        "notch_position_frac": 0.0,
        "saddle_depth_relative": 0.0,
        "monotonic_fraction": 1.0,
    })
    base["_extrema_list"] = []
    return base


# ---------------------------------------------------------------------------
# Per-recording: iterate beats × leads × regions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-beat 2-D plane analysis (limb + chest)
# ---------------------------------------------------------------------------
#
# qpsi runs an average-beat plane decomposition AND a per-beat re-decomposition
# (for variability features). Our direct-measurement pipeline so far does
# only per-lead per-beat measurements; this section adds the symmetric
# per-beat plane primitives.
#
# For each beat:
#   1. Slice the 6 leads of each plane (limb: I,II,III,aVR,aVL,aVF;
#      chest: V1..V6).
#   2. Project through LIMB_MAT / CHEST_MAT (2x6) → 2-D trace v(t) ∈ ℝ².
#   3. Compute magnitude |v(t)| and angle ∠v(t).
#   4. For each wave region (P, QRS, T), run the same direct measurement
#      on |v(t)| within that region — peak, centroid, width, extrema,
#      morphology — plus an *axis angle* at the dominant peak time.
#
# This gives us angle/axis features (P-axis, T-axis, QRS-axis) per beat
# that the per-lead pipeline cannot produce, with the same baseline-
# invariant, derivative-based shape primitives.

# Default lead-position indices in the input ECG. Caller can override
# if their layout differs.
LIMB_LEAD_IDX_DEFAULT  = (0, 1, 2, 3, 4, 5)   # I, II, III, aVR, aVL, aVF
CHEST_LEAD_IDX_DEFAULT = (6, 7, 8, 9, 10, 11) # V1, V2, V3, V4, V5, V6


def _project_plane(beat_ecg: np.ndarray, lead_idx: tuple,
                    plane_mat: np.ndarray) -> np.ndarray:
    """Project the 6-lead slice through the 2x6 plane matrix → (2, T)."""
    seg6 = beat_ecg[list(lead_idx), :]   # (6, T)
    return plane_mat @ seg6              # (2, T)


def measure_plane_window(v: np.ndarray, t_ms: np.ndarray
                          ) -> dict[str, float]:
    """Direct measurement on the 2-D plane trace within a wave window.

    Inputs
    ------
    v     : (2, T) — 2-D plane trace (x and y components)
    t_ms  : (T,)   — signed time axis

    Returns the same 11-primitive + extrema + morphology schema as
    ``measure_window`` (computed on |v(t)|), PLUS three angle features:
        peak_angle_deg     : ∠v at the time of the dominant |v| peak
        mean_angle_deg     : weighted mean direction over the window
                              (weights = |v|)
        angle_dispersion_deg : circular MAD of ∠v(t) weighted by |v|

    Angles are in degrees on (-180, +180].
    """
    if v.shape[1] < 5:
        out = _empty_window()
        out.update({"peak_angle_deg": 0.0, "mean_angle_deg": 0.0,
                    "angle_dispersion_deg": 0.0})
        return out

    mag = np.sqrt(v[0] ** 2 + v[1] ** 2)
    out = measure_window(mag, t_ms)

    # Angle at dominant peak (the magnitude peak — always pos amplitude here)
    i_peak = int(np.argmax(mag))
    peak_angle = float(np.degrees(np.arctan2(v[1, i_peak], v[0, i_peak])))

    # Weighted mean direction — use unit vectors of v weighted by |v|
    eps = 1e-9
    ux = v[0] / (mag + eps)
    uy = v[1] / (mag + eps)
    w = mag
    mean_x = float(np.sum(ux * w) / max(w.sum(), eps))
    mean_y = float(np.sum(uy * w) / max(w.sum(), eps))
    mean_angle = float(np.degrees(np.arctan2(mean_y, mean_x)))

    # Circular dispersion: weighted MAD of (angle - mean_angle), folded
    angles = np.degrees(np.arctan2(v[1], v[0]))
    diffs = (angles - mean_angle + 180.0) % 360.0 - 180.0   # in (-180, 180]
    # weighted MAD: median of |diffs| weighted by w
    order = np.argsort(np.abs(diffs))
    w_sorted = w[order]; d_sorted = np.abs(diffs)[order]
    cw = np.cumsum(w_sorted)
    if cw[-1] > 0:
        median_idx = int(np.searchsorted(cw, cw[-1] * 0.5))
        median_idx = min(median_idx, len(d_sorted) - 1)
        ang_disp = float(d_sorted[median_idx])
    else:
        ang_disp = 0.0

    out["peak_angle_deg"] = peak_angle
    out["mean_angle_deg"] = mean_angle
    out["angle_dispersion_deg"] = ang_disp
    return out


def measure_plane_beat(beat_ecg: np.ndarray, beat_time_ms: np.ndarray,
                        rr_ms: float,
                        limb_mat: np.ndarray | None,
                        chest_mat: np.ndarray | None,
                        limb_lead_idx: tuple = LIMB_LEAD_IDX_DEFAULT,
                        chest_lead_idx: tuple = CHEST_LEAD_IDX_DEFAULT,
                        ) -> dict[str, dict[str, dict[str, float]]]:
    """Run the plane direct measurement for one beat, both planes."""
    out: dict[str, dict[str, dict[str, float]]] = {"limb": {}, "chest": {}}
    windows = _windows_for_beat(rr_ms)

    if limb_mat is not None and beat_ecg.shape[0] >= max(limb_lead_idx) + 1:
        v_limb = _project_plane(beat_ecg, limb_lead_idx, limb_mat)
        for region, (lo, hi) in windows.items():
            mask = (beat_time_ms >= lo) & (beat_time_ms <= hi)
            if mask.sum() < 5:
                out["limb"][region] = _empty_plane_window()
                continue
            out["limb"][region] = measure_plane_window(
                v_limb[:, mask], beat_time_ms[mask],
            )

    if chest_mat is not None and beat_ecg.shape[0] >= max(chest_lead_idx) + 1:
        v_chest = _project_plane(beat_ecg, chest_lead_idx, chest_mat)
        for region, (lo, hi) in windows.items():
            mask = (beat_time_ms >= lo) & (beat_time_ms <= hi)
            if mask.sum() < 5:
                out["chest"][region] = _empty_plane_window()
                continue
            out["chest"][region] = measure_plane_window(
                v_chest[:, mask], beat_time_ms[mask],
            )
    return out


def _empty_plane_window() -> dict[str, float]:
    base = _empty_window()
    base["peak_angle_deg"] = 0.0
    base["mean_angle_deg"] = 0.0
    base["angle_dispersion_deg"] = 0.0
    return base


WAVE_REGIONS = ("P", "QRS", "T")
SCALAR_KEYS = (
    # Existing 11 primitives
    "pos_peak_amp_mv", "pos_peak_time_ms",
    "neg_peak_amp_mv", "neg_peak_time_ms",
    "peak_to_peak_mv", "centroid_ms",
    "duration_ms_rms", "duration_ms_fwhm",
    "max_abs_deriv_mv_per_ms", "rms_amplitude_mv", "snr_db",
    # Fixed-arity extrema list (4 × 4 = 16)
    "extr_1_t_ms", "extr_1_amp_mv", "extr_1_kind", "extr_1_prom",
    "extr_2_t_ms", "extr_2_amp_mv", "extr_2_kind", "extr_2_prom",
    "extr_3_t_ms", "extr_3_amp_mv", "extr_3_kind", "extr_3_prom",
    "extr_4_t_ms", "extr_4_amp_mv", "extr_4_kind", "extr_4_prom",
    # Derived morphology (7)
    "n_significant_extrema", "slope_reversal_count",
    "biphasic_ratio", "notch_depth_mv", "notch_position_frac",
    "saddle_depth_relative", "monotonic_fraction",
)


def _windows_for_beat(rr_ms: float,
                      pad_head_r_ms: float = 70.0,
                      offset_ms: float = 75.0,
                      qrs_frac_ms: float = 80.0) -> dict[str, tuple[float, float]]:
    """Return (t_min, t_max) for each of the three wave regions.

    Note: pad_head_r_ms is 70 (not 40) here — the P-window's right edge is
    pushed further from the QRS-onset slope to prevent the rising edge of
    QRS from contaminating P-window measurements. The textbook P-Q
    junction sits ~60-80 ms before R; 70 ms is a safe default. The
    extrema-detector is intrinsically robust to monotonic edges, but
    moving the boundary is defence in depth and additionally cleans the
    summary stats (centroid, second moment, RMS amp).
    """
    return {
        "P":   (-0.5 * rr_ms + offset_ms, -pad_head_r_ms),
        "QRS": (-offset_ms,                qrs_frac_ms),
        "T":   ( qrs_frac_ms,              0.5 * rr_ms),
    }


def measure_beat(beat_ecg: np.ndarray, beat_time_ms: np.ndarray,
                  rr_ms: float, lead_names: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Direct measurements for one beat across all leads.

    Returns: {lead: {region: {scalar: value}}}
    """
    windows = _windows_for_beat(rr_ms)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for li, lead in enumerate(lead_names):
        out[lead] = {}
        for region, (t_lo, t_hi) in windows.items():
            mask = (beat_time_ms >= t_lo) & (beat_time_ms <= t_hi)
            if mask.sum() < 5:
                out[lead][region] = _empty_window()
                continue
            out[lead][region] = measure_window(
                beat_ecg[li, mask], beat_time_ms[mask],
            )
    return out


def measure_all_beats(beats_all: list, lead_names: list[str], fs: int,
                       limb_mat: np.ndarray | None = None,
                       chest_mat: np.ndarray | None = None,
                       limb_lead_idx: tuple = LIMB_LEAD_IDX_DEFAULT,
                       chest_lead_idx: tuple = CHEST_LEAD_IDX_DEFAULT,
                       ) -> tuple[list[dict], list[dict]]:
    """Run direct measurement on every beat (per lead AND per plane).

    Returns ``(per_beat_lead, per_beat_plane)`` — two parallel lists of
    per-beat dicts, one for the per-lead measurements and one for the
    per-plane measurements. If ``limb_mat`` / ``chest_mat`` are None,
    the second list is filled with empty dicts.

    When ``qpsi_native.measure_window_batch_native`` is available
    (Phase B.5 / Brief 2), all per-beat / per-lead / per-region and
    per-beat / per-plane / per-region windows are gathered into a flat
    batch, scipy-savgol'd in a single Python loop, and dispatched to
    the rayon-parallel Rust kernel in one PyO3 boundary crossing.
    Output is bit-identical to the per-call path
    (``measure_window`` / ``measure_plane_window``) by construction —
    the batch kernel composes the same ``measure_window_core_rust``
    that Item #2a deployed.
    """
    if _RUST_MEASURE_WINDOW_BATCH is not None:
        return _measure_all_beats_via_batch(
            beats_all, lead_names, fs,
            limb_mat=limb_mat, chest_mat=chest_mat,
            limb_lead_idx=limb_lead_idx, chest_lead_idx=chest_lead_idx,
        )
    # Python fallback — per-call path (still uses Item #2a's per-call
    # Rust kernel internally via measure_window).
    per_beat_lead: list[dict] = []
    per_beat_plane: list[dict] = []
    for beat in beats_all:
        seg = beat.ecg_12             # (n_leads, T_beat)
        t_ms = beat.time_ms           # (T_beat,)
        rr_ms = seg.shape[1] / fs * 1_000.0
        per_beat_lead.append(measure_beat(seg, t_ms, rr_ms, lead_names))
        per_beat_plane.append(measure_plane_beat(
            seg, t_ms, rr_ms,
            limb_mat=limb_mat, chest_mat=chest_mat,
            limb_lead_idx=limb_lead_idx, chest_lead_idx=chest_lead_idx,
        ))
    return per_beat_lead, per_beat_plane


# ---------------------------------------------------------------------------
# Phase B.5 / Brief 2 — batched dispatch path
# ---------------------------------------------------------------------------

def _measure_all_beats_via_batch(
    beats_all: list,
    lead_names: list[str],
    fs: int,
    limb_mat: np.ndarray | None = None,
    chest_mat: np.ndarray | None = None,
    limb_lead_idx: tuple = LIMB_LEAD_IDX_DEFAULT,
    chest_lead_idx: tuple = CHEST_LEAD_IDX_DEFAULT,
) -> tuple[list[dict], list[dict]]:
    """Batched-dispatch variant of ``measure_all_beats``.

    Per-record flow:
      1. Iterate (beat, lead, region) + (beat, plane, region), collect
         x slices + t_ms slices. Per-plane windows also retain the 2-D
         v(t) projection for the post-batch angle computation.
      2. Detrend each x slice (edge-mean subtract, Python).
      3. Batch-smooth via ``qpsi.savgol_batch.savgol_pack_batch`` — one
         scipy savgol call per window, no PyO3 round-trip per window.
      4. Pack (x_d_flat, x_s_flat, t_ms_flat) with shared offsets / lengths.
      5. Single call to ``qpsi_native.measure_window_batch_native`` —
         rayon parallel across windows, GIL released for the duration.
      6. Unpack: per-window argsort by descending prominence (numpy
         quicksort tie behaviour matching Apr28 GT, same hybrid contract
         as Items #1 and #2a), then ``derived_morphology`` per window,
         then dict assembly. Per-plane windows also receive the 3 angle
         features computed from v(t).

    Returns bit-identical output to the per-call path (verified by
    ``test_measure_window_batch_native.py`` + the Phase B.5 20-patient
    probe).
    """
    from qpsi.savgol_batch import savgol_pack_batch

    region_names = ("P", "QRS", "T")

    # ---- Step 1: collect windows ----
    # Per-lead jobs: (beat_idx, lead_idx, region_idx, x_slice, t_slice)
    # Per-plane jobs: (beat_idx, plane_name, region_idx, mag_slice, t_slice, v_slice)
    lead_jobs: list[tuple] = []
    plane_jobs: list[tuple] = []

    n_beats = len(beats_all)
    n_leads = len(lead_names)
    n_regions = len(region_names)

    for bi, beat in enumerate(beats_all):
        seg = beat.ecg_12              # (n_leads, T_beat)
        t_ms = beat.time_ms            # (T_beat,)
        rr_ms = seg.shape[1] / fs * 1_000.0
        windows = _windows_for_beat(rr_ms)

        # Per-lead
        for li in range(n_leads):
            for ri, region in enumerate(region_names):
                t_lo, t_hi = windows[region]
                mask = (t_ms >= t_lo) & (t_ms <= t_hi)
                if mask.sum() < 5:
                    lead_jobs.append((bi, li, ri, None, None))
                    continue
                lead_jobs.append((bi, li, ri, seg[li, mask], t_ms[mask]))

        # Per-plane
        for plane_name, plane_mat, plane_lead_idx in (
            ("limb", limb_mat, limb_lead_idx),
            ("chest", chest_mat, chest_lead_idx),
        ):
            if plane_mat is None or seg.shape[0] < max(plane_lead_idx) + 1:
                for ri, region in enumerate(region_names):
                    plane_jobs.append((bi, plane_name, ri, None, None, None))
                continue
            v_full = _project_plane(seg, plane_lead_idx, plane_mat)
            for ri, region in enumerate(region_names):
                t_lo, t_hi = windows[region]
                mask = (t_ms >= t_lo) & (t_ms <= t_hi)
                if mask.sum() < 5:
                    plane_jobs.append((bi, plane_name, ri, None, None, None))
                    continue
                v_slice = v_full[:, mask]
                mag = np.sqrt(v_slice[0] ** 2 + v_slice[1] ** 2)
                plane_jobs.append(
                    (bi, plane_name, ri, mag, t_ms[mask], v_slice)
                )

    # ---- Step 2: detrend each non-empty window ----
    # Mirror measure_window's: edge = max(3, int(0.10 * n)); edge_mean =
    # mean of concat(x[:edge], x[-edge:]).
    active_indices: list[int] = []  # index into combined (lead_jobs + plane_jobs)
    x_d_list: list[np.ndarray] = []
    t_ms_list: list[np.ndarray] = []
    smooth_ms = 8.0
    prominence_frac = 0.03

    def _detrend(x: np.ndarray) -> np.ndarray:
        n = len(x)
        edge = max(3, int(0.10 * n))
        edge_mean = np.concatenate([x[:edge], x[-edge:]]).mean()
        return x - edge_mean

    n_lead_jobs = len(lead_jobs)
    n_plane_jobs = len(plane_jobs)
    for i, job in enumerate(lead_jobs):
        bi, li, ri, x, t_slice = job
        if x is None:
            continue
        active_indices.append(i)
        x_d_list.append(_detrend(x.astype(np.float64)))
        t_ms_list.append(t_slice.astype(np.float64))
    for j, job in enumerate(plane_jobs):
        bi, plane_name, ri, mag, t_slice, v_slice = job
        if mag is None:
            continue
        active_indices.append(n_lead_jobs + j)
        x_d_list.append(_detrend(mag.astype(np.float64)))
        t_ms_list.append(t_slice.astype(np.float64))

    if not active_indices:
        # All windows short-circuited — assemble empty structure.
        per_beat_lead: list[dict] = [
            {lead: {r: _empty_window() for r in region_names} for lead in lead_names}
            for _ in range(n_beats)
        ]
        per_beat_plane: list[dict] = [
            {"limb": {r: _empty_plane_window() for r in region_names},
             "chest": {r: _empty_plane_window() for r in region_names}}
            for _ in range(n_beats)
        ]
        return per_beat_lead, per_beat_plane

    # ---- Step 3: batch-smooth via scipy savgol (one tight Python loop) ----
    # Use win/poly that find_extrema would compute internally; for our
    # uniform t_ms (500 Hz) this is win=5, poly=2 in the vast majority
    # of cases. Per-window win may shift slightly when n < 5 fallback
    # rules apply, so we batch by win first.
    # NOTE: for simplicity we group all windows by the (win, poly) the
    # internal find_extrema would pick. In practice with smooth_ms=8.0
    # and 500 Hz this is always (5, 2).
    smoothed_per_window: list[np.ndarray] = []
    win_groups: dict[tuple[int, int], list[int]] = {}
    win_inputs: dict[tuple[int, int], list[np.ndarray]] = {}
    for w_idx, (x_d, t_slice) in enumerate(zip(x_d_list, t_ms_list)):
        n = len(x_d)
        fs_implied = 1000.0 / max(t_slice[1] - t_slice[0], 1e-6)
        win = max(5, int(round(smooth_ms * fs_implied / 1000.0)) | 1)
        if win >= n:
            win = max(5, (n // 2) * 2 - 1)
        poly = min(2, win - 1)
        key = (win, poly)
        win_groups.setdefault(key, []).append(w_idx)
        win_inputs.setdefault(key, []).append(x_d)

    # Initialize smoothed_per_window with placeholders, then fill by group
    smoothed_per_window = [np.zeros(0, dtype=np.float64)] * len(x_d_list)
    for (win, poly), idxs in win_groups.items():
        from qpsi.savgol_batch import savgol_filter_batch
        outputs = savgol_filter_batch(win_inputs[(win, poly)], win, poly)
        for slot, smoothed in zip(idxs, outputs):
            smoothed_per_window[slot] = smoothed

    # ---- Step 4: pack flat arrays ----
    lengths = np.array([len(s) for s in x_d_list], dtype=np.int64)
    offsets = np.zeros(len(x_d_list), dtype=np.int64)
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    x_d_flat = np.ascontiguousarray(np.concatenate(x_d_list), dtype=np.float64)
    x_s_flat = np.ascontiguousarray(
        np.concatenate(smoothed_per_window), dtype=np.float64
    )
    t_ms_flat = np.ascontiguousarray(np.concatenate(t_ms_list), dtype=np.float64)

    # ---- Step 5: single PyO3 boundary crossing into Rust ----
    scalars_arr, extrema_flat, counts = _RUST_MEASURE_WINDOW_BATCH(
        x_d_flat, x_s_flat, t_ms_flat, offsets, lengths, float(prominence_frac),
    )
    scalars_arr = np.asarray(scalars_arr)
    extrema_flat = np.asarray(extrema_flat)
    counts = np.asarray(counts)

    # ---- Step 6: per-window dict assembly ----
    # Build empty per-beat structures first.
    per_beat_lead = [
        {lead: {r: _empty_window() for r in region_names} for lead in lead_names}
        for _ in range(n_beats)
    ]
    per_beat_plane = [
        {"limb": {r: _empty_plane_window() for r in region_names},
         "chest": {r: _empty_plane_window() for r in region_names}}
        for _ in range(n_beats)
    ]

    cumcounts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=cumcounts[1:])

    # ---- Pre-pass: per-window sorted extrema (prominence-descending) +
    # window sizes. Sorting stays on the Python side per the Phase B.5
    # hybrid contract (numpy quicksort tie-break — see FIDELITY_REPORT H.B.4).
    n_active = len(active_indices)
    sorted_extrema_per_window: list[list[dict]] = []
    window_sizes_arr = np.zeros(n_active, dtype=np.float64)
    # Also pre-build the flat (total_extrema, 4) array for the batched
    # Rust derived_morphology call.
    sorted_rows: list[np.ndarray] = []
    sorted_counts = np.zeros(n_active, dtype=np.int64)

    for active_pos in range(n_active):
        n_ext = int(counts[active_pos])
        ext_slice = extrema_flat[cumcounts[active_pos] : cumcounts[active_pos + 1]]
        if n_ext > 0:
            order = np.argsort(-ext_slice[:, 3])
            sorted_arr = ext_slice[order]
            extrema_list = [
                {"time_ms": float(sorted_arr[i, 0]),
                 "amp_mv": float(sorted_arr[i, 1]),
                 "kind": int(sorted_arr[i, 2]),
                 "prominence": float(sorted_arr[i, 3])}
                for i in range(n_ext)
            ]
            sorted_rows.append(sorted_arr)
        else:
            extrema_list = []
        sorted_extrema_per_window.append(extrema_list)
        sorted_counts[active_pos] = n_ext
        t_slice = t_ms_list[active_pos]
        window_sizes_arr[active_pos] = float(t_slice[-1] - t_slice[0])

    # ---- Batched derived_morphology dispatch ----
    # Single Rust call (or per-window Python fallback). Returns
    # (n_active, 7) — the 7 morphology scalars per window in the
    # canonical order documented in derived_morphology.rs.
    if _RUST_DERIVED_MORPHOLOGY_BATCH is not None and n_active > 0:
        if sorted_rows:
            sorted_extrema_flat = np.ascontiguousarray(
                np.concatenate(sorted_rows, axis=0), dtype=np.float64
            )
        else:
            sorted_extrema_flat = np.zeros((0, 4), dtype=np.float64)
        morph_results = np.asarray(_RUST_DERIVED_MORPHOLOGY_BATCH(
            sorted_extrema_flat, sorted_counts, window_sizes_arr,
        ))
    else:
        morph_results = np.zeros((n_active, 7), dtype=np.float64)
        for active_pos in range(n_active):
            m = derived_morphology(
                sorted_extrema_per_window[active_pos],
                window_size_ms=float(window_sizes_arr[active_pos]),
            )
            morph_results[active_pos, 0] = m["n_significant_extrema"]
            morph_results[active_pos, 1] = m["slope_reversal_count"]
            morph_results[active_pos, 2] = m["biphasic_ratio"]
            morph_results[active_pos, 3] = m["notch_depth_mv"]
            morph_results[active_pos, 4] = m["notch_position_frac"]
            morph_results[active_pos, 5] = m["saddle_depth_relative"]
            morph_results[active_pos, 6] = m["monotonic_fraction"]

    for active_pos, global_job_idx in enumerate(active_indices):
        sc = scalars_arr[active_pos]
        extrema_sorted = sorted_extrema_per_window[active_pos]

        # Build the base + extrema-slot fields.
        out: dict[str, Any] = {
            "pos_peak_amp_mv": float(sc[1]),
            "pos_peak_time_ms": float(sc[0]),
            "neg_peak_amp_mv": float(sc[3]),
            "neg_peak_time_ms": float(sc[2]),
            "peak_to_peak_mv": float(sc[4]),
            "centroid_ms": float(sc[5]),
            "duration_ms_rms": float(sc[6]),
            "duration_ms_fwhm": float(sc[7]),
            "max_abs_deriv_mv_per_ms": float(sc[8]),
            "rms_amplitude_mv": float(sc[9]),
            "snr_db": float(sc[10]),
        }
        for k in range(K_EXTREMA):
            e = extrema_sorted[k] if k < len(extrema_sorted) else None
            out[f"extr_{k+1}_t_ms"] = float(e["time_ms"]) if e else 0.0
            out[f"extr_{k+1}_amp_mv"] = float(e["amp_mv"]) if e else 0.0
            out[f"extr_{k+1}_kind"] = float(e["kind"]) if e else 0.0
            out[f"extr_{k+1}_prom"] = float(e["prominence"]) if e else 0.0

        # Derived morphology — populated from the batched Rust result
        # (or Python fallback). Field ordering mirrors the canonical
        # ``derived_morphology`` dict keys; n_* fields are coerced to
        # int to match the per-call schema exactly.
        mr = morph_results[active_pos]
        out["n_significant_extrema"] = int(mr[0])
        out["slope_reversal_count"] = int(mr[1])
        out["biphasic_ratio"] = float(mr[2])
        out["notch_depth_mv"] = float(mr[3])
        out["notch_position_frac"] = float(mr[4])
        out["saddle_depth_relative"] = float(mr[5])
        out["monotonic_fraction"] = float(mr[6])
        out["_extrema_list"] = extrema_sorted

        # Route to per_beat_lead or per_beat_plane.
        if global_job_idx < n_lead_jobs:
            bi, li, ri, _, _ = lead_jobs[global_job_idx]
            per_beat_lead[bi][lead_names[li]][region_names[ri]] = out
        else:
            j = global_job_idx - n_lead_jobs
            bi, plane_name, ri, _, _, v_slice = plane_jobs[j]
            # Add the 3 angle features (computed from v_slice, NOT mag).
            # Mirror measure_plane_window's angle math exactly.
            mag = np.sqrt(v_slice[0] ** 2 + v_slice[1] ** 2)
            i_peak = int(np.argmax(mag))
            peak_angle = float(np.degrees(np.arctan2(
                v_slice[1, i_peak], v_slice[0, i_peak]
            )))
            eps = 1e-9
            ux = v_slice[0] / (mag + eps)
            uy = v_slice[1] / (mag + eps)
            w = mag
            mean_x = float(np.sum(ux * w) / max(w.sum(), eps))
            mean_y = float(np.sum(uy * w) / max(w.sum(), eps))
            mean_angle = float(np.degrees(np.arctan2(mean_y, mean_x)))
            angles = np.degrees(np.arctan2(v_slice[1], v_slice[0]))
            diffs = (angles - mean_angle + 180.0) % 360.0 - 180.0
            order_ang = np.argsort(np.abs(diffs))
            w_sorted = w[order_ang]
            d_sorted = np.abs(diffs)[order_ang]
            cw = np.cumsum(w_sorted)
            if cw[-1] > 0:
                median_idx = int(np.searchsorted(cw, cw[-1] * 0.5))
                median_idx = min(median_idx, len(d_sorted) - 1)
                ang_disp = float(d_sorted[median_idx])
            else:
                ang_disp = 0.0
            out["peak_angle_deg"] = peak_angle
            out["mean_angle_deg"] = mean_angle
            out["angle_dispersion_deg"] = ang_disp
            per_beat_plane[bi][plane_name][region_names[ri]] = out

    return per_beat_lead, per_beat_plane


# ---------------------------------------------------------------------------
# Per-recording aggregation
# ---------------------------------------------------------------------------

PLANE_SCALAR_KEYS = SCALAR_KEYS + (
    "peak_angle_deg", "mean_angle_deg", "angle_dispersion_deg",
)

# Primitives used to define the per-beat morphology vector for cluster
# analysis. Chosen so each axis is independent and clinically meaningful;
# cluster algorithm (k-means with k=1..3, gap-statistic-style) decides
# how many beat morphologies the recording contains.
_CLUSTER_PRIMITIVES = (
    "pos_peak_amp_mv", "neg_peak_amp_mv",
    "duration_ms_rms", "centroid_ms",
    "n_significant_extrema",
)


def _beat_cluster_summary(per_beat_for_cell: list[dict]) -> dict[str, float]:
    """Cluster analysis on per-beat parameter vectors.

    Operates on a single (lead, region) cell's per-beat data. For each
    of K ∈ {1, 2, 3}, fits k-means and computes inertia. Picks the
    smallest K such that adding one more cluster doesn't improve
    inertia by more than 25 % — a coarse "elbow" rule.

    Returns:
        n_morphology_clusters     ∈ {1, 2, 3}    — number of distinct beat shapes
        cluster_balance_ratio     ∈ [0, 1]       — size ratio min/max cluster
        longest_run_in_dominant   ∈ [0, n_beats] — longest consecutive same-cluster run
        cluster_alternation       ∈ [0, 1]       — fraction of consecutive
                                                     beat pairs in different clusters
        n_beats                   — count of beats used in clustering
    """
    if len(per_beat_for_cell) < 4:
        return {"n_morphology_clusters": 1.0,
                "cluster_balance_ratio": 1.0,
                "longest_run_in_dominant": float(len(per_beat_for_cell)),
                "cluster_alternation": 0.0,
                "n_beats": float(len(per_beat_for_cell))}

    # Build the per-beat matrix (N, F) over selected primitives
    rows = []
    for d in per_beat_for_cell:
        rows.append([float(d.get(k, 0.0)) for k in _CLUSTER_PRIMITIVES])
    X = np.array(rows, dtype=float)
    if X.shape[0] < 4:
        return {"n_morphology_clusters": 1.0,
                "cluster_balance_ratio": 1.0,
                "longest_run_in_dominant": float(X.shape[0]),
                "cluster_alternation": 0.0,
                "n_beats": float(X.shape[0])}
    # Z-score per column (avoid features with large scale dominating)
    means = X.mean(axis=0); stds = X.std(axis=0)
    stds = np.where(stds > 1e-9, stds, 1.0)
    Xz = (X - means) / stds

    # Tiny custom k-means (avoids sklearn import cost in worker; deterministic)
    def _kmeans(Xz, k, n_iter=20, seed=0):
        rng = np.random.default_rng(seed)
        n = Xz.shape[0]
        # k-means++ init: first centre random, subsequent farthest from existing
        idx0 = int(rng.integers(0, n))
        centres = [Xz[idx0]]
        for _ in range(k - 1):
            d = np.min([np.sum((Xz - c) ** 2, axis=1) for c in centres], axis=0)
            if d.sum() == 0:
                centres.append(Xz[int(rng.integers(0, n))])
            else:
                centres.append(Xz[int(rng.choice(n, p=d / d.sum()))])
        centres = np.array(centres)
        for _ in range(n_iter):
            dists = np.array([np.sum((Xz - c) ** 2, axis=1) for c in centres])
            assign = np.argmin(dists, axis=0)
            new = np.array([Xz[assign == ki].mean(axis=0)
                             if (assign == ki).any() else centres[ki]
                             for ki in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = float(sum(np.sum((Xz[assign == ki] - centres[ki]) ** 2)
                             for ki in range(k) if (assign == ki).any()))
        return assign, inertia

    inertias = []
    assigns = []
    for k in (1, 2, 3):
        if X.shape[0] < k * 2:
            break
        a, inertia = _kmeans(Xz, k)
        inertias.append(inertia)
        assigns.append(a)

    # Pick K via elbow: smallest K where (Δ_inertia / inertia_K) < 25%
    chosen_k = 1
    for i in range(1, len(inertias)):
        if inertias[i - 1] > 0:
            improvement = (inertias[i - 1] - inertias[i]) / inertias[i - 1]
            if improvement >= 0.25:
                chosen_k = i + 1
            else:
                break
    assignment = assigns[chosen_k - 1] if chosen_k <= len(assigns) else assigns[-1]

    # Cluster sizes
    sizes = np.bincount(assignment, minlength=chosen_k)
    if chosen_k > 1 and sizes.max() > 0:
        balance = float(sizes.min() / sizes.max())
    else:
        balance = 1.0

    # Longest run in the dominant cluster
    dom_cluster = int(np.argmax(sizes))
    longest = 0; cur = 0
    for c in assignment:
        if c == dom_cluster:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    # Cluster alternation: fraction of consecutive pairs that switch cluster
    if len(assignment) >= 2:
        transitions = float(np.sum(assignment[:-1] != assignment[1:]))
        alternation = transitions / (len(assignment) - 1)
    else:
        alternation = 0.0

    return {"n_morphology_clusters": float(chosen_k),
            "cluster_balance_ratio": balance,
            "longest_run_in_dominant": float(longest),
            "cluster_alternation": float(alternation),
            "n_beats": float(X.shape[0])}


def _autocorrelation_lags(vals: np.ndarray, max_lag: int = 10) -> list[float]:
    """Sample autocorrelation function (ACF) at lags 1..max_lag.

    Returns a fixed-arity list of length ``max_lag``. A lag is populated
    only when the number of sample-pairs at that lag is ≥4 (i.e. when
    n_beats - k ≥ 4); otherwise the slot is 0. Bounded to [-1, 1].

    Default max_lag was raised from 5 to 10 so 60-second recordings
    (typically n=60-100 beats) can resolve longer-period patterns
    (Wenckebach 5:4, autonomic cycles, slow drift) up to lag 10. For
    short 10-second recordings (n=10 beats) lags 1..6 still populate
    while 7..10 stay zero — schema consistency without spurious noise.

    ACF(k) detects period-k structure across beats:
        - ACF(1) high+ → smooth drift across beats (correlated neighbours)
        - ACF(1) high− → strict alternation (anti-correlated neighbours)
        - ACF(2) high+ → period-2 alternation reinforced
        - ACF(3) high+ → trigeminy-like structure
        - ACF(5..10) high+ → long-period or respiratory-modulated patterns
        - ACF(k) ≈ 0 → uncorrelated / random
    """
    n = len(vals)
    out = [0.0] * max_lag
    if n < 5:
        return out
    if (np.abs(vals) > 1e-9).mean() < 0.5:
        return out          # mostly zeros → undefined, skip
    mean = float(np.mean(vals))
    centred = vals - mean
    var = float(np.mean(centred ** 2))
    if var < 1e-12:
        return out          # constant sequence → ACF undefined
    # Require ≥4 sample pairs at each lag for a stable estimate.
    for k in range(1, max_lag + 1):
        n_pairs = n - k
        if n_pairs < 4:
            break
        num = float(np.mean(centred[:n_pairs] * centred[k:]))
        out[k - 1] = num / var
    return out


def _dft_alternation_index(vals: np.ndarray, med: float, mad: float,
                            period: int) -> float:
    """Magnitude of the period-N DFT component, normalised by MAD.

    Detects whether a periodic pattern of period N exists in the
    sequence. Period 2 = alternans, period 3 = trigeminy, period 4 =
    quadrigeminy. Generalises the sign-pattern alternation index to
    arbitrary period and to non-square waveforms.

    Returns 0 when the sequence has insufficient variation to support
    a meaningful alternation claim — specifically, MAD must exceed
    1% of |median| (or an absolute floor) AND the sequence must have
    at least some nonzero values. This prevents the division-by-tiny
    explosion for sparse / padded / quasi-constant feature columns.
    """
    n = len(vals)
    if n < 2 * period:
        return 0.0
    # Require nontrivial variation
    rel_floor = 0.01 * max(abs(med), 1e-3)
    abs_floor = 1e-3
    if mad < max(rel_floor, abs_floor):
        return 0.0
    # Require ≥80% nonzero entries (sparse features padded with 0
    # when extrema are absent should not register alternation — those
    # are "feature presence variation", not "feature value alternation")
    if (np.abs(vals) > 1e-9).mean() < 0.8:
        return 0.0
    devs = vals - med
    omega = 2.0 * np.pi / float(period)
    cos_proj = float(np.mean(devs * np.cos(omega * np.arange(n))))
    sin_proj = float(np.mean(devs * np.sin(omega * np.arange(n))))
    amp = float(np.hypot(cos_proj, sin_proj))
    return amp / (mad + 1e-9)


def _aggregate_one(vals_arr: np.ndarray, *, is_angle: bool = False) -> dict[str, float]:
    """Compute robust statistics + multi-period alternation indices.

    Returns dict with keys:
        median, mad, iqr, n,
        alternation_index, alternation_index_p2, alternation_index_p3,
        alternation_index_p4
    """
    n = len(vals_arr)
    if n == 0:
        return {"median": 0.0, "mad": 0.0, "iqr": 0.0, "n": 0,
                "alternation_index": 0.0,
                "alternation_index_p2": 0.0,
                "alternation_index_p3": 0.0,
                "alternation_index_p4": 0.0}
    if is_angle:
        rad = np.deg2rad(vals_arr)
        cx = float(np.mean(np.cos(rad)))
        cy = float(np.mean(np.sin(rad)))
        med = float(np.degrees(np.arctan2(cy, cx)))
        diffs = ((vals_arr - med + 180.0) % 360.0) - 180.0
        mad = float(np.median(np.abs(diffs)))
        iqr_v = float(np.percentile(np.abs(diffs), 75) -
                      np.percentile(np.abs(diffs), 25))
        # Period-N DFT on the wrapped diffs (treats angle as ordinary scalar
        # after centring on the circular mean — fine for small dispersions)
        alt_p2 = _dft_alternation_index(diffs, 0.0, mad, 2)
        alt_p3 = _dft_alternation_index(diffs, 0.0, mad, 3)
        alt_p4 = _dft_alternation_index(diffs, 0.0, mad, 4)
    else:
        med = float(np.median(vals_arr))
        mad = float(np.median(np.abs(vals_arr - med)))
        q1, q3 = (float(x) for x in np.percentile(vals_arr, [25, 75]))
        iqr_v = q3 - q1
        alt_p2 = _dft_alternation_index(vals_arr, med, mad, 2)
        alt_p3 = _dft_alternation_index(vals_arr, med, mad, 3)
        alt_p4 = _dft_alternation_index(vals_arr, med, mad, 4)
    # Legacy alternation_index = sign-pattern p2 (kept for backward compat
    # with the stability_compare driver and any downstream code referencing
    # this key). The DFT version is more rigorous; we surface both.
    if n >= 4 and not is_angle:
        signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)])
        alt_amp = abs(np.mean(signs * (vals_arr - med)))
        alt_legacy = float(alt_amp / (mad + 1e-9))
    else:
        alt_legacy = 0.0
    # Sample ACF at lags 1..10 — captures slow drift, alternation, and
    # longer-period beat-to-beat patterns. For 10s recordings (n≈10
    # beats) only lags 1..6 populate; for 60s recordings (n≈60 beats)
    # all 10 lags populate. Schema is fixed-arity (always 10 slots) for
    # consistency across cohorts; sparse lags are zero.
    acfs = _autocorrelation_lags(vals_arr if not is_angle else
                                  ((vals_arr - med + 180.0) % 360.0) - 180.0,
                                  max_lag=10)
    return {
        "median": med, "mad": mad, "iqr": iqr_v, "n": n,
        "alternation_index": alt_legacy,
        "alternation_index_p2": alt_p2,
        "alternation_index_p3": alt_p3,
        "alternation_index_p4": alt_p4,
        "acf_lag_1":  acfs[0], "acf_lag_2":  acfs[1],
        "acf_lag_3":  acfs[2], "acf_lag_4":  acfs[3],
        "acf_lag_5":  acfs[4], "acf_lag_6":  acfs[5],
        "acf_lag_7":  acfs[6], "acf_lag_8":  acfs[7],
        "acf_lag_9":  acfs[8], "acf_lag_10": acfs[9],
    }


# ---------------------------------------------------------------------------
# Phase C — vectorized aggregator: 4-D batch over (groups, regions, scalars, beats)
# ---------------------------------------------------------------------------

def _aggregate_batch(vals: np.ndarray, *, is_angle: bool = False
                     ) -> dict[str, np.ndarray]:
    """Vectorized batch variant of ``_aggregate_one``.

    Input ``vals`` shape ``(..., n_beats)`` — any leading dims preserved.
    Returns a dict of 18 numpy arrays, each shaped ``vals.shape[:-1]``,
    matching ``_aggregate_one``'s key set.

    Bit-equality is preserved because numpy axis reductions use the
    same pairwise-summation tree shape as 1-D calls on a size-``n_beats``
    array; ``np.median`` / ``np.percentile`` are deterministic across
    axis position. Guards from the per-cell helpers
    (``_dft_alternation_index``, ``_autocorrelation_lags``) translate to
    vectorized boolean masks applied via ``np.where``.

    See ``test_aggregate_vectorized.py`` for the lock.
    """
    n_beats = vals.shape[-1]
    L = vals.shape[:-1]

    def _zeros() -> np.ndarray:
        return np.zeros(L, dtype=np.float64)

    if n_beats == 0:
        out_empty: dict[str, np.ndarray] = {
            "median": _zeros(), "mad": _zeros(), "iqr": _zeros(),
            "n": np.zeros(L, dtype=np.int64),
            "alternation_index": _zeros(),
            "alternation_index_p2": _zeros(),
            "alternation_index_p3": _zeros(),
            "alternation_index_p4": _zeros(),
        }
        for k in range(1, 11):
            out_empty[f"acf_lag_{k}"] = _zeros()
        return out_empty

    # --- Central tendency + dispersion + DFT input setup ---
    if is_angle:
        rad = np.deg2rad(vals)
        cx = np.cos(rad).mean(axis=-1)
        cy = np.sin(rad).mean(axis=-1)
        med = np.degrees(np.arctan2(cy, cx))
        diffs = ((vals - med[..., None] + 180.0) % 360.0) - 180.0
        abs_diffs = np.abs(diffs)
        mad = np.median(abs_diffs, axis=-1)
        qs = np.percentile(abs_diffs, [25, 75], axis=-1)
        iqr_v = qs[1] - qs[0]
        # DFT alternation: per _aggregate_one is_angle path passes
        # (diffs, med=0.0, mad). Centred devs = diffs - 0 = diffs.
        alt_signal = diffs
        alt_med = np.zeros_like(med)
        alt_devs = diffs
    else:
        med = np.median(vals, axis=-1)
        abs_devs = np.abs(vals - med[..., None])
        mad = np.median(abs_devs, axis=-1)
        qs = np.percentile(vals, [25, 75], axis=-1)
        iqr_v = qs[1] - qs[0]
        alt_signal = vals
        alt_med = med
        alt_devs = vals - med[..., None]

    # --- DFT alternation_index_pN (vectorized _dft_alternation_index) ---
    # Per-cell guards from _dft_alternation_index:
    #   (1) n < 2*period → 0 (applied as a scalar gate per period below)
    #   (2) mad < max(0.01*|med|, 1e-3) → 0
    #   (3) (|vals| > 1e-9).mean() < 0.8 → 0
    mad_floor = np.maximum(0.01 * np.maximum(np.abs(alt_med), 1e-3), 1e-3)
    mad_ok = mad >= mad_floor
    nonzero_frac_alt = (np.abs(alt_signal) > 1e-9).mean(axis=-1)
    sparsity_ok_alt = nonzero_frac_alt >= 0.8
    alt_valid = mad_ok & sparsity_ok_alt

    alt_pN: dict[int, np.ndarray] = {}
    arange_n = np.arange(n_beats, dtype=np.float64)
    for period in (2, 3, 4):
        if n_beats < 2 * period:
            alt_pN[period] = _zeros()
            continue
        omega = 2.0 * np.pi / float(period)
        cos_kernel = np.cos(omega * arange_n)
        sin_kernel = np.sin(omega * arange_n)
        cos_proj = (alt_devs * cos_kernel).mean(axis=-1)
        sin_proj = (alt_devs * sin_kernel).mean(axis=-1)
        amp = np.hypot(cos_proj, sin_proj)
        raw = amp / (mad + 1e-9)
        alt_pN[period] = np.where(alt_valid, raw, 0.0)

    # --- Legacy alternation_index (sign-pattern p2, non-angle only) ---
    if n_beats >= 4 and not is_angle:
        # _aggregate_one builds `signs = np.array([1 if i%2==0 else -1, ...])`
        # which becomes int64; int64*float64 promotes to float64 — using
        # float64 directly gives the same bit-equal product.
        signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_beats)],
                          dtype=np.float64)
        alt_amp = np.abs((signs * (vals - med[..., None])).mean(axis=-1))
        alt_legacy = alt_amp / (mad + 1e-9)
    else:
        alt_legacy = _zeros()

    # --- ACF lags 1..10 (vectorized _autocorrelation_lags) ---
    # Per-cell guards:
    #   (1) n < 5 → all zeros
    #   (2) (|vals| > 1e-9).mean() < 0.5 → all zeros
    #   (3) var < 1e-12 → all zeros
    #   (4) per-lag: n_pairs < 4 → break (this and all higher lags zero)
    # Angle path uses wrapped diffs as input (per _aggregate_one's call).
    acf_input = diffs if is_angle else vals
    acf_out: dict[str, np.ndarray] = {}
    if n_beats < 5:
        for k in range(1, 11):
            acf_out[f"acf_lag_{k}"] = _zeros()
    else:
        nonzero_frac_acf = (np.abs(acf_input) > 1e-9).mean(axis=-1)
        sparsity_ok_acf = nonzero_frac_acf >= 0.5
        means_acf = acf_input.mean(axis=-1)
        centred = acf_input - means_acf[..., None]
        var_acf = (centred ** 2).mean(axis=-1)
        var_ok = var_acf >= 1e-12
        acf_valid = sparsity_ok_acf & var_ok
        var_safe = np.where(var_ok, var_acf, 1.0)
        for k in range(1, 11):
            n_pairs = n_beats - k
            if n_pairs < 4:
                acf_out[f"acf_lag_{k}"] = _zeros()
                continue
            num = (centred[..., :n_pairs] * centred[..., k:]).mean(axis=-1)
            raw = num / var_safe
            acf_out[f"acf_lag_{k}"] = np.where(acf_valid, raw, 0.0)

    return {
        "median": med,
        "mad": mad,
        "iqr": iqr_v,
        "n": np.full(L, n_beats, dtype=np.int64),
        "alternation_index": alt_legacy,
        "alternation_index_p2": alt_pN[2],
        "alternation_index_p3": alt_pN[3],
        "alternation_index_p4": alt_pN[4],
        **acf_out,
    }


def _pack_summary(stats: dict[str, np.ndarray], idx: tuple,
                   n_beats: int) -> dict[str, float]:
    """Extract a single (group, region, scalar) cell from the batched stats.

    Matches ``_aggregate_one``'s output schema: 8 keys when ``n_beats==0``
    (no ``acf_lag_*``, ``n=0``); 18 keys otherwise.
    """
    if n_beats == 0:
        return {"median": 0.0, "mad": 0.0, "iqr": 0.0, "n": 0,
                "alternation_index": 0.0,
                "alternation_index_p2": 0.0,
                "alternation_index_p3": 0.0,
                "alternation_index_p4": 0.0}
    return {
        "median": float(stats["median"][idx]),
        "mad": float(stats["mad"][idx]),
        "iqr": float(stats["iqr"][idx]),
        "n": int(stats["n"][idx]),
        "alternation_index": float(stats["alternation_index"][idx]),
        "alternation_index_p2": float(stats["alternation_index_p2"][idx]),
        "alternation_index_p3": float(stats["alternation_index_p3"][idx]),
        "alternation_index_p4": float(stats["alternation_index_p4"][idx]),
        "acf_lag_1":  float(stats["acf_lag_1"][idx]),
        "acf_lag_2":  float(stats["acf_lag_2"][idx]),
        "acf_lag_3":  float(stats["acf_lag_3"][idx]),
        "acf_lag_4":  float(stats["acf_lag_4"][idx]),
        "acf_lag_5":  float(stats["acf_lag_5"][idx]),
        "acf_lag_6":  float(stats["acf_lag_6"][idx]),
        "acf_lag_7":  float(stats["acf_lag_7"][idx]),
        "acf_lag_8":  float(stats["acf_lag_8"][idx]),
        "acf_lag_9":  float(stats["acf_lag_9"][idx]),
        "acf_lag_10": float(stats["acf_lag_10"][idx]),
    }


def _composite_alternans(comp: dict[str, float | str]) -> dict[str, float | str]:
    """Build the ``_morphology_alternans`` synthetic per-region scalar from
    accumulated (max_alt, driver_key) tracking."""
    return {
        "alt_p2_max": comp["max_p2"],
        "alt_p3_max": comp["max_p3"],
        "alt_p4_max": comp["max_p4"],
        "alt_p2_driver": comp["max_p2_key"],
        "alt_p3_driver": comp["max_p3_key"],
        "alt_p4_driver": comp["max_p4_key"],
        # By construction, perfect period-N alternation yields alt_pN=1.0;
        # white-noise expectation at n≈10 is ~0.5. Threshold 1.0 ⇒
        # "as strong as perfect alternation".
        "alternans_present_p2": float(comp["max_p2"] > 1.0),
        "alternans_present_p3": float(comp["max_p3"] > 1.0),
        "alternans_present_p4": float(comp["max_p4"] > 1.0),
    }


def aggregate_recording(per_beat: list[dict], lead_names: list[str]
                         ) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Robust-statistics aggregation across beats (vectorized).

    Schema: ``{lead: {region: {scalar: {median, mad, iqr, n,
    alternation_index, alternation_index_p2/p3/p4, acf_lag_1..10}}}}``
    plus the synthetic ``_morphology_alternans`` and ``_beat_clusters``
    cells per (lead, region).

    Phase C: one ``_aggregate_batch`` call replaces ~1224 per-cell
    ``_aggregate_one`` calls. Bit-identical to the per-cell path —
    verified by ``test_aggregate_vectorized.py`` + the 20-patient
    deep-equality probe.
    """
    n_beats = len(per_beat)
    n_leads = len(lead_names)
    n_regions = len(WAVE_REGIONS)
    n_scalars = len(SCALAR_KEYS)

    # ---- Step 1: materialize 4-D array (n_leads × n_regions × n_scalars × n_beats) ----
    vals_4d = np.zeros((n_leads, n_regions, n_scalars, n_beats), dtype=np.float64)
    for bi in range(n_beats):
        cell_b = per_beat[bi]
        for li, lead in enumerate(lead_names):
            cell_l = cell_b.get(lead, {})
            for ri, region in enumerate(WAVE_REGIONS):
                cell_r = cell_l.get(region, {})
                for si, key in enumerate(SCALAR_KEYS):
                    vals_4d[li, ri, si, bi] = cell_r.get(key, 0.0)

    # ---- Step 2: batched stats (replaces ~1224 _aggregate_one calls) ----
    stats = _aggregate_batch(vals_4d, is_angle=False)

    # ---- Step 3: per (lead, region) reassembly + composite + clusters ----
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for li, lead in enumerate(lead_names):
        out[lead] = {}
        for ri, region in enumerate(WAVE_REGIONS):
            out[lead][region] = {}
            comp = {"max_p2": 0.0, "max_p3": 0.0, "max_p4": 0.0,
                    "max_p2_key": "", "max_p3_key": "", "max_p4_key": ""}
            for si, key in enumerate(SCALAR_KEYS):
                summary = _pack_summary(stats, (li, ri, si), n_beats)
                out[lead][region][key] = summary
                for p, suffix in (("p2", "alternation_index_p2"),
                                   ("p3", "alternation_index_p3"),
                                   ("p4", "alternation_index_p4")):
                    if summary[suffix] > comp[f"max_{p}"]:
                        comp[f"max_{p}"] = summary[suffix]
                        comp[f"max_{p}_key"] = key
            out[lead][region]["_morphology_alternans"] = _composite_alternans(comp)
            cell_per_beat = [per_beat[i].get(lead, {}).get(region, {})
                              for i in range(n_beats)]
            out[lead][region]["_beat_clusters"] = _beat_cluster_summary(cell_per_beat)
    return out


def aggregate_plane_recording(per_beat_plane: list[dict]
                               ) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Aggregate per-beat plane measurements (vectorized).

    Same robust statistics as ``aggregate_recording`` plus circular
    handling for angle-suffixed keys, and the same composite
    morphology-alternans summary per (plane, region).

    PLANE_SCALAR_KEYS = SCALAR_KEYS + 3 angle keys. The vectorized path
    splits the (n_regions × n_total_scalars × n_beats) materialized
    array into non-angle (first 34) and angle (last 3) sub-batches and
    calls ``_aggregate_batch`` once for each.

    When the entire plane is absent (``limb_mat`` / ``chest_mat`` None
    at runtime ⇒ ``per_beat_plane[i][plane] == {}`` for every beat),
    falls back to the empty-vals schema per cell — mirrors per-cell
    behaviour bit-for-bit.

    Returns: ``{plane ∈ {limb, chest}: {region: {scalar: {...}}}}``
    """
    n_beats = len(per_beat_plane)
    n_regions = len(WAVE_REGIONS)
    n_non_angle = len(SCALAR_KEYS)
    angle_keys = tuple(k for k in PLANE_SCALAR_KEYS if "angle" in k)
    assert PLANE_SCALAR_KEYS[:n_non_angle] == SCALAR_KEYS, (
        "PLANE_SCALAR_KEYS layout assumption violated"
    )

    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "limb": {}, "chest": {}
    }

    for plane in ("limb", "chest"):
        # Detect plane presence. ``measure_plane_beat`` leaves
        # ``out[plane] = {}`` (empty dict) when the matrix is None or
        # lead-count guard fails; otherwise fills all regions with
        # ``measure_plane_window`` / ``_empty_plane_window`` (both full
        # 37-key dicts). Plane presence is therefore constant across
        # beats — but check defensively in case a future caller passes
        # heterogeneous data.
        plane_present = (
            n_beats > 0 and
            all(bool(per_beat_plane[i].get(plane, {})) for i in range(n_beats))
        )

        if not plane_present:
            for region in WAVE_REGIONS:
                out[plane][region] = {}
                for key in PLANE_SCALAR_KEYS:
                    out[plane][region][key] = _pack_summary({}, (), 0)
                out[plane][region]["_morphology_alternans"] = _composite_alternans({
                    "max_p2": 0.0, "max_p3": 0.0, "max_p4": 0.0,
                    "max_p2_key": "", "max_p3_key": "", "max_p4_key": "",
                })
                cell_per_beat = [per_beat_plane[i].get(plane, {}).get(region, {})
                                  for i in range(n_beats)]
                out[plane][region]["_beat_clusters"] = _beat_cluster_summary(cell_per_beat)
            continue

        # ---- Materialize (n_regions × n_total_scalars × n_beats) ----
        n_total = len(PLANE_SCALAR_KEYS)
        vals_3d = np.zeros((n_regions, n_total, n_beats), dtype=np.float64)
        for bi in range(n_beats):
            cell_p = per_beat_plane[bi].get(plane, {})
            for ri, region in enumerate(WAVE_REGIONS):
                cell_r = cell_p.get(region, {})
                for si, key in enumerate(PLANE_SCALAR_KEYS):
                    v = cell_r.get(key)
                    vals_3d[ri, si, bi] = float(v) if v is not None else 0.0

        # ---- Two-batch dispatch: non-angle (34 keys) + angle (3 keys) ----
        stats_na = _aggregate_batch(vals_3d[:, :n_non_angle, :], is_angle=False)
        stats_ang = _aggregate_batch(vals_3d[:, n_non_angle:, :], is_angle=True)

        for ri, region in enumerate(WAVE_REGIONS):
            out[plane][region] = {}
            comp = {"max_p2": 0.0, "max_p3": 0.0, "max_p4": 0.0,
                    "max_p2_key": "", "max_p3_key": "", "max_p4_key": ""}
            for si, key in enumerate(SCALAR_KEYS):
                summary = _pack_summary(stats_na, (ri, si), n_beats)
                out[plane][region][key] = summary
                for p, suffix in (("p2", "alternation_index_p2"),
                                   ("p3", "alternation_index_p3"),
                                   ("p4", "alternation_index_p4")):
                    if summary[suffix] > comp[f"max_{p}"]:
                        comp[f"max_{p}"] = summary[suffix]
                        comp[f"max_{p}_key"] = key
            for ai, key in enumerate(angle_keys):
                summary = _pack_summary(stats_ang, (ri, ai), n_beats)
                out[plane][region][key] = summary
                for p, suffix in (("p2", "alternation_index_p2"),
                                   ("p3", "alternation_index_p3"),
                                   ("p4", "alternation_index_p4")):
                    if summary[suffix] > comp[f"max_{p}"]:
                        comp[f"max_{p}"] = summary[suffix]
                        comp[f"max_{p}_key"] = key
            out[plane][region]["_morphology_alternans"] = _composite_alternans(comp)
            cell_per_beat = [per_beat_plane[i].get(plane, {}).get(region, {})
                              for i in range(n_beats)]
            out[plane][region]["_beat_clusters"] = _beat_cluster_summary(cell_per_beat)
    return out


# ---------------------------------------------------------------------------
# Flatten to feature dict (xgb_batch_backend-style "ml::*" naming)
# ---------------------------------------------------------------------------

def flatten_features(agg: dict, prefix: str = "ml_direct") -> dict[str, float]:
    """Flatten the aggregation dict to {feature_name: value}.

    Schema:
        <prefix>::<key1>::<key2>::<scalar>::<stat>
    e.g.
        ml_direct::II::P::pos_peak_amp_mv::median        (per-lead)
        plane_direct::limb::QRS::peak_angle_deg::median   (per-plane)

    Non-numeric values (e.g. the ``alt_p2_driver`` string fields in the
    composite morphology-alternans summary) are skipped here. They are
    available in the unflattened ``agg`` for human inspection.
    """
    out: dict[str, float] = {}
    for k1, regions in agg.items():
        for region, scalars in regions.items():
            for key, summary in scalars.items():
                for stat, v in summary.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out[f"{prefix}::{k1}::{region}::{key}::{stat}"] = float(v)
                    elif isinstance(v, bool):
                        out[f"{prefix}::{k1}::{region}::{key}::{stat}"] = float(v)
                    # else (str, etc.) — silently skip
    return out
