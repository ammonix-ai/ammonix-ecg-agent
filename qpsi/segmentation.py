"""
Beat segmentation — R-peak to beat boundaries, padding, RR intervals,
and cross-correlation synchronisation.

Source: Cell 3 of q_psi_ai_for_ecg_Feb14_Adele.ipynb

Depends on:
    constants  — Beat, FS_HZ, PAD_HEAD_R_MS, PAD_TAIL_T_MS
    features.helpers — _get  (polymorphic accessor for Beat / dict)

Functions:
    robust_rr_from_rpeaks  — CANONICAL version (H31 / Q-A6-6 consolidated
                             from Cells 2, 3, 13; previously duplicated in
                             pipeline.py + features/rhythm.py with hardcoded
                             [240, 3000] no-MAD Cell 13 bounds. Now
                             parameterised so a single function serves both
                             contexts via kwargs).
    segment_rr             — Cut 12-lead ECG into Beat objects around R-peaks.
    pad_beats              — Zero-pad a list of beats to a common length.
    synchronise_beats      — Cross-correlation alignment (coarse + fine).
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.signal import correlate

from qpsi.constants import Beat, FS_HZ, PAD_HEAD_R_MS, PAD_TAIL_T_MS
from qpsi.features.helpers import _get

logger = logging.getLogger("qpsi")


# --------------------------------------------------------------------------- #
#  RR interval extraction (CANONICAL — H31 / Q-A6-6)                          #
# --------------------------------------------------------------------------- #

def robust_rr_from_rpeaks(
    r_peaks: Any,
    fs: int,
    *,
    min_ms: float = 300.0,
    max_ms: float = 2000.0,
    mad_trim: Optional[float] = 3.5,
) -> List[float]:
    """RR intervals (ms) from raw R-peak indices with light artifact trimming.

    Two-stage filter:
        1. Hard bounds — discard intervals outside [*min_ms*, *max_ms*].
        2. MAD trimming (skipped when *mad_trim* is ``None``) — discard intervals
           further than *mad_trim* scaled-MADs from the median (scale factor
           1.4826 converts MAD to sigma-equiv).

    Tolerant of ``None``/empty input (returns ``[]``) and ``fs == 0``
    (defaults to 500 Hz). Wrapped in ``try/except`` so any unexpected
    input shape returns ``[]`` rather than raising.

    Defaults match Cell 3's segment_rr context: ``[300, 2000]`` + MAD trim 3.5.
    Cell 13/17 (rhythm/pipeline) callers pass ``min_ms=240, max_ms=3000,
    mad_trim=None`` to opt into the wider bounds without MAD trimming
    (matching the notebook's runtime where Cell 13 shadows Cell 3).

    Args:
        r_peaks: Sample indices of detected R-peaks (ndarray, list, or None).
        fs: Sampling frequency in Hz (0 falls back to 500).
        min_ms: Minimum acceptable RR interval in milliseconds.
        max_ms: Maximum acceptable RR interval in milliseconds.
        mad_trim: MAD multiplier for outlier rejection, or ``None`` to skip.

    Returns:
        List of RR intervals in milliseconds after filtering.
    """
    try:
        r = np.asarray(r_peaks, dtype=float)
        if r.size < 2:
            return []

        rr = np.diff(r) * 1000.0 / float(fs if fs else 500)

        # Stage 1: hard bounds
        rr = rr[(rr >= min_ms) & (rr <= max_ms)]
        if rr.size == 0:
            return []

        # Stage 2: MAD trim (skipped when mad_trim is None)
        if mad_trim is not None:
            med = np.median(rr)
            mad = np.median(np.abs(rr - med)) + 1e-9
            keep = np.abs(rr - med) <= mad_trim * 1.4826 * mad
            rr = rr[keep]
        return rr.tolist()
    except Exception:
        return []


# --------------------------------------------------------------------------- #
#  Beat segmentation on a signed axis                                         #
# --------------------------------------------------------------------------- #

def segment_rr(
    ecg_bs: np.ndarray,
    r_peaks: np.ndarray,
    *,
    fs: int = FS_HZ,
) -> List[Beat]:
    """Cut a 12-lead ECG into per-beat segments centred on R-peaks.

    For every *interior* R-peak (i = 1 ... len-2) the segment runs from::

        mid(prev, this) - PAD_HEAD  ...  mid(this, next) + PAD_TAIL

    so each beat is flanked by guard padding and always has a left and right
    neighbour.  The signed time axis places 0 ms exactly at ``r_peaks[i]``.

    Args:
        ecg_bs: Baseline-subtracted ECG, shape ``(12, total_samples)``.
        r_peaks: R-peak sample indices.
        fs: Sampling frequency in Hz.

    Returns:
        List of :class:`Beat` objects (one per interior R-peak).
    """
    beats: List[Beat] = []
    n_tot = ecg_bs.shape[1]

    for i in range(1, len(r_peaks) - 1):
        r_prev, r_curr, r_next = map(int, r_peaks[i - 1 : i + 2])

        mid_l = (r_prev + r_curr) // 2
        mid_r = (r_curr + r_next) // 2

        start = max(0, int(mid_l - PAD_HEAD_R_MS * fs / 1_000))
        end = min(n_tot, int(mid_r + PAD_TAIL_T_MS * fs / 1_000))

        seg = ecg_bs[:, start:end]
        idx0 = r_curr - start  # sample index of R-peak within segment
        time = (np.arange(seg.shape[1]) - idx0) / fs * 1_000.0
        beats.append(Beat(seg, time))

    logger.debug("segment_rr: produced %d beats", len(beats))
    return beats


# --------------------------------------------------------------------------- #
#  Cross-correlation helpers                                                  #
# --------------------------------------------------------------------------- #

def _energy(lead_block: np.ndarray) -> np.ndarray:
    """Per-sample energy envelope = sum(|lead|) over limb leads only.

    Restricts the sum to rows 0-5 (I, II, III, aVR, aVL, aVF) so that both
    the coarse and fine synchronisation stages are driven by the limb-plane
    signal.  Chest-lead rows (6-11) are carried along by the same shift.

    Args:
        lead_block: Shape ``(n_leads, n_samples)`` array.

    Returns:
        1-D energy envelope, shape ``(n_samples,)``.
    """
    return np.sum(np.abs(lead_block[:6]), axis=0)


def _shift_with_zeros(arr: np.ndarray, shift: int) -> np.ndarray:
    """Roll *arr* along axis=1 but zero-fill the exposed edge instead of wrapping.

    Args:
        arr: 2-D array, shape ``(n_leads, n_samples)``.
        shift: Positive = shift left, negative = shift right.

    Returns:
        Shifted copy of *arr* (same shape).
    """
    if shift == 0:
        return arr
    rolled = np.roll(arr, -shift, axis=1)
    if shift > 0:  # moved left
        rolled[:, -shift:] = 0.0
    else:  # moved right
        rolled[:, :-shift] = 0.0
    return rolled


# --------------------------------------------------------------------------- #
#  Cross-correlation synchronisation (integer-shift, no resample)             #
# --------------------------------------------------------------------------- #

def synchronise_beats(
    beats: List[Beat],
    *,
    fs: int,
    win_ms: float = 60.0,
    peak_align_ms: float = 80.0,
) -> List[Beat]:
    """Align beats via two-stage cross-correlation.

    Stage 1 (coarse): align beats 1..N to beat 0 using limb-lead energy
    envelope cross-correlation within a +/- *win_ms* window.

    Stage 2 (fine): peak-centre *all* beats so the maximum energy sample
    sits at 0 ms, searching within +/- *peak_align_ms*.

    Alignment is by integer sample shift (no resampling). Exposed edges
    are zero-filled.

    Args:
        beats: List of :class:`Beat` objects (modified in-place).
        fs: Sampling frequency in Hz.
        win_ms: Coarse alignment window in ms (default 60).
        peak_align_ms: Fine peak-centring window in ms (default 80).

    Returns:
        The same list of beats (modified in-place).
    """
    if len(beats) < 2:
        return beats

    win = int(round(win_ms * fs / 1000))
    fine = int(round(peak_align_ms * fs / 1000))

    # ---------- reference envelope from beat 0 -------------------------
    ref = beats[0]
    cen_ref = int(np.argmin(np.abs(ref.time_ms)))
    sli_ref = slice(
        max(0, cen_ref - win),
        min(ref.ecg_12.shape[1], cen_ref + win + 1),
    )
    ref_en = _energy(ref.ecg_12[:, sli_ref])  # limb rows only

    # ---------- Stage 1: coarse alignment on beats 1... -----------------
    for beat in beats[1:]:
        cen = int(np.argmin(np.abs(beat.time_ms)))
        sli = slice(
            max(0, cen - win),
            min(beat.ecg_12.shape[1], cen + win + 1),
        )
        en = _energy(beat.ecg_12[:, sli])

        L = max(len(ref_en), len(en))
        lag = np.argmax(
            correlate(
                np.pad(en, (0, L - len(en))),
                np.pad(ref_en, (0, L - len(ref_en))),
                mode="full",
            )
        ) - (L - 1)
        lag = int(np.clip(lag, -win, win))

        if lag:
            beat.ecg_12 = _shift_with_zeros(beat.ecg_12, lag)
            beat.time_ms = beat.time_ms - lag / fs * 1000.0
            beat.synced_shift = getattr(beat, "synced_shift", 0) + lag

    # ---------- Stage 2: fine peak-centring on *all* beats --------------
    for beat in beats:
        cen = int(np.argmin(np.abs(beat.time_ms)))

        fine = int(round(peak_align_ms * fs / 1000))
        lo = max(0, cen - fine)
        hi = min(beat.ecg_12.shape[1] - 1, cen + fine)

        mags = _energy(beat.ecg_12[:, lo : hi + 1])
        idx_max = int(np.argmax(mags)) + lo
        diff = idx_max - cen

        if diff:  # roll without time-axis shift
            beat.ecg_12 = _shift_with_zeros(beat.ecg_12, diff)
            # keep time axis fixed -> 0 ms stays in the middle
            beat.synced_shift = getattr(beat, "synced_shift", 0) + diff

        # hard-reassert that the centre sample is 0 ms
        cen_new = int(np.argmin(np.abs(beat.time_ms)))
        beat.time_ms = (np.arange(beat.ecg_12.shape[1]) - cen_new) / fs * 1000.0

    return beats


# --------------------------------------------------------------------------- #
#  Common zero-padding (12- or 6-lead snippets)                               #
# --------------------------------------------------------------------------- #

def pad_beats(
    beats: List[Any],
    *,
    fs: int = FS_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-pad beats to a common length and stack into a 3-D array.

    Accepts :class:`Beat` dataclass objects **or** dicts produced by
    ``slice_beats_to_plane()`` (with ``"ecg_segment"`` and ``"time_ms"``
    keys).

    The common length is derived from the widest left/right extent across
    all beats.  Shorter beats are zero-padded symmetrically so that their
    0 ms point aligns.

    Args:
        beats: Non-empty list of Beat objects or dicts.
        fs: Sampling frequency in Hz.

    Returns:
        Tuple of:
            stack: ``(n_beats, n_leads, L)`` zero-padded array.
            time_ms: ``(L,)`` common time axis in ms, 0 = R-peak.

    Raises:
        ValueError: If *beats* is empty.
    """
    if not beats:
        raise ValueError("pad_beats: empty list")

    # Compute common extent from all beats
    left_ms = max(-_get(b, "time_ms")[0] for b in beats)
    right_ms = max(_get(b, "time_ms")[-1] for b in beats)
    L_left = int(math.ceil(left_ms * fs / 1_000))
    L_right = int(math.ceil(right_ms * fs / 1_000))
    L = L_left + L_right + 1
    time_ms = (np.arange(L) - L_left) / fs * 1_000.0

    # Derive lead count from first beat (6 or 12)
    first_seg = (
        _get(beats[0], "ecg_segment")
        if isinstance(beats[0], dict) and "ecg_segment" in beats[0]
        else _get(beats[0], "ecg_12")
    )
    n_leads = first_seg.shape[0]

    stack = np.zeros((len(beats), n_leads, L), dtype=float)

    for i, b in enumerate(beats):
        seg = (
            _get(b, "ecg_segment")
            if isinstance(b, dict) and "ecg_segment" in b
            else _get(b, "ecg_12")
        )
        zero = int(np.argmin(np.abs(_get(b, "time_ms"))))
        start = L_left - zero
        stack[i, :, start : start + seg.shape[1]] = seg

    return stack, time_ms
