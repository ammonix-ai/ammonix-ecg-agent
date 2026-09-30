"""
Beat processing -- filtering, plane slicing, averaging, RR summary.

Source: Cell 4 of q_psi_ai_for_ecg_Feb14_Adele.ipynb
Depends on: constants (PLANE_IDX), features.helpers (_get)
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
from typing import Any, Dict, List, Tuple

import numpy as np

from qpsi.constants import PLANE_IDX
from qpsi.features.helpers import _get

logger = logging.getLogger("qpsi")


# ---------------------------------------------------------------------------
# 1.  Filter beats whose length deviates > tolerance_ms from median
# ---------------------------------------------------------------------------

def filter_beats_by_length(
    beats: List[Any], *, tolerance_ms: int, fs: int
) -> List[Any]:
    """Remove beats whose RR-interval deviates more than *tolerance_ms* from
    the median beat length.

    Parameters
    ----------
    beats:
        List of beat objects (dataclass or dict).  Each must expose
        ``length_samples`` (via :func:`_get`).
    tolerance_ms:
        Maximum allowed deviation from the median length, in milliseconds.
    fs:
        Sampling frequency in Hz (used to convert samples -> ms).

    Returns
    -------
    List of beats that fall within the tolerance band.
    """
    if not beats:
        return []
    lengths = [_get(b, "length_samples") for b in beats]
    med_len = statistics.median_low(lengths)
    keep: List[Any] = []
    for b in beats:
        diff_ms = abs(_get(b, "length_samples") - med_len) / fs * 1_000
        if diff_ms <= tolerance_ms:
            keep.append(b)
    return keep


# ---------------------------------------------------------------------------
# 2.  Plane slicing (-> 6-lead dict list)
# ---------------------------------------------------------------------------

def slice_beats_to_plane(
    beats: List[Any], plane: str
) -> List[Dict[str, Any]]:
    """Extract a 6-lead subset (limb or chest) from each 12-lead beat.

    Returns a *new* list of dicts, each containing:

    * ``ecg_segment`` -- ``(6, N)`` ndarray for the requested plane
    * ``time_ms`` -- signed time axis (length *N*, 0 ms = R-peak)
    * ``length_samples`` -- number of samples in the segment

    All other original fields from the beat are preserved.

    Parameters
    ----------
    beats:
        List of beat objects (dataclass or dict).  Each must have an
        ``ecg_12`` array of shape ``(12, N)`` and a ``time_ms`` array.
    plane:
        ``"limb"`` (leads I--aVF, indices 0--5) or
        ``"chest"`` (leads V1--V6, indices 6--11).
    """
    if not beats:
        return []

    idx = PLANE_IDX[plane]
    out: List[Dict[str, Any]] = []
    for b in beats:
        base = dataclasses.asdict(b) if dataclasses.is_dataclass(b) else dict(b)
        seg6 = base["ecg_12"][idx, :].copy()
        d = base.copy()
        d["ecg_segment"] = seg6
        d["time_ms"] = base["time_ms"]           # unchanged
        d["length_samples"] = seg6.shape[1]
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# 3.  Build averaged 6-lead snippet
# ---------------------------------------------------------------------------

def build_average_snippet(
    beats: List[Dict[str, Any]], *, fs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-pad beats to a common span and return the mean 6-lead snippet.

    All beats **must** already share the same signed time-axis semantics
    (0 ms = R-peak).  Beats may differ in length; the routine zero-pads
    left and right so the concatenated matrix is ``(n_beats, 6, T)``.

    Parameters
    ----------
    beats:
        List of beat dicts, each with ``ecg_segment`` (6, N_i),
        ``time_ms`` (N_i,) where 0 ms marks the R-peak.
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    (avg6, time_ms)
        ``avg6`` has shape ``(6, T)``; ``time_ms`` has shape ``(T,)``
        with 0 ms aligned to the R-peak.

    Raises
    ------
    ValueError
        If *beats* is empty.
    """
    if not beats:
        raise ValueError("build_average_snippet: empty list")

    # --- global span -------------------------------------------------------
    left_ms = max(-b["time_ms"][0] for b in beats)
    right_ms = max(b["time_ms"][-1] for b in beats)
    L_left = int(np.ceil(left_ms * fs / 1_000))
    L_right = int(np.ceil(right_ms * fs / 1_000))
    T = L_left + L_right + 1
    time_ms = (np.arange(T) - L_left) / fs * 1_000

    # --- stack with zero-padding -------------------------------------------
    n_beats = len(beats)
    stack = np.zeros((n_beats, 6, T), dtype=float)
    for i, b in enumerate(beats):
        seg = b["ecg_segment"]
        zero = int(np.argmin(np.abs(b["time_ms"])))    # index of 0 ms
        start = L_left - zero
        stack[i, :, start : start + seg.shape[1]] = seg

    avg6 = np.mean(stack, axis=0)
    return avg6, time_ms


# ---------------------------------------------------------------------------
# 4.  rr_summary -- tiny helper
# ---------------------------------------------------------------------------

def rr_summary(beats: List[Any], *, fs: int) -> Dict[str, float]:
    """Compute mean and standard deviation of RR intervals in milliseconds.

    Parameters
    ----------
    beats:
        List of beat objects (dataclass or dict).
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    Dict with keys ``rr_mean_ms``, ``rr_sd_ms``, ``beats_used``.
    Returns NaN values and ``beats_used=0`` when *beats* is empty.
    """
    if not beats:
        return {"rr_mean_ms": np.nan, "rr_sd_ms": np.nan, "beats_used": 0}
    rr_ms = [(_get(b, "length_samples") / fs) * 1_000 for b in beats]
    return {
        "rr_mean_ms": float(np.mean(rr_ms)),
        "rr_sd_ms": float(np.std(rr_ms)),
        "beats_used": len(beats),
    }
