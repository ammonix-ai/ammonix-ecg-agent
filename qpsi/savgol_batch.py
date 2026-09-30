"""Batched scipy Savitzky-Golay smoothing helper.

This module exists as the Python-side infrastructure for Phase B.5's
``qpsi_native.measure_all_beats_full``. The Rust kernel does the per-
beat / per-lead / per-region orchestration with rayon parallelism but
**cannot run scipy savgol internally** — see ``FIDELITY_REPORT.md``
Phase H.C for the full investigation. Briefly:

  * The existing Rust ``seeded_fitter::savgol_filter_quadratic_pub``
    uses a closed-form quadratic SG coefficient formula that diverges
    from scipy's Vandermonde + ``np.linalg.lstsq`` (LAPACK ``dgelsd``)
    path by ~1 ULP per tap.
  * The pixi-env standalone ``liblapack.dll`` is a different LAPACK
    distribution than the OpenBLAS that scipy bundles in
    ``site-packages/scipy.libs/libscipy_openblas-*.dll``; calling
    ``dgelsd_`` against either drifts from scipy's bit pattern by 1-4
    ULP, and the scipy DLL filename has a hash suffix that changes
    per scipy build (brittle FFI target).
  * PyO3 callbacks from rayon workers serialise on the GIL, defeating
    the point of parallelisation.

The robust answer: keep savgol on the scipy side, pre-batch it once
per record in Python, then hand the smoothed batch to Rust as a
single PyO3 boundary crossing. This module is the batch helper.

API:
  * ``savgol_filter_batch(windows, window_length, polyorder)`` —
    returns a list of smoothed arrays, scipy-bit-identical to calling
    ``scipy.signal.savgol_filter`` on each window individually.
  * ``savgol_pack_batch(windows, window_length, polyorder)`` — same
    smoothing but returns ``(data_flat, offsets, lengths)`` so the
    Rust kernel can address arbitrary windows via slice indexing
    without dict marshalling overhead.

Bit-fidelity to ``scipy.signal.savgol_filter`` is locked by
``qpsi/test_savgol_batch.py``.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.signal import savgol_filter


def savgol_filter_batch(
    windows: Sequence[np.ndarray],
    window_length: int,
    polyorder: int = 2,
) -> list[np.ndarray]:
    """Apply scipy.signal.savgol_filter to each input in a tight loop.

    Each input array in ``windows`` must have length >= ``window_length``.
    The Rust crate isn't involved; this is a pure Python convenience
    that batches the scipy C-implemented savgol path.

    Bit-identical to:
        ``[scipy.signal.savgol_filter(y, window_length, polyorder)
            for y in windows]``
    """
    return [savgol_filter(y, window_length, polyorder) for y in windows]


def savgol_pack_batch(
    windows: Sequence[np.ndarray],
    window_length: int,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth + pack a heterogeneous batch of windows for Rust consumption.

    Returns:
        data_flat (n_total_samples,) float64 — concatenated smoothed signals
        offsets  (n_windows,) int64 — start index of each window's output
        lengths  (n_windows,) int64 — number of samples per window

    The Rust caller reconstructs window ``i``'s smoothed signal via:
        ``data_flat[offsets[i] : offsets[i] + lengths[i]]``

    Heterogeneous lengths (caused by variable RR per beat / variable
    region timing) are supported natively — there's no padding, no
    wasted memory.

    Bit-identical to ``savgol_filter_batch(windows, ...)`` when
    reassembled.
    """
    smoothed = [savgol_filter(y, window_length, polyorder) for y in windows]
    if not smoothed:
        empty = np.zeros(0, dtype=np.float64)
        empty_i = np.zeros(0, dtype=np.int64)
        return empty, empty_i, empty_i
    lengths = np.asarray([len(s) for s in smoothed], dtype=np.int64)
    offsets = np.empty(len(smoothed), dtype=np.int64)
    offsets[0] = 0
    if len(smoothed) > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    data_flat = np.ascontiguousarray(np.concatenate(smoothed), dtype=np.float64)
    return data_flat, offsets, lengths
