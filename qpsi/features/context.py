"""
FeatureContext — the central data container passed to all 93 feature extractors.

Source: Cell 9 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (lines 29–84)
        + 3 new fields (meta, sex, extras) per extraction plan §4.9

No internal dependencies. External: numpy (type hints only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _f(x: Any) -> float:
    """Return a safe float (handles numpy scalars / 1-element arrays).

    Args:
        x: Any numeric value, numpy scalar, or 1-element array.

    Returns:
        float value, or 0.0 on failure.
    """
    try:
        a = np.asarray(x)
        return float(a.reshape(()))
    except Exception:
        try:
            return float(x)
        except Exception:
            return 0.0


@dataclass
class FeatureContext:
    """Central data container passed to all feature extractors.

    Four fields are required (avg_plane, segments, beat_plane, leads).
    All others default to sensible empty values.

    Args:
        avg_plane: {"limb": [lump_dict], "chest": [lump_dict]}
        segments: {"averaged": {"by_lead": {lead: {raw, signal, time_ms}}}, "raw": {...}}
        beat_plane: {"limb": {"P": [...], "QRS": [...], "T": [...]}, "chest": {...}}
        leads: ["I","II","III","aVR","aVL","aVF","V1"..."V6"]
        composites: Accumulator populated by extractors.
        params: {"HR": ..., "QT": ..., etc.}
        segment_bounds: Per-lead timing bounds.
        fs: Sampling rate, default 500.
        avg_lead: Per-lead averaged waveforms.
        beat_vectors: Per-beat 2D vectors.
        lead_vectors: Per-lead 2D vectors.
        rr_intervals: List of RR intervals in ms.
        lead_fits: {lead: {"avg": [Gaussian dicts], "by_beat": [...]}}
        residuals_by_lead: {lead: residual_array}
        r_peaks: R-peak sample indices.
        stack_sync: (nBeats, 12, T) aligned beat stack.
        avg_per_lead: (12, T) per-lead beat-axis mean of stack_sync.
        raw_ecg_12: (12, N) raw signal.
        time_ms: (T,) signed time axis.
        baseline_lead_fits: Baseline-corrected lead fits.
        meta: Patient metadata (age, sex, etc.).
        sex: Patient sex for QTc thresholds.
        extras: Catch-all for runtime state (rhythm_panorama, semantic_flags, quality).
    """
    __slots__ = (
        "avg_plane", "segments", "beat_plane", "leads", "composites",
        "params", "segment_bounds", "fs",
        "avg_lead", "beat_vectors", "lead_vectors", "rr_intervals",
        "lead_fits", "residuals_by_lead", "r_peaks",
        "stack_sync", "avg_per_lead", "raw_ecg_12", "time_ms",
        "baseline_lead_fits", "meta", "sex", "extras",
    )

    avg_plane: dict
    segments: dict
    beat_plane: dict
    leads: list
    composites: dict
    params: dict
    segment_bounds: dict
    fs: int
    avg_lead: dict
    beat_vectors: dict
    lead_vectors: dict
    rr_intervals: list
    lead_fits: dict
    residuals_by_lead: dict
    r_peaks: list
    stack_sync: Optional[np.ndarray]
    avg_per_lead: Optional[np.ndarray]
    raw_ecg_12: Optional[np.ndarray]
    time_ms: Optional[np.ndarray]
    baseline_lead_fits: Optional[dict]
    meta: Optional[dict]
    sex: Optional[str]
    extras: dict

    def __init__(
        self,
        *,
        avg_plane: dict,
        segments: dict,
        beat_plane: dict,
        leads: list,
        fs: int | None = None,
        params: dict | None = None,
        segment_bounds: dict | None = None,
        avg_lead: dict | None = None,
        beat_vectors: dict | None = None,
        lead_vectors: dict | None = None,
        rr_intervals: list | None = None,
        lead_fits: dict | None = None,
        residuals_by_lead: dict | None = None,
        r_peaks: Any = None,
        stack_sync: np.ndarray | None = None,
        avg_per_lead: np.ndarray | None = None,
        raw_ecg_12: np.ndarray | None = None,
        time_ms: np.ndarray | None = None,
        baseline_lead_fits: dict | None = None,
        meta: dict | None = None,
        sex: str | None = None,
        extras: dict | None = None,
    ) -> None:
        self.avg_plane = avg_plane
        self.segments = segments
        self.segment_bounds = segment_bounds or segments
        self.beat_plane = beat_plane
        self.leads = leads
        self.composites = {}
        self.params = params or {}
        self.fs = fs or 500
        self.avg_lead = avg_lead or {}
        self.beat_vectors = beat_vectors or {}
        self.lead_vectors = lead_vectors or {}
        self.rr_intervals = rr_intervals or []
        self.lead_fits = lead_fits or {}
        self.residuals_by_lead = residuals_by_lead or {}
        self.baseline_lead_fits = baseline_lead_fits
        self.r_peaks = (
            list(r_peaks) if isinstance(r_peaks, (list, tuple))
            else (r_peaks if r_peaks is not None else [])
        )
        self.stack_sync = stack_sync
        self.avg_per_lead = avg_per_lead
        self.raw_ecg_12 = raw_ecg_12
        self.time_ms = time_ms
        self.meta = meta
        self.sex = sex
        self.extras = extras or {}
