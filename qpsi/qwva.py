"""
Q-Wave Variability Analysis — RAT/RVT detection, DBSCAN clustering, PRD metrics.

RAT = Recurrent Atrial Transient (P-wave variants per beat).
RVT = Recurrent Ventricular Transient (T-wave variants per beat).
PRD = Periodic Repolarisation Dynamics (Lomb-Scargle of T-angle series).

Source: Cell 15 lines 514-540 (utility helpers), 782-1066 (QWVA core)
No internal dependencies. External: numpy, scipy.signal.lombscargle,
sklearn.cluster.DBSCAN.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import lombscargle
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Utility helpers (Cell 15 lines 514-540)
# ---------------------------------------------------------------------------

def _normalize_angle(deg: float) -> float:
    """Normalize angle to [-180, 180] range."""
    deg = deg % 360
    if deg > 180:
        deg -= 360
    return deg


def _amplitude(w: np.ndarray) -> float:
    """Peak-to-peak amplitude in mV (handles None / empty)."""
    return float(np.max(w) - np.min(w)) if w is not None and len(w) else 0.0


def _duration(
    idx_pair: Tuple[Optional[int], Optional[int]] | None,
    fs: int,
) -> float:
    """Return segment duration in ms; 0 if bounds are missing."""
    if idx_pair is None:
        return 0.0
    i0, i1 = idx_pair
    if i0 is None or i1 is None:
        return 0.0
    return 1000.0 * (i1 - i0) / fs


def _zero_crossings(w: np.ndarray) -> int:
    """Count zero-crossings in a waveform."""
    if w is None or len(w) == 0:
        return 0
    return int(np.sum(np.signbit(w[:-1]) != np.signbit(w[1:])))


def _peak_positions(w: np.ndarray, rel: float = 0.30) -> List[int]:
    """Lightweight local-max finder used for notch / split checks."""
    if w is None or len(w) < 5:
        return []
    pmax = np.max(w)
    thr = pmax * rel
    return [
        i for i in range(1, len(w) - 1)
        if w[i - 1] < w[i] > w[i + 1] and w[i] > thr
    ]


# ---------------------------------------------------------------------------
# Internal: RAT mechanism scoring
# ---------------------------------------------------------------------------

def _score_rat_mechanism(
    similarity: float,
    residual_frac: float,
    phase_var: float,
) -> str:
    """Return 'coexistent', 'ectopic' or 'indeterminate'.

    Args:
        similarity: Cosine-amplitude similarity to reference P.
        residual_frac: Fractional amplitude difference from reference.
        phase_var: Phase difference relative to RR interval.

    Returns:
        Mechanism classification string.
    """
    if similarity >= 0.85 and residual_frac >= 0.3 and phase_var > 0.4:
        return "coexistent"
    if similarity < 0.7 and residual_frac < 0.3 and phase_var < 0.3:
        return "ectopic"
    return "indeterminate"


# ---------------------------------------------------------------------------
# 1. RAT analysis (P-wave)
# ---------------------------------------------------------------------------

def analyse_rat_variants(
    beat_idx: int,
    ref_wave: Dict[str, Any],
    lumps_xy: List[Dict[str, Any]],
    fs: int,
    rr_ms: float,
) -> List[Dict[str, Any]]:
    """Collect P-wave variants and classify mechanism (coexistent vs ectopic).

    Args:
        beat_idx: Index of the current beat.
        ref_wave: Reference P-wave dict with amplitude_mV, peak_time,
                  angle_degrees.
        lumps_xy: List of LumpDicts for this beat in one plane.
        fs: Sampling rate in Hz.
        rr_ms: Current RR interval in ms.

    Returns:
        List of RAT event dicts.
    """
    rat_events: List[Dict[str, Any]] = []
    if not ref_wave:
        return rat_events

    ref_amp = ref_wave.get("amplitude_mV", 0.0)
    ref_time = ref_wave.get("peak_time", 0.0)
    ref_angle = _normalize_angle(float(ref_wave.get("angle_degrees", 0.0)))

    for w in [l for l in lumps_xy if "P" in l["wave_type"]]:
        amp = float(w.get("amplitude_mV", 0.0))
        width = float(abs(w.get("end_time", 0) - w.get("start_time", 0)))
        angle = _normalize_angle(float(w.get("angle_degrees", 0.0)))
        peak_t = float(w.get("peak_time", 0.0))

        # similarity to reference P (cosine of angle + amplitude ratio)
        amp_ratio = (amp + 1e-6) / (ref_amp + 1e-6)
        angle_diff = abs(angle - ref_angle)
        similarity = (
            np.exp(-abs(np.log(amp_ratio)))
            * np.cos(np.radians(angle_diff))
        )

        # residual energy proxy
        residual_frac = abs(amp - ref_amp) / (ref_amp + 1e-6)

        # phase difference relative to reference P within RR
        phase_rel_ms = peak_t - ref_time
        phase_var = abs(phase_rel_ms) / max(rr_ms, 1.0)

        mechanism = _score_rat_mechanism(similarity, residual_frac, phase_var)

        rat_events.append({
            "beat_index": beat_idx,
            "amp_mv": amp,
            "width_ms": width,
            "angle_deg": angle,
            "phase_rel_ms": round(phase_rel_ms, 2),
            "template_similarity": round(float(similarity), 3),
            "residual_fraction": round(residual_frac, 3),
            "mechanism": mechanism,
        })
    return rat_events


# ---------------------------------------------------------------------------
# 2. RVT analysis (T-wave)
# ---------------------------------------------------------------------------

def analyse_rvt_variants(
    beat_idx: int,
    ref_wave: Dict[str, Any],
    lumps_xy: List[Dict[str, Any]],
    fs: int,
    rr_ms: float,
) -> List[Dict[str, Any]]:
    """Collect T-wave variants (no subtraction).

    Args:
        beat_idx: Index of the current beat.
        ref_wave: Reference T-wave dict.
        lumps_xy: List of LumpDicts for this beat in one plane.
        fs: Sampling rate in Hz.
        rr_ms: Current RR interval in ms.

    Returns:
        List of RVT event dicts.
    """
    rvt_events: List[Dict[str, Any]] = []
    if not ref_wave:
        return rvt_events

    for w in [l for l in lumps_xy if "T" in l["wave_type"]]:
        amp = float(w.get("amplitude_mV", 0.0))
        width = float(abs(w.get("end_time", 0) - w.get("start_time", 0)))
        angle = _normalize_angle(float(w.get("angle_degrees", 0.0)))
        peak_t = float(w.get("peak_time", 0.0))
        rvt_events.append({
            "beat_index": beat_idx,
            "amp_mv": amp,
            "width_ms": width,
            "angle_deg": angle,
            "peak_time_ms": peak_t,
        })
    return rvt_events


# ---------------------------------------------------------------------------
# 3. Clustering of RAT / RVT events
# ---------------------------------------------------------------------------

def cluster_wave_variants(
    events: List[Dict[str, Any]],
    kind: str,
) -> List[Dict[str, Any]]:
    """Cluster RAT / RVT events by amplitude, width, angle using DBSCAN.

    Args:
        events: List of event dicts (must have amp_mv, width_ms, angle_deg).
        kind: Label prefix (e.g. "RAT_limb", "RVT_chest").

    Returns:
        List of cluster summary dicts.
    """
    clusters: List[Dict[str, Any]] = []
    if not events:
        return clusters

    X = np.array([[e["amp_mv"], e["width_ms"], e["angle_deg"]]
                   for e in events])
    if len(X) < 3:
        # too few for DBSCAN -> single cluster summary
        m = np.mean(X, axis=0)
        return [{
            "n_events": len(X),
            "mean_amp_mv": round(float(m[0]), 3),
            "mean_width_ms": round(float(m[1]), 2),
            "mean_angle_deg": round(float(m[2]), 2),
            "temporal_pattern": "single",
        }]

    db = DBSCAN(
        eps=0.15 * np.std(X, axis=0).mean(), min_samples=3
    ).fit(X)
    labels = db.labels_
    for label in sorted(set(labels)):
        idx = labels == label
        pts = X[idx]
        if pts.size == 0:
            continue
        mean = np.mean(pts, axis=0)
        clusters.append({
            "cluster_id": f"{kind.lower()}_{int(label)}",
            "n_events": int(idx.sum()),
            "mean_amp_mv": round(float(mean[0]), 3),
            "mean_width_ms": round(float(mean[1]), 2),
            "mean_angle_deg": round(float(mean[2]), 2),
        })
    return clusters


# ---------------------------------------------------------------------------
# 4. Temporal pattern analysis
# ---------------------------------------------------------------------------

def analyse_temporal_patterns(
    events: List[Dict[str, Any]],
    rr_series: List[float],
) -> Dict[str, Any]:
    """Analyse periodic or grouped repetition of variant events.

    Args:
        events: List of event dicts (must have beat_index).
        rr_series: RR intervals in ms.

    Returns:
        Dict with pattern, period_ms, score.
    """
    label = "random"
    period = None
    score = 0.0
    if not events or len(events) < 4:
        return {"pattern": label, "period_ms": period, "score": score}

    beat_idx = np.array([e["beat_index"] for e in events])
    if len(beat_idx) < 4:
        return {"pattern": label, "period_ms": period, "score": score}

    diffs = np.diff(beat_idx)
    if len(set(diffs)) == 1:
        label, score = "every_nth", 1.0
        period = float(np.mean(diffs) * np.mean(rr_series))
    else:
        ac = np.correlate(
            diffs - diffs.mean(), diffs - diffs.mean(), mode="full"
        )
        ac = ac[len(ac) // 2:]
        if len(ac) > 5 and np.max(ac[1:5]) > 0.5 * ac[0]:
            label, score = "grouped", float(np.max(ac[1:5]) / ac[0])
            period = float(np.argmax(ac[1:5]) * np.mean(rr_series))
    return {"pattern": label, "period_ms": period, "score": score}


# ---------------------------------------------------------------------------
# 5. PRD metrics (T-angle fluctuation)
# ---------------------------------------------------------------------------

def detect_prd_metrics(
    t_angle_series: List[float],
    rr_series: List[float],
) -> Dict[str, Any]:
    """Compute PRD power and dominant period from T-angle fluctuations.

    Uses Lomb-Scargle periodogram on the non-uniform beat-by-beat
    T-wave angle series, looking for periodic components in the
    0.04-0.15 Hz band (6.7-25 s).

    Args:
        t_angle_series: Per-beat T-wave angles in degrees.
        rr_series: RR intervals in ms.

    Returns:
        Dict with prd_power, prd_period_s, prd_score.
    """
    out: Dict[str, Any] = {
        "prd_power": 0.0, "prd_period_s": None, "prd_score": 0.0
    }

    if not t_angle_series or not rr_series:
        return out

    # Build nonuniform time axis (seconds)
    t = np.cumsum(np.asarray(rr_series, dtype=float)) / 1000.0
    t -= t[0]
    duration_s = float(t[-1]) if t.size else 0.0

    # Angles in radians, demeaned
    angles = np.asarray(t_angle_series, dtype=float)
    n_beats = int(min(len(angles), len(t)))
    if n_beats < 6 or duration_s < 6.0:
        return out

    t = t[:n_beats]
    angles = angles[:n_beats]
    angles_rad = np.radians(angles - np.mean(angles))

    # Frequency grid & band
    freqs = np.linspace(0.02, 0.25, 200)
    band_mask = (freqs >= 0.04) & (freqs <= 0.15)
    if not np.any(band_mask):
        return out

    # Lomb-Scargle (guarded)
    try:
        pxx = lombscargle(t, angles_rad, 2 * np.pi * freqs)
    except Exception:
        return out

    # Band-limited vectors
    freqs_band = freqs[band_mask]
    p_band = pxx[band_mask]
    if p_band.size < 3:
        return out

    # Band power and dominant period
    prd_power = float(np.trapezoid(p_band, freqs_band))
    peak_idx = int(np.argmax(p_band))
    peak_freq = float(freqs_band[peak_idx])
    prd_period = 1.0 / peak_freq if peak_freq > 0 else None

    # Base regularity score = peak / mean within band
    base_score = float(np.max(p_band) / (np.mean(p_band) + 1e-9))

    # Small-sample penalties
    # (A) Beat-count penalty (reference ~30 beats)
    beat_pen = min(1.0, n_beats / 30.0)

    # (B) Cycle-count penalty
    if prd_period is None or duration_s <= 0:
        cycle_pen = 0.0
    else:
        n_cycles = duration_s / prd_period
        cycle_pen = max(0.0, min(1.0, (n_cycles - 1.0) / 2.0))

    prd_score = base_score * beat_pen * cycle_pen

    # Suppress period if < ~2 cycles
    if cycle_pen < 0.25:
        prd_period = None

    return {
        "prd_power": round(prd_power, 5),
        "prd_period_s": (
            None if prd_period is None else round(prd_period, 2)
        ),
        "prd_score": round(prd_score, 2),
    }


# ---------------------------------------------------------------------------
# 6. Final summary bundler (separate limb / chest planes)
# ---------------------------------------------------------------------------

def summarize_qwva_results(
    rat_events_limb: List[Dict[str, Any]],
    rvt_events_limb: List[Dict[str, Any]],
    rat_events_chest: List[Dict[str, Any]],
    rvt_events_chest: List[Dict[str, Any]],
    rr_series: List[float],
) -> Dict[str, Any]:
    """Bundle all QWVA results for JSON export.

    Each plane is analysed independently so we avoid ambiguous
    pairing of multiple sub-waves within a beat.

    Args:
        rat_events_limb: RAT events from limb plane.
        rvt_events_limb: RVT events from limb plane.
        rat_events_chest: RAT events from chest plane.
        rvt_events_chest: RVT events from chest plane.
        rr_series: RR intervals in ms.

    Returns:
        Dict with per-plane cluster summaries and PRD metrics.
    """
    # --- Limb plane ---
    rat_clusters_limb = cluster_wave_variants(rat_events_limb, "RAT_limb")
    rvt_clusters_limb = cluster_wave_variants(rvt_events_limb, "RVT_limb")
    prd_limb = detect_prd_metrics(
        [e["angle_deg"] for e in rvt_events_limb], rr_series
    )

    # --- Chest plane ---
    rat_clusters_chest = cluster_wave_variants(rat_events_chest, "RAT_chest")
    rvt_clusters_chest = cluster_wave_variants(rvt_events_chest, "RVT_chest")
    prd_chest = detect_prd_metrics(
        [e["angle_deg"] for e in rvt_events_chest], rr_series
    )

    return {
        "RAT_clusters_limb": rat_clusters_limb,
        "RVT_clusters_limb": rvt_clusters_limb,
        "PRD_metrics_limb": prd_limb,
        "RAT_clusters_chest": rat_clusters_chest,
        "RVT_clusters_chest": rvt_clusters_chest,
        "PRD_metrics_chest": prd_chest,
    }
