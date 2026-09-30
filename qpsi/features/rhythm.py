"""
Rhythm feature extractors -- 21 extractors + RhythmPanorama dataclass.

Source: Cell 13 (Cell 9A5) of q_psi_ai_for_ecg_Feb14_Adele.ipynb (2,275 lines)

Provides:
    ``RhythmPanorama``
        Dataclass holding windowed rhythm analysis results.

    ``rr_from_context(ctx)``
        Extract RR intervals prioritising raw R-peaks.

    ``RHYTHM_FEATURES``
        Registry list of the 21 extractor callables.

    Various helper functions used internally by the extractors.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass as _dataclass, field as _field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert
from scipy import stats
from scipy.stats import iqr
from numpy.fft import rfft, irfft, rfftfreq
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from qpsi.compat import trapezoid as _trapezoid

from qpsi.features.context import FeatureContext, _f
from qpsi.features.helpers import format_lead_list, _get, get_qrs_gaussians
from qpsi.features.p_wave import get_p_wave_gaussians
from qpsi.features.qrs_complex import qrs_morphology_clusters
from qpsi.segmentation import robust_rr_from_rpeaks as _robust_rr_canonical

# H31 / Q-A6-6: Cell 13 shadows Cell 3 with wider bounds [240, 3000] and NO MAD
# trimming. The canonical robust_rr_from_rpeaks now lives in qpsi.segmentation
# with kwargs (min_ms, max_ms, mad_trim) — call sites in this file pass the
# Cell-13 kwargs explicitly via robust_rr_from_rpeaks() below.


def robust_rr_from_rpeaks(r_peaks: Any, fs: int) -> list:
    """Cell 13/17 wrapper — wide bounds [240, 3000] ms, no MAD trimming.

    Thin wrapper around qpsi.segmentation.robust_rr_from_rpeaks with the
    rhythm-context kwargs locked in. Kept as a function (not deleted) so
    the in-file call sites and any external imports of
    ``qpsi.features.rhythm.robust_rr_from_rpeaks`` continue to resolve.
    """
    return _robust_rr_canonical(r_peaks, fs, min_ms=240.0, max_ms=3000.0, mad_trim=None)


# =============================================================================
# Rhythm Panorama -- 10-second global rhythm view + pattern tokens
# =============================================================================

# ---- 0) RR extraction + robust features ------------------------------------

def rr_from_context(ctx: FeatureContext) -> List[float]:
    """
    Extract RR intervals for rhythm analysis.
    CRITICAL: Prioritize raw R-peaks ('r_peaks') to capture arrhythmias.
    'ctx.rr_intervals' often contains only 'good' beats (filtered for morphology),
    which hides irregularity.
    """
    # 1. Try raw R-peaks first (The Truth)
    # FIX: Use explicit None check to avoid "Truth value of an array" error
    rpeaks = getattr(ctx, "r_peaks", None)
    if rpeaks is None:
        rpeaks = []

    fs = float(getattr(ctx, "fs", 500) or 500.0)

    if len(rpeaks) >= 3:
        rpeaks = np.asarray(rpeaks, dtype=float)
        rr_ms = np.diff(rpeaks) / fs * 1000.0
        rr_ms = rr_ms[np.isfinite(rr_ms)]
        # Filter physics-impossible values (e.g. < 200ms or > 4000ms)
        rr_ms = rr_ms[(rr_ms > 200) & (rr_ms < 4000)]
        if len(rr_ms) >= 3:
            return rr_ms.tolist()

    # 2. Fallback to pre-calculated intervals (The Subset)
    # FIX: Explicit None check
    rr = getattr(ctx, "rr_intervals", None)
    if rr is None:
        rr = []

    if len(rr) >= 3:
        return [float(x) for x in rr]

    return []


def _rr_features_robust(rr_intervals_ms: List[float]) -> Dict[str, Any]:
    """
    Basic time-domain features, robust to NaNs; also returns simple entropy proxy.
    """
    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < 3:
        return {"mean_ms": None, "sdnn_ms": None, "rmssd_ms": None, "pnn50": None, "cv": None, "rr_entropy": None}
    diffs = np.diff(rr)
    absdiff = np.abs(diffs)
    mean_ms = float(rr.mean())
    sdnn_ms = float(rr.std())
    rmssd_ms = float(np.sqrt(np.mean(diffs**2)))
    pnn50 = float((absdiff >= 50.0).sum()) / max(1, absdiff.size) * 100.0
    cv = (sdnn_ms / mean_ms) if mean_ms else None

    # Simple Shannon entropy of the RR histogram (proxy for irregularity)
    try:
        hist, _ = np.histogram(rr, bins=min(20, max(5, rr.size // 5)))
        p = hist.astype(float); p = p / (p.sum() + 1e-12)
        rr_entropy = float(-np.nansum(p * np.log(p + 1e-12)))
    except Exception:
        rr_entropy = None

    return {
        "mean_ms": mean_ms, "sdnn_ms": sdnn_ms, "rmssd_ms": rmssd_ms,
        "pnn50": pnn50, "cv": cv, "rr_entropy": rr_entropy
    }

# ---- 1) Lightweight 1-D K-Means (k=2/3) for RR modes -----------------------

def _kmeans_1d(x: np.ndarray, k: int, n_init: int = 6, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Tiny 1-D k-means over RR values.
    Returns labels, centers(sorted), inertia.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    rng = np.random.default_rng(12345)
    if n < k:
        return np.zeros(n, dtype=int), np.array([np.mean(x)]), 0.0
    # smart inits: percentiles + random jitters
    pct = np.linspace(0, 100, k+2)[1:-1]
    init_centers = np.percentile(x, pct).reshape(-1, 1)
    candidates = [init_centers]
    for _ in range(n_init-1):
        idx = rng.choice(n, size=k, replace=False)
        candidates.append(x[idx])
    for C0 in candidates:
        C = C0.copy()
        for _ in range(max_iter):
            # assign
            d = np.abs(x - C.T)  # n x k
            labels = np.argmin(d, axis=1)
            # update
            newC = np.vstack([x[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(np.sum((x - C[labels])**2))
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels.copy(), C.copy()
    # sort by center
    order = np.argsort(best_centers[:, 0])
    maplbl = {old: new for new, old in enumerate(order)}
    labels_sorted = np.array([maplbl[l] for l in best_labels], dtype=int)
    centers_sorted = best_centers[order, 0]
    return labels_sorted, centers_sorted, best_inertia


def _cluster_rr_values(rr_ms: np.ndarray) -> Dict[str, Any]:
    """
    Try k=2 and k=3. Compute simple separation metric: min gap / max within-SD.
    """
    out: Dict[str, Any] = {"k": 1, "centers_ms": [np.mean(rr_ms)], "labels": np.zeros(len(rr_ms), int), "separation": 0.0}
    if rr_ms.size < 6:
        return out

    best = out
    for k in (2, 3):
        labels, centers, _ = _kmeans_1d(rr_ms, k)
        # within SD per cluster
        wsd = []
        for j in range(k):
            s = rr_ms[labels == j]
            wsd.append(s.std() if s.size > 1 else 1e-6)
        wsd_max = float(max(wsd) if wsd else 1e-6)
        centers = np.sort(centers)
        gaps = np.diff(centers)
        sep = float(gaps.min() / (wsd_max + 1e-9)) if gaps.size else 0.0
        # pick the model with highest separation
        if sep > best.get("separation", 0.0):
            best = {
                "k": k,
                "centers_ms": centers.tolist(),
                "labels": labels,
                "separation": sep
            }
    return best

# ---- 2) Sliding windows over 10 s with steps -------------------------------

def _windows_over_time(rr_ms: np.ndarray, win_sec: float, step_sec: float, fs: float = 1000.0) -> List[Tuple[int, int]]:
    """
    Produce index windows over a tachogram timeline. RR array in ms.
    Returns list of (i_start, i_end) indices (inclusive start, exclusive end) into rr_ms.
    """
    if rr_ms.size == 0:
        return []
    # Build cumulative time (ms) aligned to RR boundaries
    t = np.concatenate(([0.0], np.cumsum(rr_ms)))  # beat times (ms)
    total_ms = t[-1]
    win_ms, step_ms = win_sec * 1000.0, step_sec * 1000.0
    starts = np.arange(0.0, max(1.0, total_ms - win_ms + 1.0), step_ms)
    spans: List[Tuple[int, int]] = []
    for s in starts:
        e = s + win_ms
        # indices of RR intervals fully inside [s, e]
        # rr[i] spans t[i]..t[i+1]; include if both endpoints inside
        i0 = np.searchsorted(t, s, side='left')
        i1 = np.searchsorted(t, e, side='right') - 1
        # intervals are 0..n-1; valid if i1 - 1 >= i0
        lo = max(0, i0)
        hi = min(rr_ms.size, max(i0, i1 - 1))
        if hi - lo >= 4:  # need at least ~5 beats to be meaningful
            spans.append((lo, hi))
    return spans


def rhythm_window_scan(rr_ms: np.ndarray, win_sec: float = 10.0, step_sec: float = 2.0) -> List[Dict[str, Any]]:
    """
    Slide 10-s windows and compute RR features + clustering per window.
    """
    rr = np.asarray(rr_ms, float)
    out: List[Dict[str, Any]] = []
    spans = _windows_over_time(rr, win_sec, step_sec)
    for (i0, i1) in spans:
        seg = rr[i0:i1]
        feats = _rr_features_robust(seg)
        clus = _cluster_rr_values(seg)
        out.append({
            "i0": int(i0), "i1": int(i1),
            "n": int(len(seg)),
            "features": feats,
            "cluster": {"k": int(clus["k"]), "centers_ms": list(map(float, clus["centers_ms"])),
                        "labels": clus["labels"].tolist() if isinstance(clus["labels"], np.ndarray) else [int(x) for x in clus["labels"]],
                        "separation": float(clus["separation"])},
        })
    return out

# ---- 3) Pattern detectors (bigeminy/trigeminy/flutter/AF/ectopy) -----------

def _is_regular_window(feats: Dict[str, Any], cv_thr: float = 0.08, pnn_thr: float = 10.0) -> bool:
    # FIX: Treat missing data (None) as NOT regular
    cv = feats.get("cv")
    if cv is None:
        return False
    pnn50 = feats.get("pnn50") or 0.0
    return (cv <= cv_thr) and (pnn50 <= pnn_thr)

def _is_irregular_window(feats: Dict[str, Any], cv_thr: float = 0.12, pnn_thr: float = 20.0) -> bool:
    cv = feats.get("cv")
    if cv is None:
        return False  # Don't count insufficient data as irregular either
    pnn50 = feats.get("pnn50") or 0.0
    return (cv >= cv_thr) or (pnn50 >= pnn_thr)

def _detect_bige_trige(labels: np.ndarray) -> Optional[str]:
    """
    Simple alternans pattern on labels (for k=2): ABAB... -> 'BIGEM'
    For k=3: repeating ABC pattern -> 'TRIGEM'
    """
    if labels.size < 6:
        return None
    # normalize labels to consecutive integers starting at 0
    labs = labels.astype(int)
    # bigeminy heuristic (k=2)
    if np.unique(labs).size == 2:
        # Count alternations
        alt = np.mean(labs[1:] != labs[:-1])
        if alt > 0.8:
            return "BIGEM"
    # trigeminy heuristic (k=3)
    if np.unique(labs).size == 3:
        # 0,1,2 repeating?
        seq = labs[:6] % 3
        if np.all(seq == np.array([0, 1, 2, 0, 1, 2])):
            return "TRIGEM"
    return None

def _detect_flutter_ratio(centers_ms: List[float], tol: float = 0.15) -> Optional[str]:
    """
    Given sorted RR centers (k=2 or 3), check if ratios ~2:1 or ~3:1.
    Returns 'FLT2' or 'FLT3' or None.
    """
    c = np.sort(np.asarray(centers_ms, float))
    if c.size == 2:
        r = c.max() / c.min()
        if abs(r - 2.0) <= tol * 2.0:
            return "FLT2"
    if c.size == 3:
        r = c.max() / c.min()
        if abs(r - 3.0) <= tol * 3.0:
            return "FLT3"
    return None

def _detect_ectopy(rr_ms: np.ndarray) -> Dict[str, Any]:
    """
    Detect premature beats by short-long coupling (very simple heuristic).
    Returns counts and indices.
    """
    rr = np.asarray(rr_ms, float)
    if rr.size < 5:
        return {"pvcs": 0, "pacs": 0, "pvcs_idx": [], "pacs_idx": []}
    med = np.median(rr)
    short = rr < 0.8 * med
    long_after = np.r_[False, rr[1:] > 1.2 * med]
    pvcs_idx = np.where(short & long_after)[0].tolist()
    # PACs often less pronounced: 0.9*med with small compensatory pause
    short_pac = (rr < 0.9 * med) & (rr >= 0.8 * med)
    small_pause = np.r_[False, rr[1:] > 1.05 * med]
    pacs_idx = np.where(short_pac & small_pause)[0].tolist()
    return {"pvcs": len(pvcs_idx), "pacs": len(pacs_idx),
            "pvcs_idx": pvcs_idx, "pacs_idx": pacs_idx}

# ---- 4) Panorama container + builder ---------------------------------------

@_dataclass
class RhythmPanorama:
    rr_ms: List[float]
    windows: List[Dict[str, Any]]
    tokens: List[str] = _field(default_factory=list)
    regular_fraction: float = 0.0
    irregular_fraction: float = 0.0
    ectopy: Dict[str, Any] = _field(default_factory=dict)
    dominant_rate_bpm: Optional[float] = None
    notes: List[str] = _field(default_factory=list)
    # Extended fields for direct feature support
    cv_global: float = 0.0
    n_clusters_global: int = 1
    separation_global: float = 0.0


def rhythm_panorama_from_ctx(ctx: FeatureContext, plot_enabled: bool = False) -> RhythmPanorama:
    """
    Build a 10-s windowed panorama and infer compact tokens:
      STA (sinus-like tachy if regular & fast), LRR (low-rate regular), REG, HIR,
      BIGEM/TRIGEM, FLT2/FLT3 (conduction patterns), AAR (absolute arrhythmia).
    """
    rr = np.asarray(rr_from_context(ctx), float)
    win = rhythm_window_scan(rr, win_sec=10.0, step_sec=2.0)

    # Global stats
    if rr.size > 1:
        cv_global = float(np.std(rr) / (np.mean(rr) + 1e-9))
    else:
        cv_global = 0.0

    if not win:
        return RhythmPanorama(rr_ms=rr.tolist(), windows=[], tokens=[], notes=["insufficient data"], cv_global=cv_global)

    # fractions of regular vs irregular windows
    reg_mask = np.array([_is_regular_window(w["features"]) for w in win], dtype=bool)
    irr_mask = np.array([_is_irregular_window(w["features"]) for w in win], dtype=bool)

    # Valid windows count (avoid divide by zero if all windows invalid)
    n_valid = len(win)
    regular_fraction = float(reg_mask.sum()) / n_valid if n_valid > 0 else 0.0
    irregular_fraction = float(irr_mask.sum()) / n_valid if n_valid > 0 else 0.0

    # global clustering on full RR
    full_clus = _cluster_rr_values(rr)
    bigtri = _detect_bige_trige(np.array(full_clus["labels"]))
    flt = _detect_flutter_ratio(full_clus["centers_ms"])
    ect = _detect_ectopy(rr)

    # dominant rate
    mean_rr = rr.mean() if rr.size else np.nan
    rate_bpm = 60000.0 / mean_rr if mean_rr and np.isfinite(mean_rr) else None

    tokens: List[str] = []
    # AAR vs REG/HIR
    if irregular_fraction >= 0.60 and full_clus["k"] == 1:
        tokens.append("AAR")  # absolute arrhythmia signature (diffuse irregular, 1 mode)
    elif irregular_fraction >= 0.50 and full_clus["k"] >= 2 and full_clus["separation"] < 1.5:
        tokens.append("HIR")  # highly irregular, multi-modal but poorly separated
    elif regular_fraction >= 0.60:
        tokens.append("REG")

    # rate tags (if regular dominates)
    if rate_bpm:
        if regular_fraction >= 0.60 and rate_bpm >= 100:
            tokens.append("STA")  # sinus-like tachy panorama
        elif regular_fraction >= 0.60 and rate_bpm < 60:
            tokens.append("LRR")  # low-rate regular

    # conduction-ratio hints (flutter patterns) or ectopy runs
    if flt:
        tokens.append(flt)  # FLT2 / FLT3
    if bigtri:
        tokens.append(bigtri)

    # ectopy presence tags
    if ect.get("pvcs", 0) >= 3:
        tokens.append("PVCx")
    if ect.get("pacs", 0) >= 3:
        tokens.append("PACx")

    pano = RhythmPanorama(
        rr_ms=rr.tolist(),
        windows=win,
        tokens=tokens,
        regular_fraction=regular_fraction,
        irregular_fraction=irregular_fraction,
        ectopy=ect,
        dominant_rate_bpm=float(rate_bpm) if rate_bpm is not None else None,
        notes=[
            f"k={full_clus['k']}, centers={[np.round(c, 1) for c in full_clus['centers_ms']]}, sep={full_clus['separation']:.2f}"
        ],
        cv_global=cv_global,
        n_clusters_global=int(full_clus["k"]),
        separation_global=float(full_clus["separation"])
    )

    return pano


# ---- 6) Emit flags + concise summary ---------------------------------------
# ---- Helper to ensure panorama exists ------------------------------------
def _get_panorama(ctx: FeatureContext) -> RhythmPanorama:
    """Ensure we have a panorama object, computing it if missing."""
    # FeatureContext uses __slots__; use extras dict for caching
    extras = getattr(ctx, "extras", None)
    if extras is not None and isinstance(extras, dict):
        p = extras.get("rhythm_panorama")
        if p is not None and isinstance(p, RhythmPanorama):
            return p

    # Also check direct attribute (legacy path)
    p = getattr(ctx, "rhythm_panorama", None)
    if p is not None and isinstance(p, RhythmPanorama):
        return p

    # Compute on the fly if missing (fast)
    p = rhythm_panorama_from_ctx(ctx, plot_enabled=False)
    # Cache it in extras if possible
    if extras is not None and isinstance(extras, dict):
        extras["rhythm_panorama"] = p
    return p


# ============================================================================
# HELPER FUNCTIONS FOR RHYTHM ANALYSIS
# ============================================================================

# H30 / Q-A6-12: get_qrs_gaussians imported from qpsi.features.helpers (canonical).


def calculate_qrs_duration(lead_fits: Dict, lead: str) -> Optional[float]:
    """Calculate QRS duration from Gaussian parameters"""
    qrs_gaussians = get_qrs_gaussians(lead_fits, lead)
    if not qrs_gaussians:
        return None

    start = min(g["center_ms"] - 2*g["sigma_ms"] for g in qrs_gaussians)
    end = max(g["center_ms"] + 2*g["sigma_ms"] for g in qrs_gaussians)
    return end - start


def _rr_features(rr_intervals_ms: Any) -> Dict[str, Any]:
    """
    Basic time-domain RR features (robust to outliers).
    Returns: {mean_ms, sdnn_ms, rmssd_ms, pnn50, cv}
    """
    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < 3:
        return {"mean_ms": None, "sdnn_ms": None, "rmssd_ms": None, "pnn50": None, "cv": None}

    # mild artifact-trim if long enough
    if rr.size >= 20:
        lo, hi = np.percentile(rr, [2.5, 97.5])
        rr = rr[(rr >= lo) & (rr <= hi)]

    diffs   = np.diff(rr)
    absdiff = np.abs(diffs)
    mean_ms = float(rr.mean())
    sdnn_ms = float(rr.std())
    rmssd_ms = float(np.sqrt(np.mean(diffs**2))) if diffs.size else None
    pnn50   = float((absdiff >= 50.0).sum()) / max(1, absdiff.size) * 100.0 if absdiff.size else None
    cv      = (sdnn_ms / mean_ms) if (mean_ms and sdnn_ms is not None) else None
    return {"mean_ms": mean_ms, "sdnn_ms": sdnn_ms, "rmssd_ms": rmssd_ms, "pnn50": pnn50, "cv": cv}

def _rr_hist_peaks(rr_ms: Any, bin_ms: int = 20, rel_height: float = 0.35, min_count: int = 3) -> Tuple[list, list, Any]:
    """
    Histogram peak finder for RR intervals.
    Returns (centers_ms, counts, hist) where 'centers_ms' are the peak bin centers.
    """
    rr = np.asarray(rr_ms, float)
    rr = rr[np.isfinite(rr)]
    if rr.size < 3 or rr.max() <= rr.min():
        return [], [], None

    bins = np.arange(rr.min(), rr.max() + bin_ms, bin_ms)
    hist, edges = np.histogram(rr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    if not np.any(hist):
        return [], [], (hist, edges)

    # relative height finds "modes" robustly
    pk, _ = find_peaks(hist, height=max(1, int(rel_height * np.max(hist))))
    pk = [i for i in pk if hist[i] >= min_count]
    return centers[pk].tolist(), hist[pk].tolist(), (hist, edges)


def _assign_rr_to_peaks(rr_ms: Any, centers_ms: list, tol_ms: float = 25) -> Tuple[list, float]:
    """
    Assign each RR to nearest center within tolerance and return:
      (cluster_stds_ms, coverage_fraction)
    """
    rr = np.asarray(rr_ms, float)
    rr = rr[np.isfinite(rr)]
    n = rr.size
    if n == 0 or not centers_ms:
        return [], 0.0

    C = np.asarray(centers_ms, float)
    d = np.abs(rr[:, None] - C[None, :])
    nearest = np.argmin(d, axis=1)
    ok = d[np.arange(n), nearest] <= tol_ms
    labels = np.where(ok, nearest, -1)

    stds: list = []
    for k in range(len(C)):
        vals = rr[labels == k]
        stds.append(float(vals.std()) if vals.size > 1 else None)

    coverage = float((labels >= 0).sum()) / float(n)
    return stds, coverage


def analyze_rr_signature(rr_ms: Any, *, mean_rr_guard_ms: float = 280, min_points: int = 12) -> Dict[str, Any]:
    """
    Decide 'random_scatter' (AF-like) vs 'organized_av_conduction' (flutter-like)
    using RR only.  Also returns labels for plotting.
    """
    rr = np.asarray(rr_ms, float)
    rr = rr[np.isfinite(rr)]
    n = rr.size
    if n < min_points:
        return {"enough_data": False}

    feats = _rr_features(rr)  # mean_ms, sdnn_ms, rmssd_ms, pnn50, cv
    mean_rr = feats.get("mean_ms") or 0.0
    if mean_rr < mean_rr_guard_ms:   # avoid extremely fast/short segments
        return {"enough_data": False}

    cv    = feats.get("cv") or 0.0
    rmssd = feats.get("rmssd_ms") or 0.0
    pnn50 = feats.get("pnn50") or 0.0

    # Adaptive binning/tolerance vs mean RR
    bin_ms = int(max(15.0, 0.04 * mean_rr))          # ~4% of mean RR
    tol_ms = float(max(20.0, 0.06 * mean_rr))        # +/-6% window

    centers, counts, _ = _rr_hist_peaks(rr, bin_ms=bin_ms, rel_height=0.35, min_count=3)

    # Assignments & coverage
    labels = np.full(n, -1, int)
    if centers:
        C = np.asarray(centers, float)
        d = np.abs(rr[:, None] - C[None, :])
        j = np.argmin(d, axis=1)
        mask = d[np.arange(n), j] <= tol_ms
        labels[mask] = j[mask]
    coverage = float((labels >= 0).sum()) / float(n)

    # Tightness of clusters
    stds, _ = _assign_rr_to_peaks(rr, centers, tol_ms=tol_ms)
    tight = [s for s in stds if s is not None and s <= 0.5 * tol_ms]  # tighter than tol
    tight_frac = (len(tight) / max(1, len(centers))) if centers else 0.0

    # Transition dominance in RR return map
    trans_dom = 0.0
    if n >= 2:
        pairs = [(labels[i], labels[i+1]) for i in range(n-1)
                 if labels[i] >= 0 and labels[i+1] >= 0]
        if pairs:
            cnt = Counter(pairs)
            top2 = cnt.most_common(2)
            trans_dom = sum(c for _, c in top2) / float(len(pairs))

    n_peaks = len(centers)

    # AF-like "random" conditions (more permissive)
    randomlike = (
        (n_peaks == 0) or
        (n_peaks >= 4) or
        (coverage < 0.65) or
        (trans_dom < 0.45) or
        ((cv >= 0.12 or pnn50 >= 20.0) and rmssd >= 40.0)
    )

    # Flutter-like organized conditions (stricter)
    organized = (
        (n_peaks in (2, 3)) and
        (tight_frac >= 0.60) and
        (coverage >= 0.75) and
        (trans_dom >= 0.60)
    )

    pattern = "organized_av_conduction" if organized and not randomlike else \
              ("random_scatter" if randomlike else "undetermined")

    ratios: list = []
    if organized and n_peaks >= 2:
        cs = sorted(centers)
        base = cs[0]
        for c in cs[1:]:
            if base > 0:
                ratios.append(round(c / base, 2))

    return {
        "enough_data": True,
        "cv": cv, "rmssd_ms": rmssd, "mean_rr_ms": mean_rr, "pnn50": pnn50,
        "n_peaks": n_peaks, "centers_ms": centers, "cluster_stds_ms": stds,
        "coverage": coverage, "tight_frac": tight_frac, "trans_dom": trans_dom,
        "pattern": pattern, "ratios_approx": ratios,
        "labels": labels.tolist(), "tol_ms": tol_ms, "bin_ms": bin_ms,
    }


# ============================================================================
# Integer Ratio Prevalence (Metric for Flutter)
# ============================================================================
def _calculate_integer_ratio_prevalence(rr_ms: Any, tolerance: float = 0.1) -> float:
    """
    Calculates the percentage of RR intervals that are integer multiples (or simple halves)
    of the dominant short-interval mode.
    High values (>30%) suggest Variable Block Flutter (regular irregularity).
    Low values suggest AFib (random irregularity).
    """
    if len(rr_ms) < 10:
        return 0.0

    # Find the "base" cycle (shortest significant mode)
    # Use basic histogram peaks
    hist, edges = np.histogram(rr_ms, bins=int(len(rr_ms)/3))
    centers = (edges[:-1] + edges[1:]) / 2

    # Smooth slightly
    hist_smooth = gaussian_filter1d(hist, sigma=1)

    peaks, _ = find_peaks(hist_smooth, height=max(hist_smooth)*0.2)
    if len(peaks) == 0:
        base_cycle = float(np.median(rr_ms))
    else:
        # Take the shortest peak that isn't noise (e.g., > 200ms)
        peak_centers = centers[peaks]
        valid_peaks = [p for p in peak_centers if p > 200]
        base_cycle = float(min(valid_peaks)) if valid_peaks else float(np.median(rr_ms))

    # Count alignment
    aligned_count = 0
    for rr_val in rr_ms:
        ratio = rr_val / base_cycle
        # Check closeness to 1.0, 1.5, 2.0, 3.0, 4.0 within tolerance
        # 1.5 handles 3:2 Wenckebach-like patterns in flutter
        if (abs(ratio - 1.0) < tolerance) or \
           (abs(ratio - 1.5) < tolerance) or \
           (abs(ratio - 2.0) < tolerance) or \
           (abs(ratio - 3.0) < tolerance) or \
           (abs(ratio - 4.0) < tolerance):
            aligned_count += 1

    return float(aligned_count) / len(rr_ms)


# Helper function to ensure JSON serializable output
def ensure_json_serializable(obj: Any) -> Any:
    """
    Convert common non-JSON Python/NumPy/pandas/PyTorch types to JSON-safe
    Python natives (str, int, float, bool, None, list, dict).
    Robust across NumPy versions where certain aliases (e.g., np.bool8) may not exist.
    """
    # Fast-path primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        # Replace non-finite floats with None to keep strict JSON
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    # Optional imports (kept inside to avoid hard deps)
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        import torch
    except Exception:
        torch = None

    # ------------------------
    # NumPy handling
    # ------------------------
    # Scalars (bool, int, float, complex)
    np_bool_ = getattr(np, "bool_", None)
    if np_bool_ is not None and isinstance(obj, np_bool_):
        return bool(obj)

    np_integer = getattr(np, "integer", None)
    if np_integer is not None and isinstance(obj, np_integer):
        return int(obj)

    np_floating = getattr(np, "floating", None)
    if np_floating is not None and isinstance(obj, np_floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val

    np_complexfloating = getattr(np, "complexfloating", None)
    if np_complexfloating is not None and isinstance(obj, np_complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}

    # Arrays
    if hasattr(np, "ndarray") and isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            # Zero-d array -> scalar
            return ensure_json_serializable(obj.item())
        # Regular array -> list (recursively sanitize elements)
        return [ensure_json_serializable(x) for x in obj.tolist()]

    # Void / structured dtypes -> None
    np_void = getattr(np, "void", None)
    if np_void is not None and isinstance(obj, np_void):
        return None

    # ------------------------
    # pandas handling
    # ------------------------
    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(pd, "Timedelta") and isinstance(obj, pd.Timedelta):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return [ensure_json_serializable(x) for x in obj.to_list()]
        if isinstance(obj, pd.DataFrame):
            return [ensure_json_serializable(rec) for rec in obj.to_dict("records")]

    # ------------------------
    # PyTorch handling (optional)
    # ------------------------
    if torch is not None:
        if isinstance(obj, getattr(torch, "Tensor", ())):
            if obj.ndim == 0:
                return ensure_json_serializable(obj.item())
            return [ensure_json_serializable(x) for x in obj.detach().cpu().tolist()]

    # ------------------------
    # Bytes-like
    # ------------------------
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj).decode("utf-8", errors="replace")

    # ------------------------
    # Collections
    # ------------------------
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = ensure_json_serializable(k)
            vs = ensure_json_serializable(v)
            # JSON object keys must be strings; coerce if needed
            if not isinstance(ks, str):
                ks = "null" if ks is None else str(ks)
            out[ks] = vs
        return out

    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(x) for x in obj]

    if isinstance(obj, set):
        # Sets are unordered; convert to list
        return [ensure_json_serializable(x) for x in list(obj)]

    # ------------------------
    # Fallback
    # ------------------------
    # Last resort: string representation (prevents crashes, keeps pipeline flowing)
    return str(obj)


# ============================================================================
# RHYTHM FEATURE EXTRACTORS (21 public functions)
# ============================================================================

def rhythm_extrasystole_burden(ctx: FeatureContext) -> dict:
    """
    Summarize ectopy using QRS cluster minority + RR timing.
    Uses Panorama to skip if rhythm is organized-variable (flutter).
    """
    # Get panorama
    pano = _get_panorama(ctx)

    # Guard: if panorama says FLT2/FLT3, ignore simple ectopy logic
    if any(tok in pano.tokens for tok in ("FLT2", "FLT3")):
        return {}

    cl = qrs_morphology_clusters(ctx).get("qrs_clusters", {})
    if not cl:
        return {}

    rr = getattr(ctx, "rr_intervals", []) or []
    n  = len(rr)
    ect = cl.get("minority_indices", []) or []
    prem = cl.get("premature_indices", []) or []

    if n < 8 or not ect:
        return {}

    # --- conservative timing gate for "real" premature beats ---------
    med = float(np.median(rr)) if rr else None
    if med:
        short_pre = [i for i in ect if (i-1) >= 0 and rr[i-1] < 0.82*med]    # was 0.85 -> tighter
        long_post = [i for i in ect if i < len(rr) and rr[i]   > 1.18*med]   # was 1.10-1.15 -> tighter
        # require >=50% of minority beats to meet (short CI OR compensatory pause)
        good_timing = set(short_pre) | set(long_post)
        if len(good_timing) < max(2, int(0.5*len(ect))):
            return {}

    burden = 100.0 * len(ect) / float(n)
    kind = "premature" if len(prem) >= max(1, len(ect)//2) else "ectopic"
    return {
        "extrasystoles": {
            "present": True,
            "description": f"{kind} beats {len(ect)}/{n} (~{burden:.1f}%)",
            "count": int(len(ect)), "total_beats": int(n),
            "burden_percent": burden,
            "indices": ect
        }
    }


def rhythm_panorama_to_flags(ctx: FeatureContext) -> dict:
    """
    Turn RR-based panorama metrics into semantic rhythm flags.
    This is the PRIMARY source of global rhythm semantics.
    """
    pano = _get_panorama(ctx)
    if not pano.rr_ms:
        return {}

    # Get robust features for fallback check
    feats = _rr_features(pano.rr_ms)
    pnn50 = feats.get("pnn50", 0.0)

    # Try to retrieve PRD power if available in ctx
    try:
        prd_limb = getattr(ctx, "computed_parameters", {}).get("rvt_prd_power_limb")
        prd_chest = getattr(ctx, "computed_parameters", {}).get("rvt_prd_power_chest")
    except Exception:
        prd_limb, prd_chest = None, None

    # Expose raw metrics, now including PRD
    flags: Dict[str, Any] = {
        "rhythm_panorama": {
            "present": True,
            "description": ("RR panorama summary"),
            "regular_fraction": round(pano.regular_fraction, 2),
            "cv": round(pano.cv_global, 3),
            "n_clusters": pano.n_clusters_global,
            "irregularity_score": round(pano.irregular_fraction, 2),
            "beats_used": len(pano.rr_ms),
            "dominant_rate": round(pano.dominant_rate_bpm) if pano.dominant_rate_bpm else None,
            "tokens": pano.tokens,
        }
    }

    # Add PRD to metrics if available
    # FIX: Explicit None check to avoid array ambiguity
    if prd_limb is not None or prd_chest is not None:
        flags["metrics"] = {
            "present": True,
            "rvt_prd_power_limb": round(prd_limb, 5) if prd_limb is not None else None,
            "rvt_prd_power_chest": round(prd_chest, 5) if prd_chest is not None else None
        }

    # FIX: Explicit check for high pNN50 = Absolute Arrhythmia regardless of clustering
    # 40% is a very high threshold; regular rhythms with ectopy rarely exceed 20%
    is_highly_irregular = (pano.cv_global >= 0.15 and pnn50 > 40.0)

    # ---- 1. Absolute Arrhythmia (AF-like) ----
    # High irregularity, low regular fraction, unimodal or diffuse clustering OR explicit high pNN50
    if "AAR" in pano.tokens or (pano.irregular_fraction > 0.50 and pano.regular_fraction < 0.40) or is_highly_irregular:
        flags["absolute_arrhythmia"] = {
            "present": True,
            "description": (f"Absolute arrhythmia (irregularity {pano.irregular_fraction:.2f}, "
                            f"CV {pano.cv_global:.2f}, pNN50 {pnn50:.1f}%) \u2013 consistent with atrial fibrillation")
        }

    # ---- 2. Flutter-like (Organized Irregularity) ----
    # High regular_fraction locally, but multiple clusters globally (e.g. 2:1 / 3:1 switching)
    # Suppress if it is actually Absolute Arrhythmia
    if not is_highly_irregular:
        if any(t in pano.tokens for t in ("FLT2", "FLT3")) or (pano.n_clusters_global >= 2 and pano.regular_fraction > 0.60):
            flags["flutter_like"] = {
                "present": True,
                "description": (f"Organized variable conduction ({pano.n_clusters_global} RR modes, "
                                f"regularity {pano.regular_fraction:.2f})")
            }

    # ---- 3. Regular Rhythm ----
    # Suppress if high irregularity detected
    if not is_highly_irregular:
        if "REG" in pano.tokens or pano.regular_fraction > 0.80:
            flags["regular_rhythm_check"] = {
                 "present": True,
                 "description": f"Regular rhythm ({int(pano.regular_fraction*100)}%)"
            }

    return flags


def rhythm_panorama_feature(ctx: FeatureContext) -> dict:
    """Feature extractor wrapper for rhythm panorama flags."""
    # keep plotting off here to avoid side effects inside the feature loop
    _get_panorama(ctx)
    return rhythm_panorama_to_flags(ctx)  # Pass ctx, it calls get_panorama internally

def rhythm_panorama_to_summary(pano: RhythmPanorama) -> str:
    """
    Short text line to inject into clinical_summaries['rhythm_analysis'].
    """
    pieces: List[str] = []
    if pano.dominant_rate_bpm:
        pieces.append(f"~{int(round(pano.dominant_rate_bpm))} bpm")
    pieces.append(f"{int(round(pano.regular_fraction*100))}% regular")
    if pano.tokens:
        pieces.append("tokens: " + ",".join(pano.tokens))
    return "Rhythm (10-s view): " + "; ".join(pieces)


def merge_panorama_into_semantics(semantic_flags: Dict[str, Any],
                                  pano: RhythmPanorama) -> None:
    """
    In-place merge into semantic_flags['rhythm'] WITHOUT overwriting
    existing specific rhythm findings you already compute.
    """
    rhythm_block = semantic_flags.setdefault("rhythm", {})
    # Re-implementing logic from rhythm_panorama_to_flags using 'pano' directly

    # Get pNN50 from scratch locally since we don't have ctx
    feats = _rr_features(pano.rr_ms)
    pnn50 = feats.get("pnn50", 0.0)

    flags: Dict[str, Any] = {
        "rhythm_panorama": {
            "present": True,
            "description": ("RR panorama summary"),
            "regular_fraction": round(pano.regular_fraction, 2),
            "cv": round(pano.cv_global, 3),
            "n_clusters": pano.n_clusters_global,
            "irregularity_score": round(pano.irregular_fraction, 2),
            "beats_used": len(pano.rr_ms),
            "dominant_rate": round(pano.dominant_rate_bpm) if pano.dominant_rate_bpm else None,
            "tokens": pano.tokens
        }
    }

    is_highly_irregular = (pano.cv_global >= 0.15 and pnn50 > 40.0)

    if "AAR" in pano.tokens or (pano.irregular_fraction > 0.50 and pano.regular_fraction < 0.40) or is_highly_irregular:
        flags["absolute_arrhythmia"] = {
            "present": True,
            "description": (f"Absolute arrhythmia (irregularity {pano.irregular_fraction:.2f}, "
                            f"CV {pano.cv_global:.2f}, pNN50 {pnn50:.1f}%) \u2013 consistent with atrial fibrillation")
        }

    if not is_highly_irregular:
        if any(t in pano.tokens for t in ("FLT2", "FLT3")) or (pano.n_clusters_global >= 2 and pano.regular_fraction > 0.60):
            flags["flutter_like"] = {
                "present": True,
                "description": (f"Organized variable conduction ({pano.n_clusters_global} RR modes, "
                                f"regularity {pano.regular_fraction:.2f})")
            }

    rhythm_block.update(flags)


def rhythm_flutter_like(ctx: FeatureContext, plot_enabled: bool = False) -> dict:
    """
    Detect flutter-like RR clustering AND calculate Integer Ratio Prevalence.
    """
    rr_list = getattr(ctx, "rr_intervals", None)
    if rr_list is None:
        rr_list = []
    rr_raw = rr_list or robust_rr_from_rpeaks(getattr(ctx, "r_peaks", []), getattr(ctx, "fs", 500))
    rr = np.asarray(rr_raw, float)
    rr = rr[np.isfinite(rr)]

    if rr.size < 12:
        return {}

    # --- Calculate Integer Ratio Prevalence ---
    int_ratio_prev = _calculate_integer_ratio_prevalence(rr)

    # Existing GMM Logic (Cluster finding)
    X = rr.reshape(-1, 1)
    best, best_k, best_bic = None, 1, np.inf
    for k in (1, 2, 3):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
            gmm.fit(X)
            bic = gmm.bic(X)
        except Exception:
            continue
        if bic < best_bic:
            best, best_k, best_bic = gmm, k, bic

    # Always return the integer ratio metric if we have data, even if GMM fails
    res: Dict[str, Any] = {
        "metrics": {
            "integer_ratio_prevalence": int_ratio_prev
        }
    }

    if best is None or best_k < 2:
        return res

    labels = best.predict(X)
    means  = best.means_.flatten()
    weights = best.weights_.flatten()

    # Order by population
    pop_idx = np.argsort(weights)[::-1]
    m1, m2 = float(means[pop_idx[0]]), float(means[pop_idx[1]])
    w1, w2 = float(weights[pop_idx[0]]), float(weights[pop_idx[1]])
    delta  = abs(m2 - m1)

    sil = 0.0
    if len(np.unique(labels)) > 1:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = 0.0

    # Gate for "Flutter-Like" Diagnosis
    if (w1 >= 0.20 and w2 >= 0.20) and (delta >= 60.0) and (sil >= 0.12):
        res["flutter_like_rr_clustering"] = {
            "present": True,
            "description": f"RR clustering into {best_k} modes (delta~{delta:.0f}ms, sil={sil:.2f})",
            "centers_ms": sorted([m1, m2]),
            "silhouette": sil
        }

    return res


def rhythm_absolute_arrhythmia(ctx: FeatureContext) -> dict:
    """
    Flag 'absolute arrhythmia' only when BOTH:
      - RR is highly irregular, and
      - there is no dominant QRS morphology (or P sequence).
    Avoids triggering on "regular + ectopy".
    """
    # USE NEW PANORAMA METRICS if available
    pano = _get_panorama(ctx)

    # FIX: Explicit None check
    rr_list = getattr(ctx, "rr_intervals", None)
    if rr_list is None:
        rr_list = []
    feats = _rr_features(rr_list)

    cv = feats.get("cv") or 0.0
    pnn50 = feats.get("pnn50") or 0.0

    # dominant morphology?
    cl = qrs_morphology_clusters(ctx).get("qrs_clusters", {})
    dom = float(cl.get("dominant_fraction", 1.0))
    ect_burden = len(cl.get("minority_indices", []) or []) / max(1, len(rr_list))

    # stable P fraction (limb plane P collection)
    # FIX: Safer dict access
    beat_plane = getattr(ctx, "beat_plane", {}) or {}
    limb_plane = beat_plane.get("limb", {}) or {}
    p_list = limb_plane.get("P", []) or []
    p_presence = (sum(w is not None for w in p_list) / max(1, len(p_list))) if p_list else 0.0

    # -- Updated Logic using Panorama --
    # Irregular if panorama says high irregularity OR old fallback stats are high
    irregular = (pano.irregular_fraction > 0.50) or (cv > 0.12 and pnn50 > 20.0)
    no_stable_p = (p_presence < 0.4)

    # Don't call AF-like if there is a clear dominant template and ectopy explains the scatter
    if irregular and no_stable_p and (dom < 0.70 or ect_burden < 0.15):
        return {
            "absolute_arrhythmia": {
                "present": True,
                "description": (f"Irregular RR without stable P sequence (Irregularity {pano.irregular_fraction:.2f}, "
                                f"CV {pano.cv_global:.2f}), no dominant QRS morphology"),
                "rr_cv": pano.cv_global, "pnn50": pnn50,
                "rmssd": feats.get("rmssd_ms"), "sdnn": feats.get("sdnn_ms"),
                "mean_rr_ms": feats.get("mean_ms"),
            }
        }
    return {}


# ============================================================================
# RHYTHM FEATURES 1-9
# ============================================================================

# 1. SINUS RHYTHM
def rhythm_sinus(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect normal sinus rhythm using panorama robustness.
    Criteria: HR 60-100 bpm, regular rhythm, P wave before each QRS
    """
    pano = _get_panorama(ctx)
    if not ctx.rr_intervals:
        return {}

    # Calculate heart rate and rhythm regularity
    mean_rr = np.mean(ctx.rr_intervals)
    hr = 60000 / mean_rr  # Convert RR interval to heart rate

    # Use Panorama Regularity if available (more robust than raw CV)
    is_regular = (pano.regular_fraction > 0.80)

    # Check for P waves before each QRS (using lead II as reference)
    p_present = False
    if "II" in ctx.lead_fits:
        p_gaussians = get_p_wave_gaussians(ctx.lead_fits, "II")
        # P waves should be present and upright in lead II for sinus rhythm
        p_present = len(p_gaussians) > 0 and any(g["amp_mv"] > 0.05 for g in p_gaussians)

    # Normal sinus rhythm criteria
    if 60 <= hr <= 100 and is_regular and p_present:
        return {
            "normal_sinus_rhythm": {
                "present": True,
                "description": f"Normal sinus rhythm at {hr:.0f} bpm with regular RR intervals ({int(pano.regular_fraction*100)}% regular)",
                "heart_rate": hr,
                "rr_variation": pano.cv_global
            }
        }
    return {}

# 2. SINUS BRADYCARDIA
def rhythm_sinus_bradycardia(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect sinus bradycardia (sinus rhythm <60 bpm)
    """
    if not ctx.rr_intervals:
        return {}

    mean_rr = np.mean(ctx.rr_intervals)
    hr = 60000 / mean_rr

    # Check for P waves (sinus origin)
    p_present = False
    if "II" in ctx.lead_fits:
        p_gaussians = get_p_wave_gaussians(ctx.lead_fits, "II")
        p_present = len(p_gaussians) > 0 and any(g["amp_mv"] > 0.05 for g in p_gaussians)

    if hr < 60 and p_present:
        # Classify severity
        if hr >= 50:
            severity = "mild"
            clinical = "often benign, common in athletes"
        elif hr >= 40:
            severity = "moderate"
            clinical = "evaluate for symptoms (dizziness, fatigue)"
        else:
            severity = "severe"
            clinical = "high risk for syncope, consider pacemaker"

        return {
            "sinus_bradycardia": {
                "present": True,
                "description": f"Sinus bradycardia at {hr:.0f} bpm ({severity}), {clinical}",
                "heart_rate": hr,
                "severity": severity
            }
        }
    return {}

# 3. SINUS TACHYCARDIA
def rhythm_sinus_tachycardia(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect sinus tachycardia (sinus rhythm >100 bpm)
    """
    if not ctx.rr_intervals:
        return {}

    mean_rr = np.mean(ctx.rr_intervals)
    hr = 60000 / mean_rr

    # Check for P waves (sinus origin)
    p_present = False
    if "II" in ctx.lead_fits:
        p_gaussians = get_p_wave_gaussians(ctx.lead_fits, "II")
        p_present = len(p_gaussians) > 0 and any(g["amp_mv"] > 0.05 for g in p_gaussians)

    if hr > 100 and p_present:
        # Classify severity and likely causes
        if hr <= 120:
            severity = "mild"
            causes = "anxiety, fever, dehydration"
        elif hr <= 150:
            severity = "moderate"
            causes = "hyperthyroidism, anemia, infection"
        else:
            severity = "severe"
            causes = "shock, severe hypoxia, thyrotoxicosis"

        return {
            "sinus_tachycardia": {
                "present": True,
                "description": f"Tachycardia at {hr:.0f} bpm ({severity}), ",
                "heart_rate": hr,
                "severity": severity
            }
        }
    return {}

# 4. IRREGULAR RHYTHM
def rhythm_irregular(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Irregular rhythm detector updated to use Panorama stats.
    """
    pano = _get_panorama(ctx)
    rr = np.asarray(getattr(ctx, "rr_intervals", []), dtype=float)

    if rr.size < 12:
        return {}

    mean_rr = float(np.mean(rr)) if rr.size > 0 else 0.0

    # Use global stats from Panorama
    cv_global = pano.cv_global
    irregular_frac = pano.irregular_fraction

    # Triggers: High global CV OR High irregular window fraction
    trigger = (cv_global >= 0.12) or (irregular_frac >= 0.40)

    if not trigger:
        return {}

    # Pattern classification
    if cv_global > 0.25 or irregular_frac > 0.7:
        pattern = "grossly irregular"
    elif cv_global > 0.15:
        pattern = "moderately irregular"
    else:
        pattern = "mildly irregular"

    return {
        "irregular_rhythm": {
            "present": True,
            "description": f"Irregular rhythm ({pattern}); Irregularity Score={irregular_frac:.2f}, CV={cv_global:.3f}",
            "pattern": pattern,
            "rr_variation": cv_global,
            "irregularity_fraction": irregular_frac
        }
    }

# 5. VENTRICULAR BIGEMINY
def rhythm_ventricular_bigeminy(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect ventricular bigeminy (PVC every other beat)
    Pattern: normal-PVC-normal-PVC
    """
    if not ctx.rr_intervals or len(ctx.rr_intervals) < 6:
        return {}

    # Look for alternating short-normal RR pattern
    # PVCs typically come early (short RR), followed by compensatory pause (long RR)
    pattern_matches = 0
    total_pairs = 0

    for i in range(0, len(ctx.rr_intervals) - 1, 2):
        if i + 1 < len(ctx.rr_intervals):
            total_pairs += 1
            # Short RR (PVC) followed by longer RR (compensatory)
            if ctx.rr_intervals[i] < ctx.rr_intervals[i + 1] * 0.8:
                pattern_matches += 1

    # Need consistent pattern in majority of beats
    if total_pairs >= 3 and pattern_matches >= total_pairs * 0.75:
        burden = 50  # Bigeminy means 50% PVC burden

        return {
            "ventricular_bigeminy": {
                "present": True,
                "description": f"Ventricular bigeminy detected - PVC every other beat (50% burden), ",
                "pvc_burden_percent": burden,
                "pattern_consistency": pattern_matches / total_pairs
            }
        }
    return {}

# 6. VENTRICULAR TRIGEMINY
def rhythm_ventricular_trigeminy(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect ventricular trigeminy (PVC every third beat)
    Pattern: normal-normal-PVC-normal-normal-PVC
    """
    if not ctx.rr_intervals or len(ctx.rr_intervals) < 9:
        return {}

    # Look for pattern: two normal RR intervals followed by short RR (PVC)
    pattern_matches = 0
    total_triplets = 0

    for i in range(0, len(ctx.rr_intervals) - 2, 3):
        if i + 2 < len(ctx.rr_intervals):
            total_triplets += 1
            # First two beats normal, third is PVC (short RR)
            mean_normal = (ctx.rr_intervals[i] + ctx.rr_intervals[i + 1]) / 2
            if ctx.rr_intervals[i + 2] < mean_normal * 0.8:
                pattern_matches += 1

    # Need consistent pattern
    if total_triplets >= 2 and pattern_matches >= total_triplets * 0.67:
        burden = 33  # Trigeminy means ~33% PVC burden

        return {
            "ventricular_trigeminy": {
                "present": True,
                "description": f"Ventricular trigeminy detected - PVC every third beat (33% burden), ",
                "pvc_burden_percent": burden,
                "pattern_consistency": pattern_matches / total_triplets
            }
        }
    return {}

# 7. HEART RATE VARIABILITY
def rhythm_hrv_analysis(ctx: FeatureContext) -> dict:
    """
    HRV summary (only when meaningful):
      - requires enough beats (~>=60)
      - requires organized RR (not absolute arrhythmia)
      - optional quality gate if a prior quality object is present
    Emits *raw metrics only* by default; 'reduced_hrv' is conservative.
    """
    # 1) RR source
    rr = getattr(ctx, "rr_intervals", []) or robust_rr_from_rpeaks(getattr(ctx, "r_peaks", []), getattr(ctx, "fs", 500))
    if len(rr) < 60:           # ~1 minute of beats
        return {}

    # 2) Skip if the RR-signature isn't organized (prevents AF cases)
    sig = analyze_rr_signature(rr)
    if sig.get("pattern") != "organized_av_conduction":
        return {}

    # 3) Optional quality gate (if a global quality object already exists in ctx)
    poor = False
    try:
        g = getattr(ctx, "global_quality", None) or getattr(ctx, "quality", None)
        poor = bool(g and g.get("poor_quality_ecg", {}).get("present"))
    except Exception:
        pass
    if poor:
        return {}

    # 4) Metrics
    feats  = _rr_features(rr)
    rmssd  = feats.get("rmssd_ms")
    sdnn   = feats.get("sdnn_ms")
    pnn50  = feats.get("pnn50")
    if rmssd is None or sdnn is None:
        return {}

    # 5) Emit conservatively: mostly metrics; flag only when clearly reduced
    if (rmssd < 20.0) and (sdnn < 30.0):   # strict AND
        return {"reduced_hrv": {
            "present": True,
            "description": f"Heart rate variability reduced on a regular segment (RMSSD={rmssd:.1f} ms, SDNN={sdnn:.1f} ms)",
            "rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50, "status": "reduced"
        }}

    return {"hrv_metrics": {
        "present": True,
        "description": f"HRV metrics (regular rhythm segment): RMSSD={rmssd:.1f} ms, SDNN={sdnn:.1f} ms",
        "rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50
    }}

# 8. LONG PAUSE
def rhythm_long_pause(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect significant pauses in rhythm
    May indicate sinus arrest, atrioventricular conduction delay, or sick sinus syndrome
    """
    if not ctx.rr_intervals:
        return {}

    mean_rr = np.mean(ctx.rr_intervals)
    max_rr = max(ctx.rr_intervals)

    # Check for significant pause
    pause_ratio = max_rr / mean_rr

    # Pause is significant if >2x normal RR or >2000ms absolute
    if pause_ratio > 2.0 or max_rr > 2000:
        # Classify severity and likely cause
        if max_rr > 3000:
            severity = "critical"
            cause = "complete heart block or severe sick sinus syndrome"
            action = "URGENT: Immediate pacemaker evaluation required"
        elif max_rr > 2500:
            severity = "severe"
            cause = "high-grade atrioventricular conduction delay or sick sinus syndrome"
            action = "Urgent cardiology consultation for pacemaker consideration"
        elif max_rr > 2000:
            severity = "significant"
            cause = "sinus pause or second-degree atrioventricular conduction delay"
            action = "Holter monitoring and cardiology evaluation recommended"
        else:
            severity = "moderate"
            cause = "sinus arrhythmia or blocked PACs"
            action = "Monitor for symptoms and progression"

        return {
            "long_pause": {
                "present": True,
                "description": f"{severity.capitalize()} pause detected ({max_rr:.0f}ms, "
                               f"{pause_ratio:.1f}x normal RR), suggesting {cause}. {action}",
                "pause_duration": max_rr,
                "pause_ratio": pause_ratio,
                "severity": severity
            }
        }
    return {}

# 9. narrow-complex tachycardia pattern (SVT)
def rhythm_svt(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect narrow-complex tachycardia pattern
    Criteria: Regular narrow-complex tachycardia >150 bpm
    """
    if not ctx.rr_intervals:
        return {}

    mean_rr = np.mean(ctx.rr_intervals)
    hr = 60000 / mean_rr
    rr_cv = np.std(ctx.rr_intervals) / mean_rr

    # SVT criteria: fast (>150), regular, narrow QRS
    if hr > 150 and rr_cv < 0.1:  # Fast and regular
        # Check QRS width to confirm narrow complex
        qrs_duration = None
        if "II" in ctx.lead_fits:
            qrs_duration = calculate_qrs_duration(ctx.lead_fits, "II")

        # Narrow complex if QRS < 120ms
        if qrs_duration and qrs_duration < 120:
            # Classify SVT type based on rate and P waves
            svt_type = ""
            p_visible = False

            if "II" in ctx.lead_fits:
                p_gaussians = get_p_wave_gaussians(ctx.lead_fits, "II")
                p_visible = len(p_gaussians) > 0

            if hr > 250:
                svt_type = "Atrial f_wave with 1:1 conduction"
                urgency = "EMERGENCY"
            elif hr > 220:
                svt_type = "AVRT (accessory pathway)"
                urgency = "URGENT"
            elif not p_visible:
                svt_type = "AVNRT (AV nodal reentrant)"
                urgency = "URGENT"
            else:
                svt_type = "Atrial tachycardia"
                urgency = "URGENT"

            return {
                "supraventricular_tachycardia": {
                    "present": True,
                    "description": f"{urgency}: narrow-complex tachycardia pattern at {hr:.0f} bpm, "
                                   f"likely {svt_type}. Consider vagal maneuvers or adenosine",
                    "heart_rate": hr,
                    "qrs_duration": qrs_duration,
                    "likely_type": svt_type
                }
            }
    return {}


# 10. wide-complex tachycardia pattern
def rhythm_ventricular_tachycardia(ctx: FeatureContext) -> Dict[str, Any]:
    """VT detection - wide complex tachycardia"""
    if not ctx.rr_intervals:
        return {}

    hr = 60000 / np.mean(ctx.rr_intervals)

    if hr > 120:
        qrs_duration = calculate_qrs_duration(ctx.lead_fits, "II") if "II" in ctx.lead_fits else 80

        if qrs_duration and qrs_duration > 120:
            return {
                "ventricular_tachycardia": {
                    "present": True,
                    "description": f"Wide complex tachycardia at {hr:.0f} bpm (QRS {qrs_duration:.0f}ms), ",
                    "heart_rate": hr,
                    "qrs_duration": qrs_duration
                }
            }
    return {}

# 11. ACCELERATED IDIOVENTRICULAR RHYTHM
def rhythm_aivr(ctx: FeatureContext) -> Dict[str, Any]:
    """AIVR - slow VT (60-120 bpm)"""
    if not ctx.rr_intervals:
        return {}

    hr = 60000 / np.mean(ctx.rr_intervals)

    if 60 <= hr <= 120:
        qrs_duration = calculate_qrs_duration(ctx.lead_fits, "II") if "II" in ctx.lead_fits else 80

        # Check for absent P waves
        p_absent = True
        if "II" in ctx.lead_fits:
            p_gaussians = get_p_wave_gaussians(ctx.lead_fits, "II")
            if p_gaussians and any(abs(g["amp_mv"]) > 0.05 for g in p_gaussians):
                p_absent = False

        if qrs_duration and qrs_duration > 120 and p_absent:
            return {
                "accelerated_idioventricular_rhythm": {
                    "present": True,
                    "description": f"AIVR at {hr:.0f} bpm, often seen in reperfusion after ischemic pattern",
                    "heart_rate": hr
                }
            }
    return {}

# 12. JUNCTIONAL RHYTHM
def rhythm_junctional(ctx: FeatureContext) -> Dict[str, Any]:
    """Junctional rhythm detection"""
    if not ctx.rr_intervals:
        return {}

    hr = 60000 / np.mean(ctx.rr_intervals)

    # Check for absent/retrograde P waves
    p_abnormal = False
    if "II" in ctx.lead_fits:
        p_gaussians = get_p_wave_gaussians(ctx.lead_fits, "II")
        if not p_gaussians or (p_gaussians and any(g["amp_mv"] < -0.05 for g in p_gaussians)):
            p_abnormal = True

    if 40 <= hr <= 60 and p_abnormal:
        return {
            "junctional_rhythm": {
                "present": True,
                "description": f"Junctional rhythm at {hr:.0f} bpm",
                "heart_rate": hr
            }
        }
    return {}

# 13. ESCAPE RHYTHM
def rhythm_escape(ctx: FeatureContext) -> Dict[str, Any]:
    """Escape rhythm detection"""
    if not ctx.rr_intervals:
        return {}

    hr = 60000 / np.mean(ctx.rr_intervals)

    if hr < 40:
        return {
            "escape_rhythm": {
                "present": True,
                "description": f"Escape rhythm at {hr:.0f} bpm - CRITICAL: ",
                "heart_rate": hr
            }
        }
    return {}

# 14. R-ON-T PHENOMENON
def rhythm_r_on_t(ctx: FeatureContext) -> Dict[str, Any]:
    """R-on-T detection - high risk for VF"""
    # Check for premature beats landing on T wave
    # This requires beat-to-beat analysis

    if len(ctx.rr_intervals) < 3:
        return {}

    # Look for very short coupling intervals
    min_coupling = min(ctx.rr_intervals)
    mean_rr = np.mean(ctx.rr_intervals)

    if min_coupling < 0.4 * mean_rr:  # Very short coupling
        return {
            "r_on_t_phenomenon": {
                "present": True,
                "description": "R-on-T phenomenon detected - HIGH RISK for ventricular fibrillation",
                "coupling_interval": min_coupling
            }
        }
    return {}

# 15. PACED RHYTHM
def rhythm_paced(ctx: FeatureContext) -> Dict[str, Any]:
    """Paced rhythm detection"""
    # Look for pacing spikes (very narrow, high amplitude deflections)
    pacing_detected = False

    for lead in ["II", "V1"]:
        if lead not in ctx.lead_fits:
            continue

        # Check for very narrow spikes before QRS
        all_gaussians = ctx.lead_fits[lead].get("avg", [])

        for g in all_gaussians:
            # Pacing spike: very narrow (<5ms) and occurs before QRS
            if g["sigma_ms"] < 5 and g["center_ms"] < -50:
                pacing_detected = True
                break

    if pacing_detected:
        return {
            "paced_rhythm": {
                "present": True,
                "description": "Paced rhythm detected - evaluate pacemaker function and capture"
            }
        }
    return {}


# ============================================================================
# 16. ATRIAL f_wave - Advanced Lag-Lag Return Map (Retuned for 2-9 Hz)
# ============================================================================
def rhythm_f_sawtooth_autocorr(ctx: FeatureContext,
                                *,
                                plot_enabled: bool = False,
                                leads_pref: tuple = ("II", "aVF", "V1", "III"),
                                # RETUNED: Tighter bandpass to isolate atrial activity
                                band: tuple = (1.0, 20.0),
                                lag_win_ms: tuple = (120, 390),
                                agree_tol_ms: float = 12,
                                need_agree_leads: int = 2) -> dict:
    """
    Atrial flutter sawtooth via autocorrelation (ACF) after QRS blanking.
    UPDATED: Uses 2.0-9.0 Hz bandpass to maximize SNR for Flutter vs T-waves.
    """
    # ---- thresholds (tuned for subtle ~0.2 peaks) ------------------------
    PI_THR       = 1.8     # prominence / MAD
    HEIGHT_THR   = 0.15    # normalized ACF height at the fundamental
    CONTRAST_THR = 1.6     # (height - region_median) / MAD
    COH_THR      = 0.25    # median |corr| across lead pairs

    fs = int(getattr(ctx, "fs", 500))
    BLANK_MS = 120
    blank = int((BLANK_MS/1000.0) * fs)

    # ---------- gather traces ----------
    traces: Dict[str, np.ndarray] = {}
    if hasattr(ctx, "raw_ecg_12"):
        raw_ecg = getattr(ctx, "raw_ecg_12", None)
        if raw_ecg is not None:
            raw = np.asarray(raw_ecg, float)
            names = list(getattr(ctx, "leads", [])) or ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
            idx = {n: i for i, n in enumerate(names[:raw.shape[0]])}
            for ld in leads_pref:
                if ld in idx:
                    traces[ld] = raw[idx[ld]]
    if not traces and hasattr(ctx, "segments"):
        by_lead = getattr(ctx, "segments", {}).get("raw", {}).get("by_lead", {})
        for ld in leads_pref:
            if ld in by_lead:
                arr = by_lead[ld].get("raw", by_lead[ld].get("signal", []))
                arr = np.asarray(arr, float)
                if arr.ndim == 1 and arr.size > fs:
                    traces[ld] = arr
    if not traces:
        return {}

    # ---------- QRS blanking ----------
    rpeaks = np.asarray(getattr(ctx, "r_peaks", []), int)
    for ld, x in list(traces.items()):
        x = x.copy()
        if rpeaks.size:
            for rp in rpeaks:
                a = max(0, rp - blank); b = min(len(x), rp + blank)
                if a < b:
                    x[a:b] = np.median(x[max(0, a-60):min(len(x), b+60)])
        traces[ld] = x - np.median(x)

    # ---------- UPDATED: 2-9 Hz band-pass ----------
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    for ld in list(traces.keys()):
        traces[ld] = sosfiltfilt(sos, traces[ld])

    # ---------- ACF + fundamental lag per lead ----------
    def _acf_and_peak(sig: np.ndarray) -> Optional[Dict[str, Any]]:
        x = sig - np.mean(sig)
        if np.allclose(np.std(x), 0):
            return None
        n = int(2**np.ceil(np.log2(len(x)*2)))
        X_fft = np.fft.rfft(x, n=n)
        acf = np.fft.irfft(X_fft * np.conj(X_fft), n=n)[:len(x)]
        acf = acf / (acf[0] + 1e-12)

        if lag_win_ms:
            pmin = int(fs * (lag_win_ms[0]/1000.0))
            pmax = int(fs * (lag_win_ms[1]/1000.0))
        else:
            pmin = int(fs / band[1])
            pmax = int(fs / band[0])
        pmax = min(pmax, len(acf))
        if pmax - pmin < 5:
            return {"acf": acf, "lag": None, "height": 0.0, "pi": 0.0, "contrast": 0.0}

        region = acf[pmin:pmax]
        baseline = float(np.median(region))
        mad = float(np.median(np.abs(region - baseline))) + 1e-12

        prom_arg = max(0.05, 0.4 * mad)
        pk, props = find_peaks(region, prominence=prom_arg, distance=int(0.18*(pmax-pmin)))
        if pk.size == 0:
            return {"acf": acf, "lag": None, "height": 0.0, "pi": 0.0, "contrast": 0.0}

        ki = int(pk[np.argmax(props["prominences"])])
        lag_idx = pmin + ki
        height  = float(region[ki])
        prom    = float(props["prominences"][np.argmax(props["prominences"])])
        pi      = prom / mad
        contrast = (height - baseline) / mad

        harm_ok = False
        harm_lag = 2 * lag_idx
        if harm_lag < len(acf):
            hwin = acf[max(0, harm_lag-5):min(len(acf), harm_lag+6)]
            if hwin.size and np.max(hwin) > 0.10:
                harm_ok = True

        lead_accept = (height >= HEIGHT_THR) or ((pi >= PI_THR and contrast >= CONTRAST_THR) or (pi >= 1.4 and contrast >= 1.4 and harm_ok))
        return {"acf": acf, "lag": int(lag_idx), "height": height, "pi": pi,
                "contrast": contrast, "lead_accept": lead_accept}

    acf_map = {ld: _acf_and_peak(x) for ld, x in traces.items()}

    # ---------- coherence ----------
    pair_corr: List[float] = []
    g_leads = [ld for ld, d in acf_map.items() if d]
    for a, b in combinations(g_leads, 2):
        xa = traces[a] - np.mean(traces[a])
        xb = traces[b] - np.mean(traces[b])
        c = np.corrcoef(xa, xb)[0, 1]
        pair_corr.append(abs(float(c)))
    median_coh = float(np.median(pair_corr)) if pair_corr else 0.0

    best_height = max([d["height"] for d in acf_map.values() if d], default=0.0)

    # Diagnostic metrics
    metrics = {
        "median_coherence": median_coh,
        "max_acf_height": best_height,
        "leads_tested": len(traces)
    }

    good = {ld: d for ld, d in acf_map.items() if d and (d["lag"] is not None) and d.get("lead_accept", False)}
    if len(good) < need_agree_leads:
        return {"metrics": metrics}

    # period agreement
    periods = {ld: 1000.0 * d["lag"] / fs for ld, d in good.items()}
    vals = sorted(periods.values())
    ok = False
    rep = None
    for i, v in enumerate(vals):
        cnt = 1
        for j in range(i+1, len(vals)):
            if abs(vals[j] - v) <= agree_tol_ms:
                cnt += 1
        if cnt >= need_agree_leads:
            ok = True; rep = v; break

    if not ok or median_coh < COH_THR:
        return {"metrics": metrics}

    f0 = 1000.0 / rep
    used = sorted([ld for ld, p in periods.items() if abs(p - rep) <= agree_tol_ms])

    return {
        "atrial_f_wave": {
            "present": True,
            "description": f"Regular atrial activity by ACF: ~{f0:.2f} Hz (period \u2248 {rep:.0f} ms)",
            "f_wave_freq_hz": float(f0),
            "leads_used": used,
            "method": "QRS-blanked 2\u20139 Hz bandpass + ACF",
        },
        "metrics": metrics
    }


def rhythm_atrial_f_wave(ctx: FeatureContext, *, debug: bool = False, plot_enabled: bool = None) -> dict:
    """Wrapper for atrial F-wave detection. Removed spectral ratios."""
    if plot_enabled is None:
        plot_enabled = bool(debug)

    # Use the new tighter band
    acf_res = rhythm_f_sawtooth_autocorr(ctx, plot_enabled=plot_enabled, band=(1.0, 20.0))

    return acf_res


# ============================================================================
# 17. PVC detection
# ============================================================================
def rhythm_pvc_detection(ctx: FeatureContext) -> dict:
    """
    Detect premature ventricular contractions (PVCs) and short runs.
    Approach:
      - Use R-peaks + raw precordial leads (or II) to build a vector-magnitude signal
      - Per-beat QRS width by 20% threshold crossing (+/-150 ms around R)
      - Build a "normal" template from narrow, well-spaced beats; correlate each beat
      - Timing rules: short coupling interval (CI) and compensatory pause (PP)
      - Classify PVC if (wide OR low corr) AND (CI short AND/OR PP present)
      - Summarize burden and detect NSVT (>=3 consecutive ventricular beats)
    Emits semantic_flags['rhythm']['ventricular_premature_beats'] and optional
    ['rhythm']['non_sustained_vt'] if runs detected.
    """
    fs = int(getattr(ctx, "fs", 500))
    raw_rpeaks = getattr(ctx, "r_peaks", None)
    if raw_rpeaks is None:
        rpeaks = np.array([], dtype=int)
    else:
        rpeaks = np.asarray(raw_rpeaks, dtype=int)

    raw_data = getattr(ctx, "raw_ecg_12", None)
    if raw_data is None:
        raw = np.empty((0, 0), dtype=float)
    else:
        raw = np.asarray(raw_data, dtype=float)
    leads = list(getattr(ctx, "leads", [])) or ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    if raw.ndim != 2 or raw.shape[1] < 5 or rpeaks.size < 6:
        return {}

    # --- choose channels for morphology ---
    lead_to_idx = {nm: i for i, nm in enumerate(leads)}
    # Prefer chest vector magnitude for QRS width/corr
    chest_idxs = [lead_to_idx.get(x) for x in ["V1","V2","V3","V4","V5","V6"] if lead_to_idx.get(x) is not None]
    if len(chest_idxs) >= 3:
        Z = (raw[chest_idxs,:] - raw[chest_idxs,:].mean(axis=1, keepdims=True)) / \
            (raw[chest_idxs,:].std(axis=1, keepdims=True) + 1e-9)
        vm = np.sqrt((Z**2).sum(axis=0))           # vector magnitude
        morph_sig = vm
    else:
        # fallback: Lead II or the channel with highest R amplitude
        if "II" in lead_to_idx:
            morph_sig = raw[lead_to_idx["II"], :]
        else:
            idx_max = int(np.argmax(np.ptp(raw, axis=1)))
            morph_sig = raw[idx_max, :]

    # --- basic RR and windows ---
    rr_ms = np.diff(rpeaks) * 1000.0 / fs
    if rr_ms.size < 5:
        return {}

    # baseline RR: median of middle 60% (robust to ectopy)
    sorted_rr = np.sort(rr_ms)
    k1, k2 = int(0.2*len(sorted_rr)), int(0.8*len(sorted_rr))
    rr_base = float(np.median(sorted_rr[k1:k2])) if k2 > k1 else float(np.median(sorted_rr))

    pre_win = int(0.15 * fs)     # +/-150 ms for width
    post_win = int(0.15 * fs)

    widths_ms_list: List[float] = []
    corr_vals_list: List[float] = []
    # Build a normal template from narrow beats separated by ~rr_base
    # Use +/-80 ms window for the template correlation
    wtemp = int(0.16 * fs)

    # collect candidate "normal" beats (narrow and RR near baseline)
    narrow_idx: List[int] = []
    segs_for_template: List[np.ndarray] = []

    # First pass: get width with 20% threshold around local peak
    for i, rp in enumerate(rpeaks):
        lo = max(0, rp - pre_win); hi = min(len(morph_sig), rp + post_win)
        seg = morph_sig[lo:hi]
        if seg.size < 10:
            widths_ms_list.append(np.nan)
            continue
        # peak index relative to seg
        pk = int(np.argmax(np.abs(seg)))
        peak = float(np.max(np.abs(seg)))
        thr = 0.20 * peak
        # left threshold crossing
        L = pk
        while L > 0 and np.abs(seg[L]) > thr:
            L -= 1
        # right threshold crossing
        R = pk
        while R < seg.size-1 and np.abs(seg[R]) > thr:
            R += 1
        qrs_w_ms = (R - L) * 1000.0 / fs
        widths_ms_list.append(float(qrs_w_ms))

    widths_ms = np.asarray(widths_ms_list, float)

    # Find beats with narrow width (likely supraventricular) to build template
    for i in range(1, len(rpeaks)-1):
        if not np.isfinite(widths_ms[i]):
            continue
        if widths_ms[i] < 115.0 and 0.85*rr_base < rr_ms[i-1] < 1.15*rr_base:  # narrow + near-baseline RR
            lo = max(0, rpeaks[i]-wtemp); hi = min(len(morph_sig), rpeaks[i]+wtemp)
            seg = morph_sig[lo:hi]
            if seg.size == 2*wtemp:
                segs_for_template.append(seg)
                narrow_idx.append(i)

    if len(segs_for_template) >= 5:
        template = np.median(np.vstack(segs_for_template), axis=0)
    else:
        template = None

    # Correlate each beat with the template
    def _xcorr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean(); b = b - b.mean()
        den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / den)

    for i in range(len(rpeaks)):
        if template is None:
            corr_vals_list.append(np.nan)
            continue
        lo = max(0, rpeaks[i]-wtemp); hi = min(len(morph_sig), rpeaks[i]+wtemp)
        seg = morph_sig[lo:hi]
        if seg.size == template.size:
            corr_vals_list.append(_xcorr(seg, template))
        else:
            corr_vals_list.append(np.nan)

    corr_vals = np.asarray(corr_vals_list, float)

    # --- timing features for PVC classification ---
    # CI = RR(i-1); PP = RR(i) after the ectopic beat
    ci = np.r_[np.nan, rr_ms]          # align to beat i (pre-RR)
    pp = np.r_[rr_ms, np.nan]          # post pause align

    # Rules of thumb (conservative):
    #  - wide OR poorly correlated to normal template (corr<0.80)
    #  - AND (CI < 0.85*baseline OR PP > 1.15*baseline)
    pvc_mask = np.zeros(len(rpeaks), dtype=bool)
    for i in range(1, len(rpeaks)-1):
        wide = (widths_ms[i] >= 120.0)
        lowcorr = (np.isfinite(corr_vals[i]) and corr_vals[i] < 0.80)
        short_ci = (np.isfinite(ci[i]) and ci[i] < 0.85*rr_base)
        comp_pause = (np.isfinite(pp[i]) and pp[i] > 1.15*rr_base)
        if (wide or lowcorr) and (short_ci or comp_pause):
            pvc_mask[i] = True

    pvc_indices = np.where(pvc_mask)[0].tolist()
    if not pvc_indices:
        return {}

    # detect short runs (NSVT): >=3 consecutive ventricular beats
    runs: List[Tuple[int, int]] = []
    run_start: Optional[int] = None
    for i in range(1, len(pvc_mask)):
        if pvc_mask[i] and pvc_mask[i-1]:
            if run_start is None:
                run_start = i-1
        else:
            if run_start is not None:
                runs.append((run_start, i-1))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(pvc_mask)-1))

    nsvt_spans = [span for span in runs if (span[1] - span[0] + 1) >= 3]

    # summary numbers
    pvc_count = int(pvc_mask.sum())
    total_beats = int(len(rpeaks))
    burden_pct = 100.0 * pvc_count / max(1, total_beats)

    mean_ci = float(np.nanmean(ci[pvc_mask])) if pvc_count else np.nan
    mean_pp = float(np.nanmean(pp[pvc_mask])) if pvc_count else np.nan
    mean_w  = float(np.nanmean(widths_ms[pvc_mask])) if pvc_count else np.nan

    desc = (f"Frequent premature wide beats compatible with ventricular origin "
            f"(PVCs {pvc_count}/{total_beats}, {burden_pct:.1f}%; "
            f"mean width {mean_w:.0f} ms; CI {mean_ci:.0f} ms; PP {mean_pp:.0f} ms)")

    out: Dict[str, Any] = {
        "ventricular_premature_beats": {
            "present": True,
            "description": desc,
            "count": pvc_count,
            "burden_percent": burden_pct,
            "mean_qrs_width_ms": mean_w,
            "mean_coupling_ms": mean_ci,
            "mean_pause_ms": mean_pp,
            }
    }

    if nsvt_spans:
        max_run = max((e-s+1) for (s, e) in nsvt_spans)
        out["non_sustained_vt"] = {
            "present": True,
            "description": f"Short run of monomorphic wide-complex tachycardia (NSVT), longest {max_run} beats",
            "longest_run_beats": int(max_run),
            "runs": [{"start_index": int(s), "end_index": int(e)} for (s, e) in nsvt_spans],
        }

    return out


# ============================================================================
# RHYTHM_FEATURES registry -- 21 extractors
# ============================================================================
RHYTHM_FEATURES = [
    rhythm_panorama_feature,
    rhythm_extrasystole_burden,
    rhythm_flutter_like,
    rhythm_absolute_arrhythmia,
    rhythm_sinus,
    rhythm_sinus_bradycardia,
    rhythm_sinus_tachycardia,
    rhythm_irregular,
    rhythm_ventricular_bigeminy,
    rhythm_ventricular_trigeminy,
    rhythm_hrv_analysis,
    rhythm_long_pause,
    rhythm_svt,
    rhythm_ventricular_tachycardia,
    rhythm_aivr,
    rhythm_junctional,
    rhythm_escape,
    rhythm_r_on_t,
    rhythm_paced,
    rhythm_atrial_f_wave,
    rhythm_pvc_detection,
]


# ===========================================================================
# Apr28 Item 4.5.4 — Cell-13 Complex Arrhythmia / TdP detection
# Verbatim port of Apr28 notebook lines 10622-11248. 7 functions covering
# energy-based beat classification, SLS trigger detection, TdP amplitude
# modulation, ectopic pattern detection (bigeminy/trigeminy), TdP risk
# assessment, and the analyse_complex_arrhythmias orchestrator that
# produces 15 Complex_* keys for Computed Parameters.
# ===========================================================================


def compute_per_beat_energy(
    stack_sync: np.ndarray,
    time_ms: np.ndarray,
    *,
    qrs_window_ms: Tuple[float, float] = (-60.0, 100.0),
) -> Dict[str, Any]:
    """Compute per-beat energy features from the synchronized beat stack.

    Source: Apr28 notebook line 10622.

    Parameters
    ----------
    stack_sync : ndarray, shape (n_beats, 12, L)
        Synchronized, zero-padded beat stack (12 leads).
    time_ms : ndarray, shape (L,)
        Signed time axis (0 = R-peak).
    qrs_window_ms : tuple
        Time window around R-peak to measure QRS energy.

    Returns
    -------
    dict with:
        peak_energies   : ndarray (n_beats,) — max RMS energy per beat
        qrs_widths_ms   : ndarray (n_beats,) — energy-based QRS width per beat
        energy_traces   : ndarray (n_beats, L) — full per-beat RMS trace
        qrs_integrals   : ndarray (n_beats,) — integral of energy in QRS window
    """
    n_beats = stack_sync.shape[0]
    energy_traces = np.sqrt(np.mean(stack_sync ** 2, axis=1))

    qrs_mask = (time_ms >= qrs_window_ms[0]) & (time_ms <= qrs_window_ms[1])
    dt_ms = float(np.median(np.abs(np.diff(time_ms)))) if len(time_ms) > 1 else 1.0

    peak_energies = np.zeros(n_beats)
    qrs_widths_ms = np.zeros(n_beats)
    qrs_integrals = np.zeros(n_beats)

    for i in range(n_beats):
        qrs_seg = energy_traces[i, qrs_mask]
        if qrs_seg.size == 0:
            continue
        peak_e = np.max(qrs_seg)
        peak_energies[i] = peak_e

        # QRS width: time above 25% of peak energy (full-width at quarter-max)
        if peak_e > 1e-6:
            above = qrs_seg >= 0.25 * peak_e
            qrs_widths_ms[i] = float(np.sum(above)) * dt_ms

        # Integral of energy in QRS window (total QRS energy).
        # Uses qpsi.compat.trapezoid (np.trapezoid on numpy >=2.0,
        # np.trapz fallback on numpy <2.0) — Item 4.5.4 originally
        # used bare np.trapz which is deprecated in numpy 2.x.
        qrs_integrals[i] = float(_trapezoid(qrs_seg, dx=dt_ms))

    return {
        "peak_energies": peak_energies,
        "qrs_widths_ms": qrs_widths_ms,
        "energy_traces": energy_traces,
        "qrs_integrals": qrs_integrals,
    }


def detect_sls_triggers(
    rr_ms: np.ndarray,
    *,
    short_ratio: float = 0.80,
    long_ratio: float = 1.20,
    min_triplets: int = 1,
) -> Dict[str, Any]:
    """Detect Short-Long-Short R-R interval sequences that precede TdP.

    Source: Apr28 notebook line 10686.

    An SLS trigger is: RR[i] < short_ratio * median (short / premature beat),
    followed by RR[i+1] > long_ratio * median (compensatory pause),
    followed by RR[i+2] < short_ratio * median (trigger beat).
    """
    rr = np.asarray(rr_ms, float)
    if rr.size < 4:
        return {"sls_count": 0, "sls_indices": [], "sls_present": False}

    med = float(np.median(rr))
    short_thr = short_ratio * med
    long_thr = long_ratio * med

    sls_indices: List[int] = []
    for i in range(len(rr) - 2):
        if rr[i] < short_thr and rr[i + 1] > long_thr and rr[i + 2] < short_thr:
            sls_indices.append(i)

    return {
        "sls_count": len(sls_indices),
        "sls_indices": sls_indices,
        "sls_present": len(sls_indices) >= min_triplets,
    }


def detect_tdp_amplitude_modulation(
    peak_energies: np.ndarray,
    *,
    min_beats: int = 8,
    mod_depth_threshold: float = 0.30,
    min_envelope_cycles: float = 0.5,
) -> Dict[str, Any]:
    """Detect the sinusoidal amplitude modulation characteristic of TdP.

    Source: Apr28 notebook line 10734.

    TdP causes the QRS peak energy to oscillate sinusoidally as the axis
    "twists" around the isoelectric line. The typical modulation period
    is 5-20 beats (corresponding to the visible spindle).

    Method:
      1. Normalize peak energies to [0, 1].
      2. Remove linear trend (detrend).
      3. Compute analytic signal via Hilbert transform → envelope.
      4. Measure modulation depth = (env_max - env_min) / env_mean.
      5. Count zero-crossings of detrended signal (phase reversals).
      6. Estimate dominant modulation frequency via autocorrelation.
    """
    pe = np.asarray(peak_energies, float)
    n = pe.size

    if n < min_beats or np.ptp(pe) < 1e-9:
        return {
            "modulation_depth": 0.0,
            "modulation_period_beats": 0.0,
            "phase_reversals": 0,
            "envelope_cycles": 0.0,
            "tdp_amplitude_modulation_detected": False,
        }

    # Normalize
    pe_norm = (pe - np.min(pe)) / (np.ptp(pe) + 1e-12)

    # Detrend (remove linear drift)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, pe_norm, 1)
    pe_detrended = pe_norm - (slope * x + intercept)

    # Hilbert transform for amplitude envelope
    analytic = hilbert(pe_detrended)
    envelope = np.abs(analytic)

    # Modulation depth
    env_mean = float(np.mean(envelope))
    env_max = float(np.max(envelope))
    env_min = float(np.min(envelope))
    mod_depth = (env_max - env_min) / max(env_mean, 1e-9)

    # Phase reversals: zero-crossings of detrended signal
    signs = np.sign(pe_detrended)
    sign_changes = np.diff(signs)
    phase_reversals = int(np.sum(np.abs(sign_changes) > 0))

    # Estimate modulation period via autocorrelation
    acf = np.correlate(pe_detrended, pe_detrended, mode="full")
    acf = acf[n - 1:]
    acf = acf / (acf[0] + 1e-12)

    search_lo = 3
    search_hi = min(20, n // 2)
    mod_period = 0.0
    if search_hi > search_lo:
        acf_segment = acf[search_lo:search_hi + 1]
        peaks, _ = find_peaks(acf_segment, height=0.15)
        if len(peaks) > 0:
            mod_period = float(peaks[0] + search_lo)

    envelope_cycles = phase_reversals / 2.0 if phase_reversals > 0 else 0.0

    detected = (
        mod_depth >= mod_depth_threshold
        and envelope_cycles >= min_envelope_cycles
        and phase_reversals >= 2
    )

    return {
        "modulation_depth": float(mod_depth),
        "modulation_period_beats": float(mod_period),
        "phase_reversals": int(phase_reversals),
        "envelope_cycles": float(envelope_cycles),
        "tdp_amplitude_modulation_detected": bool(detected),
    }


def classify_beats_by_energy(
    peak_energies: np.ndarray,
    qrs_widths_ms: np.ndarray,
    *,
    width_ratio_threshold: float = 1.30,
    amplitude_ratio_threshold: float = 0.70,
) -> Dict[str, Any]:
    """Classify beats as "normal" vs "ectopic" using energy morphology.

    Source: Apr28 notebook line 10839.

    Ectopic beats (PVCs) typically have:
      - Lower peak energy (aberrant conduction → less synchronous depolarization)
      - Wider QRS duration (ventricular origin → slower cell-to-cell conduction)
    """
    pe = np.asarray(peak_energies, float)
    qw = np.asarray(qrs_widths_ms, float)
    n = pe.size

    if n < 4:
        return {
            "labels": ["N"] * n,
            "n_ectopic": 0,
            "ectopic_fraction": 0.0,
            "ectopic_indices": [],
            "energy_bimodality": 0.0,
        }

    med_e = float(np.median(pe))
    med_w = float(np.median(qw))

    labels: List[str] = []
    ectopic_idx: List[int] = []
    for i in range(n):
        is_ectopic = False
        if pe[i] < amplitude_ratio_threshold * med_e:
            is_ectopic = True
        elif qw[i] > width_ratio_threshold * med_w and pe[i] < med_e:
            is_ectopic = True
        if is_ectopic:
            labels.append("E")
            ectopic_idx.append(i)
        else:
            labels.append("N")

    # Ashman bimodality coefficient
    normal_e = pe[[i for i in range(n) if labels[i] == "N"]]
    ectopic_e = pe[[i for i in range(n) if labels[i] == "E"]]
    ashman = 0.0
    if len(normal_e) >= 2 and len(ectopic_e) >= 2:
        mu_n, mu_e = float(np.mean(normal_e)), float(np.mean(ectopic_e))
        var_n, var_e = float(np.var(normal_e)), float(np.var(ectopic_e))
        denom = np.sqrt(var_n + var_e)
        if denom > 1e-9:
            ashman = abs(mu_n - mu_e) / denom

    return {
        "labels": labels,
        "n_ectopic": len(ectopic_idx),
        "ectopic_fraction": len(ectopic_idx) / max(n, 1),
        "ectopic_indices": ectopic_idx,
        "energy_bimodality": float(ashman),
    }


def detect_ectopic_patterns(
    labels: List[str],
    rr_ms: np.ndarray,
    *,
    min_repeats: int = 3,
    compensatory_ratio: float = 1.15,
) -> Dict[str, Any]:
    """Detect bigeminy and trigeminy patterns from beat labels + RR intervals.

    Source: Apr28 notebook line 10927.

    Bigeminy:  N-E-N-E-N-E  (period 2)
    Trigeminy: N-N-E-N-N-E  (period 3)
    """
    rr = np.asarray(rr_ms, float)
    n = len(labels)
    result: Dict[str, Any] = {
        "bigeminy": {"present": False, "runs": 0, "longest_run": 0, "consistency": 0.0},
        "trigeminy": {"present": False, "runs": 0, "longest_run": 0, "consistency": 0.0},
    }

    if n < 6:
        return result

    # --- Bigeminy: N-E repeating (period = 2) ---
    bigem_matches = 0
    bigem_total = 0
    bigem_run = 0
    bigem_longest = 0
    bigem_runs = 0

    for i in range(0, n - 1, 2):
        bigem_total += 1
        if labels[i] == "N" and labels[i + 1] == "E":
            if i + 1 < len(rr) and i > 0:
                has_pause = rr[i + 1] > compensatory_ratio * rr[i - 1] if i > 0 else True
            else:
                has_pause = True
            if has_pause:
                bigem_matches += 1
                bigem_run += 1
                bigem_longest = max(bigem_longest, bigem_run)
            else:
                if bigem_run >= min_repeats:
                    bigem_runs += 1
                bigem_run = 0
        else:
            if bigem_run >= min_repeats:
                bigem_runs += 1
            bigem_run = 0

    if bigem_run >= min_repeats:
        bigem_runs += 1

    bigem_consistency = bigem_matches / max(bigem_total, 1)
    if bigem_matches >= min_repeats and bigem_consistency >= 0.60:
        result["bigeminy"] = {
            "present": True,
            "runs": max(bigem_runs, 1),
            "longest_run": bigem_longest,
            "consistency": float(bigem_consistency),
            "pvc_burden_pct": 50.0,
        }

    # --- Trigeminy: N-N-E repeating (period = 3) ---
    trigem_matches = 0
    trigem_total = 0
    trigem_run = 0
    trigem_longest = 0
    trigem_runs = 0

    for i in range(0, n - 2, 3):
        trigem_total += 1
        if labels[i] == "N" and labels[i + 1] == "N" and labels[i + 2] == "E":
            trigem_matches += 1
            trigem_run += 1
            trigem_longest = max(trigem_longest, trigem_run)
        else:
            if trigem_run >= min_repeats:
                trigem_runs += 1
            trigem_run = 0

    if trigem_run >= min_repeats:
        trigem_runs += 1

    trigem_consistency = trigem_matches / max(trigem_total, 1)
    if trigem_matches >= min_repeats and trigem_consistency >= 0.50:
        result["trigeminy"] = {
            "present": True,
            "runs": max(trigem_runs, 1),
            "longest_run": trigem_longest,
            "consistency": float(trigem_consistency),
            "pvc_burden_pct": 33.3,
        }

    return result


def assess_tdp_risk(
    sls_result: Dict[str, Any],
    amp_mod_result: Dict[str, Any],
    beat_class: Dict[str, Any],
    rr_ms: np.ndarray,
    peak_energies: np.ndarray,
) -> Dict[str, Any]:
    """Combine SLS triggers, amplitude modulation, and beat classification
    into an overall TdP risk assessment.

    Source: Apr28 notebook line 11039.

    TdP scoring:
      - SLS trigger present:                    +2 points
      - Amplitude modulation detected:          +3 points
      - Modulation depth > 0.50:                +1 point (severe oscillation)
      - Phase reversals >= 4:                   +1 point
      - High ectopic fraction (>20%):           +1 point (substrate)
      - Energy bimodality (Ashman > 1.5):       +1 point (distinct populations)

    Risk levels:
      0-1: negligible | 2-3: low | 4-5: moderate | 6+: high
    """
    score = 0
    components: Dict[str, bool] = {}

    if sls_result.get("sls_present", False):
        score += 2
        components["sls_trigger"] = True
    else:
        components["sls_trigger"] = False

    if amp_mod_result.get("tdp_amplitude_modulation_detected", False):
        score += 3
        components["amplitude_modulation"] = True
        if amp_mod_result.get("modulation_depth", 0) > 0.50:
            score += 1
            components["severe_modulation"] = True
        if amp_mod_result.get("phase_reversals", 0) >= 4:
            score += 1
            components["multiple_phase_reversals"] = True
    else:
        components["amplitude_modulation"] = False

    ect_frac = beat_class.get("ectopic_fraction", 0.0)
    if ect_frac > 0.20:
        score += 1
        components["high_ectopic_burden"] = True

    bimod = beat_class.get("energy_bimodality", 0.0)
    if bimod > 1.5:
        score += 1
        components["bimodal_energy"] = True

    if score >= 6:
        risk_level = "high"
    elif score >= 4:
        risk_level = "moderate"
    elif score >= 2:
        risk_level = "low"
    else:
        risk_level = "negligible"

    return {
        "tdp_score": int(score),
        "risk_level": risk_level,
        "components": components,
    }


def analyse_complex_arrhythmias(
    stack_sync: np.ndarray,
    time_ms: np.ndarray,
    rr_ms: np.ndarray,
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    """Master function: detect TdP signatures and triggered ectopic patterns
    from the synchronized beat stack and RR intervals.

    Source: Apr28 notebook line 11123.

    Works even when the underlying rhythm is AFib because it uses beat-level
    energy morphology rather than assuming a stable baseline rhythm.

    Returns dict with: per_beat_energy, sls_detection, amplitude_mod,
    beat_classification, ectopic_patterns, tdp_risk, computed_metrics
    (15 flat numeric features for the model).
    """
    n_beats = stack_sync.shape[0]
    rr = np.asarray(rr_ms, float)

    energy_data = compute_per_beat_energy(stack_sync, time_ms)
    peak_e = energy_data["peak_energies"]
    qrs_w = energy_data["qrs_widths_ms"]

    sls_result = detect_sls_triggers(rr)
    amp_mod = detect_tdp_amplitude_modulation(peak_e)
    beat_class = classify_beats_by_energy(peak_e, qrs_w)
    ectopic_patterns = detect_ectopic_patterns(beat_class["labels"], rr)
    tdp_risk = assess_tdp_risk(sls_result, amp_mod, beat_class, rr, peak_e)

    metrics: Dict[str, Any] = {}
    metrics["Complex_Energy_Peak_CV"] = float(np.std(peak_e) / max(np.mean(peak_e), 1e-9))
    metrics["Complex_Energy_Peak_IQR"] = float(iqr(peak_e))
    metrics["Complex_QRS_Width_CV"] = float(np.std(qrs_w) / max(np.mean(qrs_w), 1e-9))
    metrics["Complex_SLS_Count"] = int(sls_result["sls_count"])
    metrics["Complex_SLS_Present"] = float(sls_result["sls_present"])
    metrics["Complex_TdP_Mod_Depth"] = float(amp_mod["modulation_depth"])
    metrics["Complex_TdP_Mod_Period"] = float(amp_mod["modulation_period_beats"])
    metrics["Complex_TdP_Phase_Reversals"] = int(amp_mod["phase_reversals"])
    metrics["Complex_TdP_Detected"] = float(amp_mod["tdp_amplitude_modulation_detected"])
    metrics["Complex_Ectopic_Fraction"] = float(beat_class["ectopic_fraction"])
    metrics["Complex_Energy_Bimodality"] = float(beat_class["energy_bimodality"])
    metrics["Complex_Bigeminy_Present"] = float(ectopic_patterns["bigeminy"]["present"])
    metrics["Complex_Bigeminy_Consistency"] = float(ectopic_patterns["bigeminy"]["consistency"])
    metrics["Complex_Trigeminy_Present"] = float(ectopic_patterns["trigeminy"]["present"])
    metrics["Complex_Trigeminy_Consistency"] = float(ectopic_patterns["trigeminy"]["consistency"])
    metrics["Complex_TdP_Risk_Score"] = int(tdp_risk["tdp_score"])

    return {
        "per_beat_energy": energy_data,
        "sls_detection": sls_result,
        "amplitude_mod": amp_mod,
        "beat_classification": beat_class,
        "ectopic_patterns": ectopic_patterns,
        "tdp_risk": tdp_risk,
        "computed_metrics": metrics,
    }
