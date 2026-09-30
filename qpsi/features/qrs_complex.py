"""QRS complex feature extractors.

Source: Cell 12 (Cell 9A4) of the QPSI notebook — 21 extractor functions
that analyse QRS morphology, axis, voltage, conduction patterns, and
beat-to-beat variability.

Also exposes:
    ``inject_qrs_descriptors(semantic_flags, lead_fits)``
        Pre-pass that MUTATES *semantic_flags* in-place by adding QRS
        morphology descriptors (M-shaped V1, wide slurred S).  Must run
        BEFORE the extractor loop.

    ``generate_qrs_clinical_summary(features)``
        Utility that produces a one-paragraph clinical summary from the
        collected QRS features dict.

    ``QRS_FEATURES``
        Registry list of the 21 extractor callables.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from qpsi.features.context import FeatureContext, _f
from qpsi.features.helpers import (
    format_lead_list,
    _contiguous_groups,
    _keep_significant_groups,
    get_qrs_gaussians,
)


# ============================================================================
# HELPER FUNCTIONS FOR QRS ANALYSIS
# ============================================================================

def _lead_index_map(leads: list) -> dict[str, int]:
    return {name: i for i, name in enumerate(leads)}


def _qrs_window_idx(
    ctx: FeatureContext,
    default: tuple[float, float] = (-50, 80),
) -> tuple[int, int]:
    """Return (i0, i1) indices for QRS window using ctx.segment_bounds or default ms."""
    try:
        lo_ms, hi_ms = (ctx.segment_bounds or {}).get("QRS", default)
    except Exception:
        lo_ms, hi_ms = default
    t = np.asarray(ctx.time_ms, float)
    i0 = int(np.clip(np.searchsorted(t, lo_ms), 0, len(t) - 1))
    i1 = int(np.clip(np.searchsorted(t, hi_ms), i0 + 8, len(t)) - 1)
    return i0, i1


def _extract_qrs_matrix(
    ctx: FeatureContext,
    leads_subset: tuple[str, ...] = ("V1", "I", "V6"),
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    """Return per-beat QRS matrices (B x L x T) and a flattened, z-scored
    feature matrix (B x F) for clustering.  Falls back gracefully if a lead
    is missing.
    """
    if not hasattr(ctx, "stack_sync") or ctx.stack_sync is None:
        return None, None, []
    i0, i1 = _qrs_window_idx(ctx)
    B, L, T = ctx.stack_sync.shape
    li = _lead_index_map(ctx.leads)
    keep = [ld for ld in leads_subset if ld in li]
    if not keep:
        keep = list(ctx.leads[: min(6, L)])  # safest fallback
    idxs = [li[ld] for ld in keep]
    qrs = ctx.stack_sync[:, idxs, i0:i1].copy()  # B x len(keep) x T

    # z-score per lead & beat to remove amplitude bias from clustering
    eps = 1e-6
    mu = qrs.mean(axis=2, keepdims=True)
    sd = qrs.std(axis=2, keepdims=True)
    qrs_z = (qrs - mu) / np.maximum(sd, eps)

    # flatten to features
    feats = qrs_z.reshape(qrs_z.shape[0], -1)
    return qrs, feats, keep


try:
    import qpsi_native as _qpsi_native
    if not hasattr(_qpsi_native, "kmeans_best_of_k"):
        raise ImportError("qpsi_native found but missing Rust KMeans")
    _HAS_RUST_KMEANS = True
except ImportError:
    _HAS_RUST_KMEANS = False


def _kmeans_best_of_k(
    X: Optional[np.ndarray],
    k_min: int = 1,
    k_max: int = 3,
    random_state: int = 7,
) -> tuple[np.ndarray, int]:
    """Pick k by silhouette (k=1 allowed).  Returns labels and chosen k.

    Guard: if fewer than 2 clusters are feasible, returns all-zero labels
    immediately — avoids ``silhouette_score`` crash on degenerate input.
    """
    if X is None or len(X) < 3:
        return np.zeros(len(X) if X is not None else 0, dtype=int), 1

    # Rust-accelerated KMeans (if available)
    if _HAS_RUST_KMEANS:
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        labels, k = _qpsi_native.kmeans_best_of_k(
            X_c, X_c.shape[0], X_c.shape[1],
            k_min, k_max, 10, random_state,
        )
        return np.asarray(labels, dtype=int), int(k)

    best_k, best_lab, best_s = 1, np.zeros(len(X), int), -1.0

    for k in range(k_min, min(k_max, len(X)) + 1):
        if k < 1:
            continue
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            lab = km.fit_predict(X)

            if k < 2:
                s = -1.0  # silhouette undefined for k=1
            else:
                n_labels = len(set(lab))
                # Silhouette is undefined if 1 label OR if n_labels == n_samples
                if n_labels < 2 or n_labels >= len(X):
                    s = -1.0
                else:
                    s = silhouette_score(X, lab)

            if s > best_s:
                best_k, best_lab, best_s = k, lab, s
        except Exception:
            continue

    return best_lab, best_k


# --- Constrained QRS fitter helpers (avoid ST/T leakage) ------------------

@dataclass
class QRSFit:
    params: list[tuple[float, float, float]]  # list of (A, mu, sigma)
    energy_primary_frac: float
    used_two_lobes: bool
    t_on_ms: float
    t_off_ms: float


def _deriv_ms(x: np.ndarray, t_ms: np.ndarray) -> np.ndarray:
    dt = float(np.mean(np.diff(t_ms)))
    return np.gradient(x, dt)


def _adaptive_amp_tau(xw: np.ndarray, AR: float) -> float:
    noise = np.median(np.abs(np.diff(xw))) / 0.6745  # MAD of diff
    return float(np.clip(3 * noise / (AR + 1e-6), 0.25, 0.5))


def _lead_gaussians_from_fits(lead_fits: dict, lead: str) -> list[dict]:
    """Return avg_gaussians for a lead from a *lead_fits* dict directly.

    Tolerant to shape variants — checks both ``"avg"`` and
    ``"avg_gaussians"`` keys.
    """
    try:
        lf = lead_fits.get(lead) or {}
        return lf.get("avg") or lf.get("avg_gaussians") or []
    except Exception:
        return []


def _m_shaped_qrs_v1(lead_fits: dict) -> dict:
    """Detect an RSR'-like ('M'-shaped) pattern in V1."""
    G = _lead_gaussians_from_fits(lead_fits, "V1")
    # take QRS-family lobes near 0 ms window; use positive peaks
    pos = [
        (g["center_ms"], g["amp_mv"], g.get("sigma_ms", 8.0))
        for g in G
        if g.get("wave_type", "").upper() in ("QRS", "R", "S")
        and (g["amp_mv"] > 0)
    ]
    pos = sorted(pos, key=lambda z: z[0])
    present = False
    if len(pos) >= 2:
        # Two positive lobes separated >= 15 ms and both reasonably sized
        for (t1, a1, s1), (t2, a2, s2) in zip(pos, pos[1:]):
            if (t2 - t1) >= 15.0 and min(a1, a2) >= 0.12:
                present = True
                break
    if present:
        return {
            "m_shaped_qrs_v1": {
                "present": True,
                "description": "'M'-shaped QRS in V1",
            }
        }
    return {}


def _wide_slurred_s_I_V6(lead_fits: dict) -> dict:
    """Detect wide terminal S in I and/or V6.

    A negative lobe after the main R with width proxy sigma >= 15 ms
    and |amp| >= 0.1 mV.
    """
    out: dict[str, Any] = {}
    for lead in ("I", "V6"):
        G = _lead_gaussians_from_fits(lead_fits, lead)
        if not G:
            continue
        # Find R (largest positive near 0) then any subsequent negative with sigma wide
        R = max(
            [g for g in G if g["amp_mv"] > 0],
            key=lambda g: g["amp_mv"],
            default=None,
        )
        S_wide = None
        if R:
            candidates = [
                g for g in G if g["amp_mv"] < 0 and g["center_ms"] > R["center_ms"]
            ]
            for g in sorted(candidates, key=lambda z: z["center_ms"]):
                if abs(g["amp_mv"]) >= 0.10 and g.get("sigma_ms", 0.0) >= 15.0:
                    S_wide = g
                    break
        if S_wide:
            key = "wide_slurred_s_I" if lead == "I" else "wide_slurred_s_V6"
            label = (
                "Wide slurred S wave in lead I"
                if lead == "I"
                else "Wide slurred S wave in lead V6"
            )
            out[key] = {"present": True, "description": label}
    return out


def _masked_indices_for_qrs(
    xw: np.ndarray,
    tw_ms: np.ndarray,
    tR_ms: float,
) -> np.ndarray:
    """Build a mask that keeps only true QRS support (rejects ST/T).

    Stricter version: amplitude gate never below 0.35*AR, slope gate
    0.35*max|dV/dt|, and a latency window around the data-driven R time.
    """
    # amplitude gate
    AR = float(np.max(np.abs(xw)) + 1e-12)
    noise = np.median(np.abs(np.diff(xw))) / 0.6745
    tau_amp_adapt = float(np.clip(3 * noise / (AR + 1e-6), 0.25, 0.5))
    tau_amp = max(0.35, tau_amp_adapt)  # never let it drop below 0.35
    mask_amp = np.abs(xw) >= tau_amp * AR

    # slope gate
    dx = _deriv_ms(xw, tw_ms)
    maxdx = float(np.max(np.abs(dx)) + 1e-12)
    tau_slope = 0.35  # a bit stricter
    mask_slope = np.abs(dx) >= tau_slope * maxdx

    # latency window around R: [R-25, R+55] ms (keeps S but excludes early ST)
    mask_lat = (tw_ms >= (tR_ms - 25.0)) & (tw_ms <= (tR_ms + 55.0))

    return mask_amp & mask_slope & mask_lat


def _qrs_bounds_from_slope(
    xw_full: np.ndarray,
    tw_full: np.ndarray,
) -> tuple[float, float]:
    """QRS onset/offset using low-slope rule.

    End when |dV/dt| < 10% of in-window max and stays low >= 20 ms.
    Start mirrored to the left.  Slightly more robust to small plateaus.
    """
    dx = _deriv_ms(xw_full, tw_full)
    thr = 0.10 * float(np.max(np.abs(dx)) + 1e-12)

    below = np.abs(dx) < thr
    need = max(1, int(round(20.0 / float(np.mean(np.diff(tw_full))))))

    # Right bound
    run = 0
    end_idx = len(below) - 1
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= need:
            end_idx = max(0, i - need + 1)
            break
    t_off = float(tw_full[min(end_idx, len(tw_full) - 1)])

    # Left bound
    run = 0
    on_idx = 0
    for i in range(len(below) - 1, -1, -1):
        b = below[i]
        run = run + 1 if b else 0
        if run >= need:
            on_idx = min(len(tw_full) - 1, i + need - 1)
            break
    t_on = float(tw_full[max(0, on_idx)])
    return t_on, t_off


@dataclass
class QRSFitResult:
    params: List[Tuple[float, float, float]]  # [(A, mu_ms, sigma_ms), ...]
    yfit: np.ndarray  # same length as time_ms
    mask: np.ndarray  # boolean mask used for fitting
    r_peak_time: float  # Detected R peak location


def _bandpass_10_40_for_mask(x: np.ndarray, fs_hz: float) -> np.ndarray:
    """Very cheap 10-40 Hz 'band-pass' for support selection (no phase
    fidelity needed).  One-pole HP(10 Hz) then LP(40 Hz).
    """
    if fs_hz <= 0:
        return x.copy()

    def one_pole_hp(sig: np.ndarray, fc: float) -> np.ndarray:
        a = np.exp(-2 * np.pi * fc / fs_hz)
        y = np.zeros_like(sig)
        prev_x = 0.0
        for i, xi in enumerate(sig):
            y[i] = a * (y[i - 1] if i else 0.0) + a * (xi - prev_x)
            prev_x = xi
        return y

    def one_pole_lp(sig: np.ndarray, fc: float) -> np.ndarray:
        a = np.exp(-2 * np.pi * fc / fs_hz)
        y = np.zeros_like(sig)
        for i, xi in enumerate(sig):
            y[i] = (1 - a) * xi + a * (y[i - 1] if i else 0.0)
        return y

    y = one_pole_hp(x, 10.0)
    y = one_pole_lp(y, 40.0)
    return y


def _gauss(t: np.ndarray, A: float, mu: float, s: float) -> np.ndarray:
    z = (t - mu) / (s + 1e-9)
    return A * np.exp(-0.5 * z * z)


# H30 / Q-A6-12: get_qrs_gaussians moved to qpsi.features.helpers (canonical home).
# Imported above; previously defined locally with the same Cell-12 broader filter.


def calculate_qrs_duration(lead_fits: Dict[str, Any], lead: str) -> Optional[float]:
    """Calculate QRS duration from Gaussian parameters."""
    qrs_gaussians = get_qrs_gaussians(lead_fits, lead)
    if not qrs_gaussians:
        return None

    # Find earliest and latest points of QRS complex
    start = min(g["center_ms"] - 2 * g["sigma_ms"] for g in qrs_gaussians)
    end = max(g["center_ms"] + 2 * g["sigma_ms"] for g in qrs_gaussians)
    return end - start


def calculate_qrs_axis(lead_fits: Dict[str, Any]) -> Optional[float]:
    """Calculate QRS axis using leads I and aVF."""
    qrs_I = get_qrs_gaussians(lead_fits, "I")
    qrs_aVF = get_qrs_gaussians(lead_fits, "aVF")

    if not qrs_I or not qrs_aVF:
        return None

    # Calculate net QRS amplitude in each lead
    net_I = sum(g["amp_mv"] for g in qrs_I)
    net_aVF = sum(g["amp_mv"] for g in qrs_aVF)

    if net_I == 0 and net_aVF == 0:
        return None

    # Calculate axis in degrees
    return np.degrees(np.arctan2(net_aVF, net_I))


def calculate_pr_interval(
    lead_fits: Dict[str, Any],
    lead: str = "II",
) -> Optional[float]:
    """Calculate PR interval from P-wave start to QRS start."""
    if lead not in lead_fits:
        return None

    # Get P and QRS components
    all_gaussians = lead_fits[lead].get("avg", [])
    p_gaussians = [g for g in all_gaussians if g.get("wave_type") == "P"]
    qrs_gaussians = [
        g for g in all_gaussians if g.get("wave_type") in ["Q", "R", "S", "QRS"]
    ]

    if not p_gaussians or not qrs_gaussians:
        return None

    p_start = min(g["center_ms"] - g["sigma_ms"] for g in p_gaussians)
    qrs_start = min(g["center_ms"] - g["sigma_ms"] for g in qrs_gaussians)

    return qrs_start - p_start


# --- Second variant: ctx-based lead-gaussian accessor ---------------------
# Used by qrs_bundle_branch_patterns (takes ctx, not lead_fits directly)

def _lead_gaussians(ctx: FeatureContext, lead: str) -> list[dict]:
    """Return avg-gaussians list for a lead if available (ctx-based)."""
    try:
        lf = (ctx.lead_fits or {}).get(lead, {})
        return lf.get("avg", []) or lf.get("avg_gaussians", [])
    except Exception:
        return []


def _qrs_components_from_gauss(
    gauss_list: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Pull simple Q/R/S positives and negatives from fitted gaussians."""
    pos = [
        g
        for g in gauss_list
        if g.get("wave_type", "").upper() == "QRS"
        and float(g.get("amp_mv", 0.0)) > 0
    ]
    neg = [
        g
        for g in gauss_list
        if g.get("wave_type", "").upper() == "QRS"
        and float(g.get("amp_mv", 0.0)) < 0
    ]
    return pos, neg


def _qrs_duration_est_ms(gauss_list: list[dict]) -> Optional[float]:
    """Crude width from earliest to latest |amp|>0 gaussians (+/-2sigma)."""
    if not gauss_list:
        return None
    t0 = []
    for g in gauss_list:
        if g.get("wave_type", "").upper() != "QRS":
            continue
        c = float(g.get("center_ms", g.get("center", 0.0)))
        s = float(g.get("sigma_ms", g.get("sigma", 10.0)))
        t0.append((c - 2 * s, c + 2 * s))
    if not t0:
        return None
    lo = min(a for a, b in t0)
    hi = max(b for a, b in t0)
    return max(0.0, hi - lo)


# ============================================================================
# inject_qrs_descriptors — pre-pass (MUTATES semantic_flags in-place)
# ============================================================================

def inject_qrs_descriptors(semantic_flags: dict, lead_fits: dict) -> dict:
    """Add QRS morphology descriptors and neutralise diagnosis-like wording.

    **IMPORTANT**: This is a pre-pass that MUTATES *semantic_flags* in-place.
    It must run BEFORE the main extractor loop so that downstream extractors
    can see the injected descriptors.

    Args:
        semantic_flags: The mutable semantic-flags dict (will be modified).
        lead_fits: Per-lead Gaussian fit data.

    Returns:
        The (mutated) *semantic_flags* dict.
    """
    sf = dict(semantic_flags or {})
    sf.setdefault("QRS", {})
    # Add morphology descriptors
    sf["QRS"].update(_m_shaped_qrs_v1(lead_fits))
    sf["QRS"].update(_wide_slurred_s_I_V6(lead_fits))

    # Neutralise any diagnosis-y wording that may slip in
    for name, fd in list(sf["QRS"].items()):
        if not isinstance(fd, dict):
            continue
        desc = (fd.get("description") or "").strip()
        # Strip trailing "suggesting ... / may indicate ..."
        desc = re.sub(
            r"\s*(suggesting|may indicate[s]?|indicating).*$",
            "",
            desc,
            flags=re.IGNORECASE,
        ).strip()
        if name == "wide_qrs_complex":
            dur = fd.get("duration_ms")
            if dur:
                desc = f"QRS duration {float(dur):.0f} ms"
            else:
                desc = "QRS duration prolonged"
        fd["description"] = desc
        sf["QRS"][name] = fd
    return sf


# ============================================================================
# QRS FEATURE EXTRACTORS (21 functions, all ctx -> dict[str, Any])
# ============================================================================

# ---------- 0. QRS Morphology Clusters ------------------------------------

def qrs_morphology_clusters(ctx: FeatureContext) -> dict[str, Any]:
    """Cluster per-beat QRS morphology across a compact lead subset.

    Emits dominant cluster share and indices of minority (ectopic) beats.
    """
    rr = getattr(ctx, "rr_intervals", []) or []
    qrs, X, used_leads = _extract_qrs_matrix(ctx)
    if X is None:
        return {}

    labels, k = _kmeans_best_of_k(X, k_min=1, k_max=3)
    # dominant vs minority sets
    if labels.size == 0:
        return {}

    values, counts = np.unique(labels, return_counts=True)
    dom_lab = int(values[np.argmax(counts)])
    dom_frac = float(counts.max()) / float(labels.size)
    ectopic_idx = [int(i) for i, lab in enumerate(labels) if lab != dom_lab]

    # optional "prematurity" tag: short RR before the ectopic, pause after
    prem: list[int] = []
    if rr and len(rr) == labels.size:
        med = float(np.median(rr))
        for i in ectopic_idx:
            prev_rr = rr[i - 1] if i - 1 >= 0 else None
            next_rr = rr[i] if i < len(rr) else None
            if (
                prev_rr
                and next_rr
                and prev_rr < 0.85 * med
                and next_rr > 1.10 * med
            ):
                prem.append(i)

    return {
        "qrs_clusters": {
            "present": True,
            "n_clusters": int(k),
            "dominant_fraction": dom_frac,
            "minority_indices": ectopic_idx,
            "premature_indices": prem,
            "leads_used": used_leads,
        }
    }


# ---------- 1. Wide QRS Complex -------------------------------------------

def qrs_wide_complex(ctx: FeatureContext) -> dict[str, Any]:
    """Detect wide QRS complex (>120 ms) — intraventricular conduction delay."""
    wide_leads: dict[str, float] = {}

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        duration = calculate_qrs_duration(ctx.lead_fits, lead)
        if duration and duration > 120:
            wide_leads[lead] = duration

    if wide_leads:
        max_duration = max(wide_leads.values())

        # Determine type of intraventricular conduction delay
        v1_morph = ctx.lead_fits.get("V1", {}).get("avg", [])
        v6_morph = ctx.lead_fits.get("V6", {}).get("avg", [])

        block_type = "Intraventricular conduction delay"
        if max_duration > 120:
            if v1_morph and v6_morph:
                # right IVCD pattern: RSR' in V1, wide S in V6
                # left IVCD pattern: QS or rS in V1, monophasic R in V6
                r_in_v1 = sum(
                    g["amp_mv"]
                    for g in v1_morph
                    if g.get("wave_type") == "R" and g["amp_mv"] > 0
                )
                s_in_v6 = sum(
                    abs(g["amp_mv"])
                    for g in v6_morph
                    if g.get("wave_type") == "S" and g["amp_mv"] < 0
                )

                if r_in_v1 > 0.5 and s_in_v6 > 0.1:
                    block_type = "Right intraventricular conduction delay (right intraventricular conduction delay)"
                elif r_in_v1 < 0.1:
                    block_type = "Left intraventricular conduction delay (left intraventricular conduction delay)"

        return {
            "wide_qrs_complex": {
                "present": True,
                "description": f"{block_type} with QRS duration {max_duration:.0f}ms, ",
                "duration_ms": max_duration,
                "affected_leads": list(wide_leads.keys()),
            }
        }
    return {}


# ---------- 2. Pathological Q Waves --------------------------------------

def qrs_pathological_q_waves(ctx: FeatureContext) -> dict[str, Any]:
    """Detect pathological Q waves (>40 ms duration and >25% of R wave)
    — indicates prior ST-T abnormality pattern.
    """
    path_q_leads: dict[str, dict] = {}

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # Identify Q wave (negative deflection before R) and R wave
        q_components = [
            g for g in qrs_gaussians if g["amp_mv"] < -0.1 and g["center_ms"] < 0
        ]
        r_components = [g for g in qrs_gaussians if g["amp_mv"] > 0.1]

        if q_components and r_components:
            q_depth = abs(min(g["amp_mv"] for g in q_components))
            r_height = max(g["amp_mv"] for g in r_components)
            q_duration = max(2 * g["sigma_ms"] for g in q_components)

            # Pathological if Q > 25% of R and duration > 40 ms
            if q_depth > 0.25 * r_height and q_duration > 40:
                path_q_leads[lead] = {
                    "depth_ratio": q_depth / r_height,
                    "duration": q_duration,
                }

    if path_q_leads:
        # Determine location based on affected leads
        location = ""
        if any(l in path_q_leads for l in ["V1", "V2", "V3", "V4"]):
            location = "anterior"
        elif any(l in path_q_leads for l in ["II", "III", "aVF"]):
            location = "inferior"
        elif any(l in path_q_leads for l in ["I", "aVL", "V5", "V6"]):
            location = "lateral"

        return {
            "pathological_q_waves": {
                "present": True,
                "description": (
                    f"Pathological Q waves in"
                    f" {format_lead_list(list(path_q_leads.keys()))}, "
                ),
                "location": location,
                "affected_leads": list(path_q_leads.keys()),
            }
        }
    return {}


# ---------- 3. Low Voltage QRS -------------------------------------------

def qrs_low_voltage(ctx: FeatureContext) -> dict[str, Any]:
    """Detect low voltage QRS complexes.

    <5 mm (0.5 mV) in limb leads or <10 mm (1.0 mV) in precordial leads.
    """
    limb_low: list[str] = []
    precordial_low: list[str] = []

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)
        if qrs_gaussians:
            # Calculate total QRS amplitude
            total_amp = sum(abs(g["amp_mv"]) for g in qrs_gaussians)

            if lead in ["I", "II", "III", "aVR", "aVL", "aVF"]:
                if total_amp < 0.5:  # 5 mm = 0.5 mV
                    limb_low.append(lead)
            elif lead.startswith("V"):
                if total_amp < 1.0:  # 10 mm = 1.0 mV
                    precordial_low.append(lead)

    if len(limb_low) >= 3 or len(precordial_low) >= 3:
        return {
            "qrs_low_voltage": {
                "present": True,
                "description": (
                    "Low voltage QRS complexes suggesting low-voltage with"
                    " electrical alternans pattern, cardiomyopathy, or obesity"
                ),
                "limb_leads_affected": limb_low,
                "precordial_leads_affected": precordial_low,
            }
        }
    return {}


# ---------- 4. QRS Fragmentation -----------------------------------------

def qrs_fragmentation(ctx: FeatureContext) -> dict[str, Any]:
    """Fragmented QRS: >= 3 distinct intra-QRS peaks, peak-to-peak separation
    >= 15 ms, at least one QRS peak >= 0.20 mV, with territorial support.
    """
    frag: dict[str, bool] = {}

    for lead in getattr(ctx, "leads", []):
        lf = getattr(ctx, "lead_fits", {}) or {}
        if lead not in lf:
            continue
        try:
            qrs_gaussians = get_qrs_gaussians(lf, lead)
        except Exception:
            qrs_gaussians = [
                g
                for g in lf.get(lead, {}).get("avg", [])
                if g.get("wave_type") in ("Q", "R", "S", "QRS")
            ]

        if len(qrs_gaussians) < 3:
            continue

        centers = sorted([g.get("center_ms", 0.0) for g in qrs_gaussians])
        amps = [abs(g.get("amp_mv", 0.0)) for g in qrs_gaussians]
        if max(amps, default=0.0) < 0.20:
            continue

        distinct = 1
        for i in range(1, len(centers)):
            if centers[i] - centers[i - 1] > 15.0:  # raised from 10 ms
                distinct += 1

        if distinct >= 3:
            frag[lead] = True

    if not frag:
        return {}

    leads = list(frag.keys())
    try:
        groups = _contiguous_groups(leads)
    except Exception:
        groups = [sorted(leads)]
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        return {}

    best = max(groups, key=len)
    return {
        "qrs_fragmentation": {
            "present": True,
            "description": f"Fragmented QRS in {', '.join(best)} (\u22653 peaks, \u226515 ms apart; \u22650.20 mV).",
            "affected_leads": best,
        }
    }


# ---------- 5. Poor R Wave Progression ------------------------------------

def qrs_poor_r_progression(ctx: FeatureContext) -> dict[str, Any]:
    """Detect poor R wave progression across precordial leads.

    R wave should progressively increase from V1 to V6.
    """
    r_amplitudes: dict[str, float] = {}

    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)
        r_components = [g for g in qrs_gaussians if g["amp_mv"] > 0]

        if r_components:
            r_amplitudes[lead] = max(g["amp_mv"] for g in r_components)

    if len(r_amplitudes) >= 4:
        # Check for poor progression
        v3_r = r_amplitudes.get("V3", 0)
        v4_r = r_amplitudes.get("V4", 0)

        # Poor progression if R wave still small in V3/V4
        if v3_r < 0.3 or v4_r < 0.5:
            return {
                "poor_r_wave_progression": {
                    "present": True,
                    "description": (
                        "Poor R wave progression in precordial leads, "
                        "suggesting anterior ischemic pattern or cardiomyopathy"
                    ),
                    "r_amplitudes": r_amplitudes,
                }
            }
    return {}


# ---------- 6. QRS Axis Deviation -----------------------------------------

def qrs_axis_deviation(ctx: FeatureContext) -> dict[str, Any]:
    """Detect abnormal QRS axis.  Normal: -30 deg to +90 deg."""
    axis = calculate_qrs_axis(ctx.lead_fits)

    if axis is not None:
        # Convert numpy array to scalar BEFORE any comparisons
        if isinstance(axis, np.ndarray):
            axis = float(axis.item())
        else:
            axis = float(axis)

        if axis < -30:
            return {
                "left_axis_deviation": {
                    "present": True,
                    "description": (
                        f"Left axis deviation ({axis:.0f}\u00b0), suggesting "
                        f"left anterior fascicular block or LVH"
                    ),
                    "axis_degrees": axis,
                }
            }
        elif axis > 90:
            return {
                "right_axis_deviation": {
                    "present": True,
                    "description": (
                        f"Right axis deviation ({axis:.0f}\u00b0), suggesting "
                        f"RVH or lateral ischemic pattern"
                    ),
                    "axis_degrees": axis,
                }
            }
    return {}


# ---------- 7. Delta Wave (pre-excitation) --------------------------------

def qrs_delta_wave(ctx: FeatureContext) -> dict[str, Any]:
    """Detect delta wave (slurred QRS upstroke) — pre-excitation pattern."""
    delta_leads: list[str] = []

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # Delta wave = early, slow-rising component before main QRS
        early_components = [g for g in qrs_gaussians if g["center_ms"] < -20]

        if early_components:
            for comp in early_components:
                if comp["sigma_ms"] > 20:  # Wide, slurred component
                    delta_leads.append(lead)
                    break

    # Also check for short PR interval (characteristic of pre-excitation)
    pr_interval = calculate_pr_interval(ctx.lead_fits, "II")

    if delta_leads and pr_interval and pr_interval < 120:
        return {
            "delta_wave": {
                "present": True,
                "description": (
                    f"Delta wave in {format_lead_list(delta_leads)} with short"
                    " PR interval, may indicates pre-excitation pattern"
                    " syndrome — risk for SVT"
                ),
                "affected_leads": delta_leads,
                "pr_interval": pr_interval,
            }
        }
    return {}


# ---------- 8. Ventricular Hypertrophy (Sokolow-Lyon) --------------------

def qrs_ventricular_hypertrophy(ctx: FeatureContext) -> dict[str, Any]:
    """Detect voltage criteria for ventricular increased QRS voltage pattern.

    Sokolow-Lyon criteria for LVH: S in V1 + R in V5/V6 > 35 mm.
    """
    s_v1 = 0.0
    r_v5 = 0.0
    r_v6 = 0.0

    # Measure S wave in V1
    if "V1" in ctx.lead_fits:
        v1_qrs = get_qrs_gaussians(ctx.lead_fits, "V1")
        s_components = [g for g in v1_qrs if g["amp_mv"] < 0]
        if s_components:
            s_v1 = abs(min(g["amp_mv"] for g in s_components))

    # Measure R wave in V5
    if "V5" in ctx.lead_fits:
        v5_qrs = get_qrs_gaussians(ctx.lead_fits, "V5")
        r_components = [g for g in v5_qrs if g["amp_mv"] > 0]
        if r_components:
            r_v5 = max(g["amp_mv"] for g in r_components)

    # Measure R wave in V6
    if "V6" in ctx.lead_fits:
        v6_qrs = get_qrs_gaussians(ctx.lead_fits, "V6")
        r_components = [g for g in v6_qrs if g["amp_mv"] > 0]
        if r_components:
            r_v6 = max(g["amp_mv"] for g in r_components)

    # Calculate Sokolow-Lyon index
    sokolow = s_v1 + max(r_v5, r_v6)

    if sokolow > 3.5:  # 35 mm = 3.5 mV
        return {
            "left_ventricular_hypertrophy": {
                "present": True,
                "description": (
                    f"Voltage criteria for LVH (Sokolow-Lyon {sokolow:.1f}mV), "
                ),
                "sokolow_lyon_voltage": sokolow,
            }
        }
    return {}


# ---------- 9. Epsilon Wave (ARVD) ---------------------------------------

def qrs_epsilon_wave(ctx: FeatureContext) -> dict[str, Any]:
    """Detect epsilon wave (small positive deflection after QRS) — ARVD."""
    epsilon_leads: list[str] = []

    for lead in ["V1", "V2", "V3"]:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # Look for small positive deflection after main QRS (>80 ms after R peak)
        late_components = [
            g
            for g in qrs_gaussians
            if g["center_ms"] > 80 and 0 < g["amp_mv"] < 0.2
        ]

        if late_components:
            epsilon_leads.append(lead)

    if epsilon_leads:
        return {
            "epsilon_wave": {
                "present": True,
                "description": (
                    f"Epsilon wave in {format_lead_list(epsilon_leads)},"
                    " suggestive of arrhythmogenic right ventricular"
                    " dysplasia (ARVD)"
                ),
                "affected_leads": epsilon_leads,
            }
        }
    return {}


# ---------- 10. QRS Notching ---------------------------------------------

def qrs_notching(ctx: FeatureContext) -> dict[str, Any]:
    """Detect notched QRS complex — intraventricular conduction delay."""
    notched_leads: dict[str, int] = {}

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # Count distinct positive and negative peaks
        pos_peaks = [g for g in qrs_gaussians if g["amp_mv"] > 0.1]
        neg_peaks = [g for g in qrs_gaussians if g["amp_mv"] < -0.1]

        # Notching if multiple peaks of same polarity
        if len(pos_peaks) >= 2 or len(neg_peaks) >= 2:
            notched_leads[lead] = max(len(pos_peaks), len(neg_peaks))

    if len(notched_leads) >= 2:
        return {
            "qrs_notching": {
                "present": True,
                "description": (
                    f"Notched QRS in"
                    f" {format_lead_list(list(notched_leads.keys()))},"
                    " suggesting intraventricular conduction delay"
                ),
                "affected_leads": list(notched_leads.keys()),
            }
        }
    return {}


# ---------- 11. RSR' Pattern ---------------------------------------------

def qrs_rsr_pattern(ctx: FeatureContext) -> dict[str, Any]:
    """Detect RSR' pattern in V1-V2 — classic right IVCD pattern."""
    rsr_leads: list[str] = []

    for lead in ["V1", "V2"]:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # Look for R-S-R' pattern (two R waves with S between)
        r_components = [g for g in qrs_gaussians if g["amp_mv"] > 0.1]
        s_components = [g for g in qrs_gaussians if g["amp_mv"] < -0.05]

        if len(r_components) >= 2 and len(s_components) >= 1:
            # Check temporal order: R1 < S < R2
            r_times = sorted([g["center_ms"] for g in r_components])
            s_times = [g["center_ms"] for g in s_components]

            # Verify RSR' sequence
            if s_times and r_times[0] < min(s_times) < r_times[-1]:
                rsr_leads.append(lead)

    if rsr_leads:
        return {
            "rsr_pattern": {
                "present": True,
                "description": (
                    f"RSR' pattern in {format_lead_list(rsr_leads)},"
                    " may indicates right intraventricular conduction delay"
                ),
                "affected_leads": rsr_leads,
            }
        }
    return {}


# ---------- 12. Prominent S Waves ----------------------------------------

def qrs_prominent_s_waves(ctx: FeatureContext) -> dict[str, Any]:
    """Detect deep S waves in lateral leads — may indicate RVH or right IVCD."""
    deep_s_leads: dict[str, float] = {}

    for lead in ["I", "aVL", "V5", "V6"]:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)
        s_components = [g for g in qrs_gaussians if g["amp_mv"] < -0.3]

        if s_components:
            s_depth = abs(min(g["amp_mv"] for g in s_components))
            deep_s_leads[lead] = s_depth

    if len(deep_s_leads) >= 2:
        return {
            "prominent_s_waves": {
                "present": True,
                "description": (
                    f"Prominent S waves in"
                    f" {format_lead_list(list(deep_s_leads.keys()))},"
                    " may indicates right ventricular strain or right"
                    " intraventricular conduction delay"
                ),
                "affected_leads": list(deep_s_leads.keys()),
                "depths": deep_s_leads,
            }
        }
    return {}


# ---------- 13. QS Pattern -----------------------------------------------

def qrs_qs_pattern(ctx: FeatureContext) -> dict[str, Any]:
    """Detect QS complex (no R wave) — transmural ST-T abnormality pattern."""
    qs_leads: list[str] = []

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # QS pattern: deep negative complex without R wave
        r_components = [g for g in qrs_gaussians if g["amp_mv"] > 0.05]
        qs_components = [g for g in qrs_gaussians if g["amp_mv"] < -0.2]

        if not r_components and qs_components:
            qs_leads.append(lead)

    if qs_leads:
        # Determine location
        location = ""
        if any(l in qs_leads for l in ["V1", "V2", "V3"]):
            location = "septal/anterior"
        elif any(l in qs_leads for l in ["II", "III", "aVF"]):
            location = "inferior"

        return {
            "qs_pattern": {
                "present": True,
                "description": f"QS pattern in {format_lead_list(qs_leads)}, ",
                "affected_leads": qs_leads,
                "location": location,
            }
        }
    return {}


# ---------- 14. Tall R Waves ---------------------------------------------

def qrs_tall_r_waves(ctx: FeatureContext) -> dict[str, Any]:
    """Detect abnormally tall R waves — may indicate hypertrophy or posterior
    ischemic pattern.
    """
    tall_r_leads: dict[str, float] = {}

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)
        r_components = [g for g in qrs_gaussians if g["amp_mv"] > 0]

        if r_components:
            r_height = max(g["amp_mv"] for g in r_components)

            # Lead-specific thresholds for tall R waves
            if lead == "aVR" and r_height > 0.5:
                tall_r_leads[lead] = r_height
            elif lead in ["V1", "V2"] and r_height > 0.7:
                tall_r_leads[lead] = r_height
            elif lead in ["V5", "V6"] and r_height > 2.5:
                tall_r_leads[lead] = r_height

    if tall_r_leads:
        # Determine clinical significance
        significance = ""
        if any(l in tall_r_leads for l in ["V1", "V2"]):
            significance = "posterior ischemic pattern or RVH"
        elif any(l in tall_r_leads for l in ["V5", "V6"]):
            significance = "LVH"

        return {
            "tall_r_waves": {
                "present": True,
                "description": (
                    f"Tall R waves in"
                    f" {format_lead_list(list(tall_r_leads.keys()))},"
                    f" suggesting {significance}"
                ),
                "affected_leads": list(tall_r_leads.keys()),
                "heights": tall_r_leads,
            }
        }
    return {}


# ---------- 15. Micro QRS Voltages ---------------------------------------

def qrs_micro_voltages(ctx: FeatureContext) -> dict[str, Any]:
    """Detect very low QRS voltages (<0.1 mV) — severe cardiomyopathy or
    massive effusion.
    """
    micro_leads: list[str] = []

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)
        if qrs_gaussians:
            # Calculate total QRS amplitude
            total_amp = sum(abs(g["amp_mv"]) for g in qrs_gaussians)
            if total_amp < 0.1:  # Very low voltage
                micro_leads.append(lead)

    if len(micro_leads) >= 6:
        return {
            "micro_qrs_voltages": {
                "present": True,
                "description": (
                    f"Micro QRS voltages in {len(micro_leads)} leads, "
                ),
                "affected_leads": micro_leads,
            }
        }
    return {}


# ---------- 16. Slurred S Wave -------------------------------------------

def qrs_slurred_s_wave(ctx: FeatureContext) -> dict[str, Any]:
    """Detect slurred S wave — intraventricular conduction delay."""
    slurred_s_leads: list[str] = []

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)
        s_components = [g for g in qrs_gaussians if g["amp_mv"] < -0.1]

        for s in s_components:
            # Slurred if S wave is wide
            if s["sigma_ms"] > 25:
                slurred_s_leads.append(lead)
                break

    if len(slurred_s_leads) >= 2:
        return {
            "slurred_s_wave": {
                "present": True,
                "description": (
                    f"Slurred S waves in {format_lead_list(slurred_s_leads)},"
                    " suggesting intraventricular conduction delay"
                ),
                "affected_leads": slurred_s_leads,
            }
        }
    return {}


# ---------- 17. Absent Initial R Wave ------------------------------------

def qrs_absent_initial_r(ctx: FeatureContext) -> dict[str, Any]:
    """Detect absent initial R wave in V1-V3 — may indicate septal infarction."""
    absent_r_leads: list[str] = []

    for lead in ["V1", "V2", "V3"]:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # Check for absent initial R (starts with Q or S)
        early_r = [
            g
            for g in qrs_gaussians
            if g["amp_mv"] > 0.05 and g["center_ms"] < 20
        ]

        if not early_r:
            absent_r_leads.append(lead)

    if len(absent_r_leads) >= 2:
        return {
            "absent_initial_r_wave": {
                "present": True,
                "description": (
                    f"Absent initial R wave in"
                    f" {format_lead_list(absent_r_leads)}, "
                ),
                "affected_leads": absent_r_leads,
            }
        }
    return {}


# ---------- 18. QRS Alternans --------------------------------------------

def qrs_alternans(ctx: FeatureContext) -> dict[str, Any]:
    """Detect beat-to-beat QRS amplitude variation — associated with severe
    cardiac dysfunction.
    """
    for lead in ["II", "V3"]:
        if lead not in ctx.lead_fits:
            continue

        by_beat = ctx.lead_fits[lead].get("by_beat", [])
        if len(by_beat) < 8:
            continue

        # Get QRS amplitudes for consecutive beats
        amplitudes: list[float] = []
        for beat_idx in range(min(16, len(by_beat))):
            qrs_gaussians = [
                g
                for g in by_beat[beat_idx]
                if g.get("wave_type") in ["Q", "R", "S", "QRS"]
            ]
            if qrs_gaussians:
                total_amp = sum(abs(g["amp_mv"]) for g in qrs_gaussians)
                amplitudes.append(total_amp)

        if len(amplitudes) >= 8:
            # Check for alternating pattern (ABAB)
            even_amps = [amplitudes[i] for i in range(0, len(amplitudes), 2)]
            odd_amps = [amplitudes[i] for i in range(1, len(amplitudes), 2)]

            if len(even_amps) >= 3 and len(odd_amps) >= 3:
                even_mean = float(np.mean(even_amps))
                odd_mean = float(np.mean(odd_amps))

                # Significant alternans if >15% difference
                if abs(even_mean - odd_mean) / max(even_mean, odd_mean) > 0.15:
                    return {
                        "qrs_alternans": {
                            "present": True,
                            "description": (
                                f"QRS alternans detected in lead {lead}, "
                            ),
                            "lead": lead,
                            "variation_percent": (
                                abs(even_mean - odd_mean)
                                / max(even_mean, odd_mean)
                                * 100
                            ),
                        }
                    }
    return {}


# ---------- 19. Intrinsicoid Deflection Delay -----------------------------

def qrs_intrinsicoid_delay(ctx: FeatureContext) -> dict[str, Any]:
    """Detect delayed R peak time (intrinsicoid deflection) in V5-V6.

    Normal <45 ms; prolonged suggests LVH or left IVCD.
    """
    delayed_leads: dict[str, float] = {}

    for lead in ["V5", "V6"]:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)
        r_components = [g for g in qrs_gaussians if g["amp_mv"] > 0.1]

        if r_components:
            # Find time to peak R wave
            r_peak_time = max(g["center_ms"] for g in r_components)

            # Delayed if >45 ms
            if r_peak_time > 45:
                delayed_leads[lead] = r_peak_time

    if delayed_leads:
        return {
            "intrinsicoid_deflection_delay": {
                "present": True,
                "description": (
                    f"Delayed intrinsicoid deflection in"
                    f" {format_lead_list(list(delayed_leads.keys()))},"
                    " suggesting LVH or left intraventricular conduction delay"
                ),
                "affected_leads": list(delayed_leads.keys()),
                "peak_times": delayed_leads,
            }
        }
    return {}


# ---------- 20. Splintered QRS -------------------------------------------

def qrs_splintered(ctx: FeatureContext) -> dict[str, Any]:
    """Detect splintered QRS (multiple small deflections) — marker of
    myocardial fibrosis.
    """
    splintered_leads: list[str] = []

    for lead in ctx.leads:
        if lead not in ctx.lead_fits:
            continue

        qrs_gaussians = get_qrs_gaussians(ctx.lead_fits, lead)

        # Count small amplitude components (0.05-0.2 mV)
        small_components = [
            g for g in qrs_gaussians if 0.05 < abs(g["amp_mv"]) < 0.2
        ]

        # Splintered if 3+ small components
        if len(small_components) >= 3:
            splintered_leads.append(lead)

    if len(splintered_leads) >= 2:
        return {
            "splintered_qrs": {
                "present": True,
                "description": (
                    f"Splintered QRS in"
                    f" {format_lead_list(splintered_leads)}, "
                ),
                "affected_leads": splintered_leads,
            }
        }
    return {}


# ============================================================================
# ADDITIONAL QRS UTILITIES (not in the registry)
# ============================================================================

def qrs_bundle_branch_patterns(ctx: FeatureContext) -> dict[str, Any]:
    """Heuristics for:
      - 'M-shaped QRS in V1'
      - 'wide slurred S wave in I' (also check V6)

    Also emits a broad IVCD flag when QRS >= 120 ms without clear RBBB/LBBB.
    """
    out: dict[str, Any] = {}
    gV1 = _lead_gaussians(ctx, "V1")
    gI = _lead_gaussians(ctx, "I")
    gV6 = _lead_gaussians(ctx, "V6")
    dur_V1 = _qrs_duration_est_ms(gV1) or 0.0
    dur_I = _qrs_duration_est_ms(gI) or 0.0
    dur_V6 = _qrs_duration_est_ms(gV6) or 0.0
    qrs_wide = max(dur_V1, dur_I, dur_V6)

    # M-shaped in V1: two distinct positive lobes within QRS
    pos_V1, neg_V1 = _qrs_components_from_gauss(gV1)
    m_shape = False
    if len(pos_V1) >= 2:
        centers = sorted(float(g.get("center_ms", 0.0)) for g in pos_V1)
        # require >= 2 peaks separated by 20-60 ms
        for a, b in zip(centers, centers[1:]):
            if 20.0 <= (b - a) <= 60.0:
                m_shape = True
                break
    if m_shape and qrs_wide >= 110.0:
        out["m_shaped_v1"] = {
            "present": True,
            "description": "\u2018M\u2019-shaped QRS in V1",
        }

    # Wide slurred S in I/V6: large negative component with width > 40 ms
    def _slurred_S(gauss_list: list[dict]) -> bool:
        for g in gauss_list:
            if (
                g.get("wave_type", "").upper() == "QRS"
                and float(g.get("amp_mv", 0.0)) < -0.2
            ):
                sig = float(g.get("sigma_ms", g.get("sigma", 10.0)))
                if 2.0 * sig >= 40.0:
                    return True
        return False

    sI = _slurred_S(gI)
    sV6 = _slurred_S(gV6)
    if (sI or sV6) and qrs_wide >= 110.0:
        out["wide_slurred_s_I_V6"] = {
            "present": True,
            "description": (
                "Wide slurred S wave in lead I" + (" / V6" if sV6 else "")
            ),
        }

    # Broad IVCD fallback
    if qrs_wide >= 120.0 and not out:
        out["broad_ivcd"] = {
            "present": True,
            "description": (
                f"Broad intraventricular conduction delay"
                f" (QRS \u2248 {qrs_wide:.0f} ms)"
            ),
        }

    return out


# ============================================================================
# QRS CLINICAL SUMMARY GENERATOR
# ============================================================================

def generate_qrs_clinical_summary(features: Dict[str, Any]) -> str:
    """Generate clinical summary from QRS features."""
    if not features:
        return "QRS morphology within normal limits."

    descriptions: list[str] = []
    urgent: list[str] = []

    # Priority features that need immediate attention
    priority_features = [
        "wide_qrs_complex",
        "pathological_q_waves",
        "delta_wave",
        "epsilon_wave",
        "qrs_alternans",
        "ventricular_tachycardia",
    ]

    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict) and feature_data.get("present"):
            desc = feature_data.get("description", "")
            if (feature_name in priority_features or
                    any(word in desc.upper() for word in ["EMERGENCY", "HIGH RISK", "WARNING", "CRITICAL"])):
                urgent.append(desc)
            else:
                descriptions.append(desc)

    summary = "QRS analysis reveals: "

    # Add urgent findings first
    if urgent:
        summary += "\u26a0\ufe0f IMPORTANT: " + " ".join(urgent)
        if descriptions:
            summary += " Additional findings: "

    # Add other findings
    if descriptions:
        summary += " ".join(descriptions)

    return summary


# ============================================================================
# QRS_FEATURES — registry of the 21 extractor callables
# ============================================================================

QRS_FEATURES: list = [
    qrs_wide_complex,              # 0
    qrs_pathological_q_waves,      # 1
    qrs_low_voltage,               # 2
    qrs_fragmentation,             # 3
    qrs_poor_r_progression,        # 4
    qrs_axis_deviation,            # 5
    qrs_delta_wave,                # 6
    qrs_ventricular_hypertrophy,   # 7
    qrs_epsilon_wave,              # 8
    qrs_notching,                  # 9
    qrs_rsr_pattern,               # 10
    qrs_prominent_s_waves,         # 11
    qrs_qs_pattern,                # 12
    qrs_tall_r_waves,              # 13
    qrs_micro_voltages,            # 14
    qrs_slurred_s_wave,            # 15
    qrs_absent_initial_r,          # 16
    qrs_alternans,                 # 17
    qrs_intrinsicoid_delay,        # 18
    qrs_splintered,                # 19
    qrs_bundle_branch_patterns,    # 20
]
