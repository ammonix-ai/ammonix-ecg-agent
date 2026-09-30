"""
Global feature extractors -- 10 extractors for whole-ECG analysis.

Source: Cell 14 (9A6) of q_psi_ai_for_ecg_Feb14_Adele.ipynb (931 lines)

Extractors
----------
1. global_qtc_prolongation   -- Multi-lead QTc (Tangent Method, gold standard)
2. global_electrical_alternans -- Beat-to-beat QRS amplitude alternans
3. global_extreme_axis       -- Extreme axis deviation
4. global_poor_quality       -- ECG quality assessment (spectral + line-length)
5. global_dextrocardia       -- Dextrocardia detection
6. global_pediatric_pattern  -- Pediatric ECG pattern
7. global_athletes_heart     -- Athletic heart syndrome
8. global_critical_findings  -- Critical findings requiring immediate action
9. global_metabolic_pattern  -- Metabolic disturbance patterns
10. global_overall_impression -- Overall ECG impression summary
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qpsi.features.context import FeatureContext, _f
from qpsi.features.helpers import get_t_wave_gaussians, get_qrs_gaussians
from qpsi.features.qrs_complex import (
    calculate_qrs_axis,
    calculate_qrs_duration,
    calculate_pr_interval,
)
from qpsi.features.st_deviation import get_st_segment_info


# ---------------------------------------------------------------------------
# Local helper functions (used by global_qtc_prolongation)
# ---------------------------------------------------------------------------

def _norm_sex(x: Any) -> str:
    """Normalize sex string to 'male', 'female', or 'unknown'."""
    s = (str(x or "")).strip().lower()
    if s in {"m", "male", "man"}:
        return "male"
    if s in {"f", "female", "woman"}:
        return "female"
    return "unknown"


def _g(d: dict, *keys: str, default: Any = None) -> Any:
    """Return the value for the first key found in *d*."""
    for k in keys:
        if k in d:
            return d[k]
    return default


# ---------------------------------------------------------------------------
# 1. QTC PROLONGATION (Updated with Tangent Method)
# ---------------------------------------------------------------------------

def global_qtc_prolongation(ctx: Any, *, debug: bool = False) -> Dict[str, Any]:
    """
    Multi-lead QTc assessment using the "Tangent Method" on fitted Gaussians.
    Reconstructs the T-wave vector sum to find the steepest downslope intersection,
    ignoring U-waves.
    """
    # ---- Config --------------------------------------------------------------
    MIN_QT_MS, MAX_QT_MS = 240.0, 600.0
    MIN_VALID_BEATS_PER_LEAD = 3
    MAD_TRIM_MULT = 2.5
    MAD_MIN_ABS = 30.0
    K_QRS_ONSET = 1.5  # Feb14: start QRS slightly earlier than 1-sigma to catch Q-wave

    # ---- Guards --------------------------------------------------------------
    rr_list = getattr(ctx, "rr_intervals", None)
    if rr_list is None or len(rr_list) == 0:
        return {}

    lead_fits = getattr(ctx, "lead_fits", None)
    if not isinstance(lead_fits, dict) or not lead_fits:
        return {}

    # ---- Helpers (closure over lead_fits and constants) -----------------------

    def _get_gaussians(lead: str, beat_idx: int | None, wave_type: str) -> list[dict]:
        """Try helper accessors; else direct dict access."""
        try:
            if wave_type == "QRS":
                return get_qrs_gaussians(lead_fits, lead, beat_idx)
            if wave_type == "T":
                return get_t_wave_gaussians(lead_fits, lead, beat_idx)
        except Exception:
            pass

        src = lead_fits.get(lead, {})
        if beat_idx is None:
            arr = src.get("avg", [])
        else:
            bb = src.get("by_beat", [])
            arr = bb[beat_idx] if isinstance(bb, (list, tuple)) and beat_idx < len(bb) else []
        return [g for g in (arr or []) if g.get("wave_type") == wave_type]

    def _qrs_onset_ms(qrs_list: list[dict]) -> float | None:
        if not qrs_list:
            return None
        starts: list[float] = []
        for g_item in qrs_list:
            c = _g(g_item, "center_ms", "mu_ms", "mu", default=None)
            s = _g(g_item, "sigma_ms", "sigma", default=None)
            onset = _g(g_item, "onset_ms", "qrs_onset_ms", default=None)
            if onset is not None:
                starts.append(float(onset))
            elif c is not None and s is not None:
                # Q-onset is typically earlier than center - 1 sigma
                starts.append(float(c) - K_QRS_ONSET * float(s))
        return float(np.min(starts)) if starts else None

    def _get_qt_tangent_method(
        qrs_list: list[dict], t_list: list[dict]
    ) -> float | None:
        """
        Calculate QT using the Tangent Method (gold standard).
        1. Determine Q-onset (start of QRS).
        2. Reconstruct T-wave from Gaussians (smooth).
        3. Find T-peak.
        4. Find max slope (steepest descent) after peak.
        5. Intercept with baseline = T-end.
        """
        q0 = _qrs_onset_ms(qrs_list)
        if q0 is None or not t_list:
            return None

        # 1. Reconstruct T-wave signal (Sum of Gaussians)
        t_search_start = q0 + 100.0
        t_search_end = q0 + 800.0

        # Dense grid (1 ms resolution)
        grid_t = np.linspace(
            t_search_start, t_search_end, int(t_search_end - t_search_start)
        )
        grid_y = np.zeros_like(grid_t)

        has_valid_t = False
        for g_item in t_list:
            amp = float(_g(g_item, "amp_mv", default=0.0))
            mu = float(_g(g_item, "center_ms", "mu", default=0.0))
            sigma = float(_g(g_item, "sigma_ms", "sigma", default=10.0))

            if t_search_start < mu < t_search_end:
                grid_y += amp * np.exp(-0.5 * ((grid_t - mu) / sigma) ** 2)
                has_valid_t = True

        if not has_valid_t:
            return None

        # 2. Find T-Peak (extremum with largest absolute magnitude)
        peak_idx = np.argmax(np.abs(grid_y))
        peak_time = grid_t[peak_idx]
        peak_amp = grid_y[peak_idx]

        # 3. Find Maximum Slope point AFTER the peak
        dy = np.gradient(grid_y, 1.0)  # 1 ms step

        search_slice = slice(peak_idx + 1, None)
        dy_search = dy[search_slice]
        t_search = grid_t[search_slice]
        y_search = grid_y[search_slice]

        if len(dy_search) < 5:
            return None

        # If positive T-wave: steepest downward; if inverted: steepest upward
        if peak_amp > 0:
            slope_idx = np.argmin(dy_search)
        else:
            slope_idx = np.argmax(dy_search)

        max_slope = dy_search[slope_idx]
        val_at_slope = y_search[slope_idx]
        time_at_slope = t_search[slope_idx]

        # 4. Tangent intercept with baseline (y=0)
        if abs(max_slope) < 1e-6:
            return None

        t_end_tangent = time_at_slope - (val_at_slope / max_slope)

        # Guard: T-end must be strictly after T-peak
        if t_end_tangent <= peak_time:
            return None

        qt = t_end_tangent - q0

        if 200.0 < qt < 800.0:
            return qt
        return None

    def _mad(x: np.ndarray) -> float:
        med = np.median(x)
        return 1.4826 * float(np.median(np.abs(x - med)))

    def _trim_by_mad(x_ms: np.ndarray) -> np.ndarray:
        if x_ms.size < 3:
            return x_ms
        med = np.median(x_ms)
        mad = _mad(x_ms)
        thr = max(MAD_MIN_ABS, MAD_TRIM_MULT * (mad if np.isfinite(mad) else 0.0))
        mask = np.abs(x_ms - med) <= thr
        return x_ms[mask]

    # ---- Per-lead QT extraction -----------------------------------------------
    lead_priority = [
        "II", "V5", "V6", "I", "aVF", "V2", "III", "aVL", "V3", "aVR", "V1", "V4"
    ]
    available_leads = [ld for ld in lead_priority if ld in lead_fits]

    lead_medians: dict[str, float] = {}
    beats_kept_per_lead: dict[str, int] = {}

    for lead in available_leads:
        src = lead_fits.get(lead, {})
        by_beat = src.get("by_beat", [])
        n_beats_total = len(by_beat) if isinstance(by_beat, (list, tuple)) else 0

        # Per-beat pass collecting only VALID beats (both QRS & T present)
        perbeat_qt: list[float] = []
        if n_beats_total:
            for b in range(n_beats_total):
                qrs_g = _get_gaussians(lead, b, "QRS")
                t_g = _get_gaussians(lead, b, "T")
                qt_val = _get_qt_tangent_method(qrs_g, t_g)
                if qt_val is not None:
                    perbeat_qt.append(qt_val)

        used_fallback = False

        # If too few valid beats, fallback to 'avg' trace parameters
        if len(perbeat_qt) < MIN_VALID_BEATS_PER_LEAD:
            qrs_g = _get_gaussians(lead, None, "QRS")
            t_g = _get_gaussians(lead, None, "T")
            qt_val = _get_qt_tangent_method(qrs_g, t_g)
            if qt_val is not None:
                lead_medians[lead] = qt_val
                beats_kept_per_lead[lead] = 1
                used_fallback = True
                continue

        # If we have enough valid beats, MAD-trim & median
        if len(perbeat_qt) >= MIN_VALID_BEATS_PER_LEAD and not used_fallback:
            qts = _trim_by_mad(np.asarray(perbeat_qt, float))
            if qts.size >= MIN_VALID_BEATS_PER_LEAD:
                lead_medians[lead] = float(np.median(qts))
                beats_kept_per_lead[lead] = int(qts.size)

    if not lead_medians:
        return {}

    # ---- Fuse across leads ---------------------------------------------------
    med_list = [(ld, lead_medians[ld]) for ld in lead_priority if ld in lead_medians]
    vals = np.array([m for _, m in med_list], float)

    if vals.size >= 3:
        k = int(np.floor(0.25 * vals.size))
        vals_sorted = np.sort(vals)
        fused_vals = (
            vals_sorted[k : vals_sorted.size - k]
            if vals_sorted.size - 2 * k >= 1
            else vals_sorted
        )
        qt_ms = float(np.mean(fused_vals))
    elif vals.size == 2:
        qt_ms = float(np.mean(vals))
    else:
        qt_ms = float(vals[0])

    # Guard: drop rogue lead medians >60 ms from fused estimate and re-fuse
    if vals.size >= 3:
        close = np.abs(vals - qt_ms) <= 60.0
        if close.sum() >= 2 and close.sum() != vals.size:
            qt_ms = float(np.mean(vals[close]))

    # ---- Bazett ---------------------------------------------------------------
    rr_ms = float(np.median(np.asarray(rr_list, float)))
    rr_ms = float(np.clip(rr_ms, 300.0, 2000.0))
    qtc_ms = float(qt_ms / np.sqrt(rr_ms / 1000.0))

    # ---- Thresholds & severity ------------------------------------------------
    meta = getattr(ctx, "meta", {}) or {}
    sex = _norm_sex(
        getattr(ctx, "sex", None) or meta.get("sex") or meta.get("gender")
    )
    thr = 450.0 if sex == "male" else 460.0 if sex == "female" else 455.0

    prolonged = qtc_ms > thr
    if not prolonged:
        return {}

    severity = (
        "severe" if qtc_ms >= 500.0 else "moderate" if qtc_ms >= 480.0 else "mild"
    )

    leads_used = [ld for ld, _ in med_list]
    beats_used = int(sum(beats_kept_per_lead.get(ld, 0) for ld in leads_used))
    desc = (
        f"QTc prolongation ({qtc_ms:.0f} ms, {severity}) "
        f"from multi-lead consensus (Tangent Method, QT {qt_ms:.0f} ms, RR {rr_ms:.0f} ms)"
    )

    return {
        "qtc_prolongation": {
            "present": True,
            "description": desc,
            "qt_ms": qt_ms,
            "rr_ms": rr_ms,
            "qtc_ms": qtc_ms,
            "severity": severity,
            "sex": sex,
            "method": "Gaussian Tangent Method (Gold Standard) + multi-lead fusion",
            "leads_used": leads_used,
            "beats_used": beats_used,
        }
    }


# ---------------------------------------------------------------------------
# 2. ELECTRICAL ALTERNANS
# ---------------------------------------------------------------------------

def global_electrical_alternans(ctx: Any, *, debug: bool = False) -> Dict[str, Any]:
    """
    Detect beat-to-beat QRS amplitude alternans (2-beat periodicity)
    from Gaussian fits.
    """
    lead_fits = getattr(ctx, "lead_fits", None)
    if not isinstance(lead_fits, dict) or not lead_fits:
        return {}

    lead_order = [
        "II", "V5", "V6", "I", "aVF", "V2", "V3", "III", "aVL", "V4", "aVR", "V1"
    ]

    def _get_qrs_by_beat(lead: str) -> list:
        src = lead_fits.get(lead, {})
        return src.get("by_beat", [])

    def _qrs_amp_for_beat(beat_gaussians: list[dict]) -> float | None:
        """Robust per-beat amplitude: largest absolute Gaussian amplitude among QRS components."""
        if not beat_gaussians:
            return None
        amps = [
            abs(g_item.get("amp_mv", 0.0))
            for g_item in beat_gaussians
            if g_item.get("wave_type") == "QRS"
        ]
        return float(max(amps)) if amps else None

    def _moving_mean(x: np.ndarray, w: int) -> np.ndarray:
        if x.size == 0:
            return x
        w = max(1, int(w))
        k = np.ones(w, float) / w
        return np.convolve(x, k, mode="same")

    def _alternans_score(
        x_hp_win: np.ndarray, x_norm_win: np.ndarray
    ) -> tuple[float, float]:
        """
        r: fit x_hp_win ~ b0 + b1*s  (SNR for 2-beat alternation)
        delta_odd_even%: computed on x_norm_win (baseline, not high-pass)
        """
        n = x_hp_win.size
        if n < 6:
            return 0.0, 0.0

        s = np.fromiter(
            ((1.0 if (i % 2 == 0) else -1.0) for i in range(n)), float
        )
        A = np.vstack([np.ones(n), s]).T
        try:
            coef, _, _, _ = np.linalg.lstsq(A, x_hp_win, rcond=None)
        except Exception:
            return 0.0, 0.0
        b0, b1 = float(coef[0]), float(coef[1])
        resid = x_hp_win - (b0 + b1 * s)
        r = abs(b1) / (np.std(resid) + 1e-9)

        even = x_norm_win[::2]
        odd = x_norm_win[1::2]
        med = (
            np.median(x_norm_win) if np.isfinite(x_norm_win).all() else 0.0
        )
        delta = abs(np.mean(even) - np.mean(odd))
        delta_pct = 100.0 * (delta / (abs(med) + 1e-9))
        return float(r), float(delta_pct)

    detections: list[dict] = []

    for lead in lead_order:
        by_beat = _get_qrs_by_beat(lead)
        if not isinstance(by_beat, list) or len(by_beat) < 8:
            continue

        # Build amplitude series
        amps: list[float] = []
        for beat in by_beat:
            if isinstance(beat, list):
                a = _qrs_amp_for_beat(beat)
            elif isinstance(beat, dict):
                comps = [beat] if beat.get("wave_type") == "QRS" else []
                a = _qrs_amp_for_beat(comps)
            else:
                a = None
            if a is not None and np.isfinite(a):
                amps.append(float(a))

        x = np.asarray(amps, float)
        if x.size < 8:
            continue

        # Normalize & detrend
        med = np.median(x)
        if not np.isfinite(med) or med <= 0:
            continue
        x_norm = x / med
        x_hp = x_norm - _moving_mean(x_norm, w=5)
        x_hp = x_hp - np.mean(x_hp)  # center

        # Require some dynamic range
        if np.std(x_norm) < 0.02:
            continue

        best_r, best_delta, best_win = 0.0, 0.0, 0
        for W in (12, 10, 8):
            if x_hp.size < W:
                continue
            for start in range(0, x_hp.size - W + 1):
                r, d = _alternans_score(
                    x_hp[start : start + W], x_norm[start : start + W]
                )
                if r > best_r or (abs(r - best_r) < 1e-9 and d > best_delta):
                    best_r, best_delta, best_win = r, d, W

        pass_lead = (best_r >= 0.9) and (best_delta >= 12.0)

        detections.append(
            {
                "lead": lead,
                "best_ratio": float(best_r),
                "delta_odd_even_pct": float(best_delta),
                "window_len": int(best_win),
                "beats_available": int(x.size),
                "pass": bool(pass_lead),
            }
        )

    if not detections:
        return {}

    det_sorted = sorted(detections, key=lambda d: d["best_ratio"], reverse=True)
    best = det_sorted[0]
    n_confirm = sum(1 for d in detections if d["pass"])

    if best["pass"]:
        desc = (
            f"Beat-to-beat QRS amplitude alternans in lead {best['lead']} "
            f"(ratio {best['best_ratio']:.2f}, "
            f"\u0394odd-even {best['delta_odd_even_pct']:.1f}%, "
            f"window {best['window_len']} beats)"
        )
        return {
            "electrical_alternans": {
                "present": True,
                "description": desc,
                "lead": best["lead"],
                "alternans_ratio": best["best_ratio"],
                "delta_odd_even_pct": best["delta_odd_even_pct"],
                "window_len": best["window_len"],
                "leads_confirming": int(n_confirm),
                "details_by_lead": detections,
            }
        }

    return {}


# ---------------------------------------------------------------------------
# 3. EXTREME AXIS DEVIATION
# ---------------------------------------------------------------------------

def global_extreme_axis(ctx: Any) -> Dict[str, Any]:
    """Extreme axis deviation (-90\u00b0 to \u00b1180\u00b0)."""
    axis = calculate_qrs_axis(getattr(ctx, "lead_fits", {}))

    if axis is not None:
        # Convert numpy array to scalar BEFORE any comparisons
        if isinstance(axis, np.ndarray):
            axis = float(axis.item())
        else:
            axis = float(axis)

        if axis < -90 or axis > 180:
            return {
                "extreme_axis_deviation": {
                    "present": True,
                    "description": f"Extreme axis deviation ({axis:.0f}\u00b0) - ",
                    "axis_degrees": axis,
                }
            }
    return {}


# ---------------------------------------------------------------------------
# 4. POOR QUALITY ECG -- more sensitive & explicit
# ---------------------------------------------------------------------------

def global_poor_quality(ctx: Any) -> Dict[str, Any]:
    """
    ECG quality assessment with additional line-length roughness metric.
    Uses both per-lead spectral ratios and global time-domain roughness.
    """
    fs = getattr(ctx, "fs", 500) or 500
    leads = list(getattr(ctx, "leads", [])) or [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    LIMB = {"I", "II", "III", "aVR", "aVL", "aVF"}
    PREC = {"V1", "V2", "V3", "V4", "V5", "V6"}

    # ---- thresholds -----------------------------------------------------------
    BWI_RED, BWI_AMB = 1.50, 1.00
    EMG_RED, EMG_AMB = 1.20, 0.80
    CLIP_RED, CLIP_AMB = 0.005, 0.002
    FLAT_RED, FLAT_AMB = 0.75, 0.55
    P2P_LIMB_MIN, P2P_PREC_MIN = 0.20, 0.35
    LOW_AMP_WIDESPREAD_LEADS = 5
    LL_RED, LL_AMB = 1.40, 1.10

    # ---- helpers --------------------------------------------------------------
    def _get_raw_signal(lead: str) -> Optional[np.ndarray]:
        try:
            segs = getattr(ctx, "segments", {})
            return segs["raw"]["by_lead"][lead]["raw"].astype(float)
        except Exception:
            return None

    def _band_energy_fft(x: np.ndarray, lo: float, hi: float) -> float:
        n = len(x)
        if n < 64:
            return 0.0
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        X = np.fft.rfft(x - np.nanmedian(x))
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(np.abs(X[m]) ** 2)) if np.any(m) else 0.0

    def _flat_fraction_windows(x: np.ndarray) -> float:
        if len(x) < int(0.4 * fs):
            return 0.0
        p2p = float(np.nanmax(x) - np.nanmin(x)) + 1e-9
        thr = max(0.005 * p2p, 0.015)
        win = int(0.20 * fs)
        step = win // 2
        cnt = 0
        flat = 0
        for i in range(0, len(x) - win + 1, step):
            w = x[i : i + win]
            cnt += 1
            if np.nanstd(w) < thr:
                flat += 1
        return flat / max(1, cnt)

    def _clip_run_fraction(x: np.ndarray) -> float:
        if len(x) < 16:
            return 0.0
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        rng = xmax - xmin + 1e-9
        tol = 0.001 * rng
        near_min = np.abs(x - xmin) < tol
        near_max = np.abs(x - xmax) < tol

        def _run_frac(mask: np.ndarray) -> float:
            run = 0
            tot = 0
            for v in mask:
                if v:
                    run += 1
                else:
                    if run >= 3:
                        tot += run
                    run = 0
            if run >= 3:
                tot += run
            return tot / float(len(mask))

        return _run_frac(near_min) + _run_frac(near_max)

    # ---- global line-length roughness -----------------------------------------
    try:
        raw12 = getattr(ctx, "raw_ecg_12", None)
        if raw12 is None and hasattr(ctx, "segments"):
            raw12 = np.stack(
                [_get_raw_signal(l) for l in leads if _get_raw_signal(l) is not None]
            )

        if raw12 is not None and raw12.size > 0:
            dx = np.diff(raw12, axis=1)
            line_length_trace = np.sum(np.abs(dx), axis=0)
            line_length_trace = np.concatenate(
                [[line_length_trace[0]], line_length_trace]
            )
            # smooth to 1-s windows
            win = int(fs)
            ll_env = np.convolve(line_length_trace, np.ones(win) / win, mode="valid")
            ll_ratio = float(np.std(ll_env) / (np.median(ll_env) + 1e-9))
        else:
            ll_ratio = 0.0
    except Exception:
        ll_ratio = 0.0

    # ---- main per-lead metrics ------------------------------------------------
    issues: list[str] = []
    per_lead: dict[str, dict] = {}
    red_leads: set[str] = set()
    amber_leads: set[str] = set()
    low_amp_counter = 0
    bwi_bad: list[str] = []
    emg_bad: list[str] = []
    clip_bad: list[str] = []
    flat_bad: list[str] = []

    for ld in leads:
        xr = _get_raw_signal(ld)
        if xr is None or not np.isfinite(xr).any():
            continue
        xr = xr.astype(float)
        e_low = _band_energy_fft(xr, 0.00, 0.33)
        e_sig = _band_energy_fft(xr, 2.00, 20.0) + 1e-9
        e_emg = _band_energy_fft(xr, 35.0, 70.0)

        bwi = e_low / e_sig
        emg = e_emg / (e_sig + 1e-9)
        clip = _clip_run_fraction(xr)
        flat = _flat_fraction_windows(xr)
        p2p = float(np.nanmax(xr) - np.nanmin(xr))

        if (ld in LIMB and p2p < P2P_LIMB_MIN) or (ld in PREC and p2p < P2P_PREC_MIN):
            low_amp_counter += 1

        score = "green"
        if (bwi >= BWI_RED) or (emg >= EMG_RED) or (clip >= CLIP_RED) or (flat >= FLAT_RED):
            score = "red"
        elif (bwi >= BWI_AMB) or (emg >= EMG_AMB) or (clip >= CLIP_AMB) or (flat >= FLAT_AMB):
            score = "amber"

        per_lead[ld] = {
            "bwi": round(bwi, 3),
            "emg": round(emg, 3),
            "clip_frac": round(clip, 4),
            "flatline_frac": round(flat, 3),
            "p2p_mv": round(p2p, 3),
            "score": score,
        }
        if score == "red":
            red_leads.add(ld)
        elif score == "amber":
            amber_leads.add(ld)
        if bwi >= BWI_AMB:
            bwi_bad.append(ld)
        if emg >= EMG_AMB:
            emg_bad.append(ld)
        if clip >= CLIP_AMB:
            clip_bad.append(ld)
        if flat >= FLAT_AMB:
            flat_bad.append(ld)

    # ---- global combination with line-length ----------------------------------
    widespread_low_amp = low_amp_counter >= LOW_AMP_WIDESPREAD_LEADS
    rough_ll = ll_ratio >= LL_RED
    marginal_ll = ll_ratio >= LL_AMB

    poor = (len(red_leads) >= 3) or widespread_low_amp or rough_ll

    if bwi_bad:
        issues.append(f"baseline wander high in {', '.join(bwi_bad)}")
    if emg_bad:
        issues.append(f"EMG noise high in {', '.join(emg_bad)}")
    if clip_bad:
        issues.append(f"clipping runs in {', '.join(clip_bad)}")
    if flat_bad:
        issues.append(f"flatline spans in {', '.join(flat_bad)}")
    if widespread_low_amp:
        issues.append("widespread low amplitude")
    if rough_ll:
        issues.append("global roughness (high line-length variation)")

    desc: list[str] = []
    if poor:
        desc.append("Poor ECG quality")
    else:
        desc.append("Global quality acceptable")

    desc.append(f"Line-length roughness ratio = {ll_ratio:.2f}")

    return {
        "poor_quality_ecg": {
            "present": bool(poor or marginal_ll),
            "description": "; ".join(desc),
            "issues": issues,
            "per_lead": per_lead,
            "line_length_ratio": round(ll_ratio, 3),
        }
    }


# ---------------------------------------------------------------------------
# 5. DEXTROCARDIA
# ---------------------------------------------------------------------------

def global_dextrocardia(ctx: Any) -> Dict[str, Any]:
    """Dextrocardia detection."""
    lead_fits = getattr(ctx, "lead_fits", None)
    if not isinstance(lead_fits, dict) or not lead_fits:
        return {}

    signs = 0

    # Negative QRS in I, positive in aVR
    if "I" in lead_fits:
        qrs_I = get_qrs_gaussians(lead_fits, "I")
        if qrs_I:
            net_I = sum(g_item["amp_mv"] for g_item in qrs_I)
            if net_I < -0.2:
                signs += 1

    if "aVR" in lead_fits:
        qrs_aVR = get_qrs_gaussians(lead_fits, "aVR")
        if qrs_aVR:
            net_aVR = sum(g_item["amp_mv"] for g_item in qrs_aVR)
            if net_aVR > 0.2:
                signs += 1

    # Progressive R wave decrease V1-V6
    r_decrease = True
    prev_r = float("inf")
    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        if lead in lead_fits:
            qrs = get_qrs_gaussians(lead_fits, lead)
            r_components = [g_item for g_item in qrs if g_item["amp_mv"] > 0]
            if r_components:
                r_height = max(g_item["amp_mv"] for g_item in r_components)
                if r_height > prev_r:
                    r_decrease = False
                    break
                prev_r = r_height

    if signs >= 2 and r_decrease:
        return {
            "dextrocardia": {
                "present": True,
                "description": "Dextrocardia  - verify lead placement or confirm with imaging",
            }
        }
    return {}


# ---------------------------------------------------------------------------
# 6. PEDIATRIC PATTERN
# ---------------------------------------------------------------------------

def global_pediatric_pattern(ctx: Any) -> Dict[str, Any]:
    """Pediatric ECG pattern detection."""
    lead_fits = getattr(ctx, "lead_fits", None)
    if not isinstance(lead_fits, dict):
        lead_fits = {}

    pediatric_signs = 0

    # Right axis deviation (normal in children)
    axis = calculate_qrs_axis(lead_fits)
    if axis is not None:
        if isinstance(axis, np.ndarray):
            axis = float(axis.item())
        else:
            axis = float(axis)
        if axis > 90:
            pediatric_signs += 1

    # Dominant R in V1 (normal in children)
    if "V1" in lead_fits:
        qrs_v1 = get_qrs_gaussians(lead_fits, "V1")
        r_components = [g_item for g_item in qrs_v1 if g_item["amp_mv"] > 0]
        s_components = [g_item for g_item in qrs_v1 if g_item["amp_mv"] < 0]

        if r_components and s_components:
            r_height = max(g_item["amp_mv"] for g_item in r_components)
            s_depth = abs(min(g_item["amp_mv"] for g_item in s_components))
            if r_height > s_depth:
                pediatric_signs += 1

    # Short PR interval
    pr = calculate_pr_interval(lead_fits, "II")
    if pr and pr < 120:
        pediatric_signs += 1

    if pediatric_signs >= 2:
        return {
            "pediatric_pattern": {
                "present": True,
                "description": (
                    "ECG pattern pediatric patient - "
                    "verify patient age for appropriate interpretation"
                ),
            }
        }
    return {}


# ---------------------------------------------------------------------------
# 7. ATHLETE'S HEART
# ---------------------------------------------------------------------------

def global_athletes_heart(ctx: Any) -> Dict[str, Any]:
    """Athletic heart syndrome pattern."""
    lead_fits = getattr(ctx, "lead_fits", None)
    if not isinstance(lead_fits, dict):
        lead_fits = {}

    athletic_signs = 0

    # Sinus bradycardia
    rr_list = getattr(ctx, "rr_intervals", None)
    if rr_list:
        hr = 60000 / np.mean(rr_list)
        if 40 <= hr <= 60:
            athletic_signs += 1

    # Voltage criteria for LVH (physiologic in athletes)
    if "V5" in lead_fits and "V1" in lead_fits:
        v5_qrs = get_qrs_gaussians(lead_fits, "V5")
        v1_qrs = get_qrs_gaussians(lead_fits, "V1")

        if v5_qrs and v1_qrs:
            r_v5 = max(
                (g_item["amp_mv"] for g_item in v5_qrs if g_item["amp_mv"] > 0),
                default=0,
            )
            s_v1 = abs(
                min(
                    (g_item["amp_mv"] for g_item in v1_qrs if g_item["amp_mv"] < 0),
                    default=0,
                )
            )
            if r_v5 + s_v1 > 3.5:
                athletic_signs += 1

    # Early repolarization (common in athletes)
    early_repol_count = 0
    for lead in ["V3", "V4", "V5"]:
        if lead in lead_fits:
            st_info = get_st_segment_info(ctx, lead)
            if 70 < st_info["duration"] < 100:
                early_repol_count += 1

    if early_repol_count >= 2:
        athletic_signs += 1

    if athletic_signs >= 2:
        return {
            "athletes_heart": {
                "present": True,
                "description": (
                    "Pattern  athlete's heart - "
                    "physiologic adaptation to exercise training"
                ),
            }
        }
    return {}


# ---------------------------------------------------------------------------
# 8. CRITICAL FINDINGS
# ---------------------------------------------------------------------------

def global_critical_findings(ctx: Any) -> Dict[str, Any]:
    """Aggregate critical findings requiring immediate action."""
    lead_fits = getattr(ctx, "lead_fits", None)
    if not isinstance(lead_fits, dict):
        lead_fits = {}

    critical: list[str] = []

    rr_list = getattr(ctx, "rr_intervals", None)
    if rr_list:
        hr = 60000 / np.mean(rr_list)
        if hr > 150:
            qrs_duration = (
                calculate_qrs_duration(lead_fits, "II")
                if "II" in lead_fits
                else 80
            )
            if qrs_duration and qrs_duration > 120:
                critical.append("Possible wide-complex tachycardia pattern")

        # Check for complete heart block
        if hr < 40:
            critical.append("Severe bradycardia/complete heart block")

    # Check for ST-segment elevation pattern (simplified)
    st_elevation_count = 0
    for lead in ["V2", "V3", "V4"]:
        if lead in lead_fits:
            st_info = get_st_segment_info(ctx, lead)
            if st_info["duration"] < 60:
                st_elevation_count += 1

    if st_elevation_count >= 2:
        critical.append("Possible acute ST-segment elevation pattern")

    if critical:
        return {
            "critical_findings": {
                "present": True,
                "description": (
                    f"CRITICAL FINDINGS: {', '.join(critical)} - "
                    f"IMMEDIATE MEDICAL ATTENTION REQUIRED"
                ),
                "findings": critical,
            }
        }
    return {}


# ---------------------------------------------------------------------------
# 9. METABOLIC PATTERN
# ---------------------------------------------------------------------------

def global_metabolic_pattern(ctx: Any) -> Dict[str, Any]:
    """Metabolic disturbance patterns."""
    lead_fits = getattr(ctx, "lead_fits", None)
    if not isinstance(lead_fits, dict):
        lead_fits = {}

    metabolic_signs: list[str] = []

    # Tall, peaked T-wave pattern: peaked T waves
    peaked_t_count = 0
    leads_list = getattr(ctx, "leads", [])
    for lead in leads_list:
        if lead in lead_fits:
            t_gaussians = get_t_wave_gaussians(lead_fits, lead)
            if t_gaussians:
                for g_item in t_gaussians:
                    if g_item["amp_mv"] > 0.5 and g_item["sigma_ms"] < 40:
                        peaked_t_count += 1
                        break

    if peaked_t_count >= 3:
        metabolic_signs.append("tall, peaked T-wave pattern (peaked T waves)")

    # Prolonged QT pattern: prolonged ST segment
    long_st_count = 0
    for lead in leads_list:
        st_info = get_st_segment_info(ctx, lead)
        if st_info["duration"] > 140:
            long_st_count += 1

    if long_st_count >= 4:
        metabolic_signs.append("prolonged ST pattern (prolonged ST)")

    if metabolic_signs:
        return {
            "metabolic_pattern": {
                "present": True,
                "description": (
                    f"Metabolic disturbance pattern suggesting "
                    f"{', '.join(metabolic_signs)} - "
                ),
                "patterns": metabolic_signs,
            }
        }
    return {}


# ---------------------------------------------------------------------------
# 10. OVERALL IMPRESSION
# ---------------------------------------------------------------------------

def global_overall_impression(ctx: Any) -> Dict[str, Any]:
    """Generate overall ECG impression."""
    impressions: list[str] = []

    rr_list = getattr(ctx, "rr_intervals", None)

    # Check heart rate
    if rr_list:
        hr = 60000 / np.mean(rr_list)
        if hr < 60:
            impressions.append("bradycardic")
        elif hr > 100:
            impressions.append("tachycardic")
        else:
            impressions.append("normal rate")

    # Check rhythm regularity
    if rr_list:
        rr_cv = np.std(rr_list) / np.mean(rr_list)
        if rr_cv < 0.1:
            impressions.append("regular rhythm")
        else:
            impressions.append("irregular rhythm")

    # Check axis
    axis = calculate_qrs_axis(getattr(ctx, "lead_fits", {}))
    if axis is not None:
        if isinstance(axis, np.ndarray):
            axis = float(axis.item())
        else:
            axis = float(axis)

        if -30 <= axis <= 90:
            impressions.append("normal axis")
        elif axis < -30:
            impressions.append("left axis deviation")
        else:
            impressions.append("right axis deviation")

    return {
        "overall_impression": {
            "present": True,
            "description": f"ECG shows {', '.join(impressions)}",
            "components": impressions,
        }
    }


# ---------------------------------------------------------------------------
# GLOBAL_FEATURES registry
# ---------------------------------------------------------------------------

GLOBAL_FEATURES: List = [
    global_qtc_prolongation,
    global_electrical_alternans,
    global_extreme_axis,
    global_poor_quality,
    global_dextrocardia,
    global_pediatric_pattern,
    global_athletes_heart,
    global_critical_findings,
    global_metabolic_pattern,
    global_overall_impression,
]
