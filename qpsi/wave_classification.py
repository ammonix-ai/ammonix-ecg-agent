"""
Wave classification — map fitted Gaussians to clinical wave types.

Maps every Gaussian centre → anatomical region (P / QRS / T), merges
components into clinical "lumps", computes classical clinical intervals
(PQ, QRS, QT, ST, T, P durations).

Canonical home for ``_ensure_wave_keys`` and ``_blank_bounds``.

Source: Cell 7 of q_psi_ai_for_ecg_Feb14_Adele.ipynb.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from qpsi.constants import (
    DEADZONE_EPSILON_MS,
    EARLY_P_MS,
    NORMAL_P_MS,
    QRS_FRAC_MS,
    ST_FRAC,
    T_EARLY_FRAC,
    T_NORMAL_FRAC,
    delta_angle as _DEFAULT_DELTA_ANGLE,
)

# Dead-zone physiological-shape heuristic thresholds (parity with Apr28
# notebook lines 2695-2696). Narrow+loud Gaussians sitting on the P/QRS
# boundary are reclassified P -> QRS in build_aggregator_input.
_DEADZONE_QRS_SIGMA_MS: float = 12.0   # narrow Gaussian -> likely QRS deflection
_DEADZONE_QRS_AMP_MV: float = 0.15     # high amplitude -> likely QRS deflection
from qpsi.preprocessing import angle_diff_rad


# ---------------------------------------------------------------------------
# Guaranteed-key helpers  (CANONICAL — other modules import from here)
# ---------------------------------------------------------------------------

def _ensure_wave_keys(lumps: list) -> Dict[str, list]:
    """Return ``{"P": [..], "QRS": [..], "T": [..]}`` from *lumps*.

    Each value is a list with one element (waveform or ``None``) so that
    extractors can always iterate safely.
    """
    d: Dict[str, list] = {k: [None] for k in ("P", "QRS", "T")}
    for w in lumps:
        d[w["wave_type"]] = [w.get("waveform", None)]
    return d


def _blank_bounds(
    leads: List[str],
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return ``{lead: (None, None), ...}`` for every lead in *leads*.

    Pre-populates ``ctx.segment_bounds`` so that missing timing info
    does not raise ``KeyError`` inside extractors.
    """
    empty: Tuple[Optional[float], Optional[float]] = (None, None)
    return {ld: empty for ld in leads}


# ---------------------------------------------------------------------------
# 1)  Region mapping helpers
# ---------------------------------------------------------------------------

def time_to_region_label(
    t_ecg: float,
    rr_ms: float,
    offset_ms: float = 50.0,
    epsilon_ms: float = DEADZONE_EPSILON_MS,
) -> str:
    """Return anatomical region name for *t_ecg* (ms, signed, 0 = R-peak).

    Parity with Apr28 notebook (line 2645). Gaussians within +-epsilon_ms of
    the -offset_ms boundary are conservatively assigned to "Late-P" to
    prevent scipy-version-dependent flips between P-wave and QRS
    classification. Build-aggregator-input subsequently promotes narrow+loud
    boundary-Gaussians to QRS via the physiological-shape heuristic.
    """
    boundary = -offset_ms
    # Dead-zone: boundary +- epsilon -> conservative P-wave assignment
    if abs(t_ecg - boundary) <= epsilon_ms:
        return "Late-P"
    if t_ecg >= boundary:                          # right-hand side
        frac = t_ecg / rr_ms
        if t_ecg < QRS_FRAC_MS:
            return "QRS"
        if frac < ST_FRAC:
            return "ST"
        if frac < T_EARLY_FRAC:
            return "Early-T"
        if frac < T_NORMAL_FRAC:
            return "Normal-T"
        return "Late-T"
    else:                                             # left-hand side
        if t_ecg < EARLY_P_MS:
            return "Early-P"
        if t_ecg < NORMAL_P_MS:
            return "Normal-P"
        return "Late-P"


def unify_wave_type(raw_name: str) -> str:
    """Collapse *raw_name* to canonical class (``"P"``, ``"T"``, ``"QRS"``).

    Keeps unknown names unchanged.
    """
    name = raw_name.lower()
    for prefix in ("early-", "normal-", "late-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.startswith("p"):
        return "P"
    if name.startswith("t"):
        return "T"
    if name.startswith("qrs"):
        return "QRS"
    return raw_name


# ---------------------------------------------------------------------------
# 2)  Robust peak / angle selection (returns **radians**)
# ---------------------------------------------------------------------------

def find_true_peak(
    t_ms: np.ndarray,
    v_xy: np.ndarray,
    t0: float,
    sigma0: float,
    *,
    angle_window_deg: float = 45.0,
) -> Tuple[float, float, float]:
    """Return ``(t_peak_ms, amp_peak_mV, angle_rad)`` for one fitted wave.

    Search is constrained:
      * temporally:  ± 2·σ around *t0*
      * angularly:   ± *angle_window_deg* around the data angle at *t0*

    All outputs are raw data values — *angle* is **radians**.
    """
    idx0 = int(np.argmin(np.abs(t_ms - t0)))
    dt_ms = float(t_ms[1] - t_ms[0])

    angle_ctr = math.atan2(v_xy[1, idx0], v_xy[0, idx0])

    win_rad = max(1, int(round(2 * sigma0 / dt_ms)))
    lo = max(0, idx0 - win_rad)
    hi = min(len(t_ms) - 1, idx0 + win_rad)

    vx_w, vy_w = v_xy[0, lo:hi + 1], v_xy[1, lo:hi + 1]
    ang_w = np.arctan2(vy_w, vx_w)

    ang_diff = np.abs(np.angle(np.exp(1j * (ang_w - angle_ctr))))
    mask_good = ang_diff <= math.radians(angle_window_deg)

    if not np.any(mask_good):
        idx_peak = idx0
    else:
        idxs = np.nonzero(mask_good)[0] + lo
        mags_window = np.linalg.norm(v_xy[:, idxs], axis=0)
        idx_peak = idxs[int(np.argmax(mags_window))]

    amp_peak = float(np.linalg.norm(v_xy[:, idx_peak]))
    angle_peak = float(math.atan2(v_xy[1, idx_peak], v_xy[0, idx_peak]))
    t_peak = float(t_ms[idx_peak])

    return t_peak, amp_peak, angle_peak


# ---------------------------------------------------------------------------
# 3)  Sub-wave naming
# ---------------------------------------------------------------------------

def _choose_r_cluster(
    peak_times: List[float],
    peak_amps: List[float],
    peak_ang: List[float],
) -> int:
    """Return index of the cluster that should be labelled ``'R'``.

    * Clusters with peak angle in [−20°, +90°] are R-candidates.
    * Among candidates, pick the greatest amplitude.
    * If none match angularly, fall back to max-amplitude.
    """
    ang_lo = math.radians(-20.0)
    ang_hi = math.radians(+90.0)

    cand_idx = [i for i, a in enumerate(peak_ang) if ang_lo <= a <= ang_hi]

    if cand_idx:
        best = max(cand_idx, key=lambda i: peak_amps[i])
    else:
        best = int(np.argmax(peak_amps))

    return best


def generate_subwave_name(
    wtype: str,
    idx: int,
    n_sub: int,
    *,
    peak_times: Optional[List[float]] = None,
    peak_amps: Optional[List[float]] = None,
    peak_angles: Optional[List[float]] = None,
) -> str:
    """Return canonical sub-wave name (Q / R / S / Q2 / S2 / T / T2 / P / preP…).

    Uses angle + amplitude heuristic for R-selection.
    """
    wtype = wtype.upper()
    if wtype == "QRS":
        if (
            peak_times is None
            or peak_amps is None
            or peak_angles is None
            or len(peak_times) != n_sub
        ):
            raise ValueError("peak_* lists (len=n_sub) required for QRS")

        idx_R = _choose_r_cluster(peak_times, peak_amps, peak_angles)

        if idx == idx_R:
            return "R"
        elif idx < idx_R:
            q_num = idx_R - idx
            return "Q" if q_num == 1 else f"Q{q_num}"
        else:
            s_num = idx - idx_R
            return "S" if s_num == 1 else f"S{s_num}"

    if wtype == "T":
        return "T" if idx == 0 else f"T{idx + 1}"
    if wtype == "P":
        return "P" if idx == n_sub - 1 else f"preP{(n_sub - 1) - idx}"
    return f"{wtype}{idx + 1}"


# ---------------------------------------------------------------------------
# 4)  Build aggregator input  (raw params → wave_dict buckets)
# ---------------------------------------------------------------------------

def build_aggregator_input(
    t_array: np.ndarray,
    wave_params: np.ndarray,
    *,
    offset_ms: float,
    rr_ms: float,
    delta_rad: float = 1.0,
    amplitude_threshold: float = 0.025,
) -> Dict[str, list]:
    """Group every 4-tuple ``(t0, σ, A, α)`` into P / QRS / T buckets.

    Post-processing:
      * Last T-wave relabelled to ``"I"`` if separated and at divergent angle.
      * First QRS component demoted to ``"ST"`` if non-overlapping.
    """
    wave_dict: Dict[str, List[tuple]] = {}
    n_gauss = len(wave_params) // 4
    for i in range(n_gauss):
        t0, s, A, ang = wave_params[4 * i: 4 * i + 4]
        if A <= amplitude_threshold:
            continue
        region = time_to_region_label(t0, rr_ms, offset_ms)
        w_class = unify_wave_type(region)
        wave_dict.setdefault(w_class, []).append((t0, s, A, ang))

        # Dead-zone physiological-shape heuristic (parity with Apr28 notebook
        # lines 2721-2732). Narrow + loud Gaussians on the -offset_ms boundary
        # are QRS deflections, not P waves. Reclassify P -> QRS.
        if (
            w_class == "P"
            and abs(t0 - (-offset_ms)) <= DEADZONE_EPSILON_MS
            and s < _DEADZONE_QRS_SIGMA_MS
            and A > _DEADZONE_QRS_AMP_MV
        ):
            wave_dict["P"].remove((t0, s, A, ang))
            if not wave_dict["P"]:
                del wave_dict["P"]
            wave_dict.setdefault("QRS", []).append((t0, s, A, ang))

    for key in wave_dict:
        wave_dict[key].sort(key=lambda x: -x[0])

    # --- T wave postprocessing ---
    if "T" in wave_dict and len(wave_dict["T"]) > 1:
        t_list = sorted(wave_dict["T"], key=lambda x: -x[0])

        t0_last, s_last, A_last, ang_last = t_list[-1]
        t0_prev, s_prev, A_prev, ang_prev = t_list[-2]
        overlap_last = abs(t0_last - t0_prev) < 1.41 * (s_last + s_prev)
        angle_diff_val = abs(ang_last - ang_prev)

        if not overlap_last and angle_diff_val > delta_rad:
            wave_dict.setdefault("I", []).append(t_list[-1])
            t_list = t_list[:-1]

        wave_dict["T"] = t_list

    # --- QRS wave postprocessing ---
    if "QRS" in wave_dict and len(wave_dict["QRS"]) > 1:
        qrs_list = wave_dict["QRS"]
        t0_1, s1, A1, ang1 = qrs_list[0]
        t0_2, s2, A2, ang2 = qrs_list[1]

        overlap = abs(t0_1 - t0_2) < 1.41 * (s1 + s2)
        if not overlap:
            wave_dict.setdefault("ST", []).append(qrs_list[0])
            wave_dict["QRS"] = qrs_list[1:]

    return wave_dict


# ---------------------------------------------------------------------------
# 5)  Main aggregator  (returns *angle* in **radians**)
# ---------------------------------------------------------------------------

def aggregate_waves_by_type(
    t_ms: np.ndarray,
    v_xy: np.ndarray,
    wave_groups: Dict[str, list],
    *,
    amplitude_threshold: float = 0.04,
    delta_angle_rad: float = _DEFAULT_DELTA_ANGLE,
) -> List[Dict[str, Any]]:
    """Cluster fitted Gaussians into clinical lumps and measure true peaks.

    Parameters
    ----------
    t_ms : 1-D array — time axis in ms.
    v_xy : (2, N) array — 2-D vector trace.
    wave_groups : ``{wave_class: [(t0, σ, A, α), ...]}`` from
        :func:`build_aggregator_input`.
    amplitude_threshold : minimum amplitude (mV) to keep a component.
    delta_angle_rad : angular merging tolerance in **radians**
        (default from ``constants.delta_angle``).

    Returns
    -------
    List of lump dicts with keys ``wave_type``, ``start_time``,
    ``peak_time``, ``end_time``, ``amplitude_mV``, ``angle`` (radians).
    """
    aggregated: List[Dict[str, Any]] = []
    angle_window_deg = delta_angle_rad / np.pi * 180
    if amplitude_threshold < 0.01:
        amplitude_threshold = 0.01

    for wtype, comps in wave_groups.items():
        comps = [c for c in comps if c[2] >= amplitude_threshold]
        if not comps:
            continue
        comps.sort(key=lambda x: x[0])

        # 1) cluster nearby components
        clusters: List[List[tuple]] = []
        cur: List[tuple] = [comps[0]]
        for item in comps[1:]:
            t_i, sigma_i, amp_i, alpha_i = item
            t_p, sigma_p, amp_p, alpha_p = cur[-1]

            ang_ok = abs(alpha_i - alpha_p) <= delta_angle_rad
            time_ok = abs(t_i - t_p) <= (2 * sigma_i + 2 * sigma_p)

            ratio = (amp_i + 1e-9) / (amp_p + 1e-9)
            amp_ok = True
            if wtype.upper() == "P":
                amp_ok = math.log(ratio) <= math.log(3.0)
            elif wtype.upper() == "T":
                amp_ok = math.log(ratio) >= math.log(0.33)

            if ang_ok and time_ok and amp_ok:
                cur.append(item)
            else:
                clusters.append(cur)
                cur = [item]
        clusters.append(cur)

        # 2) reference lists for QRS sub-labelling
        peak_times: List[float] = []
        peak_amps: List[float] = []
        peak_angles: List[float] = []
        if wtype.upper() == "QRS":
            for cl in clusters:
                t_c, _, A_c, _ = max(cl, key=lambda x: x[2])
                idx_c = int(np.argmin(np.abs(t_ms - t_c)))
                ang_c = math.atan2(v_xy[1, idx_c], v_xy[0, idx_c])
                peak_times.append(t_c)
                peak_amps.append(A_c)
                peak_angles.append(ang_c)

        # 3) measure each cluster
        for idx_cl, cl in enumerate(clusters):
            t0, sigma0, _, _ = max(cl, key=lambda x: x[2])

            t_peak, amp_peak, angle_peak = find_true_peak(
                t_ms, v_xy, t0, sigma0, angle_window_deg=angle_window_deg,
            )

            aggregated.append({
                "wave_type": generate_subwave_name(
                    wtype, idx_cl, len(clusters),
                    peak_times=peak_times,
                    peak_amps=peak_amps,
                    peak_angles=peak_angles,
                ),
                "start_time": round(min(t - 2 * s for t, s, _, _ in cl), 3),
                "peak_time": round(t_peak, 3),
                "end_time": round(max(t + 2 * s for t, s, _, _ in cl), 3),
                "amplitude_mV": round(amp_peak, 3),
                "angle": round(angle_peak, 3),  # radians
            })

    return aggregated


# ---------------------------------------------------------------------------
# 6)  ST-segment helper
# ---------------------------------------------------------------------------

def st_measure(
    t_array: np.ndarray,
    v_xy: np.ndarray,
    lumps_avg: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Simple ST-segment metric. Returns angle in **degrees** for readability."""
    s_end = next(
        (w["end_time"] for w in lumps_avg if w["wave_type"].upper() == "S"),
        None,
    )
    r_end = next(
        (w["end_time"] for w in lumps_avg if w["wave_type"].upper() == "R"),
        None,
    )
    t_sta = next(
        (w["start_time"] for w in lumps_avg if w["wave_type"].upper() == "T"),
        None,
    )

    if not s_end and r_end:
        s_end = r_end
    s_end = s_end or 0.0
    t_sta = t_sta or 0.0  # noqa: F841  — kept for clarity

    st_time = s_end + 60.0  # ms past S-end
    idx = int(np.argmin(np.abs(t_array - st_time)))

    vx, vy = v_xy[:, idx]
    amp_2d = math.hypot(vx, vy)
    angle_deg = math.degrees(math.atan2(vy, vx))

    return {
        "ST_time_ms": round(st_time, 2),
        "ST_amplitude_mV": round(amp_2d, 3),
        "ST_angle": round(angle_deg, 2),
    }


# ---------------------------------------------------------------------------
# 7)  P-wave variation collector
# ---------------------------------------------------------------------------

def collect_p_variations(
    *,
    dst_list: List[Dict[str, Any]],
    ref_wave: Dict[str, Any],
    loc_lumps: List[Dict[str, Any]],
    idx: int,
    amp_thr: float,
    dt_thr: float,
    dxy_thr: float,
) -> None:
    """Append variation dicts for main P and qualifying pre-P components.

    Compares each candidate against *ref_wave*; a pre-P must have
    amplitude ≥ ⅓ of the reference P amplitude to qualify.
    """
    ref_amp = abs(ref_wave["amplitude_mV"]) + 1e-9
    cand: List[Dict[str, Any]] = []

    loc_p = next((w for w in loc_lumps if w["wave_type"] == "P"), None)
    if loc_p is not None:
        cand.append(loc_p)

    for w in loc_lumps:
        if w["wave_type"].lower().startswith("prep") and abs(
            w["amplitude_mV"]
        ) >= (ref_amp / 3.0):
            cand.append(w)

    for wav in cand:
        diffs: Dict[str, Any] = {}

        frac_amp = abs(wav["amplitude_mV"] - ref_wave["amplitude_mV"]) / ref_amp
        if frac_amp > amp_thr:
            diffs["Amplitude_MV"] = round(frac_amp, 2)

        dt = wav["peak_time"] - ref_wave["peak_time"]
        if abs(dt) > dt_thr:
            diffs["PeakTime Variation (ms)"] = round(dt, 2)

        dxy = angle_diff_rad(
            math.radians(wav["angle_degrees"]),
            math.radians(ref_wave["angle_degrees"]),
        )
        if dxy > dxy_thr:
            diffs["Angle_deg"] = round(math.degrees(dxy), 2)

        if diffs:
            diffs["Interval"] = idx
            diffs["wave_type"] = wav["wave_type"]
            dst_list.append(diffs)


# ---------------------------------------------------------------------------
# 8)  T-wave variation collector
# ---------------------------------------------------------------------------

def collect_t_variations(
    *,
    dst_list: List[Dict[str, Any]],
    ref_wave: Dict[str, Any],
    loc_lumps: List[Dict[str, Any]],
    idx: int,
    amp_thr: float,
    dt_thr: float,
    dxy_thr: float,
) -> None:
    """Append variation dict for T-wave if any metric exceeds threshold."""
    loc_t = next((w for w in loc_lumps if w["wave_type"] == "T"), None)
    if not loc_t:
        return

    diffs: Dict[str, Any] = {}

    frac_amp = abs(loc_t["amplitude_mV"] - ref_wave["amplitude_mV"]) / max(
        abs(ref_wave["amplitude_mV"]), 1e-6,
    )
    if frac_amp > amp_thr:
        diffs["Amplitude_MV"] = round(frac_amp, 2)

    dt = loc_t["peak_time"] - ref_wave["peak_time"]
    if abs(dt) > dt_thr:
        diffs["PeakTime Variation (ms)"] = round(dt, 2)

    dxy = angle_diff_rad(
        math.radians(loc_t["angle_degrees"]),
        math.radians(ref_wave["angle_degrees"]),
    )
    if dxy > dxy_thr:
        diffs["Angle_deg"] = round(math.degrees(dxy), 2)

    if diffs:
        diffs["Interval"] = idx
        dst_list.append(diffs)


# ---------------------------------------------------------------------------
# 9)  Clinical interval computation
# ---------------------------------------------------------------------------

def compute_clinical_intervals(
    aggregated_waves: List[Dict[str, Any]],
    wave_significance_threshold: float = 0.05,
    rr_ms: Optional[float] = None,
) -> Dict[str, float]:
    """Compute PQ, QRS, QT, ST, T, P intervals (all in ms).

    Parameters
    ----------
    aggregated_waves : list of sub-wave dicts from
        :func:`aggregate_waves_by_type`.
    wave_significance_threshold : unused, kept for API compat.
    rr_ms : RR interval in ms (unused internally but part of contract).

    Returns
    -------
    Dict with keys ``PQ_interval_ms``, ``QRS_interval_ms``,
    ``QT_interval_ms``, ``ST_duration_ms``, ``T_interval_ms``,
    ``P_interval_ms``.

    .. note:: **Bugfix (Feb14):** T-wave matching now includes multi-component
       T-waves (T2, T3) via ``startswith("t")``, not just ``wtype == "t"``.
    """
    intervals: Dict[str, float] = {
        "PQ_interval_ms": 0.0,
        "QRS_interval_ms": 0.0,
        "QT_interval_ms": 0.0,
        "ST_duration_ms": 0.0,
        "T_interval_ms": 0.0,
        "P_interval_ms": 0.0,
    }

    p_starts_normal: List[float] = []
    p_starts_preP: List[float] = []
    p_ends: List[float] = []

    qrs_starts: List[float] = []
    qrs_ends: List[float] = []

    t_starts: List[float] = []
    t_ends: List[float] = []

    for w in aggregated_waves:
        wtype = w["wave_type"].lower()
        st = w["start_time"]
        en = w["end_time"]

        # P group
        if wtype == "p":
            p_starts_normal.append(st)
            p_ends.append(en)
        elif wtype.startswith("prep"):
            p_starts_preP.append(st)
            p_ends.append(en)

        # QRS group: "q", "r", or starts with "s" but not "st"
        if (
            wtype == "q"
            or wtype == "r"
            or (wtype.startswith("s") and not wtype.startswith("st"))
        ):
            qrs_starts.append(st)
            qrs_ends.append(en)

        # T group: any T component (t, t1, t2, t3), excluding "st"
        if wtype.startswith("t") and not wtype.startswith("st"):
            t_starts.append(st)
            t_ends.append(en)

    # QRS interval
    qrs_start: Optional[float] = None
    qrs_end: Optional[float] = None
    if qrs_starts and qrs_ends:
        qrs_start = min(qrs_starts)
        qrs_end = max(qrs_ends)
        qrs_dur = max(0.0, qrs_end - qrs_start)
        intervals["QRS_interval_ms"] = round(qrs_dur, 2)

    # T interval
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    if t_starts and t_ends:
        t_start = min(t_starts)
        t_end = max(t_ends)
        t_dur = max(0.0, t_end - t_start)
        intervals["T_interval_ms"] = round(t_dur, 2)

    # ST duration
    if qrs_end is not None and t_starts:
        st_dur = max(0.0, min(t_starts) - qrs_end)
        intervals["ST_duration_ms"] = round(st_dur, 2)

    # P interval
    if p_starts_normal or p_starts_preP:
        p_starts_all = p_starts_normal + p_starts_preP
        earliest_p = min(p_starts_all)
        latest_p_end = max(p_ends) if p_ends else None
        if latest_p_end is not None:
            p_dur = max(0.0, latest_p_end - earliest_p)
            intervals["P_interval_ms"] = round(p_dur, 2)

    # PQ interval (prefer normal P; fallback to preP)
    p_earliest_start: Optional[float] = None
    if p_starts_normal:
        p_earliest_start = min(p_starts_normal)
    elif p_starts_preP:
        p_earliest_start = min(p_starts_preP)

    if p_earliest_start is not None and qrs_start is not None:
        pq_dur = max(0.0, qrs_start - p_earliest_start)
        intervals["PQ_interval_ms"] = round(pq_dur, 2)
    else:
        intervals["PQ_interval_ms"] = 0.0

    # QT interval
    if qrs_start is not None and t_ends:
        last_t = max(t_ends)
        qt_dur = max(0.0, last_t - qrs_start)
        intervals["QT_interval_ms"] = round(qt_dur, 2)

    return intervals
