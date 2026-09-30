"""
Plane analysis — 2D projection fitting with cheat-magnitude pattern.

Source: Cell 17 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (lines 614-858)

Key function: ``_analyse_plane()`` projects 6-lead segments onto a 2D plane,
fits Gaussian wave components (QRS, T, P), aggregates into clinical lumps,
and returns lump list + clinical intervals.

**Cheat magnitude pattern:** Direction (angle) comes from the 2D projection;
magnitude comes from the raw leads.  This hybrid avoids amplitude distortion
from projection noise while preserving directional information.

Dependencies (all Layer 0-2):
    constants: LIMB_MAT, CHEST_MAT, PAD_HEAD_R_MS, PACE_SIGMA_MS, PACE_AMP_MV,
               PipelineState, logger
    plane_fitting: fit_scalar_gain
    gaussian_fitting: fit_gaussians_hybrid_subrange, subtract_fitted_2d
    wave_classification: build_aggregator_input, aggregate_waves_by_type,
                         compute_clinical_intervals, st_measure
    preprocessing: convert_angles_to_degrees
    beats: build_average_snippet
    plotting: plot_rr_with_waves_main (optional, guarded)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from qpsi.constants import (
    DEADZONE_EPSILON_MS,
    LIMB_MAT,
    CHEST_MAT,
    PAD_HEAD_R_MS,
    PACE_SIGMA_MS,
    PACE_AMP_MV,
    PipelineState,
    QRS_HYBRID_AMP_LAMBDA,
    QRS_HYBRID_ANGLE_LAMBDA,
    QRS_HYBRID_N_WAVES,
    QRS_HYBRID_SIGMA_MAX,
    T_HYBRID_AMP_LAMBDA,
    T_HYBRID_ANGLE_LAMBDA,
    T_HYBRID_N_WAVES,
    T_HYBRID_SIGMA_MAX,
    P_HYBRID_AMP_LAMBDA,
    P_HYBRID_ANGLE_LAMBDA,
    P_HYBRID_N_WAVES,
    P_HYBRID_SIGMA_MAX,
    logger,
)
from qpsi.plane_fitting import fit_scalar_gain
from qpsi.gaussian_fitting import fit_gaussians_hybrid_subrange, subtract_fitted_2d
from qpsi.wave_classification import (
    build_aggregator_input,
    aggregate_waves_by_type,
    compute_clinical_intervals,
    st_measure,
)
from qpsi.preprocessing import convert_angles_to_degrees
from qpsi.beats import build_average_snippet


# ---------------------------------------------------------------------------
# Cheat-magnitude helper
# ---------------------------------------------------------------------------

def _compute_cheat_magnitude(seg6: np.ndarray, *, fs: int) -> np.ndarray:
    """Return a smooth magnitude surrogate built directly from the six raw leads.

    Args:
        seg6: (6, T) array — the six raw lead signals for one plane.
        fs: Sampling rate (Hz).

    Returns:
        (T,) array — mean of absolute values across the 6 leads.
    """
    cheat_mag = np.mean(np.abs(seg6), axis=0)  # shape (T,)
    return cheat_mag


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def _analyse_plane(
    plane: str,
    beats: List[Dict[str, Any]],
    rr_stats: Dict[str, float],
    *,
    fs: int,
    state: PipelineState,
    amplitude_threshold_unscaled: float,
    plot_title_prefix: str = "",
    plot_enabled: bool = True,
    individual_traces: bool = False,
    offset_ms: int = 50,
    n_extra: int = 3,
    refine_sigmas: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Analyse the limb- or chest-plane by fitting quantum-wave lumps.

    **Cheat-magnitude pattern (v2)**:
        Direction from the true 2D projection, magnitude from the raw leads.
        Avoids amplitude distortion from projection noise while preserving
        axis measurements.

    Args:
        plane: ``"limb"`` or ``"chest"``.
        beats: List of beat dicts with ``"ecg_segment"`` and ``"time_ms"`` keys.
        rr_stats: Dict with at least ``"rr_mean_ms"``.
        fs: Sampling rate (Hz).
        state: Per-recording mutable state (plane_cache, not_noisy, plot_enabled).
        amplitude_threshold_unscaled: Base noise threshold (fraction of peak).
        plot_title_prefix: Title prefix for optional diagnostic plot.
        plot_enabled: Per-call plot enable flag.
        individual_traces: If True, use single beat instead of average.
        offset_ms: Timing offset in ms for wave windows.
        n_extra: Extra Gaussian components to try during fitting.
        refine_sigmas: If True, refine sigma estimates after initial fit.

    Returns:
        (lumps, info_dict) where lumps is a list of LumpDict and info_dict
        contains clinical intervals, ST info, and pacing detection results.
    """
    # -----------------------------------------------------------------
    # 1) Obtain average snippet (or per-beat snippet)
    # -----------------------------------------------------------------
    if individual_traces:
        seg6 = beats[0]["ecg_segment"]
        t_ms = beats[0]["time_ms"]
    else:
        seg6, t_ms = build_average_snippet(beats, fs=fs)

    # -----------------------------------------------------------------
    # 2) Choose the projection matrix for this plane
    # -----------------------------------------------------------------
    E2 = LIMB_MAT if plane == "limb" else CHEST_MAT

    # -----------------------------------------------------------------
    # 3) Scalar gain: cache once per recording
    # -----------------------------------------------------------------
    if not individual_traces:
        g_plane = fit_scalar_gain(seg6, E2)
        state.plane_cache[plane] = g_plane
    else:
        g_plane = state.plane_cache.get(plane)
        if g_plane is None:
            g_plane = fit_scalar_gain(seg6, E2)
            state.plane_cache[plane] = g_plane

    # -----------------------------------------------------------------
    # 4) *Direction* from true projection
    # -----------------------------------------------------------------
    v2 = (g_plane * E2) @ seg6                       # 2 x T
    mags = np.linalg.norm(v2, axis=0)
    if mags.max() < 1e-3:                             # 1 uV threshold
        logger.warning("%s-plane: no usable signal — skipping plane", plane)
        return [], {}

    # Unit vectors (2 x T) describing direction only
    eps = 1e-6
    unit_vec = v2 / (mags + eps)

    # -----------------------------------------------------------------
    # 5) *Magnitude* from raw leads (cheat)
    # -----------------------------------------------------------------
    cheat_mag = _compute_cheat_magnitude(seg6, fs=fs)  # shape (T,)  # noqa: F841

    # Full denoised 2-D trace (NOTE: currently using projection mags, not cheat_mag)
    v2_full = unit_vec * mags                          # 2 x T

    # -----------------------------------------------------------------
    # 6) Optional spike isolation (high-frequency component)
    # -----------------------------------------------------------------
    clip_thr = np.median(mags)
    clip_amt = np.clip(mags - clip_thr / 2, 0, None)
    high_vec = unit_vec * clip_amt                     # spikes only

    v2c = v2_full if state.not_noisy else high_vec

    # -----------------------------------------------------------------
    # 7) Proceed with original fitting logic
    # -----------------------------------------------------------------
    rr_ms = rr_stats["rr_mean_ms"]

    mag2 = np.vstack([v2c[0], v2c[1], np.zeros_like(v2c[0])])  # 3 x T
    amp_thr = amplitude_threshold_unscaled * np.linalg.norm(mag2, axis=0).max()
    logger.debug("base_noise_thr %s", amp_thr)

    # Timing windows. Parity with Apr28 notebook (line 16117): P fitting upper
    # bound stops DEADZONE_EPSILON_MS before the QRS boundary so the deadzone
    # heuristic in wave_classification.build_aggregator_input has space to
    # reclassify narrow+loud boundary Gaussians as QRS rather than P.
    lhs_lo = -0.5 * rr_ms + offset_ms
    lhs_hi = -(offset_ms + DEADZONE_EPSILON_MS)
    qrs_lo = -offset_ms
    qrs_hi = 120.0
    rhs_lo = 90
    rhs_hi = 0.5 * rr_ms

    w_qrs = fit_gaussians_hybrid_subrange(
        t_ms, v2c, qrs_lo, qrs_hi,
        n_waves=QRS_HYBRID_N_WAVES, n_extra=n_extra,
        amp_thr=amp_thr, sigma_max=QRS_HYBRID_SIGMA_MAX,
        angle_lambda=QRS_HYBRID_ANGLE_LAMBDA, amplitude_lambda=QRS_HYBRID_AMP_LAMBDA,
        refine_sigmas=refine_sigmas,
    )

    res_after_qrs = subtract_fitted_2d(t_ms, v2c, w_qrs)

    w_t = fit_gaussians_hybrid_subrange(
        t_ms, res_after_qrs, rhs_lo, rhs_hi,
        n_waves=T_HYBRID_N_WAVES, n_extra=n_extra,
        amp_thr=amp_thr, sigma_max=T_HYBRID_SIGMA_MAX,
        angle_lambda=T_HYBRID_ANGLE_LAMBDA, amplitude_lambda=T_HYBRID_AMP_LAMBDA,
        refine_sigmas=refine_sigmas,
    )

    w_p = fit_gaussians_hybrid_subrange(
        t_ms, v2c, lhs_lo, lhs_hi,
        n_waves=P_HYBRID_N_WAVES, n_extra=n_extra,
        amp_thr=amp_thr, sigma_max=P_HYBRID_SIGMA_MAX,
        angle_lambda=P_HYBRID_ANGLE_LAMBDA, amplitude_lambda=P_HYBRID_AMP_LAMBDA,
        refine_sigmas=refine_sigmas,
    )

    # --- Clean P waves falling in wrong place ---
    clean_w_p: List[float] = []
    for i in range(len(w_p) // 4):
        t0_p = w_p[4 * i]
        if (-rr_ms / 2 + offset_ms) <= t0_p <= (-PAD_HEAD_R_MS):
            clean_w_p.extend(w_p[4 * i: 4 * i + 4])
    w_p = np.array(clean_w_p)

    params = np.concatenate([w_t, w_qrs, w_p])

    # --- Pacing spike detection ---
    pacing_detected = False
    for i in range(len(params) // 4):
        t0_i, sigma_i, amp_i, _ = params[4 * i: 4 * i + 4]

        is_in_window = (-250.0 <= t0_i <= -50.0)
        is_narrow = (sigma_i <= PACE_SIGMA_MS)
        is_large_amp = (amp_i >= PACE_AMP_MV)

        if is_in_window and is_narrow and is_large_amp and not individual_traces:
            logger.debug(
                "pacing parameters: t0=%.1f ms, sigma=%.1f ms, amp=%.2f mV",
                t0_i, sigma_i, amp_i,
            )
            pacing_detected = True
            break

    # --- Aggregate + clinical intervals ---
    agg_in = build_aggregator_input(
        t_ms, params,
        rr_ms=rr_ms, offset_ms=offset_ms,
        amplitude_threshold=amp_thr,
    )

    lumps_raw = aggregate_waves_by_type(
        t_ms,
        v_xy=v2c,
        wave_groups=agg_in,
        amplitude_threshold=amp_thr,
    )

    lumps = convert_angles_to_degrees(lumps_raw)
    clin = compute_clinical_intervals(lumps, rr_ms=rr_ms)
    st_info = st_measure(t_ms, v2c, lumps)

    for lump in lumps:
        t0 = lump.get("t_peak", lump.get("center", lump.get("time", 0.0)))
        sigma = lump.get(
            "sigma_ms",
            (lump.get("end_time", 0) - lump.get("start_time", 0)) / 4.0,
        )
        lump["center_ms"] = float(t0)
        lump["sigma_ms"] = float(sigma)

    # -----------------------------------------------------------------
    # 8) Optional diagnostic plot
    # -----------------------------------------------------------------
    if plot_enabled and state.plot_enabled:
        try:
            from qpsi.plotting import plot_rr_with_waves_main
        except ImportError:
            pass
        else:
            wave_list = []
            for i in range(len(params) // 4):
                t0, sigma, A, alpha = params[4 * i: 4 * i + 4]
                sigma = max(sigma, 1.0)
                env = A * np.exp(-0.5 * ((t_ms - t0) / sigma) ** 2)
                peak_idx = int(np.argmin(np.abs(t_ms - t0)))
                wave_list.append({
                    "t": t_ms,
                    "fit_amp": env,
                    "label": f"g{i + 1}",
                    "peak_idx": peak_idx,
                    "peak_amp": float(A),
                    "peak_angle": float(alpha),
                })

            plot_rr_with_waves_main(
                t_array=t_ms,
                mag_data=np.linalg.norm(mag2, axis=0),
                angle=np.arctan2(v2[1], v2[0]),
                wave_list=wave_list,
                rr_ms=rr_ms,
                amplitude_threshold=amp_thr,
                offset_ms=offset_ms,
                title=f"{plot_title_prefix} ({plane.upper()}-plane)",
                recording_name=plane,
                hide_xaxis=False,
            )

    # -----------------------------------------------------------------
    # 9) Return results
    # -----------------------------------------------------------------
    return lumps, {**clin, **st_info, "pacing_detected": pacing_detected}


# ---------------------------------------------------------------------------
# Waveform extraction helper
# ---------------------------------------------------------------------------

def get_waveforms_from_lumps(
    beatwise_lumps: List[List[Dict[str, Any]]],
    wave_type: str,
) -> List[Any]:
    """Extract the waveform array for *wave_type* from each beat's lumps.

    Args:
        beatwise_lumps: List of per-beat lump lists.
        wave_type: Wave type string (e.g. ``"P"``, ``"QRS"``, ``"T"``).

    Returns:
        List of waveform arrays (one per beat), ``None`` for missing entries.
    """
    out: List[Any] = []
    for lumps in beatwise_lumps:
        lump = next(
            (l for l in lumps if l["wave_type"] == wave_type and "waveform" in l),
            None,
        )
        out.append(lump["waveform"] if lump is not None else None)
    return out
