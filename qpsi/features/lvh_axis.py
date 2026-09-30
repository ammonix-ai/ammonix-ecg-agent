"""LVH voltage criteria + frontal-plane QRS axis features.

Source: Apr28 notebook ``q_psi_ai_for_ecg_Apr28.ipynb`` Cell 9E (lines
19920-20068), "Frontal-plane QRS axis + LVH voltage criteria (Q-PSI v3)".
Targets two weak-AUROC diagnoses identified in the Feb14 baseline:
"axis left shift" (n=1549, AUROC 0.73) and "left ventricular hypertrophy"
(n=1035, AUROC 0.75).

Phase 2 Item 4.5.1 absorption (2026-05-19) — Apr28 paired-extraction
identified 17 missing keys under ``ECG data.Computed Parameters`` at 100%
patient coverage on batch-100. The module ports the colleague's two
feature families verbatim and is wired into ``json_writer.build_final_record``
so the keys land in the final JSON output.

The functions consume the post-pipeline ``multi_gaussian_parameters`` dict
(``{lead: {"avg_gaussians": [...], "num_beats": N}}``) the pipeline already
produces — no re-fitting, no FeatureContext dependency. The cell docstring
notes that QRS Gaussian ``component`` labels (Q/R/S/R2) follow morphology
(1st/2nd/3rd deflection in time), NOT sign — so in V1 the big negative
deflection is often labelled "R". Therefore the helpers take signed extrema
of the whole QRS family (``wave_type in {"Q","R","S","QRS"}``), which is
what Sokolow-Lyon / Cornell actually measure clinically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_QRS_TYPES = ("Q", "R", "S", "QRS")


def _avg_gaussians(
    multi_gauss: Optional[Dict[str, Any]], lead: str
) -> List[Dict[str, Any]]:
    d = (multi_gauss or {}).get(lead, {}) or {}
    return d.get("avg_gaussians", []) or []


def _net_qrs_amp(multi_gauss: Optional[Dict[str, Any]], lead: str) -> float:
    """Sum of signed amp_mv across Q/R/S/QRS Gaussians on this lead."""
    return sum(
        float(g.get("amp_mv", 0.0))
        for g in _avg_gaussians(multi_gauss, lead)
        if g.get("wave_type") in _QRS_TYPES
    )


def _qrs_extrema(
    multi_gauss: Optional[Dict[str, Any]], lead: str
) -> Tuple[float, float]:
    """Peak positive (R_peak) and peak negative (S_peak, negative number) of
    QRS-family Gaussians on this lead. Returns (0.0, 0.0) if missing."""
    amps = [
        float(g.get("amp_mv", 0.0))
        for g in _avg_gaussians(multi_gauss, lead)
        if g.get("wave_type") in _QRS_TYPES
    ]
    if not amps:
        return 0.0, 0.0
    pk_pos = max(amps) if max(amps) > 0 else 0.0
    pk_neg = min(amps) if min(amps) < 0 else 0.0
    return pk_pos, pk_neg


def compute_qrs_axis_features(
    multi_gauss: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Frontal-plane QRS axis from net amplitudes in I and aVF.

    ``arctan2(net_aVF, net_I)`` puts 0deg along lead I and +90deg along aVF,
    matching the standard hexaxial reference. Returns degrees in (-180, 180].
    """
    if not multi_gauss:
        return {}
    amp_I = _net_qrs_amp(multi_gauss, "I")
    amp_aVF = _net_qrs_amp(multi_gauss, "aVF")
    if amp_I == 0.0 and amp_aVF == 0.0:
        return {}
    axis_deg = float(np.degrees(np.arctan2(amp_aVF, amp_I)))
    lad = -90.0 < axis_deg < -30.0
    rad = 90.0 < axis_deg <= 180.0
    extreme = -180.0 < axis_deg <= -90.0
    # Key names match the Apr28 GT JSONL (underscored ASCII), NOT the
    # notebook Cell 9E source dict literal which used "QRS Axis (deg)" etc.
    # The colleague's deployed pipeline normalises keys before writing JSONL.
    return {
        "QRS_Axis_Deg": round(axis_deg, 1),
        "QRS_Axis_Net_I_mV": round(amp_I, 4),
        "QRS_Axis_Net_aVF_mV": round(amp_aVF, 4),
        "QRS_Axis_Left_Deviation": int(lad),
        "QRS_Axis_Right_Deviation": int(rad),
        "QRS_Axis_Extreme_Deviation": int(extreme),
    }


def compute_lvh_voltage_features(
    multi_gauss: Optional[Dict[str, Any]], sex: str = ""
) -> Dict[str, Any]:
    """Sokolow-Lyon and Cornell LVH voltage criteria, from signed QRS extrema.

    - Sokolow-Lyon: ``|S(V1)| + max(R(V5), R(V6)) >= 3.5 mV`` (35 mm)
    - Cornell:      ``R(aVL) + |S(V3)| >= 2.8 mV`` (M, 28 mm)
                                       ``>= 2.0 mV`` (F, 20 mm)
    - Standalone R(aVL) >= 1.1 mV is also an accepted LVH marker.

    R_peak / S_peak are taken as signed extrema of the QRS-family Gaussians on
    that lead (peak positive / peak negative amplitude), NOT by filtering on
    the morphological component label — see module docstring.
    """
    if not multi_gauss:
        return {}
    r_v5, _s_v5 = _qrs_extrema(multi_gauss, "V5")
    r_v6, _s_v6 = _qrs_extrema(multi_gauss, "V6")
    _r_v1, s_v1 = _qrs_extrema(multi_gauss, "V1")
    _r_v3, s_v3 = _qrs_extrema(multi_gauss, "V3")
    r_aVL, _s_avl = _qrs_extrema(multi_gauss, "aVL")

    abs_s_v1 = abs(s_v1)
    abs_s_v3 = abs(s_v3)

    sokolow = abs_s_v1 + max(r_v5, r_v6)
    cornell = max(0.0, r_aVL) + abs_s_v3

    sex_l = (str(sex) or "").strip().lower()
    cornell_thr = 2.0 if sex_l.startswith("f") else 2.8  # default M if unknown

    return {
        "LVH_Sokolow_Lyon_mV": round(sokolow, 4),
        "LVH_Sokolow_Lyon_Positive": int(sokolow >= 3.5),
        "LVH_Cornell_Voltage_mV": round(cornell, 4),
        "LVH_Cornell_Voltage_Positive": int(cornell >= cornell_thr),
        "LVH_Cornell_Threshold_mV": cornell_thr,
        "LVH_R_aVL_mV": round(max(0.0, r_aVL), 4),
        "LVH_R_aVL_Positive": int(r_aVL >= 1.1),
        "LVH_R_V5_mV": round(max(0.0, r_v5), 4),
        "LVH_R_V6_mV": round(max(0.0, r_v6), 4),
        "LVH_S_V1_mV": round(abs_s_v1, 4),
        "LVH_S_V3_mV": round(abs_s_v3, 4),
    }
