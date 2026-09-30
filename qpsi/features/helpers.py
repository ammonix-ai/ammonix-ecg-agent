"""
Shared helper functions used by multiple feature extractors and pipeline modules.

Consolidates duplicated helpers from Cells 4, 9, 15 into a single canonical
location, breaking circular dependencies between segmentation.py and beats.py.

Source: Cell 4 (_get), Cell 9 (format_lead_list, _contiguous_groups,
        get_t_wave_gaussians, get_qrs_gaussians, _t_amp_and_duration, _sum_wave,
        _keep_significant_groups), Cell 15 (_first, _safe_bounds).

H30 / Q-A6-12 (Session 44, 2026-04-29): get_qrs_gaussians was previously
duplicated across p_wave.py / qrs_complex.py / rhythm.py / helpers.py
(as the underscore-prefixed _qrs_gaussians). Now canonical here; the
underscore alias stays for backward compatibility with t_wave.py imports
that pre-date the rename.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qpsi.features.context import _f


# ---------------------------------------------------------------------------
# Polymorphic accessor — works for Beat dataclass or dict
# ---------------------------------------------------------------------------

def _get(obj: Any, name: str) -> Any:
    """Return ``obj.name`` or ``obj[name]`` — works for dataclass OR dict.

    Special case: ``"length_samples"`` is derived on demand from
    ``obj.ecg_12.shape[1]`` when the object is a dataclass.
    """
    if isinstance(obj, dict):
        return obj[name]
    if name == "length_samples":
        return obj.ecg_12.shape[1]
    return getattr(obj, name)


# ---------------------------------------------------------------------------
# Lead-list formatting
# ---------------------------------------------------------------------------

def format_lead_list(leads: List[str]) -> str:
    """Format a list of lead names into human-readable English.

    Examples:
        [] -> ""
        ["V1"] -> "V1"
        ["V1", "V2"] -> "V1 and V2"
        ["V1", "V2", "V3"] -> "V1, V2 and V3"
    """
    if not leads:
        return ""
    if len(leads) == 1:
        return leads[0]
    if len(leads) == 2:
        return f"{leads[0]} and {leads[1]}"
    return f"{', '.join(leads[:-1])}, and {leads[-1]}"


def _contiguous_groups(leads: List[str]) -> List[List[str]]:
    """Group leads into contiguous chest runs, inferiors, and laterals.

    Returns list of lead groups, each group being a list of lead names.
    """
    chest_order = ["V1", "V2", "V3", "V4", "V5", "V6"]
    inferiors = {"II", "III", "aVF"}
    laterals = {"I", "aVL", "V5", "V6"}

    pre = sorted(
        [L for L in leads if L in chest_order],
        key=lambda L: chest_order.index(L),
    )
    groups: List[List[str]] = []
    run: List[str] = []
    prev: Optional[int] = None
    for L in pre:
        idx = chest_order.index(L)
        if prev is None or idx == prev + 1:
            run.append(L)
        else:
            if run:
                groups.append(run)
            run = [L]
        prev = idx
    if run:
        groups.append(run)
    if inferiors & set(leads):
        groups.append(sorted(inferiors & set(leads)))
    if laterals & set(leads):
        groups.append(sorted(laterals & set(leads)))
    return groups


def _keep_significant_groups(
    leads: List[str], min_len: int = 2
) -> List[List[str]]:
    """Return only contiguous lead groups with at least *min_len* members."""
    return [g for g in _contiguous_groups(leads) if len(g) >= min_len]


# ---------------------------------------------------------------------------
# Gaussian extraction helpers
# ---------------------------------------------------------------------------

def get_t_wave_gaussians(
    lead_fits: Dict[str, Any],
    lead: str,
    beat_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract T-wave Gaussian dicts from *lead_fits* for a given lead.

    Args:
        lead_fits: {lead: {"avg": [gauss_dict, ...], "by_beat": [[...], ...]}}
        lead: Lead name (e.g. "V1").
        beat_idx: If provided, fetch that beat's Gaussians; otherwise use avg.

    Returns:
        List of Gaussian dicts whose ``wave_type`` is ``"T"``.
    """
    if lead not in lead_fits:
        return []
    if beat_idx is None:
        gs = lead_fits[lead].get("avg", [])
    else:
        bb = lead_fits[lead].get("by_beat", [])
        if beat_idx < len(bb):
            gs = bb[beat_idx]
        else:
            return []
    return [g for g in gs if g.get("wave_type") == "T"]


def get_qrs_gaussians(
    lead_fits: Dict[str, Any],
    lead: str,
    beat_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract QRS-type Gaussian dicts (Q, R, S, QRS) from *lead_fits*.

    Canonical home (H30 / Q-A6-12). Cell 12 broader filter — matches the
    notebook GT runtime; previously duplicated in qrs_complex.py and rhythm.py
    with identical behavior, and in p_wave.py with a Cell-10 narrower filter
    (`wave_type == "QRS"` only) that was dead code.

    Args:
        lead_fits: {lead: {"avg": [gauss_dict, ...], "by_beat": [[...], ...]}}
        lead: Lead name (e.g. "V1").
        beat_idx: If provided, fetch that beat's Gaussians; otherwise use avg.

    Returns:
        List of Gaussian dicts whose ``wave_type`` is one of Q, R, S, QRS.
    """
    if lead not in lead_fits:
        return []
    if beat_idx is not None:
        arr = lead_fits[lead].get("by_beat", [])
        if beat_idx < len(arr):
            arr = arr[beat_idx]
        else:
            return []
    else:
        arr = lead_fits[lead].get("avg", [])
    return [g for g in arr if g.get("wave_type") in ("Q", "R", "S", "QRS")]


# Backward-compatible alias — t_wave.py imports `_qrs_gaussians` historically.
# Kept to avoid churning 4 call sites in t_wave.py.
_qrs_gaussians = get_qrs_gaussians


def _t_amp_and_duration(
    t_gaussians: List[Dict[str, Any]],
) -> Tuple[float, Optional[float], Optional[float], float]:
    """Return (peak_abs, t_start, t_end, duration) for T-wave Gaussians.

    Returns (0.0, None, None, 0.0) if *t_gaussians* is empty.
    """
    if not t_gaussians:
        return 0.0, None, None, 0.0
    peak_abs = max(_f(g.get("amp_mv", 0.0)) for g in t_gaussians)
    t_start = min(
        _f(g["center_ms"]) - 2.0 * _f(g.get("sigma_ms", 1.0))
        for g in t_gaussians
    )
    t_end = max(
        _f(g["center_ms"]) + 2.0 * _f(g.get("sigma_ms", 1.0))
        for g in t_gaussians
    )
    return peak_abs, t_start, t_end, _f(t_end - t_start)


def _sum_wave(
    gaussians: List[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Reconstruct a time-domain waveform from Gaussian parameters.

    Returns (t, y) arrays, or (None, None) if *gaussians* is empty.
    """
    if not gaussians:
        return None, None
    t0 = min(
        _f(g["center_ms"]) - 3.0 * _f(g.get("sigma_ms", 1.0))
        for g in gaussians
    )
    t1 = max(
        _f(g["center_ms"]) + 3.0 * _f(g.get("sigma_ms", 1.0))
        for g in gaussians
    )
    t = np.linspace(t0, t1, max(50, int((t1 - t0) // 2 + 50)))
    y = np.zeros_like(t)
    for g in gaussians:
        a = _f(g.get("amp_mv", 0.0))
        mu = _f(g.get("center_ms", 0.0))
        sd = max(1.0, _f(g.get("sigma_ms", 1.0)))
        y += a * np.exp(-0.5 * ((t - mu) / sd) ** 2)
    return t, y


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _first(arr: Any) -> Any:
    """Unwrap ``[wave]`` -> ``wave``, ``None`` -> ``None``.

    If *arr* is a list or tuple, return the first element.
    Otherwise return *arr* unchanged (including None).
    """
    return arr[0] if isinstance(arr, (list, tuple)) else arr


def _safe_bounds(
    bounds: Dict[str, Any], lead: str
) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(None, None)`` if *bounds* or *lead* is missing."""
    return bounds.get(lead, (None, None))
