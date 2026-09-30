"""
Core constants, dataclasses, projection matrices, and utility functions.

Source: Cell 0 of q_psi_ai_for_ecg_Feb14_Adele.ipynb
        + ensure_json_serializable from Cell 13

This is the foundation module — imported by nearly every other qpsi module.
No internal dependencies. External: numpy, math, logging, dataclasses.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("qpsi")

# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True)
class Beat:
    """Single heartbeat segment from a 12-lead ECG.

    Args:
        ecg_12: (12, N) array — the 12-lead segment.
        time_ms: (N,) signed time axis in ms, 0 = R-peak.
        synced_shift: Samples shifted during cross-correlation sync (debug).
        ecg_sync: Zero-padded aligned copy after synchronisation.
    """
    ecg_12: np.ndarray
    time_ms: np.ndarray
    synced_shift: int = 0
    ecg_sync: np.ndarray | None = None


@dataclass
class PipelineState:
    """Per-recording mutable state. Reset at start of each recording.

    Replaces the notebook's mutable globals (NOT_NOISY, _PLANE_CACHE,
    GLOBAL_PLOT_ENABLED) with an explicit, threaded-through container.
    Essential for parallelisation safety and C++ translation.
    """
    not_noisy: bool = False
    plane_cache: dict = field(default_factory=lambda: {"limb": None, "chest": None})
    plot_enabled: bool = False


class LumpDict(TypedDict, total=False):
    """Wave description produced by wave_classification, consumed by feature extractors.

    Each lump represents one anatomically labeled wave component (P, QRS, T, etc.)
    extracted from the Gaussian decomposition.
    """
    wave_type: str        # "P","preP1","Q","R","S","S2","T","T2","I","ST"
    start_time: float     # ms, signed (0=R-peak)
    peak_time: float      # ms, signed — true data peak
    end_time: float       # ms, signed
    amplitude_mV: float   # mV — true data peak magnitude
    angle: float          # radians
    angle_degrees: float  # degrees
    center_ms: float      # ms — same as peak_time
    sigma_ms: float       # ms — estimated from (end-start)/4


# ---------------------------------------------------------------------------
# Lead-plane indexing
# ---------------------------------------------------------------------------
PLANE_IDX: Dict[str, List[int]] = {
    "limb":  list(range(0, 6)),    # I, II, III, aVR, aVL, aVF
    "chest": list(range(6, 12)),   # V1–V6
}

# ---------------------------------------------------------------------------
# Sampling rate
# ---------------------------------------------------------------------------
FS_HZ: int = 500

# ---------------------------------------------------------------------------
# Beat segmentation constants
# ---------------------------------------------------------------------------
PAD_HEAD_R_MS: float = 40.0       # zero-padding before R-peak (canonical: 40.0)
PAD_TAIL_T_MS: int = 20           # zero-padding after T-wave end
CROSSCORR_WIN_MS: float = 60.0    # ± range for envelope sync
tolerance_ms: int = 50             # RR length tolerance for good beats

# ---------------------------------------------------------------------------
# Gaussian fitting thresholds
# ---------------------------------------------------------------------------
amplitude_threshold_unscaled: float = 0.025
fraction_threshold: float = 0.8
boundary_amp_frac: float = 0.33    # min fraction of peak to influence start time
delta_angle: float = math.pi / 3   # ~1.0472 radians

# ---------------------------------------------------------------------------
# Pacemaker detection
# ---------------------------------------------------------------------------
PACE_SIGMA_MS: float = 4.0
PACE_AMP_MV: float = 0.025

# ---------------------------------------------------------------------------
# Noise/baseline thresholds
# ---------------------------------------------------------------------------
BASELINE_FLUCT_MAX: float = 0.8    # % of baseline 95% of total filtered trace
MAX_NOISE_LEVEL: float = 0.005

# ---------------------------------------------------------------------------
# Feature extraction thresholds
# ---------------------------------------------------------------------------
LOG_EPS: float = 1e-8              # prevents log(0)
rr_arrhythmia_threshold: float = 0.05
P_amp_difference_threshold: float = 0.33
P_time_difference_threshold: float = 25.0
P_theta_difference_threshold: float = math.radians(90)
T_amp_thr: float = 0.33
T_dt_thr: float = 25.0
T_qrs_thr: int = 50
T_dxy_thr: float = math.radians(90)

# Per-beat P-wave variability — skip the analysis when avg-beat P amplitude
# is below noise floor (~0.01-0.05 mV typical). H33 / Q-A6-8 — moved from
# inline at pipeline.py.
P_AMP_VARIABILITY_THRESHOLD_MV: float = 0.03

# ---------------------------------------------------------------------------
# Hybrid Gaussian fitter parameters (paired QRS / T / P fits)
#
# Used by plane_analysis._analyse_plane and the Rust kernel in
# qpsi_native/src/plane.rs. H34 / Q-A6-13 — moved from inline literals to
# centralize the Python-Rust contract; the Rust side reads these via
# build-time codegen in qpsi_native/build.rs.
# ---------------------------------------------------------------------------
QRS_HYBRID_N_WAVES: int = 5
QRS_HYBRID_SIGMA_MAX: float = 15.0
QRS_HYBRID_ANGLE_LAMBDA: float = 0.2
QRS_HYBRID_AMP_LAMBDA: float = 1.0

T_HYBRID_N_WAVES: int = 3
T_HYBRID_SIGMA_MAX: float = 80.0
T_HYBRID_ANGLE_LAMBDA: float = 1.0
T_HYBRID_AMP_LAMBDA: float = 3.0

P_HYBRID_N_WAVES: int = 3
P_HYBRID_SIGMA_MAX: float = 80.0
P_HYBRID_ANGLE_LAMBDA: float = 0.02
P_HYBRID_AMP_LAMBDA: float = 1.0

# ---------------------------------------------------------------------------
# Wave region boundary constants
# ---------------------------------------------------------------------------
EARLY_P_MS: int = -210             # < –210 ms → Early-P
NORMAL_P_MS: int = -100            # [–210, –100) ms → Normal-P
QRS_FRAC_MS: int = 80             # [0, +80) ms → QRS
ST_FRAC: float = 0.20             # fraction of RR → ST
T_EARLY_FRAC: float = 0.26        # → Early-T
T_NORMAL_FRAC: float = 0.40       # → Normal-T
OFFSET_MS: int = 75               # guard band between QRS & P-side
TOLERANCE_MS: int = 60             # tolerance RR length for good beats

# Dead-zone for P/QRS classification boundary at -offset_ms.
# Parity with Apr28 notebook (line 96): a narrow ±DEADZONE_EPSILON_MS window
# around the boundary where Gaussians need physiological-shape disambiguation
# (narrow + loud → QRS, otherwise → P). Applied in wave_classification.
# Also constrains P-wave fitting bounds in plane_analysis.py (P fitting stops
# DEADZONE_EPSILON_MS before the QRS boundary).
DEADZONE_EPSILON_MS: float = 5.0

# ═══════════════════════════════════════════════════════════════════════════
# Fixed 2-D projection matrices (no per-record fitting)
# ═══════════════════════════════════════════════════════════════════════════

# XY (limb) plane: Einthoven + Wilson geometry
_LIMB_MAT = np.array([
    [ 1.0,   0.5,  -0.5,  -1.0,   1.0,   0.0],       # X-axis
    [ 0.0,  +0.866, +0.866, 0.0, -0.866, -0.866],     # Y-axis
], dtype=float)

# XZ projection for chest leads (2 × 6)
_CHEST_MAT = np.array([
    #  V1        V2        V3        V4        V5        V6
    [-0.173648,  0.173648, 0.500000, 0.866025, 1.000000, 0.866025],   # +X
    [-0.984808, -0.984808, -0.866025, -0.500000, 0.000000, 0.500000], # +Z
], dtype=float)

# Normalise average column-gain to 1
_GAIN = np.linalg.norm(_LIMB_MAT, axis=0).mean()
_LIMB_MAT /= _GAIN
_CHEST_MAT /= _GAIN

# Pre-compute pseudoinverses
_LIMB_PINV = np.linalg.pinv(_LIMB_MAT)     # 6×2
_CHEST_PINV = np.linalg.pinv(_CHEST_MAT)   # 6×2

# Public names (without underscore prefix)
LIMB_MAT: np.ndarray = _LIMB_MAT
CHEST_MAT: np.ndarray = _CHEST_MAT
LIMB_PINV: np.ndarray = _LIMB_PINV
CHEST_PINV: np.ndarray = _CHEST_PINV


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def round_json_values(data: Any, decimals: int = 2) -> Any:
    """Recursively round floats inside nested structures.

    Args:
        data: Nested dict/list/float structure.
        decimals: Number of decimal places.

    Returns:
        Same structure with floats rounded.
    """
    if isinstance(data, dict):
        return {k: round_json_values(v, decimals) for k, v in data.items()}
    if isinstance(data, list):
        return [round_json_values(v, decimals) for v in data]
    if isinstance(data, float):
        return round(data, decimals)
    return data


def pythonify(obj: Any) -> Any:
    """Convert numpy scalars/arrays to vanilla Python types for json.dumps().

    Args:
        obj: Any Python/numpy object.

    Returns:
        JSON-safe Python equivalent.
    """
    if isinstance(obj, dict):
        return {k: pythonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [pythonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return pythonify(obj.tolist())
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def ensure_json_serializable(obj: Any) -> Any:
    """Convert common non-JSON Python/NumPy types to JSON-safe natives.

    Handles: str, int, float, bool, None, list, dict, set, bytes,
    numpy scalars, numpy arrays, numpy void/structured dtypes.
    Non-finite floats (NaN, Inf) are replaced with None.

    Args:
        obj: Any Python/NumPy object.

    Returns:
        JSON-safe Python equivalent.
    """
    # Fast-path primitives
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # NumPy scalars
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return ensure_json_serializable(obj.item())
        return [ensure_json_serializable(x) for x in obj.tolist()]

    # NumPy void / structured dtypes
    if isinstance(obj, np.void):
        return None

    # Bytes-like
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj).decode("utf-8", errors="replace")

    # Collections
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = ensure_json_serializable(k)
            vs = ensure_json_serializable(v)
            if not isinstance(ks, str):
                ks = "null" if ks is None else str(ks)
            out[ks] = vs
        return out

    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(x) for x in obj]

    if isinstance(obj, set):
        return [ensure_json_serializable(x) for x in list(obj)]

    # Fallback: string representation
    return str(obj)
