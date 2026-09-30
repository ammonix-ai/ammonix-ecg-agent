"""
2D lead-matrix projection fitting.

Models 6-lead ECG in a single anatomical plane:
    ECG(6xT) ~ E2(6x2) . v2(2xT)
where E2 are the lead-projection coefficients and v2(t) the
time-varying planar dipole vector.

Source: Cell 5 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (138 lines)
No internal dependencies. External: numpy, dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Canonical in-plane lead angles (degrees)
# ---------------------------------------------------------------------------

LIMB_DIRECTIONS: Dict[str, float] = {   # XY-plane
    "I": 0.0, "II": 60.0, "III": 120.0,
    "aVR": -150.0, "aVL": -30.0, "aVF": 90.0,
}

CHEST_DIRECTIONS: Dict[str, float] = {  # XZ-plane
    "V1": 100.0, "V2": 80.0, "V3": 60.0,
    "V4": 30.0, "V5": 0.0, "V6": -30.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlaneFitResult:
    """Result of a full plane matrix fit (currently unused — kept for future).

    Args:
        E2: (6, 2) fitted lead-projection matrix.
        E2_pinv: (2, 6) pseudoinverse of E2.
        v2: (2, T) time-varying planar dipole.
        chi_sq: Residual quality metric.
        cond_E2: Condition number of E2.
    """
    E2: np.ndarray
    E2_pinv: np.ndarray
    v2: np.ndarray
    chi_sq: float
    cond_E2: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _initial_E2(angle_map: Dict[str, float],
                lead_order: List[str]) -> np.ndarray:
    """Build a 6x2 unit-circle starting guess from canonical lead angles.

    Args:
        angle_map: Lead name -> angle in degrees.
        lead_order: Ordered list of lead names (must match rows of avg6).

    Returns:
        (6, 2) array with rows = (cos(theta), sin(theta)).
    """
    rows = []
    for name in lead_order:
        theta = np.deg2rad(angle_map[name])
        rows.append([np.cos(theta), np.sin(theta)])
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_scalar_gain(seg6: np.ndarray, E2_fixed: np.ndarray) -> float:
    """Compute the least-squares scalar gain for a 6-lead projection.

    Given a 6-lead segment and a fixed 2x6 direction matrix, finds the
    scalar gain g that minimises ||g * recon6 - seg6||^2.

    Args:
        seg6: (6, T) raw limb or chest leads (baseline-subtracted).
        E2_fixed: (2, 6) direction matrix with frozen angles.

    Returns:
        Scalar gain g.
    """
    v2_tmp = E2_fixed @ seg6              # 2xT
    recon6 = E2_fixed.T @ v2_tmp          # 6xT (unit gain)

    num = (seg6 * recon6).sum()           # <A, B>
    den = (recon6 * recon6).sum() + 1e-9  # <B, B> (epsilon avoids /0)
    return num / den


def fit_plane_matrix(
    avg6: np.ndarray,
    angle_map: Dict[str, float],
    fs: int,
    *,
    epsilon: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the plane-projection matrix and planar dipole from averaged leads.

    Currently returns the unit-circle initial guess (IRLS fitting is disabled).
    The commented-out IRLS code performs robust weighted LS:
        minimise ||W . (ECG - E2.v2)||_2
    with W = diag(1 / sqrt(eps + |ECG|^2)).

    H45 / Q-A6-4 status (2026-04-30): zero in-repo callers; not exported via
    `qpsi/__init__.py`. The IRLS block is also commented out in the source
    notebook (`reference/_ref_cell5.py` lines 99-127 + cell 5 of
    `q_psi_ai_for_ecg_Feb14_Adele.ipynb`) — disabled state mirrors GT, NOT a
    regression in extraction. Re-enabling IRLS here would diverge from the
    notebook (cross-cutting rule 3 violation). Function is retained for GT
    parity; activate only if/when the GT notebook re-enables IRLS first.

    Args:
        avg6: (6, T) averaged 6-lead snippet.
        angle_map: Lead name -> canonical angle in degrees.
        fs: Sampling rate in Hz.
        epsilon: Tikhonov weight floor (unused while IRLS is disabled).

    Returns:
        Tuple of (E2, E2_pinv, v2):
            E2: (6, 2) lead-projection matrix.
            E2_pinv: (2, 6) pseudoinverse.
            v2: (2, T) time-varying planar dipole.

    Raises:
        ValueError: If avg6 doesn't have 6 rows.
    """
    if avg6.shape[0] != 6:
        raise ValueError("fit_plane_matrix: expected a 6-lead slice")

    lead_order = list(angle_map.keys())
    E0 = _initial_E2(angle_map, lead_order)  # 6x2 initial

    # IRLS fitting is disabled in GT (see docstring for rationale).
    # When the notebook re-enables it: scipy.optimize.least_squares with TRF
    # method + Tikhonov weighting W = diag(1/sqrt(eps + |ECG|^2)) to
    # down-weight high-amplitude outliers, then row-normalise E2.

    E2 = E0
    E2_pinv = np.linalg.pinv(E2)  # 2x6
    v2 = E2_pinv @ avg6           # 2xT

    return E2, E2_pinv, v2
