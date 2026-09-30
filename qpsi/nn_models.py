"""
CNN models for auxiliary ECG classification.

H46 / Q-A6-5 status (2026-04-30 → 2026-05-18): the module was previously
labelled DORMANT because all 4 CNN calls had been removed from the pipeline
in Feb14_Adele. Phase 2 Item 4.3 (2026-05-18) re-activates the AF prediction
path under K3 (the private training repository IS the GT — distil-into-platform without waiting
on the colleague's notebook to re-wire the call sites). Activation requires
the optional torch extra: `pip install 'qpsi[nn]'` (see qpsi/pyproject.toml:17-18)
— torch is also in the root pixi env so the extra is no-op there.

OPT-IN gate (2026-05-18): QPSI colleague signalled mid-Item-4.3 that they
have stopped using neural nets. The wire-in is preserved per R2 (don't
remove features) + the QPSI unbiased-measurement "toggles OK" principle,
but default behaviour matches colleague's direction. Set
``QPSI_AF_NEURAL_ENABLE=1`` (or "true"/"yes"/"on") to opt in.

Opt-in call site (V1, 3-45 Hz, calibrated AF probability):
    qpsi/pipeline.py:1232 — gated on ``QPSI_AF_NEURAL_ENABLE``;
    `_af_probability()` injected into semantic_flags as `af_neural_prob`;
    recursive_extract picks it up downstream as `Sem_af_neural_prob`.

Models (all expect 500-sample bandpass-filtered single-lead input):
    best_model_AF.pth       — V1, 3-45 Hz: Atrial fibrillation     [LIVE]
    best_model_flutter.pth  — V1, 3-45 Hz: Atrial flutter F-waves  [historical — not in Apr28]
    best_model_SinTach.pth  — V1, 3-45 Hz: Sinus tachycardia       [historical — not in Apr28]
    T_wave_change.pth       — V3, 0.1-25 Hz: T-wave variability    [historical — not in Apr28]

The 3 "historical" models were defined in Feb14 Cell 17 but absent from the
Apr28 notebook revision (verified by grep: zero references to
best_model_flutter / best_model_SinTach / T_wave_change). Their architectures
are still served by `F_waveNet` + `predict_raw` if any future call wants to
load them; the module keeps the generic CNN scaffolding regardless of which
weight files are currently wired in.

Source: Cell 17 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (lines ~500-530) +
Cell 17/19 of q_psi_ai_for_ecg_Apr28.ipynb (F_waveNet class at line 16078,
_af_probability at line 19715). The WaveMedix versions add: type hints,
weights_only=True (CVE-mitigated torch.load), graceful-degrade gate when
torch unavailable, and Platt-calibration support via calib_AF.json
(notebook hardcodes the path; WaveMedix takes it as a parameter).

Re-exported via `qpsi/__init__.py` for downstream / test access.

External: torch (optional), numpy, scipy.signal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample

logger = logging.getLogger("qpsi")

# Optional torch import — module degrades gracefully when torch absent
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]


def _check_torch() -> None:
    """Raise ImportError if torch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for nn_models but not installed. "
            "Install with: pip install 'qpsi[nn]'"
        )


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class F_waveNet(nn.Module):
        """Tiny 1D CNN for F-wave / rhythm classification.

        Architecture: Conv1d(1->16->32), FC(32*125->64->1), sigmoid output.
        Expects input of shape (batch, 1, 500) — bandpass-filtered, resampled,
        normalised single-lead ECG.
        """
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(32 * 125, 64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)
else:
    # Placeholder when torch is not available
    class F_waveNet:  # type: ignore[no-redef]
        """Placeholder — torch not installed."""
        def __init__(self) -> None:
            _check_torch()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_resample_norm(
    sig: np.ndarray,
    fs: int,
    band: Tuple[float, float],
) -> np.ndarray:
    """Bandpass filter, resample to 500 samples, and z-normalise.

    Args:
        sig: 1D raw signal from a single lead.
        fs: Sampling rate in Hz.
        band: (low_hz, high_hz) bandpass bounds.

    Returns:
        (500,) normalised signal ready for CNN input.
    """
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    filt = sosfiltfilt(sos, sig)
    rs = resample(filt, 500)
    x = (rs - np.mean(rs)) / (np.std(rs) + 1e-6)
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_raw(
    ecg12: np.ndarray,
    fs: int,
    leads: List[str],
    lead_name: str,
    model_path: str | Path,
    band: Tuple[float, float],
) -> float:
    """Run a single-lead CNN prediction.

    Loads the model from disk, bandpass-filters the target lead,
    resamples to 500 samples, normalises, and returns sigmoid probability.

    Args:
        ecg12: (12, N) raw 12-lead ECG.
        fs: Sampling rate in Hz.
        leads: List of lead names (length 12).
        lead_name: Target lead (e.g., "V1").
        model_path: Path to .pth model weights.
        band: (low_hz, high_hz) bandpass bounds.

    Returns:
        Probability in [0, 1].

    Raises:
        ImportError: If torch is not installed.
    """
    _check_torch()
    idx = leads.index(lead_name)
    x = _band_resample_norm(ecg12[idx], fs, band)
    inp = torch.tensor(x, dtype=torch.float32)[None, None]  # (1, 1, 500)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = F_waveNet().to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev, weights_only=True))
    net.eval()
    with torch.no_grad():
        return float(torch.sigmoid(net(inp.to(dev))).item())


def _af_probability(
    leads: List[str],
    raw_ecg_12: np.ndarray,
    fs: int,
    *,
    model_path: str | Path | None = None,
    calib_path: str | Path | None = None,
) -> float:
    """Platt-calibrated AF probability using CNN + calibration JSON.

    Args:
        leads: List of lead names.
        raw_ecg_12: (12, N) raw signal.
        fs: Sampling rate.
        model_path: Path to best_model_AF.pth.
        calib_path: Path to calib_AF.json.

    Returns:
        Calibrated AF probability in [0, 1], or 0.0 on failure.

    Raises:
        ImportError: If torch is not installed.
    """
    _check_torch()
    if model_path is None or calib_path is None:
        logger.warning("AF model/calib paths not provided")
        return 0.0

    try:
        raw_prob = predict_raw(raw_ecg_12, fs, leads, "V1",
                               model_path, band=(3.0, 45.0))
        with open(calib_path) as f:
            calib = json.load(f)
        a = calib.get("a", 1.0)
        b = calib.get("b", 0.0)
        # Platt scaling: P = 1 / (1 + exp(a*logit + b))
        logit = np.log(raw_prob / (1.0 - raw_prob + 1e-9))
        return float(1.0 / (1.0 + np.exp(a * logit + b)))
    except Exception as exc:
        logger.warning("AF probability failed: %s", exc)
        return 0.0
