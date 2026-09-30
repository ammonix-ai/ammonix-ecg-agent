"""QPSI pipeline configuration — toggleable feature flags.

QPSIConfig controls which optional feature modules are enabled during
pipeline execution. All toggles default to OFF for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class QPSIConfig:
    """Configuration for optional QPSI pipeline features.

    Attributes:
        compute_template_distances: When True, compute 300 template distance
            features (50 dx × 6 metrics: euclidean, cosine, zscore_max, mahalanobis, zscore_mean, zscore_gt2sigma).
            Requires waveform_templates.json to exist.
        compute_stemi_territories: When True, compute ~60 STEMI territory
            features (5 coronary territories with reciprocal pattern detection).
        compute_direct_measurements: When True, compute the qpsi_direct
            no-Gaussian direct-measurement parallel feature track (~19,500 keys
            across ``direct_features`` / ``direct_aggregated`` /
            ``plane_direct_features`` / ``plane_direct_aggregated``). Adds
            ~7 s/record on top of the Gaussian pipeline. Default OFF for
            backward compatibility; opt-in for training / evaluation /
            World-Model swarm-classifier work (Phase 2 §4.6).
        template_set_path: Path to waveform_templates.json. If empty, auto-discovered
            from repo root (domains/ecg-12lead/layer-2/waveform_templates.json).
    """

    compute_template_distances: bool = False
    compute_stemi_territories: bool = False
    compute_direct_measurements: bool = False
    template_set_path: str = ""

    def resolve_template_path(self) -> Optional[Path]:
        """Resolve template set path, auto-discovering from repo root if needed."""
        if self.template_set_path:
            p = Path(self.template_set_path)
            return p if p.exists() else None

        # Auto-discover: walk up from this file to find repo root
        # config.py is at {repo}/domains/ecg-12lead/WaveMedix/qpsi/config.py → parents[4] = repo
        here = Path(__file__).resolve()
        candidates = [
            here.parents[i] / "domains" / "ecg-12lead" / "layer-2" / "waveform_templates.json"
            for i in range(1, 7)
        ]
        # Also check relative to CWD
        candidates.append(Path("domains/ecg-12lead/layer-2/waveform_templates.json"))

        for p in candidates:
            if p.exists():
                return p
        return None

    @classmethod
    def from_dict(cls, d: dict) -> "QPSIConfig":
        """Create from a plain dict (e.g., from app_state.qpsi_config)."""
        return cls(
            compute_template_distances=bool(d.get("compute_template_distances", False)),
            compute_stemi_territories=bool(d.get("compute_stemi_territories", False)),
            compute_direct_measurements=bool(d.get("compute_direct_measurements", False)),
            template_set_path=str(d.get("template_set_path", "")),
        )

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization."""
        return {
            "compute_template_distances": self.compute_template_distances,
            "compute_stemi_territories": self.compute_stemi_territories,
            "compute_direct_measurements": self.compute_direct_measurements,
            "template_set_path": self.template_set_path,
        }
