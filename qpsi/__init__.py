"""
QPSI — ECG Signal Processing Library

Deterministic physics-based ECG signal processing pipeline. Takes raw 12-lead
ECG waveforms (WFDB format, 500 Hz) and produces ~566+ numerical features per
patient using 2D Gaussian wave decomposition.

Layer 0 — Foundation (zero internal deps):
    constants: Core dataclasses, projection matrices, thresholds, utilities
    wfdb_io: WFDB file loading and SNOMED diagnosis mapping
    plane_fitting: 2D lead-matrix projection fitting
    gaussian_fitting: 2D Gaussian decomposition engine
    wave_fitters: 1D Gaussian component fitting with Huber LS + AICc
    nn_models: CNN models for auxiliary ECG classification. F_waveNet +
        _af_probability re-exported below (Phase 2 Item 4.3 / 2026-05-18
        re-activated the AF prediction path; pipeline.py:1232 calls it).
        Other 3 historical models (flutter, SinTach, T-wave) still
        loadable via predict_raw if needed. Optional `qpsi[nn]` torch
        extra preserves graceful-degrade when torch missing. H46 / Q-A6-5.
    clinical_summaries: Clinical summary text generation
    qwva: Q-Wave Variability Analysis (RAT/RVT, DBSCAN, PRD)
    features/context: FeatureContext dataclass

Layer 1 — Basic Processing (depends on Layer 0):
    preprocessing: Filtering, baseline correction, R-peak detection, noise detection
    segmentation: Beat segmentation, RR intervals (canonical robust_rr_from_rpeaks)
    beats: Beat filtering, plane slicing, averaging, RR summary
    features/helpers: Shared helper functions (_get, format_lead_list, etc.)
    plotting: All matplotlib/plotly visualization (optional deps)

Layer 2 — Wave Analysis & Feature Extraction (depends on Layers 0-1):
    wave_classification: Gaussian→lump mapping, clinical intervals, _ensure_wave_keys
    lead_fitting_helpers: P-wave detection, bounds construction, residual helpers
    lead_fitting: Per-lead P/QRS/T fitting, energy trace analysis (Feb14)
    features/t_wave: 21 T-wave feature extractors
    features/p_wave: 20 P-wave feature extractors
    features/st_deviation: ST segment analysis (STSegmentFeatures dataclass)
    features/qrs_complex: 21 QRS feature extractors + inject_qrs_descriptors

Layer 3 — Rhythm, Global & Plane Analysis (depends on Layers 0-2):
    features/rhythm: 21 rhythm feature extractors + RhythmPanorama
    features/global_features: 10 global feature extractors (QTc tangent, axis, etc.)
    plane_analysis: 2D plane fitting with cheat-magnitude pattern

Layer 4 — Assembly (depends on Layers 0-3):
    features/registry: Domain extractor registry, extract_semantic_features, postprocess
    json_writer: JSON output assembly, build_final_record, wave variation helpers

Layer 5 — Orchestration (depends on Layers 0-4):
    pipeline: 11-step run_pipeline() orchestrator
"""

from qpsi.constants import (
    Beat,
    PipelineState,
    LumpDict,
    FS_HZ,
    LIMB_MAT,
    CHEST_MAT,
    LIMB_PINV,
    CHEST_PINV,
    PLANE_IDX,
    round_json_values,
    pythonify,
    ensure_json_serializable,
    logger,
)
from qpsi.features.context import FeatureContext
from qpsi.features.helpers import _get, format_lead_list, get_t_wave_gaussians
from qpsi.preprocessing import preprocess_ecg, detect_r_peaks, baseline_subtract_twoanchors
from qpsi.segmentation import robust_rr_from_rpeaks, segment_rr
from qpsi.beats import filter_beats_by_length, slice_beats_to_plane, build_average_snippet

# Layer 2
from qpsi.wave_classification import (
    aggregate_waves_by_type,
    compute_clinical_intervals,
    _ensure_wave_keys,
    _blank_bounds,
)
from qpsi.lead_fitting_helpers import (
    create_wave_bounds_from_lumps_enhanced,
    detect_p_wave_candidates,
    estimate_pr_interval,
)
from qpsi.lead_fitting import (
    fit_lead_waves_enhanced,
    process_leads_with_proper_timing,
    analyse_energy_trace,
)
from qpsi.features.t_wave import T_WAVE_FEATURES
from qpsi.features.p_wave import P_WAVE_FEATURES
from qpsi.features.st_deviation import STSegmentFeatures, analyze_st_segments_all_leads
from qpsi.features.stemi_territories import (
    STEMI_TERRITORY_FEATURE_NAMES,
    TERRITORIES as STEMI_TERRITORIES,
    extract_stemi_territory_features,
    extract_territories_from_st_results,
)
from qpsi.features.qrs_complex import QRS_FEATURES, inject_qrs_descriptors

# Layer 3
from qpsi.features.rhythm import RHYTHM_FEATURES, RhythmPanorama, rr_from_context
from qpsi.features.global_features import GLOBAL_FEATURES
from qpsi.features.lvh_axis import (
    compute_lvh_voltage_features,
    compute_qrs_axis_features,
)
from qpsi.plane_analysis import _analyse_plane, _compute_cheat_magnitude

# Layer 4
from qpsi.features.registry import (
    DOMAIN_EXTRACTORS,
    EXTRACTION_ORDER,
    extract_semantic_features,
    postprocess_semantic_flags,
)
from qpsi.json_writer import build_final_record

# Layer 5
from qpsi.pipeline import run_pipeline

# Layer 0 (NN models — re-exported post Phase 2 Item 4.3 activation)
from qpsi.nn_models import F_waveNet, _af_probability, predict_raw

# Layer 2.5 (qpsi_direct no-Gaussian direct-measurement track — Phase 2 §4.6)
# Opt-in via QPSIConfig(compute_direct_measurements=True). When the toggle is
# ON, qpsi/pipeline.py:run_pipeline() invokes these to emit the 4 net-new
# top-level keys: direct_features, direct_aggregated, plane_direct_features,
# plane_direct_aggregated. See qpsi/QPSI_DIRECT_DESIGN.md for the schema.
from qpsi.direct_measurements import (
    find_extrema,
    measure_window,
    measure_beat,
    measure_plane_beat,
    measure_all_beats,
    aggregate_recording,
    aggregate_plane_recording,
    flatten_features,
    K_EXTREMA,
)

# HyperParams panel (Phase 1 passive — pipeline edits gated to future HPO work).
# See qpsi/hyperparams.py for the full PARAM_META schema (~200 tunable scalars).
from qpsi.hyperparams import HyperParams, DEFAULT as DEFAULT_HYPERPARAMS, PARAM_META

# The upstream package also re-exported five ground-truth audit harnesses
# (paired-extraction divergence audit, single-patient deep dive, Rust-vs-Python
# path diagnostic, Rust-output snapshot capture, GT parity gate). They are not
# vendored here: each one drives a machine-local corpus (a WFDB root, an
# extraction index database, pinned ground-truth JSONL) that exists only inside
# the private tree. Nothing in the extraction path imports them.
