"""
qpsi_rl hyperparameter panel.

Single source of truth for every tunable scalar the q_psi pipeline depends
on. The pipeline modules read from this dataclass instead of from their own
module-level constants, which makes the entire signal-processing chain
parameterisable in one place — suitable for Bayesian optimisation, RL,
sensitivity sweeps, or manual A/B tuning.

DESIGN
------
* ``HyperParams`` is an immutable dataclass — all fields keyword-only with
  defaults that *exactly* reproduce the current qpsi behaviour, so any run
  with the default panel is bit-identical (within Rust SVD numerical noise)
  to the production qpsi.
* ``HyperParams.PARAM_META`` is a parallel dict of per-field metadata
  (search bounds, type, category, source file:line for traceability). The
  HPO loop reads this to construct the search space.
* The pipeline modules accept an optional ``cfg: HyperParams`` argument and
  read ``cfg.<field>``. If ``cfg`` is None, they fall back to the legacy
  module-level constant — so untouched code paths still work.

PHASES
------
* Phase 1 (this file): the panel exists, defaults match qpsi, metadata is
  complete. *No pipeline edits yet.*
* Phase 2: replace module-level constants with ``cfg`` reads, one module
  at a time, validating bit-identical output after each.
* Phase 3: HPO drivers (Optuna sensitivity sweep, BO loop, three-cohort
  validation cascade) live in the private training workspace and consume this panel.

CATEGORIES
----------
``preprocessing``, ``rpeak_detection``, ``segmentation``, ``plane_fit``,
``gaussian_fit``, ``wave_classification``, ``lead_refit``, ``feature_pwave``,
``feature_qrs``, ``feature_t``, ``feature_st``, ``feature_rhythm``,
``feature_global``, ``qwva``, ``physical_constant`` (NOT tunable).

See `Q_PSI_RL_CONCEPT.md` (internal, not in this repository) for the full HPO strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Param metadata — parallel dict, indexed by field name.
# Each entry: (low, high, type_str, category, source_ref, tunable_flag)
# tunable_flag ∈ {"YES", "MAYBE", "NO"}.
# ---------------------------------------------------------------------------

PARAM_META: Dict[str, Tuple[float, float, str, str, str, str]] = {
    # ── preprocessing ───────────────────────────────────────────────────
    "hp_corner_hz":          (0.05,  0.5,   "float", "preprocessing",  "preprocessing.py:163", "YES"),
    "lp_clean_hz":           (150.0, 250.0, "float", "preprocessing",  "preprocessing.py:171", "YES"),
    "lp_noisy_hz":           (25.0,  50.0,  "float", "preprocessing",  "preprocessing.py:173", "YES"),
    "hf_band_low_hz":        (40.0,  60.0,  "float", "preprocessing",  "preprocessing.py:99",  "YES"),
    "hf_band_high_hz":       (150.0, 250.0, "float", "preprocessing",  "preprocessing.py:99",  "YES"),
    "baseline_window_ms":    (400,   1000,  "int",   "preprocessing",  "preprocessing.py:137", "YES"),
    "baseline_fluct_max":    (0.5,   1.0,   "float", "preprocessing",  "constants.py:114",     "YES"),
    "max_noise_level":       (0.002, 0.02,  "float", "preprocessing",  "constants.py:115",     "YES"),

    # ── rpeak / pacing detection ────────────────────────────────────────
    "pace_sigma_ms":         (2.0,   8.0,   "float", "rpeak_detection","constants.py:108",     "YES"),
    "pace_amp_mv":           (0.01,  0.05,  "float", "rpeak_detection","constants.py:109",     "YES"),

    # ── segmentation ────────────────────────────────────────────────────
    "pad_head_r_ms":         (20.0,  60.0,  "float", "segmentation",   "constants.py:92",      "YES"),
    "pad_tail_t_ms":         (10,    40,    "int",   "segmentation",   "constants.py:93",      "YES"),
    "crosscorr_win_ms":      (40.0,  100.0, "float", "segmentation",   "constants.py:94",      "YES"),
    "xcorr_coarse_win_ms":   (40.0,  100.0, "float", "segmentation",   "segmentation.py:178",  "YES"),
    "xcorr_fine_align_ms":   (50.0,  150.0, "float", "segmentation",   "segmentation.py:179",  "YES"),
    "rr_min_ms":             (200.0, 400.0, "float", "segmentation",   "segmentation.py:43",   "YES"),
    "rr_max_ms":             (1500.0,2500.0,"float", "segmentation",   "segmentation.py:44",   "YES"),
    "rr_mad_trim":           (2.0,   5.0,   "float", "segmentation",   "segmentation.py:45",   "YES"),
    "rhythm_rr_min_ms":      (150.0, 300.0, "float", "feature_rhythm", "rhythm.py:89",         "YES"),
    "rhythm_rr_max_ms":      (3000.0,5000.0,"float", "feature_rhythm", "rhythm.py:89",         "YES"),
    "rhythm_rr_hard_min_ms": (200.0, 300.0, "float", "feature_rhythm", "rhythm.py:57",         "YES"),
    "rhythm_rr_hard_max_ms": (2500.0,3500.0,"float", "feature_rhythm", "rhythm.py:57",         "YES"),

    # Wave-region boundaries (defensibly fixed per AHA but tunable for adaptation)
    "early_p_ms":            (-300,  -150,  "int",   "segmentation",   "constants.py:133",     "MAYBE"),
    "normal_p_ms":           (-150,  -50,   "int",   "segmentation",   "constants.py:134",     "MAYBE"),
    "qrs_frac_ms":           (40,    120,   "int",   "segmentation",   "constants.py:135",     "MAYBE"),
    "st_frac":               (0.10,  0.35,  "float", "segmentation",   "constants.py:136",     "MAYBE"),
    "t_early_frac":          (0.15,  0.40,  "float", "segmentation",   "constants.py:137",     "MAYBE"),
    "t_normal_frac":         (0.30,  0.55,  "float", "segmentation",   "constants.py:138",     "MAYBE"),
    "offset_ms":             (40,    150,   "int",   "segmentation",   "constants.py:139",     "MAYBE"),
    "tolerance_ms":          (30,    100,   "int",   "segmentation",   "constants.py:140",     "MAYBE"),

    # ── gaussian fitting ────────────────────────────────────────────────
    "amplitude_threshold_unscaled": (0.01, 0.05,  "float", "gaussian_fit", "constants.py:100", "YES"),
    "fraction_threshold":    (0.5,   0.95,  "float", "gaussian_fit",   "constants.py:101",     "YES"),
    "boundary_amp_frac":     (0.15,  0.50,  "float", "gaussian_fit",   "constants.py:102",     "YES"),
    "delta_angle":           (math.radians(30), math.radians(90), "float", "gaussian_fit", "constants.py:103", "MAYBE"),
    "sigma_max_1d":          (40.0,  150.0, "float", "gaussian_fit",   "gaussian_fitting.py:115","YES"),
    "sigma_max_2d":          (40.0,  150.0, "float", "gaussian_fit",   "gaussian_fitting.py:255","YES"),
    "angle_lambda_2d":       (0.0,   0.5,   "float", "gaussian_fit",   "gaussian_fitting.py:253","YES"),
    "amp_lambda_2d":         (0.0,   1.0,   "float", "gaussian_fit",   "gaussian_fitting.py:254","YES"),
    "amp_max_mult":          (3.0,   10.0,  "float", "gaussian_fit",   "gaussian_fitting.py:284","YES"),
    "least_sq_max_nfev":     (1000,  10000, "int",   "gaussian_fit",   "gaussian_fitting.py:200","YES"),
    "energy_win_ms":         (10.0,  30.0,  "float", "gaussian_fit",   "wave_fitters.py:100",  "YES"),
    "aicc_k1_penalty":       (1,     4,     "int",   "gaussian_fit",   "wave_fitters.py:241",  "YES"),
    "aicc_k2_penalty":       (2,     8,     "int",   "gaussian_fit",   "wave_fitters.py:241",  "YES"),

    # ── per-lead refit / wave seeding ───────────────────────────────────
    "p_snr_db_min":          (-10.0, 0.0,   "float", "lead_refit",     "wave_fitters.py:272",  "YES"),
    "p_center_iqr_k":        (1.0,   3.0,   "float", "lead_refit",     "wave_fitters.py:273",  "YES"),
    "p_center_spread_guard_ms":(40.0,150.0, "float", "lead_refit",     "wave_fitters.py:274",  "YES"),
    "p_default_sigma_ms":    (15.0,  40.0,  "float", "lead_refit",     "wave_fitters.py:275",  "YES"),
    "p_max_weight_ratio":    (3.0,   15.0,  "float", "lead_refit",     "wave_fitters.py:276",  "YES"),
    "p_sigma_min_ms":        (4.0,   12.0,  "float", "lead_refit",     "lead_fitting.py:98",   "YES"),
    "p_sigma_max_ms":        (20.0,  60.0,  "float", "lead_refit",     "lead_fitting.py:98",   "YES"),
    "p_min_separation_ms":   (8.0,   30.0,  "float", "lead_refit",     "lead_fitting.py:99",   "YES"),
    "p_max_separation_ms":   (100.0, 250.0, "float", "lead_refit",     "lead_fitting.py:100",  "YES"),
    "t_sigma_min_ms":        (15.0,  50.0,  "float", "lead_refit",     "lead_fitting.py:226",  "YES"),
    "t_sigma_max_ms":        (80.0,  200.0, "float", "lead_refit",     "lead_fitting.py:226",  "YES"),
    "t_min_separation_ms":   (20.0,  80.0,  "float", "lead_refit",     "lead_fitting.py:227",  "YES"),
    "t_max_separation_ms":   (100.0, 300.0, "float", "lead_refit",     "lead_fitting.py:227",  "YES"),
    "qrs_min_ms":            (40.0,  100.0, "float", "lead_refit",     "lead_fitting.py:288",  "YES"),
    "qrs_max_ms":            (150.0, 300.0, "float", "lead_refit",     "lead_fitting.py:288",  "YES"),
    "qrs_padding_ms":        (5.0,   30.0,  "float", "lead_refit",     "lead_fitting.py:288",  "YES"),
    "qrs_sigma_min_ms":      (3.0,   15.0,  "float", "lead_refit",     "lead_fitting.py:289",  "YES"),
    "qrs_sigma_max_ms":      (30.0,  100.0, "float", "lead_refit",     "lead_fitting.py:289",  "YES"),
    "qrs_q_s_threshold_rel": (0.01,  0.10,  "float", "lead_refit",     "lead_fitting.py:290",  "YES"),
    "qrs_rprime_threshold_rel":(0.01,0.15,  "float", "lead_refit",     "lead_fitting.py:291",  "YES"),
    "qrs_min_sep_ms":        (3.0,   20.0,  "float", "lead_refit",     "lead_fitting.py:291",  "YES"),
    "qrs_max_components_clean":  (1, 2,     "int",   "lead_refit",     "lead_fitting.py:454",  "YES"),
    "qrs_max_components_default":(2, 6,     "int",   "lead_refit",     "lead_fitting.py:454",  "YES"),
    "qrs_clean_residual_threshold":(0.08,0.25,"float","lead_refit",    "lead_fitting.py:379",  "YES"),
    "qrs_onset_sigma_mult":  (1.0,   2.5,   "float", "lead_refit",     "global_features.py:74","YES"),

    # ── PR-interval gating (lead_fitting_helpers) ───────────────────────
    "p_normal_pr_min_ms":    (100.0, 140.0, "float", "feature_pwave",  "lead_fitting_helpers.py:110","YES"),
    "p_normal_pr_max_ms":    (180.0, 250.0, "float", "feature_pwave",  "lead_fitting_helpers.py:110","YES"),
    "p_fast_hr_pr_min_ms":   (80.0,  120.0, "float", "feature_pwave",  "lead_fitting_helpers.py:112","YES"),
    "p_fast_hr_pr_max_ms":   (160.0, 220.0, "float", "feature_pwave",  "lead_fitting_helpers.py:112","YES"),
    "p_slow_hr_pr_min_ms":   (120.0, 160.0, "float", "feature_pwave",  "lead_fitting_helpers.py:113","YES"),
    "p_slow_hr_pr_max_ms":   (250.0, 350.0, "float", "feature_pwave",  "lead_fitting_helpers.py:114","YES"),
    "p_hr_threshold_fast_ms":(500.0, 700.0, "float", "feature_pwave",  "lead_fitting_helpers.py:111","YES"),
    "p_hr_threshold_slow_ms":(1000.0,1400.0,"float", "feature_pwave",  "lead_fitting_helpers.py:113","YES"),
    "p_search_range_ms":     (400.0, 800.0, "float", "feature_pwave",  "lead_fitting_helpers.py:119","YES"),

    # ── ST features ─────────────────────────────────────────────────────
    "st_j60_window_ms":      (80.0,  160.0, "float", "feature_st",     "st_deviation.py:43",   "YES"),
    "st_elevation_thr_mv":   (0.05,  0.20,  "float", "feature_st",     "st_deviation.py:44",   "YES"),
    "st_depression_thr_mv":  (-0.10, -0.02, "float", "feature_st",     "st_deviation.py:45",   "YES"),
    "st_slope_thr_uv_per_ms":(2.0,   10.0,  "float", "feature_st",     "st_deviation.py:46",   "YES"),
    "st_flat_thr_mv":        (0.01,  0.05,  "float", "feature_st",     "st_deviation.py:47",   "YES"),
    "st_min_valid_seg_ms":   (20.0,  80.0,  "float", "feature_st",     "st_deviation.py:48",   "YES"),
    "st_elev_v1v2_high":     (0.12,  0.30,  "float", "feature_st",     "st_deviation.py:51",   "YES"),
    "st_elev_v1v2_low":      (0.05,  0.15,  "float", "feature_st",     "st_deviation.py:52",   "YES"),
    "st_elev_v2_man_young":  (0.15,  0.35,  "float", "feature_st",     "st_deviation.py:70",   "YES"),
    "st_elev_v2_man_old":    (0.12,  0.30,  "float", "feature_st",     "st_deviation.py:70",   "YES"),
    "st_elev_v2_woman":      (0.08,  0.25,  "float", "feature_st",     "st_deviation.py:68",   "YES"),
    "st_elev_v3_man_young":  (0.15,  0.35,  "float", "feature_st",     "st_deviation.py:70",   "YES"),
    "st_elev_v3_man_old":    (0.12,  0.30,  "float", "feature_st",     "st_deviation.py:70",   "YES"),
    "st_elev_v3_woman":      (0.08,  0.25,  "float", "feature_st",     "st_deviation.py:68",   "YES"),
    "st_elev_other_leads":   (0.05,  0.15,  "float", "feature_st",     "st_deviation.py:71",   "YES"),
    "st_depr_other_leads":   (-0.10, -0.02, "float", "feature_st",     "st_deviation.py:82",   "YES"),
    "st_j_point_offset_ms":  (20.0,  80.0,  "float", "feature_st",     "st_deviation.py:187",  "YES"),
    "st_baseline_lookback_start_ms":(-80,-10,"int",  "feature_st",     "st_deviation.py:249",  "YES"),
    "st_baseline_lookback_end_ms":(-20, 0,  "int",   "feature_st",     "st_deviation.py:249",  "YES"),
    "st_tp_lookback_start_ms":(-80, -10,    "int",   "feature_st",     "st_deviation.py:249",  "YES"),
    "st_tp_lookback_end_ms": (-20,  0,      "int",   "feature_st",     "st_deviation.py:249",  "YES"),
    "st_late_tp_start_ms":   (300,  400,    "int",   "feature_st",     "st_deviation.py:257",  "YES"),
    "st_late_tp_end_ms":     (400,  500,    "int",   "feature_st",     "st_deviation.py:257",  "YES"),

    # ── P-wave features ─────────────────────────────────────────────────
    "p_amp_difference_threshold":(0.15,0.50,"float", "feature_pwave",  "constants.py:122",     "YES"),
    "p_time_difference_threshold":(10.0,50.0,"float","feature_pwave",  "constants.py:123",     "YES"),
    "p_theta_difference_threshold":(math.radians(30), math.radians(180), "float", "feature_pwave", "constants.py:124", "MAYBE"),
    "p_duration_sigma_mult": (1.0,   3.0,   "float", "feature_pwave",  "p_wave.py:33",         "YES"),
    "p_pr_depression_thr_mv":(-0.08, -0.02, "float", "feature_pwave",  "p_wave.py:145",        "YES"),
    "p_pr_depression_min_leads":(1,  4,     "int",   "feature_pwave",  "p_wave.py:148",        "YES"),
    "p_bigeminy_long_rr_frac":(0.8, 1.0,    "float", "feature_pwave",  "p_wave.py:193",        "YES"),
    "p_bigeminy_short_rr_frac":(0.7,0.95,   "float", "feature_pwave",  "p_wave.py:193",        "YES"),
    "p_bigeminy_frac_threshold":(0.3,0.8,   "float", "feature_pwave",  "p_wave.py:199",        "YES"),
    "p_af_rr_cv_threshold":  (0.05,  0.30,  "float", "feature_pwave",  "p_wave.py:265",        "YES"),
    "p_amp_threshold_av_block":(0.02,0.10,  "float", "feature_pwave",  "p_wave.py:259",        "YES"),

    # ── T-wave features ─────────────────────────────────────────────────
    "t_amp_thr":             (0.15,  0.50,  "float", "feature_t",      "constants.py:125",     "YES"),
    "t_dt_thr":              (10.0,  50.0,  "float", "feature_t",      "constants.py:126",     "YES"),
    "t_qrs_thr":             (20,    100,   "int",   "feature_t",      "constants.py:127",     "YES"),
    "t_dxy_thr":             (math.radians(30), math.radians(180), "float", "feature_t", "constants.py:128", "MAYBE"),
    "t_alternans_min_beats": (4,     12,    "int",   "feature_t",      "t_wave.py:60",         "YES"),
    "t_alternans_sign_change_frac":(0.5,0.9,"float", "feature_t",      "t_wave.py:71",         "YES"),
    "t_alternans_var_threshold":(8.0,25.0,  "float", "feature_t",      "t_wave.py:73",         "YES"),
    "t_inverted_amp_threshold":(-0.40,-0.10,"float", "feature_t",      "t_wave.py:89",         "YES"),
    "t_flattened_amp_min":   (0.04,  0.15,  "float", "feature_t",      "t_wave.py:111",        "YES"),
    "t_flattened_qrs_frac":  (0.05,  0.25,  "float", "feature_t",      "t_wave.py:109",        "YES"),
    "t_notched_amp_min":     (0.05,  0.20,  "float", "feature_t",      "t_wave.py:133",        "YES"),
    "t_notched_sep_min_ms":  (20.0,  80.0,  "float", "feature_t",      "t_wave.py:135",        "YES"),
    "t_notched_valley_frac": (0.15,  0.50,  "float", "feature_t",      "t_wave.py:144",        "YES"),
    "t_biphasic_min_amp":    (0.08,  0.30,  "float", "feature_t",      "t_wave.py:166",        "YES"),
    "t_biphasic_amp_ratio":  (0.15,  0.50,  "float", "feature_t",      "t_wave.py:171",        "YES"),
    "t_biphasic_sep_min_ms": (15.0,  60.0,  "float", "feature_t",      "t_wave.py:172",        "YES"),
    "t_prolonged_duration_ms":(300.0,450.0, "float", "feature_t",      "t_wave.py:192",        "YES"),
    "t_prolonged_amp_min":   (0.05,  0.20,  "float", "feature_t",      "t_wave.py:192",        "YES"),

    # ── QRS features ────────────────────────────────────────────────────
    "qrs_axis_extreme_threshold_deg":(60.0, 120.0, "float", "feature_qrs", "qrs_complex.py:?", "YES"),
    "qrs_voltage_high_limb": (1.0,   3.0,   "float", "feature_qrs",    "qrs_complex.py:?",     "YES"),
    "qrs_voltage_high_precordial":(2.0,5.0, "float", "feature_qrs",    "qrs_complex.py:?",     "YES"),
    "qrs_voltage_low_limb":  (0.2,   1.0,   "float", "feature_qrs",    "qrs_complex.py:?",     "YES"),
    "qrs_m_shaped_amp_min":  (0.05,  0.25,  "float", "feature_qrs",    "qrs_complex.py:196",   "YES"),
    "qrs_m_shaped_sep_min_ms":(5.0,  30.0,  "float", "feature_qrs",    "qrs_complex.py:196",   "YES"),

    # ── rhythm ──────────────────────────────────────────────────────────
    "rr_arrhythmia_threshold":(0.02, 0.15,  "float", "feature_rhythm", "constants.py:121",     "YES"),
    "rr_kmeans_k_min":       (1,     2,     "int",   "feature_rhythm", "rhythm.py:99",         "YES"),
    "rr_kmeans_k_max":       (2,     5,     "int",   "feature_rhythm", "rhythm.py:100",        "YES"),
    "rr_histogram_bins":     (10,    40,    "int",   "feature_rhythm", "rhythm.py:123",        "YES"),
    "pnn50_threshold_ms":    (20.0,  100.0, "float", "feature_rhythm", "rhythm.py:118",        "YES"),

    # ── global features (QT, QTc, axis, signal-quality) ─────────────────
    "qt_min_ms":             (200.0, 280.0, "float", "feature_global", "global_features.py:70","YES"),
    "qt_max_ms":             (500.0, 700.0, "float", "feature_global", "global_features.py:70","YES"),
    "min_valid_beats_per_lead":(1,   5,     "int",   "feature_global", "global_features.py:71","YES"),
    "qt_mad_trim_mult":      (1.5,   4.0,   "float", "feature_global", "global_features.py:72","YES"),
    "qt_mad_min_abs_ms":     (10.0,  80.0,  "float", "feature_global", "global_features.py:73","YES"),
    "qt_search_start_offset_ms":(50.0,200.0,"float", "feature_global", "global_features.py:136","YES"),
    "qt_search_end_offset_ms":(600.0,1000.0,"float", "feature_global", "global_features.py:137","YES"),
    "qtc_male_threshold_ms": (430.0, 470.0, "float", "feature_global", "global_features.py:294","YES"),
    "qtc_female_threshold_ms":(440.0,480.0, "float", "feature_global", "global_features.py:294","YES"),
    "qtc_neutral_threshold_ms":(435.0,475.0,"float", "feature_global", "global_features.py:294","YES"),
    "qtc_mild_threshold_ms": (460.0, 500.0, "float", "feature_global", "global_features.py:301","YES"),
    "qtc_moderate_threshold_ms":(480.0,520.0,"float","feature_global", "global_features.py:301","YES"),
    "alternans_std_threshold":(0.01, 0.05,  "float", "feature_global", "global_features.py:431","YES"),
    "alternans_r_threshold": (0.8,   0.95,  "float", "feature_global", "global_features.py:445","YES"),
    "alternans_delta_threshold":(5.0,20.0,  "float", "feature_global", "global_features.py:445","YES"),
    "bwi_threshold_red":     (1.0,   2.0,   "float", "feature_global", "global_features.py:532","YES"),
    "bwi_threshold_amb":     (0.7,   1.3,   "float", "feature_global", "global_features.py:532","YES"),
    "emg_threshold_red":     (0.8,   1.8,   "float", "feature_global", "global_features.py:533","YES"),
    "emg_threshold_amb":     (0.5,   1.2,   "float", "feature_global", "global_features.py:533","YES"),
    "clip_threshold_red":    (0.002, 0.010, "float", "feature_global", "global_features.py:534","YES"),
    "clip_threshold_amb":    (0.001, 0.005, "float", "feature_global", "global_features.py:534","YES"),
    "flat_threshold_red":    (0.5,   1.0,   "float", "feature_global", "global_features.py:535","YES"),
    "flat_threshold_amb":    (0.3,   0.8,   "float", "feature_global", "global_features.py:535","YES"),
    "p2p_limb_min_mv":       (0.10,  0.35,  "float", "feature_global", "global_features.py:536","YES"),
    "p2p_prec_min_mv":       (0.20,  0.60,  "float", "feature_global", "global_features.py:536","YES"),
    "ll_threshold_red":      (1.0,   2.0,   "float", "feature_global", "global_features.py:538","YES"),
    "ll_threshold_amb":      (0.8,   1.5,   "float", "feature_global", "global_features.py:538","YES"),
    "band_freq_border_hz":   (0.1,   1.0,   "float", "feature_global", "global_features.py:637","YES"),
    "band_freq_signal_low_hz":(1.0,  5.0,   "float", "feature_global", "global_features.py:638","YES"),
    "band_freq_signal_high_hz":(15.0,40.0,  "float", "feature_global", "global_features.py:638","YES"),
    "band_freq_emg_low_hz":  (25.0,  50.0,  "float", "feature_global", "global_features.py:639","YES"),
    "band_freq_emg_high_hz": (50.0,  100.0, "float", "feature_global", "global_features.py:639","YES"),
    "clip_fraction_threshold":(0.001,0.015, "float", "feature_global", "global_features.py:561","YES"),
    "clip_fraction_min_baseline":(0.005,0.05,"float","feature_global", "global_features.py:561","YES"),
    "line_length_window_frac":(0.10,0.40,   "float", "feature_global", "global_features.py:562","YES"),

    # ── physical constants (NOT tunable) ────────────────────────────────
    "fs_hz":                 (500,   500,   "int",   "physical_constant","constants.py:87",    "NO"),
    "log_eps":               (1e-8,  1e-8,  "float", "physical_constant","constants.py:120",   "NO"),
    "gauss_min_sigma_1d_ms": (1.0,   1.0,   "float", "physical_constant","gaussian_fitting.py:168","NO"),
}


# ---------------------------------------------------------------------------
# HyperParams dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HyperParams:
    """Frozen panel of all tunable q_psi hyperparameters.

    Defaults reproduce the production qpsi behaviour exactly. Override any
    subset by constructing with keyword arguments::

        cfg = HyperParams(qrs_amp_thr=0.018, st_elevation_thr_mv=0.08)

    Use ``HyperParams.from_dict({...})`` to load from an Optuna trial,
    JSON file, etc.
    """

    # ── preprocessing ───────────────────────────────────────────────────
    hp_corner_hz:                   float = 0.1
    lp_clean_hz:                    float = 200.0
    lp_noisy_hz:                    float = 35.0
    hf_band_low_hz:                 float = 48.0
    hf_band_high_hz:                float = 200.0
    baseline_window_ms:             int   = 600
    baseline_fluct_max:             float = 0.8
    max_noise_level:                float = 0.005

    # ── rpeak / pacing detection ────────────────────────────────────────
    pace_sigma_ms:                  float = 4.0
    pace_amp_mv:                    float = 0.025

    # ── segmentation ────────────────────────────────────────────────────
    pad_head_r_ms:                  float = 40.0
    pad_tail_t_ms:                  int   = 20
    crosscorr_win_ms:               float = 60.0
    xcorr_coarse_win_ms:            float = 60.0
    xcorr_fine_align_ms:            float = 80.0
    rr_min_ms:                      float = 300.0
    rr_max_ms:                      float = 2000.0
    rr_mad_trim:                    float = 3.5
    rhythm_rr_min_ms:               float = 200.0
    rhythm_rr_max_ms:               float = 4000.0
    rhythm_rr_hard_min_ms:          float = 240.0
    rhythm_rr_hard_max_ms:          float = 3000.0
    early_p_ms:                     int   = -210
    normal_p_ms:                    int   = -100
    qrs_frac_ms:                    int   = 80
    st_frac:                        float = 0.20
    t_early_frac:                   float = 0.26
    t_normal_frac:                  float = 0.40
    offset_ms:                      int   = 75
    tolerance_ms:                   int   = 60

    # ── gaussian fitting ────────────────────────────────────────────────
    amplitude_threshold_unscaled:   float = 0.025
    fraction_threshold:             float = 0.8
    boundary_amp_frac:              float = 0.33
    delta_angle:                    float = math.pi / 3
    sigma_max_1d:                   float = 80.0
    sigma_max_2d:                   float = 80.0
    angle_lambda_2d:                float = 0.1
    amp_lambda_2d:                  float = 0.2
    amp_max_mult:                   float = 5.0
    least_sq_max_nfev:              int   = 3000
    energy_win_ms:                  float = 18.0
    aicc_k1_penalty:                int   = 2
    aicc_k2_penalty:                int   = 4

    # ── per-lead refit / wave seeding ───────────────────────────────────
    p_snr_db_min:                   float = -3.0
    p_center_iqr_k:                 float = 1.5
    p_center_spread_guard_ms:       float = 80.0
    p_default_sigma_ms:             float = 25.0
    p_max_weight_ratio:             float = 8.0
    p_sigma_min_ms:                 float = 6.0
    p_sigma_max_ms:                 float = 30.0
    p_min_separation_ms:            float = 15.0
    p_max_separation_ms:            float = 150.0
    t_sigma_min_ms:                 float = 30.0
    t_sigma_max_ms:                 float = 140.0
    t_min_separation_ms:            float = 40.0
    t_max_separation_ms:            float = 180.0
    qrs_min_ms:                     float = 60.0
    qrs_max_ms:                     float = 250.0
    qrs_padding_ms:                 float = 10.0
    qrs_sigma_min_ms:               float = 6.0
    qrs_sigma_max_ms:               float = 60.0
    qrs_q_s_threshold_rel:          float = 0.03
    qrs_rprime_threshold_rel:       float = 0.05
    qrs_min_sep_ms:                 float = 8.0
    qrs_max_components_clean:       int   = 1
    qrs_max_components_default:     int   = 4
    qrs_clean_residual_threshold:   float = 0.15
    qrs_onset_sigma_mult:           float = 1.5

    # ── PR-interval gating ──────────────────────────────────────────────
    p_normal_pr_min_ms:             float = 120.0
    p_normal_pr_max_ms:             float = 200.0
    p_fast_hr_pr_min_ms:            float = 100.0
    p_fast_hr_pr_max_ms:            float = 180.0
    p_slow_hr_pr_min_ms:            float = 140.0
    p_slow_hr_pr_max_ms:            float = 300.0
    p_hr_threshold_fast_ms:         float = 600.0
    p_hr_threshold_slow_ms:         float = 1200.0
    p_search_range_ms:              float = 600.0

    # ── ST features ─────────────────────────────────────────────────────
    st_j60_window_ms:               float = 120.0
    st_elevation_thr_mv:            float = 0.10
    st_depression_thr_mv:           float = -0.05
    st_slope_thr_uv_per_ms:         float = 5.0
    st_flat_thr_mv:                 float = 0.02
    st_min_valid_seg_ms:            float = 40.0
    st_elev_v1v2_high:              float = 0.20
    st_elev_v1v2_low:               float = 0.10
    st_elev_v2_man_young:           float = 0.25
    st_elev_v2_man_old:             float = 0.20
    st_elev_v2_woman:               float = 0.15
    st_elev_v3_man_young:           float = 0.25
    st_elev_v3_man_old:             float = 0.20
    st_elev_v3_woman:               float = 0.15
    st_elev_other_leads:            float = 0.10
    st_depr_other_leads:            float = -0.05
    st_j_point_offset_ms:           float = 40.0
    st_baseline_lookback_start_ms:  int   = -40
    st_baseline_lookback_end_ms:    int   = 0
    st_tp_lookback_start_ms:        int   = -40
    st_tp_lookback_end_ms:          int   = 0
    st_late_tp_start_ms:            int   = 350
    st_late_tp_end_ms:              int   = 450

    # ── P-wave features ─────────────────────────────────────────────────
    p_amp_difference_threshold:     float = 0.33
    p_time_difference_threshold:    float = 25.0
    p_theta_difference_threshold:   float = math.pi / 2
    p_duration_sigma_mult:          float = 2.0
    p_pr_depression_thr_mv:         float = -0.04
    p_pr_depression_min_leads:      int   = 2
    p_bigeminy_long_rr_frac:        float = 0.9
    p_bigeminy_short_rr_frac:       float = 0.85
    p_bigeminy_frac_threshold:      float = 0.50
    p_af_rr_cv_threshold:           float = 0.15
    p_amp_threshold_av_block:       float = 0.05

    # ── T-wave features ─────────────────────────────────────────────────
    t_amp_thr:                      float = 0.33
    t_dt_thr:                       float = 25.0
    t_qrs_thr:                      int   = 50
    t_dxy_thr:                      float = math.pi / 2
    t_alternans_min_beats:          int   = 8
    t_alternans_sign_change_frac:   float = 0.7
    t_alternans_var_threshold:      float = 15.0
    t_inverted_amp_threshold:       float = -0.20
    t_flattened_amp_min:            float = 0.08
    t_flattened_qrs_frac:           float = 0.12
    t_notched_amp_min:              float = 0.10
    t_notched_sep_min_ms:           float = 40.0
    t_notched_valley_frac:          float = 0.30
    t_biphasic_min_amp:             float = 0.15
    t_biphasic_amp_ratio:           float = 0.30
    t_biphasic_sep_min_ms:          float = 35.0
    t_prolonged_duration_ms:        float = 360.0
    t_prolonged_amp_min:            float = 0.10

    # ── QRS features ────────────────────────────────────────────────────
    qrs_axis_extreme_threshold_deg: float = 90.0
    qrs_voltage_high_limb:          float = 2.0
    qrs_voltage_high_precordial:    float = 3.5
    qrs_voltage_low_limb:           float = 0.5
    qrs_m_shaped_amp_min:           float = 0.12
    qrs_m_shaped_sep_min_ms:        float = 15.0

    # ── rhythm ──────────────────────────────────────────────────────────
    rr_arrhythmia_threshold:        float = 0.05
    rr_kmeans_k_min:                int   = 1
    rr_kmeans_k_max:                int   = 3
    rr_histogram_bins:              int   = 20
    pnn50_threshold_ms:             float = 50.0

    # ── global features ─────────────────────────────────────────────────
    qt_min_ms:                      float = 240.0
    qt_max_ms:                      float = 600.0
    min_valid_beats_per_lead:       int   = 3
    qt_mad_trim_mult:               float = 2.5
    qt_mad_min_abs_ms:              float = 30.0
    qt_search_start_offset_ms:      float = 100.0
    qt_search_end_offset_ms:        float = 800.0
    qtc_male_threshold_ms:          float = 450.0
    qtc_female_threshold_ms:        float = 460.0
    qtc_neutral_threshold_ms:       float = 455.0
    qtc_mild_threshold_ms:          float = 480.0
    qtc_moderate_threshold_ms:      float = 500.0
    alternans_std_threshold:        float = 0.02
    alternans_r_threshold:          float = 0.9
    alternans_delta_threshold:      float = 12.0
    bwi_threshold_red:              float = 1.50
    bwi_threshold_amb:              float = 1.00
    emg_threshold_red:              float = 1.20
    emg_threshold_amb:              float = 0.80
    clip_threshold_red:             float = 0.005
    clip_threshold_amb:             float = 0.002
    flat_threshold_red:             float = 0.75
    flat_threshold_amb:             float = 0.55
    p2p_limb_min_mv:                float = 0.20
    p2p_prec_min_mv:                float = 0.35
    ll_threshold_red:               float = 1.40
    ll_threshold_amb:               float = 1.10
    band_freq_border_hz:            float = 0.33
    band_freq_signal_low_hz:        float = 2.0
    band_freq_signal_high_hz:       float = 20.0
    band_freq_emg_low_hz:           float = 35.0
    band_freq_emg_high_hz:          float = 70.0
    clip_fraction_threshold:        float = 0.005
    clip_fraction_min_baseline:     float = 0.015
    line_length_window_frac:        float = 0.20

    # ── physical constants (do NOT tune) ────────────────────────────────
    fs_hz:                          int   = 500
    log_eps:                        float = 1e-8
    gauss_min_sigma_1d_ms:          float = 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperParams":
        """Build a panel from a dict (Optuna trial, JSON, etc.)."""
        valid = {f.name for f in fields(cls)}
        unknown = set(d.keys()) - valid
        if unknown:
            raise KeyError(f"Unknown HyperParams fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> Dict[str, Any]:
        """Export the panel as a flat dict (for logging / serialisation)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def tunable_field_names(cls, include_maybe: bool = True) -> list[str]:
        """Return the list of field names with PARAM_META tunable=YES (and MAYBE)."""
        keep = {"YES", "MAYBE"} if include_maybe else {"YES"}
        return [name for name, meta in PARAM_META.items()
                if meta[5] in keep and any(f.name == name for f in fields(cls))]


DEFAULT = HyperParams()
"""Module-level default panel — convenience for callers that don't override."""


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Confirm every PARAM_META key has a matching dataclass field."""
    field_names = {f.name for f in fields(HyperParams)}
    meta_names = set(PARAM_META.keys())
    missing_in_class = meta_names - field_names
    missing_in_meta  = field_names - meta_names
    if missing_in_class:
        raise RuntimeError(f"PARAM_META references unknown fields: {sorted(missing_in_class)}")
    if missing_in_meta:
        raise RuntimeError(f"HyperParams fields missing from PARAM_META: {sorted(missing_in_meta)}")


_self_check()


if __name__ == "__main__":
    n_total   = sum(1 for _ in fields(HyperParams))
    n_yes     = sum(1 for v in PARAM_META.values() if v[5] == "YES")
    n_maybe   = sum(1 for v in PARAM_META.values() if v[5] == "MAYBE")
    n_no      = sum(1 for v in PARAM_META.values() if v[5] == "NO")
    print(f"HyperParams panel — {n_total} fields total")
    print(f"  tunable=YES:   {n_yes}")
    print(f"  tunable=MAYBE: {n_maybe}")
    print(f"  tunable=NO:    {n_no}")
