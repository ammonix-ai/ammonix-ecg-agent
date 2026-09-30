# QPSI-Direct — Design and Feature Catalogue

**Status:** in production; trained classifiers (Apr28All adult swarm + ongoing pediatric swarm) consume this schema
**Predecessor:** `qpsi/` (Gaussian-fit pipeline, 2024–early 2026)
**This module:** `qpsi_direct/` — no-Gaussian, direct waveform-shape measurements
**Companion docs:** `FIDELITY_REPORT.md` (numerical equivalence to legacy qpsi), `QPSI_PROFILING_REPORT.md` (per-step timing)

This document is the architectural overview that's been missing — *why* we moved off Gaussian fits, *what* replaced them, *what* the ~19,500-feature schema actually contains, and *what* it bought us.

---

## 1. Why we left Gaussians behind

The legacy `qpsi` pipeline fitted a small mixture of Gaussian components per wave-region (P, QRS, T) using non-linear least-squares (`scipy.optimize.least_squares`, TRF). For each (lead, region, beat) it produced a structured tuple `(amp_mv, mu_ms, sigma_ms)` per Gaussian component, with model selection (K=1 vs K=2) by AICc.

This was elegant on paper but expensive and fragile in practice:

| pain point | what happened |
|---|---|
| **Convergence failures** | When a wave-region had a saddle, notch, biphasic shape, or low SNR, the LSQ optimiser frequently failed to converge or settled in shallow local minima. Failure rate climbed to 5–15% on noisy / pathological records (exactly the cases we most want to characterise). |
| **`sigma_ms` instability** | The Gaussian "width" parameter has no direct anatomical meaning — fits at K=1 and K=2 produce wildly different `sigma_ms` for the same wave. Aggregating across leads or beats was statistically ill-posed. |
| **Speed** | ~85% of the per-record runtime was spent inside `least_squares`. Profiling on a single Chapman–Shaoxing patient (`JS00001`) showed 18.7 s of 30.4 s wall (61%) on P-wave + T-wave fits alone. |
| **Loss of morphology** | A wave with three peaks (notch + main + secondary) can't be cleanly described by 2 Gaussians, but adding a 3rd always does worse on AICc. The model couldn't "see" what was there. |
| **Bifurcation at the AICc boundary** | Two near-identical waves could end up with very different feature vectors purely because one was fit at K=1 and the other at K=2 — a discontinuity at the model-selection boundary. |

The pipeline still worked, but every classifier downstream had to accommodate these artefacts. Years of feature engineering went into smoothing over Gaussian-fit anomalies rather than measuring the ECG itself.

**The question we asked in early 2026:** what if we just *measured* the wave directly — extrema, derivatives, durations — and skipped the fit entirely?

---

## 2. The replacement strategy: direct measurements

`qpsi_direct/direct_measurements.py` reads, verbatim from its docstring:

> Direct waveform-shape measurements — the no-Gaussian alternative.
> No fitting, no model selection, no AICc bifurcations.

Three principles:

1. **Per-beat scalars only.** Each (lead, region, beat) emits a fixed-arity vector of continuous, baseline-aware primitives. No structured Gaussian tuples, no per-component naming, no model-selection choice.

2. **Robustness over precision.** Median / MAD / IQR aggregation across beats is intrinsically robust to outliers; Gaussian-fit aggregation depended on every fit succeeding.

3. **Capture what's there.** A separate extrema-finder identifies *all* local maxima and minima in the smoothed wave (saddles, notches, secondary peaks) and emits the top-K by prominence — no upper bound on morphological complexity beyond the K=4 emit cap.

---

## 3. The per-beat measurement vector

For each `(lead, region, beat)` triple, `direct_measurements.measure_window` produces:

### 3a. Single-extremum primitives (always 5 scalars)

| field | meaning |
|---|---|
| `pos_peak_amp_mv` | signed amplitude of the largest positive extremum in the window |
| `pos_peak_time_ms` | sub-sample time of that extremum |
| `neg_peak_amp_mv` | signed amplitude of the largest negative extremum |
| `neg_peak_time_ms` | sub-sample time of that extremum |
| `peak_to_peak_mv` | `pos_peak_amp_mv − neg_peak_amp_mv` (baseline-invariant) |

### 3b. Shape primitives (4 scalars)

| field | meaning |
|---|---|
| `centroid_ms` | first moment of `\|x_detrended\|` — the wave's "centre of mass" timing |
| `duration_ms_rms` | 4 × second moment — RMS-equivalent width |
| `duration_ms_fwhm` | full-width at half-maximum amplitude |
| `max_abs_derivative_mv_per_ms` | peak slope magnitude (baseline-invariant) |

### 3c. Energy primitives (2 scalars)

| field | meaning |
|---|---|
| `rms_amplitude_mv` | RMS of the detrended signal |
| `snr_db` | peak amplitude vs RMS of the window edges (noise floor proxy) |

### 3d. Top-K extrema (8 scalars per K-slot, K=4 fixed)

For each of the four most prominent extrema (ranked by prominence-against-flanking):

| field per slot k=1..4 | meaning |
|---|---|
| `extr_k_kind` | +1 (max) or −1 (min) |
| `extr_k_amp_mv` | signed amplitude |
| `extr_k_time_ms` | sub-sample timing |
| `extr_k_prominence` | depth relative to higher of two flanking extrema |

This catches saddles (the bump that Gaussian fits would smear), notches in QRS (Brugada, RBBB hallmark), biphasic T-waves, and secondary P-wave components — *all* without committing to a parametric form.

**Total per (lead, region, beat): 11 base scalars + 4 × 4 extrema = 27 numbers.**

---

## 4. The aggregation layer

Per-beat scalars become per-record features by aggregating across each lead's beats. For each per-beat scalar `x`, we emit:

| aggregator | what it captures |
|---|---|
| `median` | central tendency robust to outliers (1–2 bad beats out of 10–30) |
| `mad` (median absolute deviation) | scale, robust |
| `iqr` (Q3 − Q1) | scale, robust, complements MAD |
| `n` (= n_beats) | how many beats actually contributed |
| `alternation_index_p2` | DFT amplitude at period-2 / MAD — beat-to-beat alternans (ABAB) |
| `alternation_index_p3` | period-3 (ABC ABC — trigeminy-like) |
| `alternation_index_p4` | period-4 (ABCD ABCD) |
| `acf_lag_1` … `acf_lag_5` | autocorrelation at lag k beats — Wiener–Khinchin equivalent of DFT but at any lag, captures longer-period drift in 60-s recordings |

**The alternation index** is calibrated such that perfect ABAB alternans (synthetic) → 1.0, white noise → ~0.5, real records with no rhythm pattern → 0.3–0.7. Anything > 1.0 means the period-p component carries more coherent energy than a perfect-alternans baseline.

**Plus the per-(lead, region) `_beat_clusters` block** — a custom k-means with k ∈ {1, 2, 3} + elbow rule on the per-beat parameter vectors. Emits:
- `n_morphology_clusters` (1, 2, or 3)
- `cluster_balance_ratio` (size of dominant cluster / total)
- `longest_run_in_dominant`
- `cluster_alternation` (does the cluster sequence alternate or run in blocks?)
- `n_beats`

Useful for VT-risk (multifocal beat morphologies), ectopic rhythms, pacing rhythm with intrinsic beats interspersed.

**So per (lead, region, primitive): 11 aggregators × 27 base scalars = 297 features.**

---

## 5. The feature namespace inventory

Roughly 19,500 numeric features per recording, organised in **two top-level flat dicts** (no nested structure, keys preserved verbatim across the pipeline):

```
record
├── "direct_features"         flat dict, ~16,300 entries
│      keys:  ml_direct::<lead>::<region>::<param>::<aggregator>
│      example:  ml_direct::aVR::QRS::pos_peak_amp_mv::median
│
├── "plane_direct_features"   flat dict, ~2,950 entries  
│      keys:  plane_direct::<plane>::<region>::<param>::<aggregator>
│      example:  plane_direct::limb::T::peak_angle_deg::median
│
├── "Label"                   recording_id + Patient Data + Diagnosis (preserved
│                              from upstream, NOT re-derived by qpsi_direct)
│
├── "Semantic Features"       legacy prototype semantic flags (ST/T/QRS/rhythm
│                              presence, urgency, affected_leads) — kept
│                              alongside direct features as a complementary
│                              view; small block (~50 keys after flattening)
│
└── "ECG data"                per-beat raw structures (waveform samples,
                               beat indices, lead names) — preserved for
                               re-plotting and re-analysis; NOT consumed
                               by the swarm classifier directly
```

Per-namespace breakdown by counting:

| namespace | count | what's in there |
|---|---:|---|
| `ml_direct::<lead>::<region>::*::*` | ~10,000 | 12 leads × 3 regions (P, QRS, T) × ~30 params (the per-beat scalars + extrema slots) × 11 aggregators |
| `ml_direct::<lead>::<region>::_morphology_alternans::alt_p2/3/4_max` | ~108 | DFT alternation indices |
| `ml_direct::<lead>::<region>::_beat_clusters::*` | ~180 | beat-cluster summaries |
| `ml_direct::<lead>::<region>::*::acf_lag_1..5` | ~6,100 | autocorrelation lags |
| `plane_direct::<plane>::<region>::*::*` | ~2,950 | 3 planes (limb, chest, V2) × per-plane aggregations of frontal/horizontal axis features |

The 12 leads are I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
The 3 regions are P, QRS, T.
The 3 planes are limb (frontal axis), chest (horizontal), and V2 (a single-lead plane).

---

## 6. What's still here from the legacy qpsi (and why)

We didn't burn the Gaussian path entirely. The pipeline can still *optionally* produce a small set of legacy features for backward compatibility:

| legacy block | kept? | rationale |
|---|---|---|
| `Computed Parameters` (HR, axis, intervals) | **yes** | These are physically-meaningful aggregate measures used by every cardiologist — keeping them is cheap and aids interpretability |
| `multi lead parameters` (Gaussian fits) | **opt-in** | Only fit if specifically requested (CLI flag); never run by default |
| `Semantic Features` (ST elevation flags, T-wave inversion, etc.) | **yes** | Categorical clinical flags that downstream tools (prototype frontend, tribes, case skills) consume |
| `ECG data` (raw waveforms + beat indices) | **yes** | Required for plotting, re-analysis, and the prototype image generator |

**Gaussian fits specifically (`fit_wave_components_core`, `fit_gaussians_2d`, `wave_fitters.py`)**: still in the codebase as `gaussian_fitting.py` and `wave_fitters.py`, but the default `pipeline.py` path doesn't call them. They run only if `pruning="gaussians"` is explicitly passed, which it isn't in any production batch.

This keeps the option open if a downstream consumer ever needs Gaussian parameters again, without paying the runtime cost on every record.

---

## 7. Empirical wins

**Speed.** Per-record qpsi_direct runtime on a typical adult Chapman–Shaoxing patient (10 s recording, ~10 beats):

| pipeline | per-record wall | breakdown |
|---|---:|---|
| legacy `qpsi` (Gaussians, default) | ~30 s | 61% in `least_squares`, 25% in plane Gaussian fits, 11% k-means |
| `qpsi_direct` (no Gaussians) | ~7 s (-77%) | dominated by FFT/SavGol smoothing + extrema-finding (Rust native) |

For pediatric recordings (18 s, ~33 beats) qpsi_direct runs in ~14 s/record per worker — still 2× faster than legacy on the same data.

**Accuracy on the curated 3-class Brugada / WPW / Sinus benchmark (cohort B, prototype-internal):**

| feature schema | macro AUROC |
|---|---:|
| legacy qpsi (~1k features, Gaussians + computed + semantic + neural-net) | 0.937 |
| qpsi_direct (~19.5k features, direct extrema + morphology + plane + ACF + alternation) | **0.966** (+0.029) |

The +0.029 lift is concentrated in **Brugada specifically** — its ST/T morphology has notches and biphasic shapes that Gaussian fits couldn't characterise but extrema-with-prominence catch directly.

**Robustness.** Convergence-failure rate on the same 14k-record adult cohort:

| pipeline | failures |
|---|---:|
| legacy qpsi | 5–15% (depending on cohort SNR) |
| qpsi_direct | <1% (only on extremely degraded recordings where even baseline detection fails) |

**Failure mode of qpsi_direct** when it does happen: the Rust-native seeded-fitter sometimes panics on records where the prominence-ranking gives < 4 extrema. We catch this in `_worker_process` and return `{"ok": False}` with the error string. ~0.4% failure rate on adult Apr28All; ~0.8% on pediatric (longer recordings → more edge-case beats).

---

## 8. The downstream classifier story

The 19,500-feature schema is consumed by the **per-diagnosis swarm classifier** documented in `XGBOOST_SWARM.md` (an internal design note, not part of this repository):

- 47–69 binary XGBoost classifiers (one per canonical diagnosis)
- Each fits a **Stage-1 importance screen** to pick its own top-500 features (out of 19,500), then a **Stage-2 5-fold CV** on that subspace, then a **Stage-3 final model** for inference
- The top-features-per-diagnosis lists from a trained swarm reveal which qpsi_direct primitives matter for which condition — e.g. `lvef ≤45%` selects `aVR::QRS::max_abs_derivative_mv_per_ms::median`, V4 R-amplitude, frontal QRS axis from `plane_direct`; AF selects P-wave timing variability via `mad` and `acf_lag_1` of `pos_peak_time_ms`; Brugada selects `alt_p2_max` of V1/V2 ST-segment morphology.

The swarm wouldn't have been worth building on top of the legacy qpsi schema (1k features, ~5% of which were noise from Gaussian fitting). On qpsi_direct's 19,500 features it's a natural fit because *most* features are signal-bearing and *each* diagnosis can find its own informative subset.

---

## 9. Open questions / future work

- **A 6-th aggregator: Hampel-filtered max.** The `acf_lag_*` and `alternation_*` aggregators capture rhythm; a Hampel-filtered max would capture the "biggest non-outlier beat amplitude" — useful for ischemic changes that show in 1–2 beats but aren't part of a rhythm pattern. ~30 lines to add.

- **Beat-cluster k=4 option.** Currently capped at k ∈ {1, 2, 3}. Allowing k=4 would distinguish four-rhythm patterns (e.g. quadrigeminy) from three-rhythm patterns (trigeminy). Cost: marginal at training time, none at inference.

- **Plane angle stability score.** `plane_direct::*::peak_angle_deg::mad` measures angle variability across beats but doesn't separate "axis genuinely shifts beat-to-beat" from "noise in axis estimation." A bootstrap stability score per beat would help.

- **Cross-lead correlation features.** Currently each lead is independent. ST elevation is more diagnostic when it correlates across V1–V3 than when it appears in one lead alone. A small block of pairwise correlations of `pos_peak_amp_mv` across coronary-territory lead groups would add ~50 features and likely push STEMI AUROC.

- **Deprecate the Gaussian fallback.** No production path uses it as of Apr 2026. After a 6-month "no one needed it" probationary period we could delete `gaussian_fitting.py` and `wave_fitters.py` entirely. That trims ~1,500 lines of code.

---

## 10. File map

```
qpsi_direct/
├── direct_measurements.py    THE no-Gaussian primitives — measure_window,
│                              find_extrema, _aggregate_one (per-beat → per-record)
├── pipeline.py               orchestration — preprocess, beat-sync, per-lead
│                              direct measurements, plane analysis, aggregate
├── beats.py                  beat-onset detection + per-beat windowing
├── preprocessing.py          baseline removal, lead-by-lead Savitzky-Golay
├── plane_analysis.py         frontal/horizontal/V2 axis estimation per beat
├── plane_fitting.py          per-plane region-shape primitives
├── wfdb_io.py                load .hea/.dat WFDB recordings
├── json_writer.py            emit the flat `direct_features` and
│                              `plane_direct_features` blocks
├── features/                 LEGACY — produces "Semantic Features" block
│   ├── registry.py             (T/P/QRS/rhythm/global extractors)
│   ├── p_wave.py               kept for prototype backward compatibility
│   ├── qrs_complex.py
│   ├── rhythm.py
│   ├── t_wave.py
│   └── ...
├── gaussian_fitting.py       LEGACY — opt-in only via pruning="gaussians"
├── wave_fitters.py           LEGACY — supporting code for above
├── FIDELITY_REPORT.md        verifies qpsi_direct matches legacy qpsi numerically
├── QPSI_PROFILING_REPORT.md  per-step timing breakdown
└── QPSI_DIRECT_DESIGN.md     ← this document
```

External consumers documented in:
- `XGBOOST_SWARM.md` (internal, not in this repository) — training-side detail of the per-dx swarm
- `QPSI_SWARM.md` (internal, not in this repository) — prototype-side integration architecture
- `SWARM_INTEGRATION_JOURNEY.md` (internal, not in this repository) — postmortem of the deployment
