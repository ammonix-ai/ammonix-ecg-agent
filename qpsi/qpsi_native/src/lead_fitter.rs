//! Full lead-level wave fitting with rayon parallelism.
//!
//! Replaces the Python loop in lead_fitting.py process_leads_with_proper_timing
//! that calls fit_lead_waves_enhanced for each of 12 leads sequentially.
//! Instead, all 12 leads are fitted in parallel using rayon.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::seeded_fitter::{fit_wave_components_core_rust, GaussComponent, WaveFitResult};
use crate::qrs_fitter::{fit_qrs_unified, QrsComponent};

/// L3: AF/flutter spectral probe — Welch periodogram with Hann window.
/// Paired 1:1 from wave_fitters.py:1026-1063 (scipy.signal.welch + ratio checks).
fn check_af_flutter(y: &[f64], step_ms: f64) -> (bool, bool) {
    let n = y.len();
    if n < 8 { return (false, false); }
    let fs = 1000.0 / step_ms.max(0.1);
    let pi2 = 2.0 * std::f64::consts::PI;

    // Welch periodogram: segment length = min(256, n), overlap = nperseg/2
    let nperseg = n.min(256);
    let noverlap = nperseg / 2;
    let step = nperseg - noverlap;
    let n_fft = nperseg;

    // Hann window
    let hann: Vec<f64> = (0..nperseg).map(|i| {
        0.5 * (1.0 - (pi2 * i as f64 / (nperseg - 1).max(1) as f64).cos())
    }).collect();
    let win_ss: f64 = hann.iter().map(|w| w * w).sum();

    // Average periodograms over segments
    let mut avg_psd = vec![0.0f64; n_fft / 2 + 1];
    let mut n_segments = 0u32;

    let mut start = 0;
    while start + nperseg <= n {
        // Apply window
        let segment: Vec<f64> = (0..nperseg).map(|i| y[start + i] * hann[i]).collect();

        // DFT (real signal → n_fft/2+1 frequency bins)
        for k in 0..=n_fft / 2 {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..nperseg {
                let angle = pi2 * k as f64 * i as f64 / n_fft as f64;
                re += segment[i] * angle.cos();
                im -= segment[i] * angle.sin();
            }
            avg_psd[k] += (re * re + im * im) / (fs * win_ss);
        }
        n_segments += 1;
        start += step;
    }

    // If no complete segments, fall back to single-segment DFT
    if n_segments == 0 {
        for k in 0..=n_fft / 2 {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..n {
                let angle = pi2 * k as f64 * i as f64 / n as f64;
                re += y[i] * angle.cos();
                im -= y[i] * angle.sin();
            }
            avg_psd[k] = re * re + im * im;
        }
        n_segments = 1;
    }

    // Average
    for p in avg_psd.iter_mut() { *p /= n_segments as f64; }

    // Compute frequency-band power ratios
    let freq_step = fs / n_fft as f64;
    let mut flutter_power = 0.0;
    let mut fib_power = 0.0;
    let mut total_power = 0.0;

    for (k, &psd) in avg_psd.iter().enumerate() {
        let freq = k as f64 * freq_step;
        total_power += psd;
        if freq >= 3.5 && freq <= 8.5 { flutter_power += psd; }
        if freq >= 7.0 && freq <= 15.0 { fib_power += psd; }
    }

    let norm = total_power.max(1e-15);
    let flutter_ratio = flutter_power / norm;
    let fib_ratio = fib_power / norm;

    // Dual-condition thresholds (paired from wave_fitters.py:1055-1063)
    let flutter_like = flutter_ratio >= 0.20 && flutter_ratio > fib_ratio;
    let fib_like = fib_ratio >= 0.20 && fib_ratio >= flutter_ratio;
    (flutter_like, fib_like)
}

/// Configuration for fitting one wave window.
struct WaveConfig {
    sigma_bounds: (f64, f64),
    min_separation: f64,
    max_separation: f64,
    allow_biphasic: bool,
    allow_two: bool,
    aic_delta: f64,
    prefer_left: bool,
    min_snr_db: f64,
}

/// Fit P, T, and QRS waves for a single lead's average trace.
/// Returns (p_comps, p_info, t_comps, t_info, qrs_comps, qrs_bounds, qrs_r_peak).
fn fit_lead_avg(
    time_ms: &[f64],
    avg_trace: &[f64],
    p_bounds: (f64, f64),
    t_bounds: (f64, f64),
    qrs_bounds: (f64, f64),
    rr_mean_ms: f64,
    offset_ms: f64,
    fs: f64,
    r_peak_hint: Option<f64>,
    max_qrs_components: usize,
) -> (Vec<GaussComponent>, WaveFitResult, Vec<GaussComponent>, WaveFitResult, Vec<QrsComponent>, (f64, f64), f64, Vec<f64>, Vec<bool>) {
    let n = time_ms.len();

    // T window mask
    let t_mask: Vec<usize> = (0..n)
        .filter(|&i| time_ms[i] >= t_bounds.0 && time_ms[i] <= t_bounds.1)
        .collect();

    // P window mask
    let p_mask: Vec<usize> = (0..n)
        .filter(|&i| time_ms[i] >= p_bounds.0 && time_ms[i] <= p_bounds.1)
        .collect();

    // Fit T first
    let t_time: Vec<f64> = t_mask.iter().map(|&i| time_ms[i]).collect();
    let t_signal: Vec<f64> = t_mask.iter().map(|&i| avg_trace[i]).collect();

    let t_result = fit_wave_components_core_rust(
        &t_time, &t_signal,
        (30.0, 140.0), 40.0, 180.0,
        true, true, 4.0, true,
        None, 35.0, false, // T: prefer right bias, center_hint_sigma=35
        true, 5, -4.0, None, // T-wave SNR threshold = -4.0 (not -3.0)
        false, 20.0, // aicc_early_term disabled
    );

    // Build T-tail residual hint for P window
    let p_time: Vec<f64> = p_mask.iter().map(|&i| time_ms[i]).collect();
    let p_signal: Vec<f64> = p_mask.iter().map(|&i| avg_trace[i]).collect();

    // A1 FIX: T-tail residual with soft erf gating (matches Python)
    let residual_hint: Option<Vec<f64>> = if !t_result.components.is_empty() {
        let shift = -rr_mean_ms;
        let gate_width = 12.0; // matches Python gate_width_ms default
        let hint: Vec<f64> = p_time.iter().map(|&t| {
            let t_shifted = t - shift; // map P time back to T coordinate
            t_result.components.iter()
                .map(|c| {
                    let z = (t_shifted - c.center_ms) / (c.sigma_ms + 1e-9);
                    let gauss = c.amp_mv * (-0.5 * z * z).exp();
                    // Soft right-half erf gate (paired from wave_fitters.py:468-472)
                    let gate = 0.5 * (1.0 + libm::erf((t_shifted - c.center_ms) / gate_width));
                    gauss * gate
                })
                .sum::<f64>()
        }).collect();
        Some(hint)
    } else {
        None
    };

    // Fit P (with T-tail hint)
    let p_result = fit_wave_components_core_rust(
        &p_time, &p_signal,
        (8.0, 60.0), 18.0, 140.0,
        true, true, 4.0, true,
        None, 25.0, true, // P: prefer left bias
        true, 5, -3.0,
        residual_hint.as_deref(),
        false, 20.0, // aicc_early_term disabled
    );

    // Filter P components that are from previous T-wave
    let limit_ms = -0.5 * rr_mean_ms + offset_ms;
    let p_comps: Vec<GaussComponent> = p_result.components.iter()
        .filter(|c| c.center_ms >= limit_ms)
        .cloned()
        .collect();

    let t_comps = t_result.components.clone();

    // QRS fitting — paired from lead_fitting.py:285-334
    let qrs_fit = fit_qrs_unified(
        avg_trace, time_ms, qrs_bounds, r_peak_hint, fs,
        60.0, 250.0, 10.0, // min_qrs_ms, max_qrs_ms, padding_ms
        false, (6.0, 60.0), // force_narrow, sigma_bounds
        max_qrs_components, 0.03, 0.05, 8.0, // max_comp, q_s_thr, rprime_thr, min_sep
    );
    let qrs_comps = qrs_fit.as_ref().map(|r| r.components.clone()).unwrap_or_default();
    let qrs_bounds_out = qrs_fit.as_ref().map(|r| r.bounds).unwrap_or((0.0, 0.0));
    let qrs_r_peak = qrs_fit.as_ref().map(|r| r.r_peak_time).unwrap_or(0.0);
    let qrs_yfit = qrs_fit.as_ref().map(|r| r.yfit.clone()).unwrap_or_default();
    let qrs_mask = qrs_fit.as_ref().map(|r| r.mask.clone()).unwrap_or_default();

    (p_comps, p_result, t_comps, t_result, qrs_comps, qrs_bounds_out, qrs_r_peak, qrs_yfit, qrs_mask)
}

/// Fit P+T average traces for ALL leads in parallel.
///
/// Takes flat arrays of 12 leads' average traces and returns fitted components.
/// This replaces the sequential Python loop in process_leads_with_proper_timing.
///
/// Args:
///   avg_traces_flat: flat array of shape (n_leads * n_samples,)
///   time_ms: shared time axis (n_samples,)
///   n_leads: number of leads (typically 12)
///   n_samples: samples per lead
///   p_lo, p_hi: P-wave window bounds
///   t_lo, t_hi: T-wave window bounds
///   rr_mean_ms: mean RR interval in ms
///   offset_ms: timing offset
///
/// Returns: list of dicts, one per lead, each with P and T components.
#[pyfunction]
#[pyo3(signature = (avg_traces_flat, time_ms, n_leads, n_samples, p_lo, p_hi, t_lo, t_hi, rr_mean_ms, offset_ms, qrs_lo=-40.0, qrs_hi=80.0, fs=500.0))]
pub fn fit_avg_traces_all_leads<'py>(
    py: Python<'py>,
    avg_traces_flat: PyReadonlyArray1<'py, f64>,
    time_ms: PyReadonlyArray1<'py, f64>,
    n_leads: usize,
    n_samples: usize,
    p_lo: f64,
    p_hi: f64,
    t_lo: f64,
    t_hi: f64,
    rr_mean_ms: f64,
    offset_ms: f64,
    qrs_lo: f64,
    qrs_hi: f64,
    fs: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let traces = avg_traces_flat.as_slice()?.to_vec();
    let time = time_ms.as_slice()?.to_vec();

    // Build per-lead slices
    let lead_traces: Vec<Vec<f64>> = (0..n_leads)
        .map(|i| traces[i * n_samples..(i + 1) * n_samples].to_vec())
        .collect();

    // First pass: parallel fitting across all leads (P + T + QRS)
    // Compute R-peak hint per lead from argmax(|avg_trace|) in QRS window (matches Python line 278)
    let r_peak_hints: Vec<Option<f64>> = lead_traces.iter().map(|trace| {
        let qrs_mask: Vec<usize> = (0..n_samples)
            .filter(|&i| time[i] >= qrs_lo && time[i] <= qrs_hi)
            .collect();
        if qrs_mask.is_empty() { return None; }
        let i_max = qrs_mask.iter()
            .max_by(|&&a, &&b| trace[a].abs().partial_cmp(&trace[b].abs()).unwrap_or(std::cmp::Ordering::Equal))
            .copied().unwrap();
        Some(time[i_max])
    }).collect();

    let mut results: Vec<(Vec<GaussComponent>, WaveFitResult, Vec<GaussComponent>, WaveFitResult, Vec<QrsComponent>, (f64, f64), f64, Vec<f64>, Vec<bool>)> =
        lead_traces.par_iter().zip(r_peak_hints.par_iter()).map(|(trace, hint)| {
            fit_lead_avg(&time, trace, (p_lo, p_hi), (t_lo, t_hi), (qrs_lo, qrs_hi),
                         rr_mean_ms, offset_ms, fs, *hint, 4)
        }).collect();

    // L1: Weighted-median P-center consensus — 2nd pass for weak/absent P-waves
    // Paired 1:1 from wave_fitters.py:268-398 (p_center_consensus)
    {
        let min_snr_db: f64 = -3.0;
        let iqr_k: f64 = 1.5;
        let spread_guard_ms: f64 = 80.0;
        let default_sigma_ms: f64 = 25.0;
        let max_weight_ratio: f64 = 8.0;

        // Collect (center, weight) from leads where P was found with sufficient SNR
        let mut mus: Vec<f64> = Vec::new();
        let mut wts: Vec<f64> = Vec::new();
        for (p_comps, p_info, _, _, _, _, _, _, _) in results.iter() {
            if !p_info.present { continue; }
            if !p_info.snr_db.is_finite() || p_info.snr_db < min_snr_db { continue; }
            // _choose_p_center_from_comps: prefer earliest component
            if let Some(c) = p_comps.iter().min_by(|a, b| a.center_ms.partial_cmp(&b.center_ms).unwrap_or(std::cmp::Ordering::Equal)) {
                if c.center_ms.is_finite() {
                    let w = (10.0f64).powf(0.05 * p_info.snr_db).max(1e-3);
                    mus.push(c.center_ms);
                    wts.push(w);
                }
            }
        }

        // Weighted quantile helper (linear interp on weighted CDF)
        let weighted_quantile = |vals: &[f64], ws: &[f64], q: f64| -> f64 {
            if vals.is_empty() { return 0.0; }
            let mut pairs: Vec<(f64, f64)> = vals.iter().copied().zip(ws.iter().copied()).collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let cumw: Vec<f64> = pairs.iter().scan(0.0, |acc, (_, w)| { *acc += w; Some(*acc) }).collect();
            let total = *cumw.last().unwrap_or(&1.0);
            if total <= 0.0 { return pairs[pairs.len() / 2].0; }
            let cdf: Vec<f64> = cumw.iter().map(|c| c / total).collect();
            // np.interp equivalent
            if q <= cdf[0] { return pairs[0].0; }
            if q >= *cdf.last().unwrap() { return pairs.last().unwrap().0; }
            for i in 1..cdf.len() {
                if cdf[i] >= q {
                    let t = (q - cdf[i - 1]) / (cdf[i] - cdf[i - 1]).max(1e-15);
                    return pairs[i - 1].0 + t * (pairs[i].0 - pairs[i - 1].0);
                }
            }
            pairs.last().unwrap().0
        };

        let consensus: Option<(f64, f64)> = if mus.len() >= 2 {
            // Cap weight ratio
            let wmin = wts.iter().filter(|&&w| w > 0.0).copied().fold(f64::INFINITY, f64::min);
            let wmax = wts.iter().copied().fold(0.0f64, f64::max);
            if wmax / wmin.max(1e-12) > max_weight_ratio {
                let scale = (max_weight_ratio * wmin) / wmax;
                for w in wts.iter_mut() { if (*w - wmax).abs() < 1e-15 { *w *= scale; } }
            }

            let q1 = weighted_quantile(&mus, &wts, 0.25);
            let q3 = weighted_quantile(&mus, &wts, 0.75);
            let iqr = q3 - q1;
            let lo = q1 - iqr_k * iqr;
            let hi = q3 + iqr_k * iqr;

            // IQR outlier rejection
            let mut mus_k = Vec::new();
            let mut wts_k = Vec::new();
            for (m, w) in mus.iter().zip(wts.iter()) {
                if *m >= lo && *m <= hi { mus_k.push(*m); wts_k.push(*w); }
            }

            if mus_k.is_empty() || (mus_k.len() == 1 && mus.len() > 2) {
                None
            } else {
                let center = weighted_quantile(&mus_k, &wts_k, 0.5);
                // Spread guard
                let spread = mus_k.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                            - mus_k.iter().copied().fold(f64::INFINITY, f64::min);
                if spread > spread_guard_ms {
                    None
                } else {
                    // MAD-based sigma
                    let abs_devs: Vec<f64> = mus_k.iter().map(|m| (m - center).abs()).collect();
                    let mad = weighted_quantile(&abs_devs, &wts_k, 0.5);
                    let sigma = (1.4826 * mad).max(12.0).min(default_sigma_ms.max(50.0));
                    Some((center, sigma))
                }
            }
        } else {
            None
        };

        // 2nd pass: re-fit leads where P is absent, weak, or edge-clipped
        if let Some((consensus_ms, consensus_sigma)) = consensus {
            let needs_refit: Vec<usize> = results.iter().enumerate()
                .filter(|(_, (_, p_info, _, _, _, _, _, _, _))| {
                    !p_info.present || p_info.edge_left || p_info.edge_right || p_info.snr_db < -2.0
                })
                .map(|(i, _)| i)
                .collect();

            let p_mask: Vec<usize> = (0..time.len())
                .filter(|&i| time[i] >= p_lo && time[i] <= p_hi)
                .collect();
            let p_time_v: Vec<f64> = p_mask.iter().map(|&i| time[i]).collect();

            for &li in &needs_refit {
                let p_signal: Vec<f64> = p_mask.iter().map(|&i| lead_traces[li][i]).collect();
                let t_comps = &results[li].3.components;
                let hint: Option<Vec<f64>> = if !t_comps.is_empty() {
                    let shift = -rr_mean_ms;
                    let gate_width = 12.0;
                    Some(p_time_v.iter().map(|&t| {
                        let t_shifted = t - shift;
                        t_comps.iter().map(|c| {
                            let z = (t_shifted - c.center_ms) / (c.sigma_ms + 1e-9);
                            let gauss = c.amp_mv * (-0.5 * z * z).exp();
                            let gate = 0.5 * (1.0 + libm::erf((t_shifted - c.center_ms) / gate_width));
                            gauss * gate
                        }).sum::<f64>()
                    }).collect())
                } else { None };

                let p_refit = fit_wave_components_core_rust(
                    &p_time_v, &p_signal,
                    (8.0, 60.0), 18.0, 140.0,
                    true, true, 4.0, true,
                    Some(consensus_ms), consensus_sigma, true,
                    true, 5, -3.0,
                    hint.as_deref(),
                    false, 20.0, // aicc_early_term disabled
                );

                if p_refit.present {
                    let limit_ms = -0.5 * rr_mean_ms + offset_ms;
                    let new_p_comps: Vec<GaussComponent> = p_refit.components.iter()
                        .filter(|c| c.center_ms >= limit_ms)
                        .cloned()
                        .collect();
                    if !new_p_comps.is_empty() {
                        results[li].0 = new_p_comps;
                        results[li].1 = p_refit;
                    }
                }
            }
        }
    }

    // Convert to Python dicts
    let mut py_results = Vec::with_capacity(n_leads);
    for (p_comps, p_info, t_comps, t_info, qrs_comps, qrs_bnd, qrs_rpk, qrs_yfit, qrs_mask) in &results {
        let d = PyDict::new(py);

        // P components — number them P1, P2 etc. to match GT format
        let p_list: Vec<Bound<'py, PyDict>> = p_comps.iter().enumerate().map(|(idx, c)| {
            let cd = PyDict::new(py);
            cd.set_item("wave_type", "P").unwrap();
            cd.set_item("amp_mv", c.amp_mv).unwrap();
            cd.set_item("center_ms", c.center_ms).unwrap();
            cd.set_item("sigma_ms", c.sigma_ms).unwrap();
            let label = if p_comps.len() == 1 { "P".to_string() } else { format!("P{}", idx + 1) };
            cd.set_item("component", label).unwrap();
            cd
        }).collect();

        // T components — number them T1, T2 etc. to match GT format
        let t_list: Vec<Bound<'py, PyDict>> = t_comps.iter().enumerate().map(|(idx, c)| {
            let cd = PyDict::new(py);
            cd.set_item("wave_type", "T").unwrap();
            cd.set_item("amp_mv", c.amp_mv).unwrap();
            cd.set_item("center_ms", c.center_ms).unwrap();
            cd.set_item("sigma_ms", c.sigma_ms).unwrap();
            let label = if t_comps.len() == 1 { "T".to_string() } else { format!("T{}", idx + 1) };
            cd.set_item("component", label).unwrap();
            cd
        }).collect();

        // QRS components
        let qrs_list: Vec<Bound<'py, PyDict>> = qrs_comps.iter().map(|c| {
            let cd = PyDict::new(py);
            cd.set_item("wave_type", "QRS").unwrap();
            cd.set_item("amp_mv", c.amp_mv).unwrap();
            cd.set_item("center_ms", c.center_ms).unwrap();
            cd.set_item("sigma_ms", c.sigma_ms).unwrap();
            cd.set_item("component", c.label.as_str()).unwrap();
            cd
        }).collect();

        d.set_item("p_components", p_list).unwrap();
        d.set_item("t_components", t_list).unwrap();
        d.set_item("qrs_components", qrs_list).unwrap();
        d.set_item("qrs_bounds", (qrs_bnd.0, qrs_bnd.1)).unwrap();
        d.set_item("qrs_r_peak_time", *qrs_rpk).unwrap();
        // Return yfit and mask arrays for per-beat clean detection
        d.set_item("qrs_yfit", numpy::PyArray1::from_vec(py, qrs_yfit.clone())).unwrap();
        d.set_item("qrs_mask", qrs_mask.clone()).unwrap();
        d.set_item("p_present", p_info.present).unwrap();
        d.set_item("p_snr_db", p_info.snr_db).unwrap();
        d.set_item("p_notched", p_info.notched).unwrap();
        d.set_item("p_edge_left", p_info.edge_left).unwrap();
        d.set_item("p_edge_right", p_info.edge_right).unwrap();
        d.set_item("t_present", t_info.present).unwrap();
        d.set_item("t_snr_db", t_info.snr_db).unwrap();

        // A3 FIX: AF/flutter spectral probe when P absent
        // Paired from wave_fitters.py:1026-1063
        if !p_info.present {
            let p_mask_l: Vec<usize> = (0..time.len())
                .filter(|&i| time[i] >= p_lo && time[i] <= p_hi)
                .collect();
            let li_idx = py_results.len(); // current lead index
            let p_sig: Vec<f64> = p_mask_l.iter().map(|&i| {
                if li_idx < lead_traces.len() { lead_traces[li_idx][i] } else { 0.0 }
            }).collect();
            let step_ms = if p_sig.len() > 1 {
                let p_t: Vec<f64> = p_mask_l.iter().map(|&i| time[i]).collect();
                (p_t.last().unwrap_or(&1.0) - p_t.first().unwrap_or(&0.0)) / (p_sig.len() - 1).max(1) as f64
            } else { 2.0 };
            let (flutter, fib) = check_af_flutter(&p_sig, step_ms);
            d.set_item("p_flutter_like", flutter).unwrap();
            d.set_item("p_fibrillatory_like", fib).unwrap();
        } else {
            d.set_item("p_flutter_like", false).unwrap();
            d.set_item("p_fibrillatory_like", false).unwrap();
        }

        py_results.push(d);
    }

    Ok(py_results)
}
