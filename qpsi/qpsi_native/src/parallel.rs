//! Parallel lead and beat fitting using rayon.
//!
//! Replaces the sequential Python loops in fit_lead_waves_enhanced and
//! process_leads_with_proper_timing with a single Rust call that
//! processes all (lead, beat) pairs in parallel.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::gauss::nearest_idx;
use crate::residuals::{gauss_k1_residual_kernel, gauss_k2_residual_kernel};
use crate::solver::{solve_lm, analytic_jacobian_k1, analytic_jacobian_k2};

/// Fit K=1 Gaussian on a signal segment. Returns [A, mu, s, b0, cost].
fn fit_k1(
    x: &[f64], y: &[f64],
    a_max: f64, x_lo: f64, x_hi: f64,
    lo_sig: f64, hi_sig: f64, b_max: f64,
    mu_seed: f64, has_hint: bool, hint_ms: f64, hint_sigma: f64,
) -> Vec<f64> {
    if x.len() < 8 { return vec![]; }

    let i_near = nearest_idx(x, mu_seed);

    let a1_0 = y[i_near].max(-a_max).min(a_max);
    let s1_0 = (0.6 * (lo_sig + hi_sig)).max(lo_sig).min(hi_sig);
    let mu1_0 = mu_seed.max(x_lo).min(x_hi);

    let x0 = vec![a1_0, mu1_0, s1_0, 0.0];
    let lb = vec![-a_max, x_lo, lo_sig, -b_max];
    let ub = vec![a_max, x_hi, hi_sig, b_max];

    let xv = x.to_vec();
    let yv = y.to_vec();
    let xr = xv.clone();

    let res_fn = move |p: &[f64]| -> Vec<f64> {
        gauss_k1_residual_kernel(&xv, &yv, p, hint_ms, hint_sigma, has_hint)
    };
    let jac_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_k1(&xr, p, has_hint, hint_sigma)
    };

    let result = solve_lm(&res_fn, Some(&jac_fn), &x0, &lb, &ub, 1000, true, 1.5, 1e-8, 1e-8);
    let mut out = result.params;
    out.push(result.cost);
    out
}

/// Fit K=2 Gaussian on a signal segment. Returns [A1, mu1, s1, A2, sep, s2, b0, cost].
fn fit_k2(
    x: &[f64], y: &[f64],
    a_max: f64, x_lo: f64, x_hi: f64,
    lo_sig: f64, hi_sig: f64, b_max: f64,
    min_sep: f64, max_sep: f64,
    mu_seed: f64, has_hint: bool, hint_ms: f64, hint_sigma: f64,
) -> Vec<f64> {
    if x.len() < 8 { return vec![]; }

    let i_near = nearest_idx(x, mu_seed);

    let a1_0 = y[i_near].max(-a_max).min(a_max);
    let s1_0 = (0.6 * (lo_sig + hi_sig)).max(lo_sig).min(hi_sig);
    let mu1_0 = mu_seed.max(x_lo).min(x_hi);
    let sep0 = ((min_sep + max_sep) * 0.5).max(min_sep).min(max_sep);

    let x0 = vec![a1_0, mu1_0, s1_0, a1_0 * 0.5, sep0, s1_0, 0.0];
    let lb = vec![-a_max, x_lo, lo_sig, -a_max, min_sep, lo_sig, -b_max];
    let ub = vec![a_max, x_hi, hi_sig, a_max, max_sep, hi_sig, b_max];

    let xv = x.to_vec();
    let yv = y.to_vec();
    let xr = xv.clone();

    let res_fn = move |p: &[f64]| -> Vec<f64> {
        gauss_k2_residual_kernel(&xv, &yv, p, hint_ms, hint_sigma, has_hint)
    };
    let jac_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_k2(&xr, p, has_hint, hint_sigma)
    };

    let result = solve_lm(&res_fn, Some(&jac_fn), &x0, &lb, &ub, 1000, true, 1.5, 1e-8, 1e-8);
    let mut out = result.params;
    out.push(result.cost);
    out
}

/// AICc computation (paired from wave_fitters.py _aicc)
fn aicc(rss: f64, k: usize, n: usize) -> f64 {
    let nf = n.max(1) as f64;
    let kf = k as f64;
    let denom = (nf - kf - 1.0).max(1.0);
    nf * (rss / nf + 1e-12).ln() + 2.0 * kf + (2.0 * kf * (kf + 1.0)) / denom
}

/// Single Gaussian evaluation
fn gauss_eval(x: &[f64], a: f64, mu: f64, s: f64) -> Vec<f64> {
    let inv_s = 1.0 / (s + 1e-9);
    x.iter().map(|&xi| {
        let z = (xi - mu) * inv_s;
        a * (-0.5 * z * z).exp()
    }).collect()
}

/// R1: MAD-based robust scale — paired 1:1 from wave_fitters.py:194-206.
fn robust_scale(y: &[f64]) -> f64 {
    if y.is_empty() { return 1e-6; }
    let n = y.len();
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
    let mut abs_dev: Vec<f64> = y.iter().map(|v| (v - med).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n % 2 == 1 { abs_dev[n / 2] } else { (abs_dev[n / 2 - 1] + abs_dev[n / 2]) / 2.0 };
    (1.4826 * (mad + 1e-12)).max(1e-6)
}

/// Fit one beat's T and P waves. Returns list of (wave_type, amp, center, sigma, component).
fn fit_one_beat_tp(
    beat_trace: &[f64],
    time_ms: &[f64],
    t_mask: &[usize],
    p_mask: &[usize],
    t_time: &[f64],
    p_time: &[f64],
    do_p: bool,
    p_limit_ms: f64,
) -> Vec<(String, f64, f64, f64, String)> {
    let mut comps = Vec::new();

    // P2+P3: T-wave fitting via full seeded_fitter (preprocessing + multi-seed + K=0/K=1/K=2)
    if t_time.len() >= 8 {
        let t_sig: Vec<f64> = t_mask.iter().map(|&i| beat_trace[i]).collect();
        let t_result = crate::seeded_fitter::fit_wave_components_core_rust(
            &t_time, &t_sig,
            (30.0, 140.0), 40.0, 180.0,  // sigma_bounds, min_sep, max_sep
            true, true, 4.0, true,         // biphasic, two, aic_delta, k0 baseline
            None, 35.0, false,             // no center hint, sigma=35, prefer_right for T
            true, 5, -4.0, None,           // multi_seed, templates, snr_thresh, no residual hint
            false, 20.0,                   // aicc_early_term disabled
        );
        let nc = t_result.components.len();
        for (i, c) in t_result.components.iter().enumerate() {
            let label = if nc == 1 { "T".to_string() }
                       else { format!("T{}", i + 1) };
            comps.push(("T".into(), c.amp_mv, c.center_ms, c.sigma_ms, label));
        }
    }

    // P2+P3: P-wave fitting via full seeded_fitter (preprocessing + multi-seed + K=0/K=1/K=2)
    if do_p && p_time.len() >= 8 {
        let p_sig_raw: Vec<f64> = p_mask.iter().map(|&i| beat_trace[i]).collect();

        // B2: Build T-tail residual hint from per-beat T components
        let t_comps_for_hint: Vec<(f64, f64, f64)> = comps.iter()
            .filter(|(wt, _, _, _, _)| wt == "T")
            .map(|(_, a, mu, s, _)| (*a, *mu, *s))
            .collect();

        let residual_hint: Option<Vec<f64>> = if !t_comps_for_hint.is_empty() {
            let gate_width = 12.0;
            Some(p_time.iter().map(|&t| {
                t_comps_for_hint.iter().map(|(a, mu, s)| {
                    let z = (t - mu) / (s + 1e-9);
                    let gauss = a * (-0.5 * z * z).exp();
                    let gate = 0.5 * (1.0 + libm::erf((t - mu) / gate_width));
                    gauss * gate
                }).sum::<f64>()
            }).collect())
        } else {
            None
        };

        let p_result = crate::seeded_fitter::fit_wave_components_core_rust(
            &p_time, &p_sig_raw,
            (8.0, 60.0), 18.0, 140.0,     // sigma_bounds, min_sep, max_sep
            true, true, 4.0, true,          // biphasic, two, aic_delta, k0 baseline
            None, 25.0, true,               // no center hint, sigma=25, prefer_left for P
            true, 5, -3.0,                  // multi_seed, templates, snr_thresh
            residual_hint.as_deref(),        // T-tail hint
            false, 20.0,                    // aicc_early_term disabled
        );
        let npc = p_result.components.len();
        for (i, c) in p_result.components.iter().enumerate() {
            let label = if npc == 1 { "P".to_string() }
                       else { format!("P{}", i + 1) };
            comps.push(("P".into(), c.amp_mv, c.center_ms, c.sigma_ms, label));
        }
    }

    // P6: Post-fit P-limit filter (paired from lead_fitting.py:505-506)
    comps.retain(|(wt, _, mu, _, _)| {
        if wt == "P" { *mu >= p_limit_ms } else { true }
    });

    comps
}

/// Process ALL beats across ALL leads in parallel using rayon.
///
/// This replaces the sequential per-beat loop in fit_lead_waves_enhanced,
/// called once for each of the 12 leads. Instead, we process all
/// (lead, beat) pairs in a single parallel batch.
///
/// Input:
///   stack_flat: flattened (n_beats * n_leads * n_samples) array
///   time_ms: shared time axis
///   n_beats, n_leads, n_samples: dimensions
///   wave bounds: T and P windows
///   avg_p_flags: per-lead boolean (1.0/0.0)
///
/// Returns list of dicts with lead_idx, beat_idx, wave_type, amp, center, sigma, component
#[pyfunction]
pub fn fit_all_leads_beats_parallel<'py>(
    py: Python<'py>,
    stack_flat: PyReadonlyArray1<'py, f64>,
    time_ms: PyReadonlyArray1<'py, f64>,
    n_beats: usize,
    n_leads: usize,
    t_lo: f64, t_hi: f64,
    p_lo: f64, p_hi: f64,
    avg_p_flags: PyReadonlyArray1<'py, f64>,
    rr_mean_ms: f64,
    offset_ms: f64,
) -> PyResult<Bound<'py, PyList>> {
    let time = time_ms.as_slice()?.to_vec();
    let n_samples = time.len();
    let stack = stack_flat.as_slice()?.to_vec();
    let p_flags = avg_p_flags.as_slice()?.to_vec();

    // Pre-compute masks
    let t_mask: Vec<usize> = (0..n_samples).filter(|&i| time[i] >= t_lo && time[i] <= t_hi).collect();
    let p_mask: Vec<usize> = (0..n_samples).filter(|&i| time[i] >= p_lo && time[i] <= p_hi).collect();
    let t_time: Vec<f64> = t_mask.iter().map(|&i| time[i]).collect();
    let p_time: Vec<f64> = p_mask.iter().map(|&i| time[i]).collect();
    let p_limit_ms = -0.5 * rr_mean_ms + offset_ms;

    // Total tasks = n_leads * n_beats
    let total = n_leads * n_beats;

    let results: Vec<(usize, usize, String, f64, f64, f64, String)> = (0..total)
        .into_par_iter()
        .flat_map(|task_idx| {
            let lead_idx = task_idx / n_beats;
            let beat_idx = task_idx % n_beats;
            let do_p = p_flags[lead_idx] > 0.5;

            let offset = (beat_idx * n_leads + lead_idx) * n_samples;
            let beat_trace = &stack[offset..offset + n_samples];

            fit_one_beat_tp(
                beat_trace, &time, &t_mask, &p_mask, &t_time, &p_time,
                do_p, p_limit_ms,
            ).into_iter()
                .map(move |(wt, amp, center, sigma, comp)| {
                    (lead_idx, beat_idx, wt, amp, center, sigma, comp)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Convert to Python list of dicts
    let py_list = PyList::empty(py);
    for (lead_idx, beat_idx, ref wave_type, amp, center, sigma, ref component) in &results {
        let dict = PyDict::new(py);
        dict.set_item("lead_idx", *lead_idx)?;
        dict.set_item("beat_idx", *beat_idx)?;
        dict.set_item("wave_type", wave_type.as_str())?;
        dict.set_item("amp_mv", *amp)?;
        dict.set_item("center_ms", *center)?;
        dict.set_item("sigma_ms", *sigma)?;
        dict.set_item("component", component.as_str())?;
        py_list.append(dict)?;
    }

    Ok(py_list.into())
}
