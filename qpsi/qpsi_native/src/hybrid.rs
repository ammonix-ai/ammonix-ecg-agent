//! Full hybrid Gaussian fitting pipeline in pure Rust.
//!
//! Paired from: gaussian_fitting.py
//!   fit_gaussians_1d           (lines 105-199)
//!   _initial_guess_2d          (lines 205-237)
//!   fit_gaussians_2d           (lines 243-345)
//!   fit_gaussians_hybrid       (lines 351-563)
//!   fit_gaussians_hybrid_subrange (lines 569-651)
//!   subtract_fitted_2d         (lines 733-747)
//!
//! This module contains NO PyO3 — pure Rust only. Called from plane.rs.

use crate::gauss::{sum_gaussians_2d_kernel, sum_gaussians_1d_kernel, fit_2d_residual_kernel, analytic_jacobian_2d, nearest_idx};
use crate::solver::solve_lm;

/// Find peaks in a 1D signal above a height threshold.
/// Returns indices of peaks (simple local maxima detection).
fn find_peaks_above(signal: &[f64], height: f64) -> Vec<usize> {
    let n = signal.len();
    if n < 3 { return vec![]; }
    let mut peaks = Vec::new();
    for i in 1..n - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] >= height {
            peaks.push(i);
        }
    }
    peaks
}

/// fit_gaussians_1d: 1D magnitude Gaussian fit.
/// Paired from gaussian_fitting.py:105-199
/// Returns flat [t0, sigma, A] * n_waves
pub fn fit_gaussians_1d(
    t: &[f64], mag: &[f64], n_waves: usize, sigma_max: f64,
) -> Vec<f64> {
    let n = t.len();
    if n < 3 || t[0] == t[n - 1] { return vec![0.0; 3 * n_waves]; }
    if mag.iter().all(|&v| v == 0.0) { return vec![0.0; 3 * n_waves]; }

    let t_min = t[0];
    let t_max = t[n - 1];
    let amp_max = mag.iter().cloned().fold(0.0f64, f64::max) * 5.0;
    let min_dist = (n / (2 * n_waves)).max(3);

    // Peak detection for initial guess
    let peaks = find_peaks_above(mag, 0.0);
    let mut sorted_peaks: Vec<(usize, f64)> = peaks.iter().map(|&i| (i, mag[i])).collect();
    sorted_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Select n_waves peaks with minimum distance
    let mut selected: Vec<usize> = Vec::new();
    for (idx, _) in &sorted_peaks {
        let too_close = selected.iter().any(|&s| (*idx as isize - s as isize).unsigned_abs() < min_dist);
        if !too_close {
            selected.push(*idx);
            if selected.len() >= n_waves { break; }
        }
    }

    // Fill remaining with evenly spaced
    while selected.len() < n_waves {
        let gap = n / (selected.len() + 2);
        selected.push(gap * (selected.len() + 1));
    }
    selected.sort();

    // Build initial guess + bounds
    let sigma_guess = (0.45 * (n as f64 / (2.0 * n_waves as f64)) * ((t_max - t_min) / n as f64)).min(0.9 * sigma_max);
    let mut init = Vec::with_capacity(3 * n_waves);
    let mut lb = Vec::with_capacity(3 * n_waves);
    let mut ub = Vec::with_capacity(3 * n_waves);
    for &idx in &selected {
        let idx = idx.min(n - 1);
        init.push(t[idx]);
        init.push(sigma_guess.max(1.0));
        init.push(mag[idx].max(0.001));
        lb.push(t_min); lb.push(1.0); lb.push(0.0);
        ub.push(t_max); ub.push(sigma_max); ub.push(amp_max.max(0.001));
    }

    // Clamp init inside bounds
    for i in 0..init.len() {
        init[i] = init[i].max(lb[i] + 1e-6).min(ub[i] - 1e-6);
    }

    // Residual: sum_1d(t, p, n_waves) - mag
    let t_v = t.to_vec();
    let mag_v = mag.to_vec();
    let nw = n_waves;
    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        let synth = sum_gaussians_1d_kernel(&t_v, p, nw);
        synth.iter().zip(mag_v.iter()).map(|(s, m)| s - m).collect()
    };

    let result = solve_lm(&residual_fn, None, &init, &lb, &ub, 3000, false, 1.0, 1e-8, 1e-8);
    result.params
}

/// _initial_guess_2d: deterministic 2D init from 1D magnitude fit.
/// Paired from gaussian_fitting.py:205-237
pub fn initial_guess_2d(
    t: &[f64], vx: &[f64], vy: &[f64], n_waves: usize, sigma_max: f64,
) -> Vec<f64> {
    let n = t.len();
    let mag: Vec<f64> = (0..n).map(|i| (vx[i]*vx[i] + vy[i]*vy[i]).sqrt()).collect();
    let g1d = fit_gaussians_1d(t, &mag, n_waves, sigma_max);

    let mut out = Vec::with_capacity(4 * n_waves);
    for w in 0..n_waves {
        let t0 = g1d[3 * w];
        let sigma = g1d[3 * w + 1];
        let amp = g1d[3 * w + 2];
        let idx = nearest_idx(t, t0);
        let alpha = vy[idx].atan2(vx[idx]);
        out.push(t0);
        out.push(sigma);
        out.push(amp);
        out.push(alpha);
    }
    out
}

/// fit_gaussians_2d: full 2D Gaussian fit with penalties.
/// Paired from gaussian_fitting.py:243-345
pub fn fit_gaussians_2d(
    t: &[f64], vx: &[f64], vy: &[f64], n_waves: usize,
    angle_lambda: f64, amplitude_lambda: f64, sigma_max: f64,
) -> Vec<f64> {
    let n = t.len();
    if n < 3 { return vec![]; }

    let init = initial_guess_2d(t, vx, vy, n_waves, sigma_max);
    let mag: Vec<f64> = (0..n).map(|i| (vx[i]*vx[i] + vy[i]*vy[i]).sqrt()).collect();
    let amp_max = (5.0 * mag.iter().cloned().fold(0.0f64, f64::max)).max(1e-3);
    let t_min = t[0];
    let t_max = t[n - 1];

    let mut lb = Vec::with_capacity(4 * n_waves);
    let mut ub = Vec::with_capacity(4 * n_waves);
    for _ in 0..n_waves {
        lb.extend_from_slice(&[t_min, 1.0, 0.0, -std::f64::consts::PI]);
        ub.extend_from_slice(&[t_max, sigma_max, amp_max, std::f64::consts::PI]);
    }

    let mut init_c = init.clone();
    for i in 0..init_c.len() {
        init_c[i] = init_c[i].max(lb[i] + 1e-6).min(ub[i] - 1e-6);
    }

    // Residual + Jacobian using the existing 2D kernels
    let t_v = t.to_vec();
    let vx_v = vx.to_vec();
    let vy_v = vy.to_vec();
    let mag_v = mag.clone();
    let nw = n_waves;
    let al = angle_lambda;
    let aml = amplitude_lambda;

    let t_r = t_v.clone();
    let vx_r = vx_v.clone();
    let vy_r = vy_v.clone();
    let mag_r = mag_v.clone();

    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        crate::gauss::fit_2d_residual_kernel(&t_v, &vx_v, &vy_v, &mag_v, p, nw, al, aml)
    };
    let jacobian_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_2d(&t_r, &vx_r, &vy_r, &mag_r, p, nw, al, aml)
    };

    let result = solve_lm(
        &residual_fn, Some(&jacobian_fn), &init_c, &lb, &ub,
        4000, false, 1.0, 1e-8, 1e-8,
    );
    result.params
}

/// subtract_fitted_2d: v2 - sum_gaussians_2d(t, params)
/// Paired from gaussian_fitting.py:733-747
pub fn subtract_fitted_2d(
    t: &[f64], vx: &[f64], vy: &[f64], params: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    if params.is_empty() { return (vx.to_vec(), vy.to_vec()); }
    let n = t.len();
    let synth = sum_gaussians_2d_kernel(t, params);
    let rx: Vec<f64> = (0..n).map(|i| vx[i] - synth[i]).collect();
    let ry: Vec<f64> = (0..n).map(|i| vy[i] - synth[n + i]).collect();
    (rx, ry)
}

/// fit_gaussians_hybrid: two-stage hybrid fitter.
/// Paired from gaussian_fitting.py:351-563
pub fn fit_gaussians_hybrid(
    t: &[f64], vx: &[f64], vy: &[f64],
    n_waves_2d: usize, n_extra: usize, amp_thr: f64,
    t_low: f64, t_high: f64,
    sigma_max: f64, sigma_min: f64,
    angle_lambda: f64, amplitude_lambda: f64,
    refine_sigmas: bool,       // H3: when true, include sigma in 1D stage (default false)
    n_restarts: usize,         // H4: multi-start count (default 0 = single run)
    _jitter_scale: f64,        // H4: jitter magnitude for restarts (unused when n_restarts=0)
) -> Vec<f64> {
    let n = t.len();
    if n < 3 || t_low >= t_high { return vec![]; }

    // Stage 1: 2D fit
    let p2d = fit_gaussians_2d(t, vx, vy, n_waves_2d, angle_lambda, amplitude_lambda, sigma_max);
    if p2d.is_empty() { return p2d; }

    let t0_seed: Vec<f64> = (0..n_waves_2d).map(|i| p2d[4 * i]).collect();
    let s_seed: Vec<f64> = (0..n_waves_2d).map(|i| p2d[4 * i + 1]).collect();

    // Stage 2: residual peak detection
    let mag: Vec<f64> = (0..n).map(|i| (vx[i]*vx[i] + vy[i]*vy[i]).sqrt()).collect();
    let synth = sum_gaussians_2d_kernel(t, &p2d);
    let resid: Vec<f64> = (0..n).map(|i| {
        let dx = vx[i] - synth[i];
        let dy = vy[i] - synth[n + i];
        (dx*dx + dy*dy).sqrt()
    }).collect();

    let peaks = find_peaks_above(&resid, amp_thr);
    let mut sorted_peaks: Vec<(usize, f64)> = peaks.iter().map(|&i| (i, resid[i])).collect();
    sorted_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut sel_peaks: Vec<usize> = sorted_peaks.iter().take(n_extra).map(|p| p.0).collect();

    // H1: Q-wave safety net — force Q-candidate if strong angle swing detected
    // Paired 1:1 from gaussian_fitting.py:441-459
    {
        let q_lo = -60.0f64; // q_zone default
        let q_hi = 0.0f64;
        let q_thresh_rad = 40.0f64.to_radians(); // q_angle_thresh_deg default

        let q_mask: Vec<usize> = (0..n).filter(|&i| t[i] >= q_lo && t[i] <= q_hi).collect();
        if !q_mask.is_empty() {
            // Compute unwrapped angle and check swing
            let angles: Vec<f64> = q_mask.iter().map(|&i| vy[i].atan2(vx[i])).collect();
            // Simple unwrap: adjust jumps > PI
            let mut unwrapped = angles.clone();
            for i in 1..unwrapped.len() {
                let mut d = unwrapped[i] - unwrapped[i - 1];
                if d > std::f64::consts::PI { d -= 2.0 * std::f64::consts::PI; }
                if d < -std::f64::consts::PI { d += 2.0 * std::f64::consts::PI; }
                unwrapped[i] = unwrapped[i - 1] + d;
            }
            let swing = unwrapped.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                      - unwrapped.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            if swing > q_thresh_rad {
                let has_q_peak = sel_peaks.iter().any(|&idx| t[idx] >= q_lo && t[idx] <= q_hi);
                if !has_q_peak {
                    // Find max residual in q_zone and inject
                    if let Some(&cand) = q_mask.iter()
                        .max_by(|&&a, &&b| resid[a].partial_cmp(&resid[b]).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        sel_peaks.push(cand);
                        // Re-limit to n_extra if needed
                        if sel_peaks.len() > n_extra {
                            sel_peaks.sort_by(|&a, &b| resid[b].partial_cmp(&resid[a]).unwrap_or(std::cmp::Ordering::Equal));
                            sel_peaks.truncate(n_extra);
                        }
                    }
                }
            }
        }
    }

    let n_sel = sel_peaks.len();

    // Build 1D optimization (amplitude refinement)
    let amp_max_factor = 5.0;
    let max_amp = amp_max_factor * mag.iter().cloned().fold(1e-6f64, f64::max);
    let dt = if n > 1 { (t[1] - t[0]).abs() } else { 1.0 };

    let mut x0 = Vec::new();
    let mut lb = Vec::new();
    let mut ub = Vec::new();

    // Seeded waves: amplitude only (or amplitude + sigma when refine_sigmas=true)
    for i in 0..n_waves_2d {
        x0.push(p2d[4 * i + 2]); // A
        lb.push(0.0);
        ub.push(max_amp);
        if refine_sigmas {
            x0.push(s_seed[i]);
            lb.push(sigma_min);
            ub.push(sigma_max);
        }
    }

    // Extra peaks: t0, sigma, A
    let sigma_guess = (0.45 * (n as f64 / (2.0 * (n_waves_2d + n_sel) as f64)) * dt).min(0.9 * sigma_max);
    for &idx in &sel_peaks {
        x0.push(t[idx]);
        x0.push(sigma_guess);
        x0.push(resid[idx]);
        lb.push(t_low); lb.push(sigma_min); lb.push(0.0);
        ub.push(t_high); ub.push(sigma_max); ub.push(max_amp);
    }

    // Clamp
    for i in 0..x0.len() {
        x0[i] = x0[i].max(lb[i] + 1e-6).min(ub[i] - 1e-6);
    }

    // 1D residual: env - mag
    let t_v = t.to_vec();
    let mag_v = mag.clone();
    let t0_s = t0_seed.clone();
    let s_s = s_seed.clone();
    let nw2d = n_waves_2d;
    let ns = n_sel;

    let ref_sig = refine_sigmas;
    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        let n_pts = t_v.len();
        let mut env = vec![0.0f64; n_pts];
        let mut ptr = 0usize;

        for i in 0..nw2d {
            let a = p[ptr]; ptr += 1;
            let s = if ref_sig { p[ptr].max(1.0) } else { s_s[i].max(1.0) };
            if ref_sig { ptr += 1; }
            let inv_s = 1.0 / s;
            for j in 0..n_pts {
                let z = (t_v[j] - t0_s[i]) * inv_s;
                env[j] += a * (-0.5 * z * z).exp();
            }
        }
        for _ in 0..ns {
            let (t0_e, s_e, a_e) = (p[ptr], p[ptr+1].max(1.0), p[ptr+2]);
            ptr += 3;
            let inv_s = 1.0 / s_e;
            for j in 0..n_pts {
                let z = (t_v[j] - t0_e) * inv_s;
                env[j] += a_e * (-0.5 * z * z).exp();
            }
        }
        env.iter().zip(mag_v.iter()).map(|(e, m)| e - m).collect()
    };

    let result = solve_lm(&residual_fn, None, &x0, &lb, &ub, 3000, false, 1.0, 1e-8, 1e-8);
    let p_opt = result.params;

    // Assemble final [t0, sigma, A, alpha] * (n_waves_2d + n_sel)
    let mut out = Vec::new();
    let mut ptr = 0usize;
    for i in 0..n_waves_2d {
        let a_i = p_opt[ptr]; ptr += 1;
        let s_i = if refine_sigmas { let s = p_opt[ptr]; ptr += 1; s } else { s_seed[i] };
        let idx_c = nearest_idx(t, t0_seed[i]);
        let alpha_i = vy[idx_c].atan2(vx[idx_c]);
        out.extend_from_slice(&[t0_seed[i], s_i, a_i, alpha_i]);
    }
    for _ in 0..n_sel {
        let (t0_e, s_e, a_e) = (p_opt[ptr], p_opt[ptr+1], p_opt[ptr+2]);
        ptr += 3;
        let idx_c = nearest_idx(t, t0_e);
        let alpha_e = vy[idx_c].atan2(vx[idx_c]);
        out.extend_from_slice(&[t0_e, s_e, a_e, alpha_e]);
    }
    out
}

/// fit_gaussians_hybrid_subrange: restrict fitting to a time window.
/// Paired from gaussian_fitting.py:569-651
pub fn fit_gaussians_hybrid_subrange(
    t: &[f64], vx: &[f64], vy: &[f64],
    t_min: f64, t_max: f64,
    n_waves: usize, n_extra: usize, amp_thr: f64,
    sigma_max: f64, angle_lambda: f64, amplitude_lambda: f64,
    refine_sigmas: bool,
    n_restarts: usize,
    jitter_scale: f64,
) -> Vec<f64> {
    // Guard: degenerate window
    if t_min >= t_max { return vec![]; }

    // Extract sub-range
    let mask: Vec<usize> = (0..t.len()).filter(|&i| t[i] >= t_min && t[i] <= t_max).collect();
    if mask.len() < 3 { return vec![]; }

    let t_sub: Vec<f64> = mask.iter().map(|&i| t[i]).collect();
    let vx_sub: Vec<f64> = mask.iter().map(|&i| vx[i]).collect();
    let vy_sub: Vec<f64> = mask.iter().map(|&i| vy[i]).collect();

    // H5: amp_min guard — skip fitting if signal is below noise floor
    let max_mag = vx_sub.iter().zip(vy_sub.iter())
        .map(|(x, y)| (x * x + y * y).sqrt())
        .fold(0.0f64, f64::max);
    if max_mag < 0.005 { return vec![]; }

    // H2: Graceful error handling for bounds violations (paired from gaussian_fitting.py:631-641)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_gaussians_hybrid(
            &t_sub, &vx_sub, &vy_sub,
            n_waves, n_extra, amp_thr,
            t_min, t_max, sigma_max, 1.0,
            angle_lambda, amplitude_lambda,
            refine_sigmas, n_restarts, jitter_scale,
        )
    }));
    match result {
        Ok(params) => params,
        Err(_) => vec![], // graceful return on bounds violation
    }
}
