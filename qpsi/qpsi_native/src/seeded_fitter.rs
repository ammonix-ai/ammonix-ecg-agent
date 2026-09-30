//! Complete wave component fitter with seeding — replaces Python fit_wave_components_core.
//!
//! Paired translation from wave_fitters.py:506-918.
//! Includes: savgol filter, peak detection, matched filter seeding,
//! K=0/K=1/K=2 model selection with AICc, notch detection.
//! Designed to be called from rayon parallel loops.

use crate::solver::{lm_solve_k1, lm_solve_k2, LsResult};

// ---------------------------------------------------------------------------
// Savitzky-Golay filter (simplified, paired from scipy.signal.savgol_filter)
// ---------------------------------------------------------------------------

/// Compute Savitzky-Golay filter coefficients for quadratic polynomial (order=2).
fn savgol_coeffs(window_len: usize) -> Vec<f64> {
    let half = (window_len / 2) as f64;
    let n = window_len;
    // Least-squares fit of quadratic to [-half..half], return smoothing row
    let mut sum_x2: f64 = 0.0;
    let mut sum_x4: f64 = 0.0;
    let mut sum_1: f64 = 0.0;
    for i in 0..n {
        let x = i as f64 - half;
        sum_1 += 1.0;
        sum_x2 += x * x;
        sum_x4 += x * x * x * x;
    }
    // Normal equations for [a0, a2] (a1=0 by symmetry for eval at center):
    // [sum_1   sum_x2] [a0]   [y_i]
    // [sum_x2  sum_x4] [a2] = [y_i * x_i^2]
    // Smoothing value at center = a0, so weights are:
    // w_i = (sum_x4 - sum_x2 * x_i^2) / (sum_1 * sum_x4 - sum_x2^2)
    let det = sum_1 * sum_x4 - sum_x2 * sum_x2;
    if det.abs() < 1e-30 {
        // Fallback: uniform weights
        return vec![1.0 / n as f64; n];
    }
    let mut coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 - half;
        coeffs.push((sum_x4 - sum_x2 * x * x) / det);
    }
    coeffs
}

/// Fit a quadratic polynomial to points (xs, ys) and evaluate at x_eval.
/// Uses normal equations: [sum1 sumx sumx2; sumx sumx2 sumx3; sumx2 sumx3 sumx4] * [a0 a1 a2] = [sumy sumxy sumx2y]
fn polyeval_quadratic(xs: &[f64], ys: &[f64], x_eval: f64) -> f64 {
    let n = xs.len();
    if n == 0 { return 0.0; }
    // Build sums for 3x3 normal equations
    let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut r0, mut r1, mut r2) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        let x2 = x * x;
        s0 += 1.0; s1 += x; s2 += x2; s3 += x * x2; s4 += x2 * x2;
        r0 += y; r1 += x * y; r2 += x2 * y;
    }
    // Solve 3x3 system via Cramer's rule
    let det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2);
    if det.abs() < 1e-30 {
        return r0 / n as f64; // fallback to mean
    }
    let a0 = (r0 * (s2 * s4 - s3 * s3) - s1 * (r1 * s4 - r2 * s3) + s2 * (r1 * s3 - r2 * s2)) / det;
    let a1 = (s0 * (r1 * s4 - r2 * s3) - r0 * (s1 * s4 - s2 * s3) + s2 * (s1 * r2 - s2 * r1)) / det;
    let a2 = (s0 * (s2 * r2 - s3 * r1) - s1 * (s1 * r2 - s2 * r1) + r0 * (s1 * s3 - s2 * s2)) / det;
    a0 + a1 * x_eval + a2 * x_eval * x_eval
}

/// Apply Savitzky-Golay smoothing filter (quadratic, mode=interp).
/// Matches scipy.signal.savgol_filter(y, window_len, 2, mode='interp').
/// Interior points: standard convolution with SG coefficients.
/// Edge points: fit quadratic to nearest window_len points, evaluate at position.
pub fn savgol_filter_quadratic_pub(y: &[f64], window_len: usize) -> Vec<f64> {
    savgol_filter_quadratic(y, window_len)
}

fn savgol_filter_quadratic(y: &[f64], window_len: usize) -> Vec<f64> {
    let n = y.len();
    let wl = if window_len % 2 == 0 { window_len + 1 } else { window_len };
    let wl = wl.max(5).min(n);
    let half = wl / 2;
    let coeffs = savgol_coeffs(wl);
    let mut out = vec![0.0; n];

    // Build integer x-positions for polynomial fitting (used at edges)
    // scipy uses integer positions 0..n-1, then shifts to center on the window
    for i in 0..n {
        if i >= half && i + half < n {
            // Interior: standard convolution
            let mut val = 0.0;
            for j in 0..wl {
                val += coeffs[j] * y[i + j - half];
            }
            out[i] = val;
        } else {
            // Edge: fit quadratic to the nearest wl points, evaluate at position i
            let (start, end) = if i < half {
                (0, wl.min(n))
            } else {
                (n.saturating_sub(wl), n)
            };
            let xs: Vec<f64> = (start..end).map(|k| k as f64).collect();
            let ys: Vec<f64> = (start..end).map(|k| y[k]).collect();
            out[i] = polyeval_quadratic(&xs, &ys, i as f64);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Peak finding (paired from scipy.signal.find_peaks with distance+prominence)
// ---------------------------------------------------------------------------

/// Find peaks in a signal with minimum distance and prominence.
/// Returns indices of peaks, sorted by amplitude (largest last).
fn find_peaks_abs(y: &[f64], min_distance: usize, min_prominence: f64) -> Vec<usize> {
    let n = y.len();
    if n < 3 {
        return vec![];
    }
    let mut peaks = Vec::new();
    for i in 1..n - 1 {
        if y[i] > y[i - 1] && y[i] > y[i + 1] {
            peaks.push(i);
        }
    }

    // Filter by distance (keep tallest when peaks are too close)
    peaks.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap().reverse());
    let mut keep = vec![true; peaks.len()];
    let mut selected = Vec::new();
    for i in 0..peaks.len() {
        if !keep[i] {
            continue;
        }
        selected.push(peaks[i]);
        for j in (i + 1)..peaks.len() {
            if keep[j] && (peaks[j] as isize - peaks[i] as isize).unsigned_abs() < min_distance {
                keep[j] = false;
            }
        }
    }

    // R2: Filter by prominence using isolation interval walk (scipy algorithm)
    selected.retain(|&pk| {
        // Walk LEFT until higher peak or edge
        let mut left_base = y[pk];
        let mut j = pk;
        while j > 0 {
            j -= 1;
            left_base = left_base.min(y[j]);
            if y[j] > y[pk] { break; }
        }
        // Walk RIGHT until higher peak or edge
        let mut right_base = y[pk];
        j = pk;
        while j < n - 1 {
            j += 1;
            right_base = right_base.min(y[j]);
            if y[j] > y[pk] { break; }
        }
        let base = left_base.max(right_base);
        (y[pk] - base) >= min_prominence
    });

    // Sort by index
    selected.sort();
    selected
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// R1: MAD-based robust scale — paired 1:1 from wave_fitters.py:194-206.
fn robust_scale(y: &[f64]) -> f64 {
    if y.is_empty() { return 1e-6; }
    let n = y.len();
    let mut sorted: Vec<f64> = y.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
    let mut abs_dev: Vec<f64> = y.iter().map(|v| (v - med).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n % 2 == 1 { abs_dev[n / 2] } else { (abs_dev[n / 2 - 1] + abs_dev[n / 2]) / 2.0 };
    (1.4826 * (mad + 1e-12)).max(1e-6)
}

/// R3: AICc — paired 1:1 from wave_fitters.py:229-241.
fn aicc(rss: f64, k: usize, n: usize) -> f64 {
    let n_f = (n as f64).max(1.0);
    let k_f = k as f64;
    let denom = (n_f - k_f - 1.0).max(1.0);
    n_f * (rss / n_f + 1e-12).ln() + 2.0 * k_f + (2.0 * k_f * (k_f + 1.0)) / denom
}

fn gauss_eval(x: f64, amp: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / (sigma + 1e-9);
    amp * (-0.5 * z * z).exp()
}

fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Clip strictly inside bounds by a margin of frac * span.
/// Paired from wave_fitters.py:209-217 `_clip_inside`.
fn clip_inside(v: f64, lo: f64, hi: f64) -> f64 {
    let frac: f64 = 1e-6;
    let span = hi - lo;
    v.max(lo + frac * span).min(hi - frac * span)
}

fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn ptp(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mn = data.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    mx - mn
}

fn argmax_abs(data: &[f64]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Seeding (paired from wave_fitters.py lines 608-677)
// ---------------------------------------------------------------------------

struct Seed {
    mu: f64,
    method: &'static str,
}

fn build_seeds(
    x: &[f64],
    y_seed: &[f64],
    y_ds: &[f64],
    x_lo: f64,
    x_hi: f64,
    lo_sig: f64,
    hi_sig: f64,
    step_ms: f64,
    min_separation_ms: f64,
    multi_seed: bool,
    max_templates: usize,
    prefer_left_bias: bool,
) -> Vec<Seed> {
    let n = x.len();
    let mut seeds_mu: Vec<f64> = Vec::new();
    let mut seed_methods: Vec<&str> = Vec::new();

    // 1) abs-peak seeds
    let y_abs: Vec<f64> = y_seed.iter().map(|v| v.abs()).collect();
    if multi_seed {
        let dist = (min_separation_ms / step_ms.max(1e-9)).max(2.0) as usize;
        let prom = (0.02 * ptp(&y_abs)).max(0.001);
        let pk_idx = find_peaks_abs(&y_abs, dist, prom);
        if !pk_idx.is_empty() {
            // Take top 3 by amplitude
            let mut by_amp: Vec<usize> = pk_idx.clone();
            by_amp.sort_by(|&a, &b| y_abs[b].partial_cmp(&y_abs[a]).unwrap_or(std::cmp::Ordering::Equal));
            by_amp.truncate(3);
            by_amp.sort(); // Back to index order
            for &i in &by_amp {
                seeds_mu.push(x[i]);
                seed_methods.push("peaks");
            }
        } else {
            let i0 = argmax_abs(y_seed);
            seeds_mu.push(x[i0]);
            seed_methods.push("peaks-fallback");
        }
    } else {
        let i0 = argmax_abs(y_seed);
        seeds_mu.push(x[i0]);
        seed_methods.push("peaks");
    }

    // 2) matched filter (multi-sigma)
    if multi_seed && max_templates >= 1 {
        let n_templates = max_templates.max(2);
        let stride = (8.0 / step_ms).max(1.0) as usize;
        let mu_grid: Vec<f64> = x.iter().step_by(stride).copied().collect();

        for t in 0..n_templates {
            let s = lo_sig + (hi_sig - lo_sig) * (t as f64) / ((n_templates - 1).max(1) as f64);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_j = 0;
            for (j, &mu) in mu_grid.iter().enumerate() {
                let inv_s = 1.0 / (s + 1e-9);
                let mut num = 0.0;
                let mut den = 0.0;
                for i in 0..n {
                    let z = (x[i] - mu) * inv_s;
                    let g = (-0.5 * z * z).exp();
                    num += y_seed[i] * g;
                    den += g * g;
                }
                let score = (num * num) / (den + 1e-12);
                if score > best_score {
                    best_score = score;
                    best_j = j;
                }
            }
            seeds_mu.push(mu_grid[best_j]);
            seed_methods.push("matched");
        }

        // Edge bias
        if !mu_grid.is_empty() {
            let q_idx = (0.25 * mu_grid.len() as f64) as usize;
            let s0 = (lo_sig + hi_sig) / 2.0;
            let (range_start, range_end) = if prefer_left_bias {
                (0, q_idx.max(1))
            } else {
                (mu_grid.len().saturating_sub(q_idx), mu_grid.len())
            };

            let mut scores_all: Vec<f64> = Vec::with_capacity(mu_grid.len());
            for &mu in &mu_grid {
                let inv_s = 1.0 / (s0 + 1e-9);
                let mut num = 0.0;
                let mut den = 0.0;
                for i in 0..n {
                    let z = (x[i] - mu) * inv_s;
                    let g = (-0.5 * z * z).exp();
                    num += y_seed[i] * g;
                    den += g * g;
                }
                scores_all.push((num * num) / (den + 1e-12));
            }

            let mut best_j = range_start;
            let mut best_s = f64::NEG_INFINITY;
            for j in range_start..range_end.min(mu_grid.len()) {
                if scores_all[j] > best_s {
                    best_s = scores_all[j];
                    best_j = j;
                }
            }
            seeds_mu.push(mu_grid[best_j]);
            seed_methods.push(if prefer_left_bias { "matched-left-bias" } else { "matched-right-bias" });
        }
    }

    // Deduplicate seeds ~12ms
    let mut seeds: Vec<Seed> = Vec::new();
    for (mu, method) in seeds_mu.into_iter().zip(seed_methods.into_iter()) {
        let mu = clip(mu, x_lo, x_hi);
        if !seeds.iter().any(|s| (s.mu - mu).abs() <= 12.0) {
            seeds.push(Seed { mu, method });
        }
    }
    if seeds.is_empty() {
        let i0 = argmax_abs(y_seed);
        seeds.push(Seed { mu: x[i0], method: "fallback" });
    }
    seeds
}

// ---------------------------------------------------------------------------
// Complete wave fitter (paired from fit_wave_components_core)
// ---------------------------------------------------------------------------

/// Result of fitting a single wave window (P or T).
#[derive(Clone, Debug)]
pub struct WaveFitResult {
    pub components: Vec<GaussComponent>,
    pub model: String,     // "K0", "K1", "K2"
    pub present: bool,
    pub snr_db: f64,
    pub notched: bool,
    pub interpeak_ms: f64,
    pub valley_drop_frac: f64,
    pub edge_left: bool,
    pub edge_right: bool,
    pub noise_sigma: f64,
}

#[derive(Clone, Debug)]
pub struct GaussComponent {
    pub amp_mv: f64,
    pub center_ms: f64,
    pub sigma_ms: f64,
}

/// Full K=0/K=1/K=2 fitter with multi-seed strategy and AICc selection.
/// Paired 1:1 from wave_fitters.py fit_wave_components_core (lines 506-918).
pub fn fit_wave_components_core_rust(
    x: &[f64],
    y: &[f64],
    sigma_bounds_ms: (f64, f64),
    min_separation_ms: f64,
    max_separation_ms: f64,
    allow_biphasic: bool,
    allow_two: bool,
    aic_delta_two: f64,
    allow_baseline_k0: bool,
    center_hint_ms: Option<f64>,
    center_hint_sigma_ms: f64,
    prefer_left_bias: bool,
    multi_seed: bool,
    max_templates: usize,
    min_peak_snr_db: f64,
    residual_hint: Option<&[f64]>,
    aicc_early_term: bool,
    aicc_early_term_delta: f64,
) -> WaveFitResult {
    let n = x.len();
    if n < 8 || ptp(x) <= 0.0 {
        return WaveFitResult {
            components: vec![], model: "NA".into(), present: false,
            snr_db: 0.0, notched: false, interpeak_ms: 0.0,
            valley_drop_frac: 0.0, edge_left: false, edge_right: false,
            noise_sigma: 1e-6,
        };
    }

    let x_lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (lo_sig, hi_sig) = sigma_bounds_ms;

    // Step size
    let mut diffs: Vec<f64> = x.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let step_ms = if diffs.is_empty() { 1.0 } else { median(&mut diffs) }.max(1e-6);

    // Savgol smoothing + detrend
    let k_sg = ((n / 15) * 2 + 1).max(5);
    let y_s = savgol_filter_quadratic(y, k_sg);

    // Linear detrend
    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let mut sx2 = 0.0;
    let mut sxy = 0.0;
    let mut sy = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        sx2 += dx * dx;
        sxy += dx * y_s[i];
        sy += y_s[i];
    }
    let slope = if sx2.abs() > 1e-30 { sxy / sx2 } else { 0.0 };
    let intercept = sy / n as f64 - slope * x_mean;
    let y_ds: Vec<f64> = (0..n)
        .map(|i| y_s[i] - (slope * (x[i] - x_mean) + intercept))
        .collect();

    // Seed signal: residual-aware or high-pass
    let y_seed_base: Vec<f64> = match residual_hint {
        Some(hint) if hint.len() == n => {
            y.iter().zip(hint.iter()).map(|(a, b)| a - b).collect()
        }
        _ => y.to_vec(),
    };

    let k_hp = ((n / 8) * 2 + 1).max(7);
    let y_hp = savgol_filter_quadratic(&y_seed_base, k_hp);
    let y_seed: Vec<f64> = y_seed_base.iter().zip(y_hp.iter()).map(|(a, b)| a - b).collect();

    // Amplitude bounds
    let abs_y_ds: Vec<f64> = y_ds.iter().map(|v| v.abs()).collect();
    let max_abs = abs_y_ds.iter().copied().fold(0.0f64, f64::max);
    let med_abs = median(&abs_y_ds);
    let a_max = (1.25 * max_abs)
        .max(0.5 * ptp(y))
        .max(5.0 * med_abs)
        .max(1e-3);
    let b_max = (0.25 * ptp(y)).max(1e-3);

    let noise_sigma = robust_scale(&y_seed);
    let snr_db_fn = |amp: f64| 20.0 * (amp.abs().max(1e-12) / noise_sigma).log10();

    // Build seeds
    let seeds = build_seeds(
        x, &y_seed, &y_ds, x_lo, x_hi,
        lo_sig, hi_sig, step_ms, min_separation_ms,
        multi_seed, max_templates, prefer_left_bias,
    );

    // K=0 baseline
    let mut aic0 = f64::INFINITY;
    let b0_hat = clip(y.iter().sum::<f64>() / n as f64, -b_max, b_max);
    if allow_baseline_k0 {
        let rss0: f64 = y.iter().map(|&yi| (yi - b0_hat).powi(2)).sum();
        aic0 = aicc(rss0, 1, n);
    }

    // Best candidate tracking
    let mut best_aic_eff = f64::INFINITY;
    let mut best_model = "K0";
    let mut best_comps: Vec<GaussComponent> = Vec::new();
    let mut best_amp_abs: f64 = 0.0;
    let mut best_aic: f64 = aic0;

    let has_hint = center_hint_ms.is_some();
    let hint_ms = center_hint_ms.unwrap_or(0.0);

    for seed in &seeds {
        let i_near = x.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| ((**a - seed.mu).abs()).partial_cmp(&((**b - seed.mu).abs())).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let a1_0 = clip_inside(y_ds[i_near], -a_max, a_max);
        let s1_0 = clip_inside(0.6 * (lo_sig + hi_sig), lo_sig, hi_sig);
        let mu1_0 = clip_inside(seed.mu, x_lo, x_hi);
        let b0_0 = clip_inside(0.0, -b_max, b_max);

        // K=1 fit
        let x0_k1 = vec![a1_0, mu1_0, s1_0, b0_0];
        let lb_k1 = vec![-a_max, x_lo, lo_sig, -b_max];
        let ub_k1 = vec![a_max, x_hi, hi_sig, b_max];

        let rs1 = lm_solve_k1(x, y, &x0_k1, &lb_k1, &ub_k1, hint_ms, center_hint_sigma_ms, has_hint);
        let (a1, mu1, s1, b01) = (rs1.params[0], rs1.params[1], rs1.params[2], rs1.params[3]);

        let rss1: f64 = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (yi - gauss_eval(xi, a1, mu1, s1) - b01).powi(2))
            .sum();
        let aic1 = aicc(rss1, 4, n);
        let cand1_aic_eff = aic1;
        let cand1_comps = vec![GaussComponent { amp_mv: a1, center_ms: mu1, sigma_ms: s1 }];
        let cand1_amp_abs = a1.abs();

        // Update best for K=1
        if cand1_aic_eff < best_aic_eff
            || ((cand1_aic_eff - best_aic_eff).abs() <= 0.75
                && !cand1_comps.is_empty()
                && !best_comps.is_empty()
                && prefer_left_bias
                && cand1_comps[0].center_ms < best_comps[0].center_ms)
            || ((cand1_aic_eff - best_aic_eff).abs() <= 0.75
                && !cand1_comps.is_empty()
                && !best_comps.is_empty()
                && !prefer_left_bias
                && cand1_comps[0].center_ms > best_comps[0].center_ms)
        {
            best_aic_eff = cand1_aic_eff;
            best_model = "K1";
            best_comps = cand1_comps.clone();
            best_amp_abs = cand1_amp_abs;
            best_aic = aic1;
        }

        // R4: AICc early termination — if K=0 dominates K=1 by delta, skip K=2
        let skip_k2_early = aicc_early_term
            && allow_baseline_k0
            && (aic0 + aicc_early_term_delta < aic1);

        // K=2 fit
        if allow_two && !skip_k2_early {
            let a2_0 = if allow_biphasic {
                clip_inside(y_ds[i_near], -a_max, a_max)
            } else {
                clip_inside(y_ds[i_near].abs() * a1_0.signum(), -a_max, a_max)
            };
            let sep0 = clip_inside(
                (min_separation_ms.max(0.5 * (hi_sig + lo_sig))).min(max_separation_ms),
                min_separation_ms, max_separation_ms,
            );
            let s2_0 = clip_inside(s1_0, lo_sig, hi_sig);

            let x0_k2 = vec![a1_0, mu1_0, s1_0, a2_0, sep0, s2_0, b0_0];
            let lb_k2 = vec![-a_max, x_lo, lo_sig, -a_max, min_separation_ms, lo_sig, -b_max];
            let ub_k2 = vec![a_max, x_hi, hi_sig, a_max, max_separation_ms, hi_sig, b_max];

            let rs2 = lm_solve_k2(x, y, &x0_k2, &lb_k2, &ub_k2, hint_ms, center_hint_sigma_ms, has_hint);
            let (a1_, mu1_, s1_) = (rs2.params[0], rs2.params[1], rs2.params[2]);
            let (a2_, sep_, s2_, b02) = (rs2.params[3], rs2.params[4], rs2.params[5], rs2.params[6]);
            let mu2_ = mu1_ + sep_;

            let rss2: f64 = x.iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| {
                    (yi - gauss_eval(xi, a1_, mu1_, s1_) - gauss_eval(xi, a2_, mu2_, s2_) - b02).powi(2)
                })
                .sum();
            let aic2 = aicc(rss2, 7, n);
            let cand2_aic_eff = aic2 + aic_delta_two;

            // Order by center
            let (ca1, cmu1, cs1, ca2, cmu2, cs2) = if mu2_ < mu1_ {
                (a2_, mu2_, s2_, a1_, mu1_, s1_)
            } else {
                (a1_, mu1_, s1_, a2_, mu2_, s2_)
            };

            let cand2_comps = vec![
                GaussComponent { amp_mv: ca1, center_ms: cmu1, sigma_ms: cs1 },
                GaussComponent { amp_mv: ca2, center_ms: cmu2, sigma_ms: cs2 },
            ];
            let cand2_amp_abs = ca1.abs().max(ca2.abs());

            if cand2_aic_eff < best_aic_eff
                || ((cand2_aic_eff - best_aic_eff).abs() <= 0.75
                    && !cand2_comps.is_empty()
                    && !best_comps.is_empty()
                    && prefer_left_bias
                    && cand2_comps[0].center_ms < best_comps[0].center_ms)
                || ((cand2_aic_eff - best_aic_eff).abs() <= 0.75
                    && !cand2_comps.is_empty()
                    && !best_comps.is_empty()
                    && !prefer_left_bias
                    && cand2_comps[0].center_ms > best_comps[0].center_ms)
            {
                best_aic_eff = cand2_aic_eff;
                best_model = "K2";
                best_comps = cand2_comps;
                best_amp_abs = cand2_amp_abs;
                best_aic = aic2;
            }
        }
    }

    // K=0 vs best: SNR gating
    let present;
    let chosen_comps;
    let chosen_model;
    if !best_comps.is_empty() {
        let ok_aic = !allow_baseline_k0 || (best_aic + 1.0 < aic0);
        let ok_snr = snr_db_fn(best_amp_abs) >= min_peak_snr_db;
        if ok_aic && ok_snr {
            present = true;
            chosen_comps = best_comps;
            chosen_model = best_model;
        } else {
            present = false;
            chosen_comps = vec![];
            chosen_model = "K0";
        }
    } else {
        present = false;
        chosen_comps = vec![];
        chosen_model = "K0";
    }

    // R6 fix: when K=0 wins (present=false), SNR should reflect no wave found.
    // Python: peak_amp=0.0 → SNR ≈ -240 dB. Rust was leaking best K≥1 amplitude.
    let snr_amp = if present { best_amp_abs } else { 0.0 };

    // Notch detection
    let mut notched = false;
    let mut interpeak_ms = 0.0;
    let mut valley_drop_frac = 0.0;
    if chosen_model == "K2" && chosen_comps.len() == 2 {
        let c1 = &chosen_comps[0];
        let c2 = &chosen_comps[1];
        if c1.amp_mv.signum() == c2.amp_mv.signum() && c1.amp_mv.signum() != 0.0 {
            // Sample between the two centers
            let npts = 200;
            let mut valley = f64::INFINITY;
            for i in 0..npts {
                let t = c1.center_ms + (c2.center_ms - c1.center_ms) * (i as f64) / (npts as f64 - 1.0);
                let v = (gauss_eval(t, c1.amp_mv, c1.center_ms, c1.sigma_ms)
                    + gauss_eval(t, c2.amp_mv, c2.center_ms, c2.sigma_ms))
                    .abs();
                valley = valley.min(v);
            }
            let minpk = c1.amp_mv.abs().min(c2.amp_mv.abs()).max(1e-9);
            let drop = (minpk - valley) / minpk;
            if drop >= 0.20 && (c2.center_ms - c1.center_ms) >= 20.0 {
                notched = true;
                interpeak_ms = c2.center_ms - c1.center_ms;
                valley_drop_frac = drop;
            }
        }
    }

    // Edge detection
    let edge_left = present && {
        let mu = chosen_comps[0].center_ms;
        let s = chosen_comps[0].sigma_ms;
        (mu - x_lo) <= 12.0f64.max(0.8 * s)
    };
    let edge_right = present && {
        let mu = chosen_comps[0].center_ms;
        let s = chosen_comps[0].sigma_ms;
        (x_hi - mu) <= 16.0f64.max(0.8 * s)
    };

    WaveFitResult {
        components: chosen_comps,
        model: chosen_model.into(),
        present,
        snr_db: snr_db_fn(snr_amp),
        notched,
        interpeak_ms,
        valley_drop_frac,
        edge_left,
        edge_right,
        noise_sigma,
    }
}
