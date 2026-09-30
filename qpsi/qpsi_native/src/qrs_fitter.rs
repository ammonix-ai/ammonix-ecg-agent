//! QRS unified fitter — paired 1:1 from wave_fitters.py _fit_qrs_unified (lines 1199-1460).
//!
//! Detects QRS boundaries via gradient analysis, seeds Q/R/S/R' components,
//! estimates sigma via FWHM, and solves amplitudes via constrained least-squares.

#[allow(unused_imports)]
use nalgebra;

/// Result of QRS fitting for one lead.
#[derive(Clone, Debug)]
pub struct QrsComponent {
    pub label: String,   // "Q", "R", "S", "R2"
    pub amp_mv: f64,
    pub center_ms: f64,
    pub sigma_ms: f64,
}

#[derive(Clone, Debug)]
pub struct QrsFitResult {
    pub components: Vec<QrsComponent>,
    pub r_peak_time: f64,
    pub bounds: (f64, f64),    // (left_ms, right_ms)
    pub yfit: Vec<f64>,        // fitted waveform (full length, zeros outside bounds+padding)
    pub mask: Vec<bool>,       // boolean mask (t >= left_ms) & (t <= right_ms)
    pub padding_ms: f64,       // padding used for yfit
}

/// Gaussian basis: exp(-0.5 * ((t - mu) / s)^2)
fn gauss_basis(t: &[f64], mu: f64, s: f64) -> Vec<f64> {
    let inv_s = 1.0 / (s + 1e-9);
    t.iter().map(|&ti| {
        let z = (ti - mu) * inv_s;
        (-0.5 * z * z).exp()
    }).collect()
}

/// Gaussian smoothing with edge padding. Paired from _fit_qrs_unified::gauss_smooth.
pub fn gauss_smooth(x: &[f64], dt_ms: f64, sigma_ms: f64) -> Vec<f64> {
    if dt_ms <= 0.0 || x.is_empty() { return x.to_vec(); }
    let sig_samp = (sigma_ms / dt_ms).round().max(1.0) as usize;
    let k = ((6 * sig_samp) | 1).max(3);

    // Build kernel
    let mut kernel = Vec::with_capacity(2 * k + 1);
    let mut sum = 0.0;
    for i in 0..=(2 * k) {
        let gi = (i as f64) - k as f64;
        let v = (-0.5 * (gi / sig_samp as f64).powi(2)).exp();
        kernel.push(v);
        sum += v;
    }
    for v in kernel.iter_mut() { *v /= sum; }

    // Pad with edge values
    let n = x.len();
    let mut padded = Vec::with_capacity(n + 2 * k);
    for _ in 0..k { padded.push(x[0]); }
    padded.extend_from_slice(x);
    for _ in 0..k { padded.push(x[n - 1]); }

    // Convolve
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut val = 0.0;
        for j in 0..kernel.len() {
            val += padded[i + j] * kernel[j];
        }
        out[i] = val;
    }
    out
}

/// Numerical gradient matching numpy.gradient(y, t, edge_order=2).
/// Interior: central differences. Edges: second-order one-sided differences.
pub fn gradient(y: &[f64], t: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n < 2 { return vec![0.0; n]; }
    let mut g = vec![0.0; n];

    // Interior: central differences (same as numpy)
    for i in 1..n - 1 {
        let dt = t[i + 1] - t[i - 1];
        if dt.abs() > 1e-15 {
            g[i] = (y[i + 1] - y[i - 1]) / dt;
        }
    }

    // Edges: second-order one-sided (edge_order=2)
    // numpy uses: (-3*f[0] + 4*f[1] - f[2]) / (x[2] - x[0]) for left edge
    //             (3*f[-1] - 4*f[-2] + f[-3]) / (x[-1] - x[-3]) for right edge
    if n >= 3 {
        let dt_left = t[2] - t[0];
        if dt_left.abs() > 1e-15 {
            g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / dt_left;
        }
        let dt_right = t[n - 1] - t[n - 3];
        if dt_right.abs() > 1e-15 {
            g[n - 1] = (3.0 * y[n - 1] - 4.0 * y[n - 2] + y[n - 3]) / dt_right;
        }
    } else {
        // n == 2: fall back to first-order
        let dt = t[1] - t[0];
        if dt.abs() > 1e-15 {
            g[0] = (y[1] - y[0]) / dt;
            g[1] = (y[1] - y[0]) / dt;
        }
    }
    g
}

/// Solve least-squares Ax = b via SVD, matching numpy.linalg.lstsq(A, b, rcond=None).
/// phi is row-major (n_rows x n_cols).
pub fn lstsq_svd(phi: &[f64], n_rows: usize, n_cols: usize, b: &[f64]) -> Vec<f64> {
    if n_cols == 0 || n_rows == 0 { return vec![0.0; n_cols]; }

    // SVD via LAPACK dgesdd (or nalgebra fallback)
    let svd = crate::lapack_svd::svd_decompose(phi, n_rows, n_cols);
    let k = svd.k;
    let s = &svd.s;

    // Threshold for near-zero singular values (matches numpy rcond=None → machine eps * max(m,n) * s[0])
    let threshold = 2.220446049250313e-16 * (n_rows.max(n_cols) as f64) * s[0];

    // U^T * b: utb[j] = sum_i U[i,j] * b[i]
    // u_col is column-major: u_col[j*m + i] = U[i,j]
    let utb: Vec<f64> = (0..k).map(|j| {
        let mut dot = 0.0;
        for i in 0..n_rows {
            dot += svd.u_col[j * n_rows + i] * b[i];
        }
        dot
    }).collect();

    // x = V * diag(1/s_i) * U^T * b
    // V[j,i] = V^T[i,j]; vt_col is column-major k×n: vt_col[j*k + i] = V^T[i,j]
    // So V[j,i] = vt_col[j*k + i]
    let mut x = vec![0.0; n_cols];
    for j in 0..n_cols {
        for i in 0..k {
            if s[i] > threshold {
                x[j] += svd.vt_col[j * k + i] * utb[i] / s[i];
            }
        }
    }
    x
}

/// Full QRS unified fitter. Paired 1:1 from _fit_qrs_unified (wave_fitters.py:1199-1460).
pub fn fit_qrs_unified(
    trace: &[f64],
    time_ms: &[f64],
    qrs_window: (f64, f64),
    t_r_ms: Option<f64>,
    fs: f64,
    // Duration control
    min_qrs_ms: f64,    // 60.0
    max_qrs_ms: f64,    // 250.0
    padding_ms: f64,    // 10.0
    // Morphology control
    force_narrow: bool,  // false
    sigma_bounds_ms: (f64, f64), // (6.0, 60.0)
    max_components: usize, // 4
    q_s_threshold_rel: f64, // 0.03
    rprime_threshold_rel: f64, // 0.05
    min_sep_ms: f64,    // 8.0
) -> Option<QrsFitResult> {
    let t = time_ms;
    let y = trace;
    let n = t.len();
    if n < 5 { return None; }

    let (mut q0, mut q1) = qrs_window;
    if q1 < q0 { std::mem::swap(&mut q0, &mut q1); }

    // Helper: indices within [lo, hi]
    let idx_window = |lo: f64, hi: f64| -> Vec<usize> {
        (0..n).filter(|&i| t[i] >= lo && t[i] <= hi).collect()
    };

    // R-peak estimate
    let mut m0 = idx_window(q0, q1);
    if m0.is_empty() {
        m0 = (0..n).collect();
        q0 = t[0]; q1 = t[n - 1];
    }

    let r_idx0 = if let Some(tr) = t_r_ms {
        if tr.is_finite() {
            t.iter().enumerate().min_by(|(_, a), (_, b)|
                ((**a - tr).abs()).partial_cmp(&((**b - tr).abs())).unwrap()
            ).map(|(i, _)| i).unwrap_or(0)
        } else {
            m0[y.iter().enumerate().filter(|(i, _)| m0.contains(i))
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .map(|(i, _)| i).unwrap_or(0)]
        }
    } else {
        // argmax(abs(y[m0]))
        let mut best_i = m0[0];
        let mut best_v = 0.0f64;
        for &i in &m0 {
            if y[i].abs() > best_v { best_v = y[i].abs(); best_i = i; }
        }
        best_i
    };

    // Refine R peak within +/-8ms
    let refine_peak = |center_idx: usize, span_ms: f64| -> usize {
        let lo = t[center_idx] - span_ms;
        let hi = t[center_idx] + span_ms;
        let ii = idx_window(lo, hi);
        if ii.is_empty() { return center_idx; }
        let sgn = if y[center_idx] >= 0.0 { 1.0 } else { -1.0 };
        let mut best_i = ii[0];
        let mut best_v = f64::NEG_INFINITY;
        for &i in &ii {
            let v = sgn * y[i];
            if v > best_v { best_v = v; best_i = i; }
        }
        best_i
    };

    let r_idx = refine_peak(r_idx0, 8.0);
    let t_r = t[r_idx];
    let a_r_obs = y[r_idx];
    let sgn_r = if a_r_obs >= 0.0 { 1.0 } else { -1.0 };

    // Boundary detection via gradient
    let dt_ms = if n > 1 {
        let mut diffs: Vec<f64> = t.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        diffs[diffs.len() / 2]
    } else {
        1000.0 / fs
    };

    let grad = gradient(y, t);
    // Python: a = np.abs(gauss_smooth(gradient, 8.0)) — smooth THEN abs
    let smoothed_grad_signed = gauss_smooth(&grad, dt_ms, 8.0);
    let smoothed_grad: Vec<f64> = smoothed_grad_signed.iter().map(|v| v.abs()).collect();

    let max_grad_in_qrs = m0.iter().map(|&i| smoothed_grad[i]).fold(0.0f64, f64::max);
    let thr = if max_grad_in_qrs > 0.0 { 0.18 * max_grad_in_qrs } else { 0.0 };

    let walk_boundary = |start_idx: usize, step: i64| -> f64 {
        let mut i = start_idx as i64;
        let mut quiet = 0;
        while i >= 0 && (i as usize) < n {
            if smoothed_grad[i as usize] < thr {
                quiet += 1;
                if quiet >= 3 { return t[i as usize]; }
            } else {
                quiet = 0;
            }
            i += step;
        }
        t[((i - step).max(0) as usize).min(n - 1)]
    };

    let mut left_ms = walk_boundary(r_idx, -1);
    let mut right_ms = walk_boundary(r_idx, 1);

    left_ms = left_ms.max(q0);
    right_ms = right_ms.min(q1);
    let mut half = 0.5 * (right_ms - left_ms);
    half = half.max(min_qrs_ms / 2.0).min(max_qrs_ms / 2.0);
    left_ms = (t_r - half).max(q0);
    right_ms = (t_r + half).min(q1);

    let m_fit: Vec<usize> = idx_window(left_ms - padding_ms, right_ms + padding_ms);
    if m_fit.is_empty() { return None; }

    // Seed components: (name, mu, sigma_fixed_or_None, sign_hint)
    let mut comps: Vec<(String, f64, Option<f64>, f64)> = Vec::new();

    let pick_extreme = |lo: f64, hi: f64, pol: f64| -> Option<(f64, f64)> {
        let ii = idx_window(lo, hi);
        if ii.is_empty() { return None; }
        let mut best_i = ii[0];
        let mut best_v = f64::NEG_INFINITY;
        for &i in &ii {
            let v = pol * y[i];
            if v > best_v { best_v = v; best_i = i; }
        }
        Some((t[best_i], y[best_i]))
    };

    // R component (always present)
    comps.push(("R".into(), t_r, if force_narrow { Some(12.0) } else { None }, sgn_r));

    // Q component (negative lobe before R)
    if let Some((mu, amp)) = pick_extreme(left_ms, t_r - min_sep_ms, -sgn_r) {
        if amp.abs() >= q_s_threshold_rel * a_r_obs.abs() {
            comps.push(("Q".into(), mu, if force_narrow { Some(10.0) } else { None }, -sgn_r));
        }
    }

    // S component (negative lobe after R)
    if let Some((mu, amp)) = pick_extreme(t_r + min_sep_ms, right_ms, -sgn_r) {
        if amp.abs() >= q_s_threshold_rel * a_r_obs.abs() {
            comps.push(("S".into(), mu, if force_narrow { Some(10.0) } else { None }, -sgn_r));
        }
    }

    // R' component (positive lobe after R, optional)
    if max_components >= 4 {
        if let Some((mu, amp)) = pick_extreme(t_r + 1.5 * min_sep_ms, right_ms, sgn_r) {
            if amp.abs() >= rprime_threshold_rel * a_r_obs.abs() {
                comps.push(("R2".into(), mu, if force_narrow { Some(12.0) } else { None }, sgn_r));
            }
        }
    }

    // Sort by center, limit to max_components
    comps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    comps.truncate(max_components);

    // Estimate sigma via FWHM if not forced narrow
    let half_width_sigma = |mu_hw: f64, sign_hint: f64| -> f64 {
        let ii = idx_window(left_ms, right_ms);
        if ii.len() < 3 { return 12.0; }
        let idx_mu = ii[ii.iter().map(|&i| (t[i] - mu_hw).abs())
            .enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)];
        let ypeak = sign_hint * y[idx_mu];
        if ypeak <= 0.0 { return 12.0; }
        let h = 0.5 * ypeak;
        // Walk left
        let mut j = idx_mu;
        while j > ii[0] && sign_hint * y[j] > h { j -= 1; }
        let t_l = t[j];
        // Walk right
        j = idx_mu;
        while j < *ii.last().unwrap() && sign_hint * y[j] > h { j += 1; }
        let t_r_hw = t[j];
        let fwhm = (t_r_hw - t_l).max(4.0);
        fwhm / (2.0 * (2.0f64 * 2.0f64.ln()).sqrt()) // FWHM to sigma
    };

    let mut comps_final: Vec<(String, f64, f64, f64)> = comps.iter().map(|(name, mu, sigma_fixed, sign_hint)| {
        let s = match sigma_fixed {
            Some(sf) => *sf,
            None => half_width_sigma(*mu, *sign_hint).max(sigma_bounds_ms.0).min(sigma_bounds_ms.1),
        };
        (name.clone(), *mu, s, *sign_hint)
    }).collect();

    // Enforce minimum separation
    for i in 1..comps_final.len() {
        let (_, prev_mu, _, _) = comps_final[i - 1];
        let (_, cur_mu, _, _) = comps_final[i];
        if cur_mu - prev_mu < min_sep_ms {
            let shift = 0.5 * (min_sep_ms - (cur_mu - prev_mu));
            comps_final[i - 1].1 -= shift;
            comps_final[i].1 += shift;
        }
    }

    // Build Gaussian basis matrix (m_fit.len() x n_comps) and solve amplitudes
    let n_fit = m_fit.len();
    let n_comps = comps_final.len();
    if n_comps == 0 { return None; }

    let tm: Vec<f64> = m_fit.iter().map(|&i| t[i]).collect();
    let ym: Vec<f64> = m_fit.iter().map(|&i| y[i]).collect();
    let sign_hints: Vec<f64> = comps_final.iter().map(|c| c.3).collect();

    // Build Phi matrix (n_fit x n_comps, row-major)
    let mut phi = vec![0.0; n_fit * n_comps];
    for j in 0..n_comps {
        let basis = gauss_basis(&tm, comps_final[j].1, comps_final[j].2);
        for i in 0..n_fit {
            phi[i * n_comps + j] = basis[i];
        }
    }

    // Constrained lstsq with polarity enforcement (3 iterations max)
    let mut keep = vec![true; n_comps];
    let mut a_vec = vec![0.0; n_comps];
    for _ in 0..3 {
        if !keep.iter().any(|&k| k) { break; }

        // Build sub-matrix for kept columns
        let kept_indices: Vec<usize> = (0..n_comps).filter(|&j| keep[j]).collect();
        let n_kept = kept_indices.len();
        if n_kept == 0 { break; }

        let mut phi_sub = vec![0.0; n_fit * n_kept];
        for i in 0..n_fit {
            for (jj, &j) in kept_indices.iter().enumerate() {
                phi_sub[i * n_kept + jj] = phi[i * n_comps + j];
            }
        }

        let ak = lstsq_svd(&phi_sub, n_fit, n_kept, &ym);
        for (jj, &j) in kept_indices.iter().enumerate() {
            a_vec[j] = ak[jj];
        }

        // Check polarity constraints — paired from Python line 1429
        // Note: Python does NOT reset A_vec for dropped components (stale values kept)
        let mut any_bad = false;
        for j in 0..n_comps {
            if keep[j] && sign_hints[j] * a_vec[j] < 0.0 {
                keep[j] = false;
                // Do NOT zero a_vec[j] — Python leaves stale values from prior iteration
                any_bad = true;
            }
        }
        if !any_bad { break; }
    }

    // Build output — include ALL components (even dropped ones with stale amplitudes)
    // This matches Python's behavior: params = [(A_vec[i], ...) for i in range(len(comps_final))]
    let components: Vec<QrsComponent> = (0..n_comps)
        .map(|j| QrsComponent {
            label: comps_final[j].0.clone(),
            amp_mv: if a_vec[j].is_finite() { a_vec[j] } else { 0.0 },
            center_ms: comps_final[j].1,
            sigma_ms: comps_final[j].2,
        })
        .collect();

    if components.is_empty() { return None; }

    // Compute yfit and mask — paired EXACTLY from wave_fitters.py:1434-1451
    let padding = padding_ms;
    let mut yfit = vec![0.0; n];
    let mask_vec: Vec<bool> = (0..n).map(|i| t[i] >= left_ms && t[i] <= right_ms).collect();

    // yfit: sum of Gaussians, but only within (left-padding, right+padding)
    // Matches Python: yfit[mask_full] = ysum[mask_full]
    for i in 0..n {
        if t[i] >= (left_ms - padding) && t[i] <= (right_ms + padding) {
            let mut val = 0.0;
            for j in 0..n_comps {
                let a = if a_vec[j].is_finite() { a_vec[j] } else { 0.0 };
                let mu = comps_final[j].1;
                let s = comps_final[j].2;
                let z = (t[i] - mu) / (s + 1e-9);
                val += a * (-0.5 * z * z).exp();
            }
            yfit[i] = val;
        }
    }

    Some(QrsFitResult {
        components,
        r_peak_time: t_r,
        bounds: (left_ms, right_ms),
        yfit,
        mask: mask_vec,
        padding_ms: padding,
    })
}
