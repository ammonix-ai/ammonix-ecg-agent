//! Bounded Levenberg-Marquardt solver — replaces scipy.optimize.least_squares.
//!
//! Implements the LM algorithm with:
//! - Numerical Jacobian (central finite differences, matching scipy)
//! - Bounded parameter clamping (trust-region reflective approximation)
//! - Huber loss support (matching scipy's loss="huber")
//!
//! The solver is generic over the residual function, allowing K=1 and K=2
//! wave fitting problems to share the same optimizer code.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::residuals::{gauss_k1_residual_kernel, gauss_k2_residual_kernel};

// ---------------------------------------------------------------------------
// LM solver core (pure Rust, no Python dependency)
// ---------------------------------------------------------------------------

/// Result of a least-squares optimization.
pub struct LsResult {
    pub params: Vec<f64>,
    pub cost: f64,       // 0.5 * sum(residuals^2)
    pub n_fev: usize,    // number of function evaluations
    pub success: bool,
}

/// Analytic Jacobian for K=1 Gaussian residual.
///
/// r(p) = y - (A*exp(-0.5*((x-mu)/s)^2) + b0)  [+ optional prior]
/// dr/dA  = -exp(-0.5*z^2)
/// dr/dmu = -A * z/s * exp(-0.5*z^2)
/// dr/ds  = -A * z^2/s * exp(-0.5*z^2)
/// dr/db0 = -1
pub fn analytic_jacobian_k1(
    x_sig: &[f64],
    p: &[f64],
    has_hint: bool,
    center_hint_sigma_ms: f64,
) -> Vec<Vec<f64>> {
    let (a, mu, s, _b0) = (p[0], p[1], p[2], p[3]);
    let n = x_sig.len();
    let n_out = if has_hint { n + 1 } else { n };
    let mut jac = vec![vec![0.0f64; 4]; n_out];
    let inv_s = 1.0 / (s + 1e-9);

    for i in 0..n {
        let z = (x_sig[i] - mu) * inv_s;
        let g = (-0.5 * z * z).exp();
        jac[i][0] = -g;                    // dr/dA
        jac[i][1] = -a * z * inv_s * g;    // dr/dmu
        jac[i][2] = -a * z * z * inv_s * g; // dr/ds
        jac[i][3] = -1.0;                  // dr/db0
    }
    if has_hint {
        let hs = center_hint_sigma_ms.max(1e-6);
        jac[n][1] = 1.0 / hs; // d(prior)/dmu
    }
    jac
}

/// Analytic Jacobian for K=2 Gaussian residual.
///
/// r(p) = y - (A1*g1 + A2*g2 + b0)  where g_k = exp(-0.5*((x-mu_k)/s_k)^2)
/// p = [A1, mu1, s1, A2, sep, s2, b0], mu2 = mu1 + sep
pub fn analytic_jacobian_k2(
    x_sig: &[f64],
    p: &[f64],
    has_hint: bool,
    center_hint_sigma_ms: f64,
) -> Vec<Vec<f64>> {
    let (a1, mu1, s1, a2, sep, s2, _b0) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
    let mu2 = mu1 + sep;
    let n = x_sig.len();
    let n_out = if has_hint { n + 1 } else { n };
    let mut jac = vec![vec![0.0f64; 7]; n_out];
    let inv_s1 = 1.0 / (s1 + 1e-9);
    let inv_s2 = 1.0 / (s2 + 1e-9);

    for i in 0..n {
        let z1 = (x_sig[i] - mu1) * inv_s1;
        let z2 = (x_sig[i] - mu2) * inv_s2;
        let g1 = (-0.5 * z1 * z1).exp();
        let g2 = (-0.5 * z2 * z2).exp();

        jac[i][0] = -g1;                       // dr/dA1
        jac[i][1] = -a1 * z1 * inv_s1 * g1;    // dr/dmu1
        jac[i][2] = -a1 * z1 * z1 * inv_s1 * g1; // dr/ds1
        jac[i][3] = -g2;                       // dr/dA2
        jac[i][4] = -a2 * z2 * inv_s2 * g2;    // dr/dsep (= dr/dmu2)
        jac[i][5] = -a2 * z2 * z2 * inv_s2 * g2; // dr/ds2
        jac[i][6] = -1.0;                      // dr/db0
    }
    if has_hint {
        let hs = center_hint_sigma_ms.max(1e-6);
        jac[n][1] = 1.0 / hs; // d(prior)/dmu1
    }
    jac
}

/// Numerical Jacobian fallback (for general residual functions).
fn numerical_jacobian(
    residual_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    r0: &[f64],
    lb: &[f64],
    ub: &[f64],
) -> Vec<Vec<f64>> {
    let n_params = x.len();
    let n_resid = r0.len();
    let mut jac = vec![vec![0.0f64; n_params]; n_resid];
    let mut x_fwd = x.to_vec();
    let mut x_bwd = x.to_vec();

    for j in 0..n_params {
        let h = 1.4901161193847656e-8 * x[j].abs().max(1.0);
        x_fwd[j] = (x[j] + h).min(ub[j]);
        x_bwd[j] = (x[j] - h).max(lb[j]);
        let actual_h = x_fwd[j] - x_bwd[j];
        if actual_h.abs() < 1e-15 {
            x_fwd[j] = x[j] + h;
            let r_fwd = residual_fn(&x_fwd);
            for i in 0..n_resid {
                jac[i][j] = (r_fwd[i] - r0[i]) / h;
            }
        } else {
            let r_fwd = residual_fn(&x_fwd);
            let r_bwd = residual_fn(&x_bwd);
            for i in 0..n_resid {
                jac[i][j] = (r_fwd[i] - r_bwd[i]) / actual_h;
            }
        }
        x_fwd[j] = x[j];
        x_bwd[j] = x[j];
    }
    jac
}

/// Solve J^T * J * delta = -J^T * r using Gauss elimination.
/// Uses flat array [n_params * n_params] for cache-friendly access.
fn solve_normal_equations(
    jac: &[Vec<f64>],
    residuals: &[f64],
    lambda: f64,
    n_params: usize,
) -> Vec<f64> {
    let n_resid = residuals.len();
    let np = n_params;

    // Build J^T * J + lambda * diag in flat row-major layout
    let mut jtj = vec![0.0f64; np * np];
    let mut jtr = vec![0.0f64; np];

    for j in 0..np {
        for k in j..np {
            let mut sum = 0.0;
            for i in 0..n_resid {
                sum += jac[i][j] * jac[i][k];
            }
            jtj[j * np + k] = sum;
            jtj[k * np + j] = sum;
        }
        let mut sum = 0.0;
        for i in 0..n_resid {
            sum += jac[i][j] * residuals[i];
        }
        jtr[j] = sum;
        jtj[j * np + j] += lambda * jtj[j * np + j].max(1e-12);
    }

    // Gaussian elimination with partial pivoting on flat array
    let mut b: Vec<f64> = jtr.iter().map(|v| -v).collect();

    for col in 0..np {
        let mut max_row = col;
        let mut max_val = jtj[col * np + col].abs();
        for row in (col + 1)..np {
            let v = jtj[row * np + col].abs();
            if v > max_val { max_val = v; max_row = row; }
        }
        if max_val < 1e-15 { continue; }
        if max_row != col {
            for k in 0..np { jtj.swap(col * np + k, max_row * np + k); }
            b.swap(col, max_row);
        }
        let pivot = jtj[col * np + col];
        for row in (col + 1)..np {
            let factor = jtj[row * np + col] / pivot;
            for k in col..np {
                jtj[row * np + k] -= factor * jtj[col * np + k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitute
    let mut delta = vec![0.0f64; np];
    for col in (0..np).rev() {
        if jtj[col * np + col].abs() < 1e-15 { continue; }
        let mut sum = b[col];
        for k in (col + 1)..np {
            sum -= jtj[col * np + k] * delta[k];
        }
        delta[col] = sum / jtj[col * np + col];
    }
    delta
}

/// Clamp parameters to bounds.
fn clamp_to_bounds(x: &mut [f64], lb: &[f64], ub: &[f64]) {
    for i in 0..x.len() {
        x[i] = x[i].max(lb[i]).min(ub[i]);
    }
}

/// Sum of squares: 0.5 * sum(r^2)
fn sum_sq(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum::<f64>() * 0.5
}

/// Apply Huber loss scaling to residuals.
/// Matches scipy's loss="huber" with f_scale parameter.
/// Huber: rho(z) = z if z <= 1, else 2*sqrt(z) - 1
fn huber_scale_residuals(residuals: &[f64], f_scale: f64) -> Vec<f64> {
    residuals
        .iter()
        .map(|&r| {
            let z = (r / f_scale).abs();
            if z <= 1.0 {
                r // linear region
            } else {
                // sqrt(2*z - 1) * f_scale * sign(r)
                f_scale * (2.0 * z - 1.0).sqrt() * r.signum()
            }
        })
        .collect()
}

/// Core Levenberg-Marquardt solver with bounds, optional Huber loss, optional analytic Jacobian.
///
/// Replaces: scipy.optimize.least_squares(residual, x0, bounds, loss="huber", f_scale=1.5)
pub fn solve_lm(
    residual_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    jacobian_fn: Option<&dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    max_nfev: usize,
    use_huber: bool,
    f_scale: f64,
    ftol: f64,
    xtol: f64,
) -> LsResult {
    let n_params = x0.len();
    let mut x = x0.to_vec();
    clamp_to_bounds(&mut x, lb, ub);

    let mut r = residual_fn(&x);
    let mut n_fev = 1;

    // Working residuals: apply Huber scaling in-place if needed
    let mut r_work = if use_huber {
        huber_scale_residuals(&r, f_scale)
    } else {
        r.clone()
    };
    let mut cost = sum_sq(&r_work);

    let mut lambda = 1e-4;
    let nu = 3.0;
    let mut x_new = vec![0.0f64; n_params]; // reusable buffer

    for _iter in 0..200 {
        if n_fev >= max_nfev { break; }

        // Compute Jacobian: analytic (fast) or numerical (fallback)
        let jac = if let Some(jac_f) = jacobian_fn {
            jac_f(&x)
        } else {
            let jac_fn = |p: &[f64]| -> Vec<f64> {
                let raw = residual_fn(p);
                if use_huber { huber_scale_residuals(&raw, f_scale) } else { raw }
            };
            n_fev += 2 * n_params;
            numerical_jacobian(&jac_fn, &x, &r_work, lb, ub)
        };

        let delta = solve_normal_equations(&jac, &r_work, lambda, n_params);
        if delta.iter().any(|d| d.is_nan() || d.is_infinite()) { break; }

        // Trial step — clamp to bounds
        for i in 0..n_params { x_new[i] = x[i] + delta[i]; }
        clamp_to_bounds(&mut x_new, lb, ub);

        let r_new_raw = residual_fn(&x_new);
        n_fev += 1;
        let r_new_work = if use_huber {
            huber_scale_residuals(&r_new_raw, f_scale)
        } else {
            r_new_raw.clone()
        };
        let cost_new = sum_sq(&r_new_work);

        if cost_new < cost {
            let cost_change = (cost - cost_new) / cost.max(1e-15);
            x.copy_from_slice(&x_new);
            r = r_new_raw;
            r_work = r_new_work;
            cost = cost_new;
            lambda = (lambda * 0.33).max(1e-10);

            if cost_change < ftol { break; }
            let x_change: f64 = delta.iter().zip(x.iter())
                .map(|(d, xi)| (d / xi.abs().max(1e-10)).abs())
                .fold(0.0f64, f64::max);
            if x_change < xtol { break; }
        } else {
            lambda = (lambda * nu).min(1e10);
        }
    }

    LsResult {
        params: x,
        cost,
        n_fev,
        success: true,
    }
}

// ===========================================================================
// Trust Region Reflective (TRF) Solver
// Paired 1:1 from scipy.optimize._lsq.trf (trf_bounds) + common.py
// ===========================================================================

const EPS: f64 = 2.220446049250313e-16; // f64::EPSILON

// --- Phase 1: Math helpers (common.py ports) ---

/// Coleman-Li scaling vector. Ref: common.py:467-508.
/// Returns (v, dv) where v scales trust region by distance to bounds.
fn cl_scaling_vector(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut v = vec![1.0; n];
    let mut dv = vec![0.0; n];
    for i in 0..n {
        if g[i] < 0.0 && ub[i] < f64::INFINITY {
            v[i] = (ub[i] - x[i]).max(0.0);
            dv[i] = -1.0;
        } else if g[i] > 0.0 && lb[i] > f64::NEG_INFINITY {
            v[i] = (x[i] - lb[i]).max(0.0);
            dv[i] = 1.0;
        }
    }
    (v, dv)
}

/// Push x strictly inside bounds. Ref: common.py:440-464.
/// rstep=1e-10 for initial feasibility; rstep=0.0 for inner-loop (nextafter precision).
fn make_strictly_feasible(x: &[f64], lb: &[f64], ub: &[f64], rstep: f64) -> Vec<f64> {
    let mut out = x.to_vec();
    for i in 0..x.len() {
        if rstep == 0.0 {
            // Machine-epsilon precision: equivalent to np.nextafter(bound, interior)
            if lb[i].is_finite() && out[i] <= lb[i] {
                let bits = lb[i].to_bits();
                out[i] = if lb[i] >= 0.0 { f64::from_bits(bits + 1) } else { f64::from_bits(bits - 1) };
            }
            if ub[i].is_finite() && out[i] >= ub[i] {
                let bits = ub[i].to_bits();
                out[i] = if ub[i] > 0.0 { f64::from_bits(bits - 1) } else { f64::from_bits(bits + 1) };
            }
        } else {
            if lb[i].is_finite() && out[i] <= lb[i] {
                out[i] = lb[i] + rstep * (1.0f64).max(lb[i].abs());
            }
            if ub[i].is_finite() && out[i] >= ub[i] {
                out[i] = ub[i] - rstep * (1.0f64).max(ub[i].abs());
            }
        }
        // If bounds too tight, midpoint
        if out[i] < lb[i] || out[i] > ub[i] {
            out[i] = 0.5 * (lb[i] + ub[i]);
        }
    }
    out
}

/// Max scalar t such that lb <= x + t*s <= ub. Ref: common.py:372-398.
/// Returns (t_max, hits) where hits[i] = sign of s[i] if bound i hit at t_max.
fn step_size_to_bound(x: &[f64], s: &[f64], lb: &[f64], ub: &[f64]) -> (f64, Vec<i8>) {
    let n = x.len();
    let mut steps = vec![f64::INFINITY; n];
    for i in 0..n {
        if s[i].abs() < 1e-300 { continue; }
        let t_lb = (lb[i] - x[i]) / s[i];
        let t_ub = (ub[i] - x[i]) / s[i];
        steps[i] = t_lb.max(t_ub); // whichever is positive
        if steps[i] < 0.0 { steps[i] = f64::INFINITY; }
    }
    let min_step = steps.iter().copied().fold(f64::INFINITY, f64::min);
    let mut hits = vec![0i8; n];
    for i in 0..n {
        if (steps[i] - min_step).abs() < 1e-15 * min_step.max(1.0) {
            hits[i] = if s[i] > 0.0 { 1 } else if s[i] < 0.0 { -1 } else { 0 };
        }
    }
    (min_step, hits)
}

/// Check if x is within bounds. Ref: common.py:367-369.
fn in_bounds(x: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    x.iter().zip(lb.iter().zip(ub.iter())).all(|(&xi, (&li, &ui))| xi >= li && xi <= ui)
}

/// Intersect ||p + t*d||^2 = delta^2. Returns (t_neg, t_pos).
fn intersect_trust_region(p: &[f64], d: &[f64], delta: f64) -> (f64, f64) {
    let a: f64 = d.iter().map(|di| di * di).sum();
    let b: f64 = p.iter().zip(d.iter()).map(|(pi, di)| pi * di).sum::<f64>() * 2.0;
    let c: f64 = p.iter().map(|pi| pi * pi).sum::<f64>() - delta * delta;
    let disc = (b * b - 4.0 * a * c).max(0.0).sqrt();
    if a < 1e-300 { return (0.0, 0.0); }
    let t1 = (-b - disc) / (2.0 * a);
    let t2 = (-b + disc) / (2.0 * a);
    (t1, t2)
}

/// Build 1D quadratic q(t) = a*t^2 + b*t [+ c]. Ref: common.py:251-299.
/// J_h is row-major (m rows, n cols), stored as flat Vec of length m*n.
fn build_quadratic_1d(
    j_h_flat: &[f64], m: usize, n: usize,
    g_h: &[f64], s: &[f64], diag: &[f64],
    s0: Option<&[f64]>,
) -> (f64, f64, f64) {
    // v = J_h @ s
    let mut v = vec![0.0; m];
    for r in 0..m {
        for c in 0..n { v[r] += j_h_flat[r * n + c] * s[c]; }
    }
    let mut a = 0.5 * v.iter().map(|vi| vi * vi).sum::<f64>();
    a += 0.5 * (0..n).map(|i| s[i] * diag[i] * s[i]).sum::<f64>();

    let mut b: f64 = g_h.iter().zip(s.iter()).map(|(gi, si)| gi * si).sum();

    let mut c_val = 0.0;
    if let Some(s0v) = s0 {
        // u = J_h @ s0
        let mut u = vec![0.0; m];
        for r in 0..m {
            for c in 0..n { u[r] += j_h_flat[r * n + c] * s0v[c]; }
        }
        b += u.iter().zip(v.iter()).map(|(ui, vi)| ui * vi).sum::<f64>();
        c_val = 0.5 * u.iter().map(|ui| ui * ui).sum::<f64>();
        c_val += g_h.iter().zip(s0v.iter()).map(|(gi, si)| gi * si).sum::<f64>();
        b += (0..n).map(|i| s0v[i] * diag[i] * s[i]).sum::<f64>();
        c_val += 0.5 * (0..n).map(|i| s0v[i] * diag[i] * s0v[i]).sum::<f64>();
    }
    (a, b, c_val)
}

/// Minimize a*t^2 + b*t + c on [lo, hi]. Ref: common.py:302-322.
fn minimize_quadratic_1d(a: f64, b: f64, lo: f64, hi: f64, c: f64) -> (f64, f64) {
    let eval = |t: f64| t * (a * t + b) + c;
    let mut best_t = lo;
    let mut best_v = eval(lo);
    let v_hi = eval(hi);
    if v_hi < best_v { best_t = hi; best_v = v_hi; }
    if a.abs() > 1e-300 {
        let ext = -0.5 * b / a;
        if ext > lo && ext < hi {
            let v_ext = eval(ext);
            if v_ext < best_v { best_t = ext; best_v = v_ext; }
        }
    }
    (best_t, best_v)
}

/// Evaluate 0.5*(||J_h*s||^2 + s^T*diag*s) + g_h^T*s. Ref: common.py:325-361.
fn evaluate_quadratic(
    j_h_flat: &[f64], m: usize, n: usize,
    g_h: &[f64], s: &[f64], diag: &[f64],
) -> f64 {
    let mut js = vec![0.0; m];
    for r in 0..m { for c in 0..n { js[r] += j_h_flat[r * n + c] * s[c]; } }
    let q = js.iter().map(|v| v * v).sum::<f64>()
        + (0..n).map(|i| s[i] * diag[i] * s[i]).sum::<f64>();
    let l: f64 = g_h.iter().zip(s.iter()).map(|(gi, si)| gi * si).sum();
    0.5 * q + l
}

/// Update trust region radius. Ref: common.py:222-245.
fn update_tr_radius(delta: f64, actual: f64, predicted: f64, step_norm: f64, bound_hit: bool) -> (f64, f64) {
    let ratio = if predicted > 0.0 {
        actual / predicted
    } else if predicted == 0.0 && actual == 0.0 {
        1.0
    } else {
        0.0
    };
    let delta_new = if ratio < 0.25 {
        0.25 * step_norm
    } else if ratio > 0.75 && bound_hit {
        delta * 2.0
    } else {
        delta
    };
    (delta_new, ratio)
}

// --- Phase 2: Trust region subproblem (SVD-based) ---

/// phi(alpha) and its derivative for the TR subproblem. Ref: common.py:106-116.
fn phi_and_derivative(alpha: f64, suf: &[f64], s: &[f64], delta: f64) -> (f64, f64) {
    let n = s.len();
    let mut p_norm_sq = 0.0;
    let mut phi_prime_num = 0.0;
    for i in 0..n {
        let denom = s[i] * s[i] + alpha;
        if denom.abs() < 1e-300 { continue; }
        let ratio = suf[i] / denom;
        p_norm_sq += ratio * ratio;
        phi_prime_num += suf[i] * suf[i] / (denom * denom * denom);
    }
    let p_norm = p_norm_sq.sqrt();
    let phi = p_norm - delta;
    let phi_prime = if p_norm > 1e-300 { -phi_prime_num / p_norm } else { 0.0 };
    (phi, phi_prime)
}

/// Solve the trust region least-squares subproblem via SVD. Ref: common.py:57-168.
/// Input: n params, m residuals, uf = U^T*f, s = singular values, V = right singular vectors.
/// Returns (step_p, alpha_out, n_iterations).
fn solve_lsq_trust_region(
    n: usize, _m: usize,
    uf: &[f64], s: &[f64], v_mat: &[f64], // v_mat is n×n row-major
    delta: f64, initial_alpha: f64,
) -> (Vec<f64>, f64, usize) {
    let k = s.len().min(n); // number of singular values
    let suf: Vec<f64> = (0..k).map(|i| s[i] * uf[i]).collect();

    // Check full rank
    let threshold = EPS * (_m.max(n) as f64) * s.get(0).copied().unwrap_or(0.0);
    let full_rank = k >= n && s.last().copied().unwrap_or(0.0) > threshold;

    // Try Gauss-Newton step (alpha = 0)
    if full_rank {
        let mut p = vec![0.0; n];
        for i in 0..n {
            for j in 0..k {
                if s[j].abs() > 1e-300 {
                    p[i] -= v_mat[i * n + j] * uf[j] / s[j]; // V is n×n, cols are right singular vectors
                }
            }
        }
        let p_norm: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        if p_norm <= delta {
            return (p, 0.0, 0);
        }
        // S1: Normalize p to trust region boundary (scipy common.py:165)
        if p_norm > 1e-15 {
            let scale = delta / p_norm;
            for v in &mut p { *v *= scale; }
        }
    }

    // Compute bounds on alpha
    let suf_norm: f64 = suf.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut alpha_upper = if delta > 1e-300 { suf_norm / delta } else { 1.0 };
    let mut alpha_lower = if full_rank {
        let (phi0, phi_prime0) = phi_and_derivative(0.0, &suf, s, delta);
        (-phi0 / phi_prime0).max(0.0)
    } else {
        0.0
    };

    // Initialize alpha
    let mut alpha = if initial_alpha > 0.0 && initial_alpha > alpha_lower && initial_alpha < alpha_upper {
        initial_alpha
    } else {
        (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt())
    };

    // Newton iterations
    let max_iter = 10;
    let rtol = 0.01;
    for it in 0..max_iter {
        if alpha < alpha_lower || alpha > alpha_upper {
            alpha = (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt());
        }
        let (phi, phi_prime) = phi_and_derivative(alpha, &suf, s, delta);
        if phi.abs() < rtol * delta {
            // Converged — compute step
            let mut p = vec![0.0; n];
            for i in 0..n {
                for j in 0..k {
                    let denom = s[j] * s[j] + alpha;
                    if denom.abs() > 1e-300 {
                        p[i] -= v_mat[i * n + j] * suf[j] / denom;
                    }
                }
            }
            // S1: Normalize p to trust region boundary
            let p_norm: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            if p_norm > delta && p_norm > 1e-15 {
                let sc = delta / p_norm;
                for v in &mut p { *v *= sc; }
            }
            return (p, alpha, it + 1);
        }
        // S3: Tighten alpha bounds (scipy common.py:151-154)
        if phi < 0.0 {
            alpha_upper = alpha;
        }
        let ratio = phi / phi_prime;
        alpha_lower = alpha_lower.max(alpha - ratio);
        alpha -= (phi + delta) * ratio / delta;
    }

    // Max iterations — return best effort
    let mut p = vec![0.0; n];
    for i in 0..n {
        for j in 0..k {
            let denom = s[j] * s[j] + alpha;
            if denom.abs() > 1e-300 {
                p[i] -= v_mat[i * n + j] * suf[j] / denom;
            }
        }
    }
    // S1: Normalize p to trust region boundary
    let p_norm: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    if p_norm > delta && p_norm > 1e-15 {
        let sc = delta / p_norm;
        for v in &mut p { *v *= sc; }
    }
    (p, alpha, max_iter)
}

// --- Phase 3: Step selection ---

/// Select best step from TR, reflected, Cauchy candidates. Ref: trf.py:129-203.
fn select_step(
    x: &[f64], j_h_flat: &[f64], m: usize, n: usize,
    diag_h: &[f64], g_h: &[f64],
    p: &[f64], p_h: &[f64], d: &[f64],
    delta: f64, lb: &[f64], ub: &[f64], theta: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    // Original-space step
    let p_orig: Vec<f64> = (0..n).map(|i| d[i] * p_h[i]).collect();

    // Candidate 1: Constrained TR step
    let x_try: Vec<f64> = x.iter().zip(p_orig.iter()).map(|(xi, pi)| xi + pi).collect();
    if in_bounds(&x_try, lb, ub) {
        let val = evaluate_quadratic(j_h_flat, m, n, g_h, p_h, diag_h);
        return (p_orig, p_h.to_vec(), -val);
    }

    // Constrain p to stay within bounds (with theta step-back)
    let (p_stride, hits) = step_size_to_bound(x, &p_orig, lb, ub);
    let p_stride = p_stride.min(1.0);

    // Build reflected direction in hat space
    let mut r_h = p_h.to_vec();
    for i in 0..n { if hits[i] != 0 { r_h[i] = -r_h[i]; } }
    let r: Vec<f64> = (0..n).map(|i| d[i] * r_h[i]).collect();

    // Constrained p
    let p_constrained: Vec<f64> = p_orig.iter().map(|v| v * p_stride * theta).collect();
    let p_h_constrained: Vec<f64> = p_h.iter().map(|v| v * p_stride * theta).collect();
    let p_value = evaluate_quadratic(j_h_flat, m, n, g_h, &p_h_constrained, diag_h);

    // Reflected step
    let x_on_bound: Vec<f64> = x.iter().zip(p_orig.iter())
        .map(|(xi, pi)| xi + pi * p_stride).collect();
    let (_, to_tr) = intersect_trust_region(
        &p_h.iter().map(|v| v * p_stride).collect::<Vec<_>>(),
        &r_h, delta);
    let (to_bound, _) = step_size_to_bound(&x_on_bound, &r, lb, ub);
    let r_stride_max = to_bound.min(to_tr).max(0.0);

    let r_value;
    let r_step_h;
    if r_stride_max > 0.0 {
        let p_h_at_bound: Vec<f64> = p_h.iter().map(|v| v * p_stride).collect();
        let (a, b, c) = build_quadratic_1d(j_h_flat, m, n, g_h, &r_h, diag_h, Some(&p_h_at_bound));
        let r_stride_lo = 0.0;
        let (r_stride, rv) = minimize_quadratic_1d(a, b, r_stride_lo, r_stride_max * theta, c);
        r_value = rv;
        r_step_h = (0..n).map(|i| p_h_at_bound[i] + r_stride * r_h[i]).collect::<Vec<_>>();
    } else {
        r_value = f64::INFINITY;
        r_step_h = p_h_constrained.clone();
    }

    // Cauchy step (steepest descent)
    let ag_h: Vec<f64> = g_h.iter().map(|v| -v).collect();
    let ag: Vec<f64> = (0..n).map(|i| d[i] * ag_h[i]).collect();
    let ag_h_norm: f64 = ag_h.iter().map(|v| v * v).sum::<f64>().sqrt();
    let to_tr_ag = if ag_h_norm > 1e-300 { delta / ag_h_norm } else { 0.0 };
    let (to_bound_ag, _) = step_size_to_bound(x, &ag, lb, ub);
    let ag_stride_max = if to_bound_ag < to_tr_ag { theta * to_bound_ag } else { to_tr_ag };

    let (a_ag, b_ag) = {
        let (a, b, _) = build_quadratic_1d(j_h_flat, m, n, g_h, &ag_h, diag_h, None);
        (a, b)
    };
    let (ag_stride, ag_value) = minimize_quadratic_1d(a_ag, b_ag, 0.0, ag_stride_max, 0.0);
    let ag_step_h: Vec<f64> = ag_h.iter().map(|v| v * ag_stride).collect();

    // Select best
    if p_value <= r_value && p_value <= ag_value {
        (p_constrained, p_h_constrained, -p_value)
    } else if r_value <= p_value && r_value <= ag_value {
        let r_step: Vec<f64> = (0..n).map(|i| d[i] * r_step_h[i]).collect();
        (r_step, r_step_h, -r_value)
    } else {
        let ag_step: Vec<f64> = ag.iter().map(|v| v * ag_stride).collect();
        (ag_step, ag_step_h, -ag_value)
    }
}

// --- Phase 4: Huber loss scaling ---

/// Compute Huber loss rho values: [rho, rho', rho''] for each residual.
fn compute_huber_rho(f: &[f64], f_scale: f64) -> Vec<[f64; 3]> {
    f.iter().map(|&fi| {
        let z = (fi / f_scale) * (fi / f_scale);
        if z <= 1.0 {
            [z, 1.0, 0.0]
        } else {
            [2.0 * z.sqrt() - 1.0, z.powf(-0.5), -0.5 * z.powf(-1.5)]
        }
    }).collect()
}

/// Scale J and f for robust loss. Ref: common.py:720-731.
/// Modifies j_flat (m×n row-major) and f in place.
fn scale_for_robust_loss(
    j_flat: &mut [f64], f: &mut [f64], m: usize, n: usize,
    rho: &[[f64; 3]], f_scale: f64,
) {
    for i in 0..m {
        let mut js = rho[i][1] + 2.0 * rho[i][2] * f[i] * f[i];
        if js < EPS { js = EPS; }
        let js_sqrt = js.sqrt();
        f[i] = f[i] * rho[i][1] / js_sqrt;
        for j in 0..n {
            j_flat[i * n + j] *= js_sqrt;
        }
    }
}

/// Compute cost under Huber loss.
fn huber_cost(f: &[f64], f_scale: f64) -> f64 {
    let rho = compute_huber_rho(f, f_scale);
    0.5 * f_scale * f_scale * rho.iter().map(|r| r[0]).sum::<f64>()
}

// --- Diagnostic logging for solver comparison ---

use std::sync::atomic::{AtomicBool, Ordering};
static SOLVER_DIAG: AtomicBool = AtomicBool::new(false);

/// Enable/disable per-iteration diagnostic output to stderr.
pub fn set_solver_diagnostics(enabled: bool) {
    SOLVER_DIAG.store(enabled, Ordering::Relaxed);
}

macro_rules! diag {
    ($($arg:tt)*) => {
        if SOLVER_DIAG.load(Ordering::Relaxed) {
            eprintln!($($arg)*);
        }
    };
}

// --- Phase 5: Main TRF solver ---

/// Trust Region Reflective solver — paired from scipy.optimize._lsq.trf.trf_bounds.
///
/// Drop-in replacement for solve_lm with identical interface plus gtol.
pub fn solve_trf(
    residual_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    jacobian_fn: Option<&dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    max_nfev: usize,
    use_huber: bool,
    f_scale: f64,
    ftol: f64,
    xtol: f64,
) -> LsResult {
    let gtol = 1e-8;
    let n = x0.len();
    let mut x = make_strictly_feasible(x0, lb, ub, 1e-10);

    // Initial evaluation
    let f_raw = residual_fn(&x);
    let m = f_raw.len();
    let mut n_fev: usize = 1;

    // Compute initial Jacobian (flat row-major: m×n)
    let jac_vecs = if let Some(jf) = jacobian_fn {
        jf(&x)
    } else {
        let jf = |p: &[f64]| residual_fn(p);
        numerical_jacobian(&jf, &x, &f_raw, lb, ub)
    };
    let mut j_flat: Vec<f64> = Vec::with_capacity(m * n);
    for row in &jac_vecs { j_flat.extend_from_slice(row); }
    let mut n_jev: usize = 1;

    // Apply Huber loss scaling
    let mut f_work = f_raw.clone();
    let mut cost = if use_huber {
        let cost_val = huber_cost(&f_raw, f_scale);
        let rho = compute_huber_rho(&f_raw, f_scale);
        scale_for_robust_loss(&mut j_flat, &mut f_work, m, n, &rho, f_scale);
        cost_val
    } else {
        0.5 * f_raw.iter().map(|v| v * v).sum::<f64>()
    };

    // Gradient: g = J^T * f_work
    let mut g = vec![0.0; n];
    for j in 0..n {
        for i in 0..m { g[j] += j_flat[i * n + j] * f_work[i]; }
    }

    // CL scaling
    let (mut v, mut dv) = cl_scaling_vector(&x, &g, lb, ub);
    let mut d: Vec<f64> = v.iter().map(|vi| vi.sqrt()).collect();

    // Initial trust region radius: norm(x / sqrt(v))
    // Matches scipy trf.py:30: Delta = norm(x0 * scale_inv / v**0.5)
    // scale_inv = 1.0 (x_scale default), so Delta = norm(x0 / sqrt(v))
    let mut delta = {
        let scaled_norm: f64 = x.iter().zip(v.iter())
            .map(|(xi, vi)| {
                let s = xi / vi.max(1e-30).sqrt();
                s * s
            })
            .sum::<f64>()
            .sqrt();
        if scaled_norm > 0.0 { scaled_norm } else { 1.0 }
    };
    let mut alpha = 0.0;

    diag!("[RUST TRF] init: n={} m={} cost={:.10e} delta={:.10e} x={:?}", n, m, cost, delta,
          &x.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>());
    diag!("[RUST TRF] init: g={:?}", &g.iter().map(|v| format!("{:.6e}", v)).collect::<Vec<_>>());
    diag!("[RUST TRF] init: v={:?} dv={:?}", &v, &dv);
    diag!("[RUST TRF] lapack_available={}", crate::lapack_svd::is_available());

    let mut step_norm: f64 = 0.0;
    let mut actual_reduction: f64 = 0.0;

    for _iteration in 0..200 {
        if n_fev >= max_nfev { break; }

        // Recompute CL scaling
        let (v_new, dv_new) = cl_scaling_vector(&x, &g, lb, ub);
        v = v_new; dv = dv_new;

        // Check gtol
        let g_norm: f64 = g.iter().zip(v.iter()).map(|(gi, vi)| (gi * vi).abs()).fold(0.0f64, f64::max);
        diag!("[RUST TRF] iter={} cost={:.10e} delta={:.10e} g_norm={:.10e} alpha={:.10e} step_norm={:.10e} actual_red={:.10e}",
              _iteration, cost, delta, g_norm, alpha, step_norm, actual_reduction);
        if g_norm < gtol {
            diag!("[RUST TRF] converged: gtol (g_norm={:.10e} < {:.10e})", g_norm, gtol);
            break;
        }

        // d = sqrt(v)
        d = v.iter().map(|vi| vi.max(1e-30).sqrt()).collect();

        // theta
        let theta = (0.995f64).max(1.0 - g_norm);

        // Hat-space quantities
        let g_h: Vec<f64> = (0..n).map(|i| d[i] * g[i]).collect();
        // J_h = J * diag(d): multiply each column j by d[j]
        let mut j_h_flat = j_flat.clone();
        for r in 0..m { for c in 0..n { j_h_flat[r * n + c] *= d[c]; } }
        // diag_h = g * dv (CL second-order diagonal)
        let diag_h: Vec<f64> = (0..n).map(|i| g[i] * dv[i]).collect();

        // Build augmented system: [J_h; diag(sqrt(|diag_h|))]
        // Augmented is (m + n_aug) × n
        let mut n_aug = 0;
        let mut aug_indices = Vec::new();
        for i in 0..n {
            if diag_h[i].abs() > 1e-30 {
                aug_indices.push(i);
                n_aug += 1;
            }
        }
        let m_aug = m + n_aug;
        let mut j_aug_flat = vec![0.0; m_aug * n];
        j_aug_flat[..m * n].copy_from_slice(&j_h_flat);
        for (row_idx, &col_idx) in aug_indices.iter().enumerate() {
            j_aug_flat[(m + row_idx) * n + col_idx] = diag_h[col_idx].abs().sqrt();
        }

        // SVD via LAPACK dgesdd (or nalgebra fallback)
        let svd_result = crate::lapack_svd::svd_decompose(&j_aug_flat, m_aug, n);
        let s_vals = &svd_result.s;
        let k = svd_result.k;

        diag!("[RUST TRF] iter={} svd: s={:?}", _iteration,
              &s_vals.iter().map(|v| format!("{:.10e}", v)).collect::<Vec<_>>());

        // V matrix as flat row-major n×n
        // V[i,j] = V^T[j,i]; vt_col is column-major k×n: vt_col[j*k + i] = V^T[i,j]
        // So V[i,j] = V^T[j,i] = vt_col[i*k + j]
        let mut v_flat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..k.min(n) {
                v_flat[i * n + j] = svd_result.vt_col[i * k + j];
            }
        }

        // f_augmented = [f_work; zeros]
        let mut f_aug = vec![0.0; m_aug];
        f_aug[..m].copy_from_slice(&f_work);

        // uf = U^T @ f_aug
        // u_col is column-major m_aug×k: u_col[j*m_aug + i] = U[i,j]
        let mut uf = vec![0.0; k];
        for j in 0..k {
            for i in 0..m_aug {
                uf[j] += svd_result.u_col[j * m_aug + i] * f_aug[i];
            }
        }

        // S2: Inner retry loop — retry with shrunk delta when reduction <= 0
        // (scipy trf.py: inner while actual_reduction <= 0 loop)
        let mut x_new = x.clone();
        let mut f_new_raw = vec![0.0; m];
        let mut cost_new = cost;
        let mut step = vec![0.0; n];
        let mut step_h_vec = vec![0.0; n];
        let mut predicted_reduction = 0.0;
        let mut ratio = 0.0;

        loop {
            // Solve TR subproblem
            let (p_h, alpha_new, _) = solve_lsq_trust_region(n, m_aug, &uf, &s_vals, &v_flat, delta, alpha);
            alpha = alpha_new;

            // Step selection
            let p_orig: Vec<f64> = (0..n).map(|i| d[i] * p_h[i]).collect();
            let (step_cand, step_h_cand, pred_red) = select_step(
                &x, &j_h_flat, m, n, &diag_h, &g_h,
                &p_orig, &p_h, &d, delta, lb, ub, theta,
            );
            predicted_reduction = pred_red;

            // Evaluate candidate
            x_new = make_strictly_feasible(
                &x.iter().zip(step_cand.iter()).map(|(xi, si)| xi + si).collect::<Vec<_>>(),
                lb, ub, 0.0,  // rstep=0: nextafter precision (matches scipy inner loop)
            );
            f_new_raw = residual_fn(&x_new);
            n_fev += 1;

            // Check for NaN
            if f_new_raw.iter().any(|v| v.is_nan() || v.is_infinite()) {
                let sh_norm: f64 = step_h_cand.iter().map(|v| v * v).sum::<f64>().sqrt();
                delta = 0.25 * sh_norm;
                if n_fev >= max_nfev { step = step_cand; step_h_vec = step_h_cand; break; }
                continue;
            }

            cost_new = if use_huber {
                huber_cost(&f_new_raw, f_scale)
            } else {
                0.5 * f_new_raw.iter().map(|v| v * v).sum::<f64>()
            };

            actual_reduction = cost - cost_new;
            let step_h_norm: f64 = step_h_cand.iter().map(|v| v * v).sum::<f64>().sqrt();

            let (delta_new, r) = update_tr_radius(
                delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * delta,
            );
            ratio = r;

            alpha *= delta / delta_new.max(1e-30);
            delta = delta_new;

            step = step_cand;
            step_h_vec = step_h_cand;

            // S2: If reduction positive or budget exhausted, break inner loop
            if actual_reduction > 0.0 || n_fev >= max_nfev {
                break;
            }
            // Otherwise retry with shrunk delta (same Jacobian)
        }

        step_norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Check convergence
        if ratio > 0.25 {
            let df = actual_reduction / cost.max(1e-15);
            let dx = step_norm / (xtol + x.iter().map(|v| v * v).sum::<f64>().sqrt());
            if df < ftol && dx < xtol { break; }
        }

        // Accept step if reduction is positive
        if actual_reduction > 0.0 {
            diag!("[RUST TRF] iter={} ACCEPT: x={:?}", _iteration,
                  &x_new.iter().map(|v| format!("{:.8}", v)).collect::<Vec<_>>());
            x = x_new;

            // Recompute J
            let jac_vecs_new = if let Some(jf) = jacobian_fn {
                jf(&x)
            } else {
                let f_new_work = if use_huber {
                    let rho = compute_huber_rho(&f_new_raw, f_scale);
                    let mut fw = f_new_raw.clone();
                    let mut jf = vec![0.0; m * n]; // placeholder
                    scale_for_robust_loss(&mut jf, &mut fw, m, n, &rho, f_scale);
                    fw
                } else {
                    f_new_raw.clone()
                };
                numerical_jacobian(&|p: &[f64]| residual_fn(p), &x, &f_new_work, lb, ub)
            };
            j_flat.clear();
            for row in &jac_vecs_new { j_flat.extend_from_slice(row); }
            n_jev += 1;
            n_fev += 2 * n; // numerical jacobian evals

            // Apply Huber scaling
            f_work = f_new_raw.clone();
            cost = cost_new;
            if use_huber {
                let rho = compute_huber_rho(&f_work, f_scale);
                scale_for_robust_loss(&mut j_flat, &mut f_work, m, n, &rho, f_scale);
            }

            // Recompute gradient
            g = vec![0.0; n];
            for j in 0..n {
                for i in 0..m { g[j] += j_flat[i * n + j] * f_work[i]; }
            }
        }
    }

    LsResult {
        params: x,
        cost,
        n_fev,
        success: true,
    }
}

// ---------------------------------------------------------------------------
// High-level wave fitting functions (expose to Python)
// ---------------------------------------------------------------------------

/// Fit K=1 wave model: A*gauss(x, mu, s) + b0, with optional center prior.
///
/// Replaces: scipy.optimize.least_squares(r1, x0, bounds, loss="huber", f_scale=1.5)
/// from wave_fitters.py fit_wave_components_core K=1 block.
///
/// Returns: [A1, mu1, s1, b0, cost]
#[pyfunction]
pub fn fit_k1_lm<'py>(
    py: Python<'py>,
    x_signal: PyReadonlyArray1<'py, f64>,
    y_signal: PyReadonlyArray1<'py, f64>,
    x0: PyReadonlyArray1<'py, f64>,
    lower_bounds: PyReadonlyArray1<'py, f64>,
    upper_bounds: PyReadonlyArray1<'py, f64>,
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_sig = x_signal.as_slice()?.to_vec();
    let y_sig = y_signal.as_slice()?.to_vec();
    let x0_v = x0.as_slice()?;
    let lb = lower_bounds.as_slice()?;
    let ub = upper_bounds.as_slice()?;

    let x_sig_r = x_sig.clone();
    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        gauss_k1_residual_kernel(&x_sig, &y_sig, p, center_hint_ms, center_hint_sigma_ms, has_hint)
    };
    let jacobian_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_k1(&x_sig_r, p, has_hint, center_hint_sigma_ms)
    };

    let result = solve_trf(
        &residual_fn, Some(&jacobian_fn), x0_v, lb, ub,
        2000, true, 1.5, 1e-8, 1e-8,
    );

    // Return [A1, mu1, s1, b0, cost]
    let mut out = result.params;
    out.push(result.cost);
    Ok(PyArray1::from_vec(py, out))
}

/// Fit K=2 wave model: A1*gauss1 + A2*gauss2 + b0, with optional center prior.
///
/// Replaces: scipy.optimize.least_squares(r2, x0, bounds, loss="huber", f_scale=1.5)
/// from wave_fitters.py fit_wave_components_core K=2 block.
///
/// Returns: [A1, mu1, s1, A2, sep, s2, b0, cost]
#[pyfunction]
pub fn fit_k2_lm<'py>(
    py: Python<'py>,
    x_signal: PyReadonlyArray1<'py, f64>,
    y_signal: PyReadonlyArray1<'py, f64>,
    x0: PyReadonlyArray1<'py, f64>,
    lower_bounds: PyReadonlyArray1<'py, f64>,
    upper_bounds: PyReadonlyArray1<'py, f64>,
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_sig = x_signal.as_slice()?.to_vec();
    let y_sig = y_signal.as_slice()?.to_vec();
    let x0_v = x0.as_slice()?;
    let lb = lower_bounds.as_slice()?;
    let ub = upper_bounds.as_slice()?;

    let x_sig_r = x_sig.clone();
    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        gauss_k2_residual_kernel(&x_sig, &y_sig, p, center_hint_ms, center_hint_sigma_ms, has_hint)
    };
    let jacobian_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_k2(&x_sig_r, p, has_hint, center_hint_sigma_ms)
    };

    let result = solve_trf(
        &residual_fn, Some(&jacobian_fn), x0_v, lb, ub,
        2000, true, 1.5, 1e-8, 1e-8,
    );

    let mut out = result.params;
    out.push(result.cost);
    Ok(PyArray1::from_vec(py, out))
}

// ---------------------------------------------------------------------------
// Pure Rust K1/K2 solvers (no PyO3, for internal use by seeded_fitter.rs)
// ---------------------------------------------------------------------------

/// Pure Rust K=1 solve — no PyO3 overhead. Used by seeded_fitter.rs.
pub fn lm_solve_k1(
    x_sig: &[f64],
    y_sig: &[f64],
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> LsResult {
    let x_sig_c = x_sig.to_vec();
    let y_sig_c = y_sig.to_vec();
    let x_sig_j = x_sig.to_vec();
    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        gauss_k1_residual_kernel(&x_sig_c, &y_sig_c, p, center_hint_ms, center_hint_sigma_ms, has_hint)
    };
    let jacobian_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_k1(&x_sig_j, p, has_hint, center_hint_sigma_ms)
    };
    solve_trf(&residual_fn, Some(&jacobian_fn), x0, lb, ub, 2000, true, 1.5, 1e-8, 1e-8)
}

/// Pure Rust K=2 solve — no PyO3 overhead. Used by seeded_fitter.rs.
pub fn lm_solve_k2(
    x_sig: &[f64],
    y_sig: &[f64],
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> LsResult {
    let x_sig_c = x_sig.to_vec();
    let y_sig_c = y_sig.to_vec();
    let x_sig_j = x_sig.to_vec();
    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        gauss_k2_residual_kernel(&x_sig_c, &y_sig_c, p, center_hint_ms, center_hint_sigma_ms, has_hint)
    };
    let jacobian_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_k2(&x_sig_j, p, has_hint, center_hint_sigma_ms)
    };
    solve_trf(&residual_fn, Some(&jacobian_fn), x0, lb, ub, 2000, true, 1.5, 1e-8, 1e-8)
}
