//! Paired translation of gaussian_fitting.py
//!
//! Source: qpsi/gaussian_fitting.py
//!   sum_gaussians_2d  — lines 23-44  (2D Gaussian synthesis)
//!   sum_gaussians_1d  — used in fit_gaussians_1d residual, lines 114-120
//!   fit_2d_residual   — lines 276-295 (2D fitting residual with penalties)
//!   fit_2d_lm         — full 2D fitter replacing scipy least_squares

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::solver::solve_lm;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Find nearest index in a sorted slice (binary-search then check neighbors).
#[inline]
pub fn nearest_idx(sorted: &[f64], target: f64) -> usize {
    let n = sorted.len();
    if n == 0 { return 0; }
    match sorted.binary_search_by(|v| v.partial_cmp(&target).unwrap()) {
        Ok(i) => i,
        Err(i) => {
            if i == 0 { 0 }
            else if i >= n { n - 1 }
            else if (sorted[i] - target).abs() < (sorted[i - 1] - target).abs() { i }
            else { i - 1 }
        }
    }
}

// ---------------------------------------------------------------------------
// Pure Rust kernels (no Python dependency — used by solver too)
// ---------------------------------------------------------------------------

/// Synthesize a 2D vector trace from oriented Gaussians.
///
/// Paired from: gaussian_fitting.py lines 23-44
/// Each component: [t0, sigma, amp, alpha] packed in `params`.
/// Returns flat vec [x0..xN, y0..yN] of length 2*N.
pub fn sum_gaussians_2d_kernel(t: &[f64], params: &[f64]) -> Vec<f64> {
    let n_pts = t.len();
    let n_waves = params.len() / 4;
    let mut out = vec![0.0f64; 2 * n_pts];

    for w in 0..n_waves {
        let t0 = params[4 * w];
        let sigma = params[4 * w + 1].max(1.0);
        let amp = params[4 * w + 2];
        let alpha = params[4 * w + 3];
        let cos_a = alpha.cos();
        let sin_a = alpha.sin();
        let inv_s = 1.0 / sigma;

        for i in 0..n_pts {
            let z = (t[i] - t0) * inv_s;
            let env = amp * (-0.5 * z * z).exp();
            out[i] += env * cos_a;
            out[n_pts + i] += env * sin_a;
        }
    }
    out
}

/// Sum of 1D Gaussians (magnitude-only).
///
/// Paired from: gaussian_fitting.py fit_gaussians_1d residual, lines 114-120
pub fn sum_gaussians_1d_kernel(t: &[f64], params: &[f64], n_waves: usize) -> Vec<f64> {
    let n_pts = t.len();
    let mut out = vec![0.0f64; n_pts];

    for w in 0..n_waves {
        let t0 = params[3 * w];
        let s = params[3 * w + 1].max(1.0);
        let a = params[3 * w + 2];
        let inv_s = 1.0 / s;

        for i in 0..n_pts {
            let z = (t[i] - t0) * inv_s;
            out[i] += a * (-0.5 * z * z).exp();
        }
    }
    out
}

/// 2D fitting residual with angle and amplitude penalties.
///
/// Paired from: gaussian_fitting.py lines 276-295
/// Returns residual of length 2*N + 2*n_waves (base + penalties).
pub fn fit_2d_residual_kernel(
    t: &[f64],
    v2_x: &[f64],
    v2_y: &[f64],
    mag: &[f64],
    params: &[f64],
    n_waves: usize,
    angle_lambda: f64,
    amplitude_lambda: f64,
) -> Vec<f64> {
    let n_pts = t.len();
    let synth = sum_gaussians_2d_kernel(t, params);

    // Base residual: (v2 - synth).ravel() = [dx0..dxN, dy0..dyN]
    let n_resid = 2 * n_pts + 2 * n_waves; // base + angle_pen + amp_pen
    let mut out = Vec::with_capacity(n_resid);

    for i in 0..n_pts {
        out.push(v2_x[i] - synth[i]);
    }
    for i in 0..n_pts {
        out.push(v2_y[i] - synth[n_pts + i]);
    }

    // Penalties per wave (nearest index via binary search for sorted t)
    for w in 0..n_waves {
        let t0 = params[4 * w];
        let a_param = params[4 * w + 2];
        let alpha = params[4 * w + 3];

        let idx = nearest_idx(t, t0);
        let alpha_ref = v2_y[idx].atan2(v2_x[idx]);
        let local_mag = mag[idx];

        // Angle penalty
        out.push(angle_lambda * local_mag * (alpha - alpha_ref));
        // Amplitude overshoot penalty
        out.push(amplitude_lambda * (a_param - local_mag).max(0.0));
    }
    out
}

/// Analytic Jacobian for the 2D fitting residual.
///
/// Derivatives of fit_2d_residual_kernel w.r.t. params [t0, sigma, amp, alpha] per wave.
pub fn analytic_jacobian_2d(
    t: &[f64],
    v2_x: &[f64],
    v2_y: &[f64],
    mag: &[f64],
    params: &[f64],
    n_waves: usize,
    angle_lambda: f64,
    amplitude_lambda: f64,
) -> Vec<Vec<f64>> {
    let n_pts = t.len();
    let n_params = 4 * n_waves;
    let n_resid = 2 * n_pts + 2 * n_waves;
    let mut jac = vec![vec![0.0f64; n_params]; n_resid];

    for w in 0..n_waves {
        let t0 = params[4 * w];
        let sigma = params[4 * w + 1].max(1.0);
        let amp = params[4 * w + 2];
        let alpha = params[4 * w + 3];
        let cos_a = alpha.cos();
        let sin_a = alpha.sin();
        let inv_s = 1.0 / sigma;

        for i in 0..n_pts {
            let z = (t[i] - t0) * inv_s;
            let g = (-0.5 * z * z).exp();

            // d(residual_x[i])/d(params) = -d(synth_x[i])/d(params)
            // synth_x = amp * cos(alpha) * g
            let dg_dt0 = g * z * inv_s;  // dg/dt0
            let dg_ds = g * z * z * inv_s; // dg/dsigma

            // X channel (rows 0..n_pts)
            jac[i][4 * w]     = -amp * cos_a * dg_dt0;     // d/dt0
            jac[i][4 * w + 1] = -amp * cos_a * dg_ds;      // d/dsigma
            jac[i][4 * w + 2] = -cos_a * g;                 // d/damp
            jac[i][4 * w + 3] = amp * sin_a * g;            // d/dalpha

            // Y channel (rows n_pts..2*n_pts)
            jac[n_pts + i][4 * w]     = -amp * sin_a * dg_dt0;
            jac[n_pts + i][4 * w + 1] = -amp * sin_a * dg_ds;
            jac[n_pts + i][4 * w + 2] = -sin_a * g;
            jac[n_pts + i][4 * w + 3] = -amp * cos_a * g;
        }

        // Penalty rows
        let pen_row_ang = 2 * n_pts + 2 * w;
        let pen_row_amp = 2 * n_pts + 2 * w + 1;

        let idx = nearest_idx(t, t0);
        let local_mag = mag[idx];

        // Angle penalty: angle_lambda * local_mag * (alpha - alpha_ref)
        // d/dalpha = angle_lambda * local_mag
        jac[pen_row_ang][4 * w + 3] = angle_lambda * local_mag;

        // Amplitude penalty: amplitude_lambda * max(A - local_mag, 0)
        let a_param = params[4 * w + 2];
        if a_param > local_mag {
            jac[pen_row_amp][4 * w + 2] = amplitude_lambda;
        }
    }
    jac
}

/// Fit 2D Gaussians to a vector trace — replaces scipy least_squares in fit_gaussians_2d.
///
/// Paired from: gaussian_fitting.py lines 308-315
/// Takes the initial guess, bounds, and signal data; returns optimized params.
#[pyfunction]
pub fn fit_2d_lm<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    v2: PyReadonlyArray2<'py, f64>,
    init_params: PyReadonlyArray1<'py, f64>,
    lower_bounds: PyReadonlyArray1<'py, f64>,
    upper_bounds: PyReadonlyArray1<'py, f64>,
    n_waves: usize,
    angle_lambda: f64,
    amplitude_lambda: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t_s = t.as_slice()?.to_vec();
    let n_pts = t_s.len();

    // Extract v2 rows
    let v2_raw = v2.as_slice()?;
    let v2_x: Vec<f64> = v2_raw[..n_pts].to_vec();
    let v2_y: Vec<f64> = v2_raw[n_pts..].to_vec();
    let mag: Vec<f64> = (0..n_pts)
        .map(|i| (v2_x[i] * v2_x[i] + v2_y[i] * v2_y[i]).sqrt())
        .collect();

    let x0 = init_params.as_slice()?;
    let lb = lower_bounds.as_slice()?;
    let ub = upper_bounds.as_slice()?;

    let t_r = t_s.clone();
    let v2x_r = v2_x.clone();
    let v2y_r = v2_y.clone();
    let mag_r = mag.clone();
    let nw = n_waves;
    let al = angle_lambda;
    let aml = amplitude_lambda;

    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        fit_2d_residual_kernel(&t_s, &v2_x, &v2_y, &mag, p, nw, al, aml)
    };

    let jacobian_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        analytic_jacobian_2d(&t_r, &v2x_r, &v2y_r, &mag_r, p, nw, al, aml)
    };

    let result = solve_lm(
        &residual_fn, Some(&jacobian_fn), x0, lb, ub,
        4000, false, 1.0, 1e-8, 1e-8,
    );

    Ok(PyArray1::from_vec(py, result.params))
}

/// Fit 1D magnitude Gaussians — replaces scipy least_squares in fit_gaussians_1d.
///
/// Paired from: gaussian_fitting.py lines 183-200
/// Residual: sum_gaussians_1d(t, p, n_waves) - mag
#[pyfunction]
pub fn fit_1d_magnitude_lm<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    mag: PyReadonlyArray1<'py, f64>,
    init_params: PyReadonlyArray1<'py, f64>,
    lower_bounds: PyReadonlyArray1<'py, f64>,
    upper_bounds: PyReadonlyArray1<'py, f64>,
    n_waves: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t_s = t.as_slice()?.to_vec();
    let mag_s = mag.as_slice()?.to_vec();
    let x0 = init_params.as_slice()?;
    let lb = lower_bounds.as_slice()?;
    let ub = upper_bounds.as_slice()?;

    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        let synth = sum_gaussians_1d_kernel(&t_s, p, n_waves);
        synth.iter().zip(mag_s.iter()).map(|(s, m)| s - m).collect()
    };

    // Analytic Jacobian for 1D Gaussian sum
    // d(residual_i)/d(t0_w) = -A_w * z_w / s_w * exp(-0.5 * z_w^2)
    // d(residual_i)/d(s_w) = -A_w * z_w^2 / s_w * exp(-0.5 * z_w^2)
    // d(residual_i)/d(A_w) = exp(-0.5 * z_w^2)
    let t_r = t.as_slice()?.to_vec();
    let jacobian_fn = move |p: &[f64]| -> Vec<Vec<f64>> {
        let n_pts = t_r.len();
        let n_params = 3 * n_waves;
        let mut jac = vec![vec![0.0f64; n_params]; n_pts];
        for w in 0..n_waves {
            let t0 = p[3 * w];
            let s = p[3 * w + 1].max(1.0);
            let a = p[3 * w + 2];
            let inv_s = 1.0 / s;
            for i in 0..n_pts {
                let z = (t_r[i] - t0) * inv_s;
                let g = (-0.5 * z * z).exp();
                // residual = synth - mag, so d(residual)/d(params) = d(synth)/d(params)
                jac[i][3 * w]     = a * z * inv_s * g;  // d/dt0
                jac[i][3 * w + 1] = a * z * z * inv_s * g;  // d/dsigma
                jac[i][3 * w + 2] = g;  // d/dA
            }
        }
        jac
    };

    let result = solve_lm(
        &residual_fn, Some(&jacobian_fn), x0, lb, ub,
        3000, false, 1.0, 1e-8, 1e-8,
    );

    Ok(PyArray1::from_vec(py, result.params))
}

/// Fit 1D magnitude residual (hybrid stage 2) — replaces scipy least_squares.
///
/// Paired from: gaussian_fitting.py _resid_1d, lines 465-487
#[pyfunction]
pub fn fit_1d_hybrid_lm<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    mag: PyReadonlyArray1<'py, f64>,
    x0: PyReadonlyArray1<'py, f64>,
    lower_bounds: PyReadonlyArray1<'py, f64>,
    upper_bounds: PyReadonlyArray1<'py, f64>,
    t0_seed: PyReadonlyArray1<'py, f64>,
    s_seed: PyReadonlyArray1<'py, f64>,
    n_waves_2d: usize,
    n_sel: usize,
    refine_sigmas: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t_s = t.as_slice()?.to_vec();
    let mag_s = mag.as_slice()?.to_vec();
    let x0_v = x0.as_slice()?;
    let lb = lower_bounds.as_slice()?;
    let ub = upper_bounds.as_slice()?;
    let t0_s = t0_seed.as_slice()?.to_vec();
    let s_s = s_seed.as_slice()?.to_vec();

    let residual_fn = move |p: &[f64]| -> Vec<f64> {
        let n_pts = t_s.len();
        let mut env = vec![0.0f64; n_pts];
        let mut ptr = 0usize;

        // Seeded waves
        for i in 0..n_waves_2d {
            let a = p[ptr];
            ptr += 1;
            let s = if refine_sigmas {
                let sv = p[ptr];
                ptr += 1;
                sv.max(1.0)
            } else {
                s_s[i].max(1.0)
            };
            let inv_s = 1.0 / s;
            for j in 0..n_pts {
                let z = (t_s[j] - t0_s[i]) * inv_s;
                env[j] += a * (-0.5 * z * z).exp();
            }
        }

        // Extra residual waves
        for _ in 0..n_sel {
            let t0_e = p[ptr];
            let s_e = p[ptr + 1].max(1.0);
            let a_e = p[ptr + 2];
            ptr += 3;
            let inv_s = 1.0 / s_e;
            for j in 0..n_pts {
                let z = (t_s[j] - t0_e) * inv_s;
                env[j] += a_e * (-0.5 * z * z).exp();
            }
        }

        // residual = env - mag
        env.iter().zip(mag_s.iter()).map(|(e, m)| e - m).collect()
    };

    let result = solve_lm(
        &residual_fn, None, x0_v, lb, ub,
        3000, false, 1.0, 1e-8, 1e-8,
    );

    Ok(PyArray1::from_vec(py, result.params))
}

// ---------------------------------------------------------------------------
// PyO3 wrappers for pure kernel functions
// ---------------------------------------------------------------------------

/// Python-callable: sum_gaussians_2d(t, params) -> ndarray (2, N)
#[pyfunction]
pub fn sum_gaussians_2d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_slice = t.as_slice()?;
    let p_slice = params.as_slice()?;
    let flat = sum_gaussians_2d_kernel(t_slice, p_slice);
    let n_pts = t_slice.len();
    let row_x = &flat[..n_pts];
    let row_y = &flat[n_pts..];
    let data = vec![row_x.to_vec(), row_y.to_vec()];
    Ok(PyArray2::from_vec2(py, &data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?)
}

/// Python-callable: sum_gaussians_1d(t, params, n_waves) -> ndarray (N,)
#[pyfunction]
pub fn sum_gaussians_1d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    n_waves: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t_slice = t.as_slice()?;
    let p_slice = params.as_slice()?;
    let result = sum_gaussians_1d_kernel(t_slice, p_slice, n_waves);
    Ok(PyArray1::from_vec(py, result))
}
