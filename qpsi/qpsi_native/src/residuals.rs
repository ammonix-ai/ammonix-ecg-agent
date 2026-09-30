//! Paired translation of wave_fitters.py residual functions.
//!
//! Source: qpsi/wave_fitters.py
//!   _gauss         — lines 151-157 (single Gaussian evaluation)
//!   r1 (K=1)       — lines 699-704 (single Gaussian + baseline + prior)
//!   r2 (K=2)       — lines 766-770 (two Gaussians + baseline + prior)

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Pure Rust kernels
// ---------------------------------------------------------------------------

/// Single Gaussian: A * exp(-0.5 * ((x - mu) / s)^2)
///
/// Paired from: wave_fitters.py lines 151-157
pub fn gauss_kernel(x: &[f64], a: f64, mu: f64, s: f64) -> Vec<f64> {
    let inv_s = 1.0 / (s + 1e-9);
    x.iter()
        .map(|&xi| {
            let z = (xi - mu) * inv_s;
            a * (-0.5 * z * z).exp()
        })
        .collect()
}

/// K=1 residual: y - (A*gauss(x, mu, s) + b0) + optional center prior
///
/// Paired from: wave_fitters.py lines 699-704
/// p = [A1, mu1, s1, b0]
pub fn gauss_k1_residual_kernel(
    x: &[f64],
    y: &[f64],
    p: &[f64],
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> Vec<f64> {
    let (a1, mu1, s1, b0) = (p[0], p[1], p[2], p[3]);
    let n = x.len();
    let n_out = if has_hint { n + 1 } else { n };
    let mut out = Vec::with_capacity(n_out);
    let inv_s = 1.0 / (s1 + 1e-9);

    for i in 0..n {
        let z = (x[i] - mu1) * inv_s;
        let g = a1 * (-0.5 * z * z).exp();
        out.push(y[i] - (g + b0));
    }

    if has_hint {
        out.push((mu1 - center_hint_ms) / center_hint_sigma_ms.max(1e-6));
    }
    out
}

/// K=2 residual: y - (A1*gauss1 + A2*gauss2 + b0) + optional center prior
///
/// Paired from: wave_fitters.py lines 766-770
/// p = [A1, mu1, s1, A2, sep, s2, b0]  (mu2 = mu1 + sep)
pub fn gauss_k2_residual_kernel(
    x: &[f64],
    y: &[f64],
    p: &[f64],
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> Vec<f64> {
    let (a1, mu1, s1, a2, sep, s2, b0) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
    let mu2 = mu1 + sep;
    let n = x.len();
    let n_out = if has_hint { n + 1 } else { n };
    let mut out = Vec::with_capacity(n_out);
    let inv_s1 = 1.0 / (s1 + 1e-9);
    let inv_s2 = 1.0 / (s2 + 1e-9);

    for i in 0..n {
        let z1 = (x[i] - mu1) * inv_s1;
        let z2 = (x[i] - mu2) * inv_s2;
        let g1 = a1 * (-0.5 * z1 * z1).exp();
        let g2 = a2 * (-0.5 * z2 * z2).exp();
        out.push(y[i] - (g1 + g2 + b0));
    }

    if has_hint {
        out.push((mu1 - center_hint_ms) / center_hint_sigma_ms.max(1e-6));
    }
    out
}

// ---------------------------------------------------------------------------
// PyO3 wrappers
// ---------------------------------------------------------------------------

/// Python-callable: gauss_single(x, A, mu, s) -> ndarray
///
/// Paired from: wave_fitters.py:151-157
#[pyfunction]
pub fn gauss_single<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: f64,
    mu: f64,
    s: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = gauss_kernel(x.as_slice()?, a, mu, s);
    Ok(PyArray1::from_vec(py, result))
}

/// Python-callable: gauss_k1_residual(x, y, p, hint_ms, hint_sigma, has_hint) -> ndarray
///
/// Paired from: wave_fitters.py:699-704
#[pyfunction]
pub fn gauss_k1_residual<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = gauss_k1_residual_kernel(
        x.as_slice()?,
        y.as_slice()?,
        p.as_slice()?,
        center_hint_ms,
        center_hint_sigma_ms,
        has_hint,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// Python-callable: gauss_k2_residual(x, y, p, hint_ms, hint_sigma, has_hint) -> ndarray
///
/// Paired from: wave_fitters.py:766-770
#[pyfunction]
pub fn gauss_k2_residual<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    center_hint_ms: f64,
    center_hint_sigma_ms: f64,
    has_hint: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = gauss_k2_residual_kernel(
        x.as_slice()?,
        y.as_slice()?,
        p.as_slice()?,
        center_hint_ms,
        center_hint_sigma_ms,
        has_hint,
    );
    Ok(PyArray1::from_vec(py, result))
}
