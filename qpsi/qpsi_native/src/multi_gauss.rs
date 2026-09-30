//! Paired translation of lead_fitting_helpers.py _multi_gauss_func.
//!
//! Source: qpsi/lead_fitting_helpers.py lines 628-639
//!   _multi_gauss_func — sum of K Gaussians + shared baseline

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Pure Rust kernel
// ---------------------------------------------------------------------------

/// Sum of K Gaussians with shared baseline.
///
/// Paired from: lead_fitting_helpers.py lines 628-639
/// params = [A1, mu1, s1, A2, mu2, s2, ..., baseline]
/// n_comp = (len(params) - 1) / 3
pub fn multi_gauss_kernel(t: &[f64], params: &[f64], n_comp: usize) -> Vec<f64> {
    let baseline = params[3 * n_comp]; // last element
    let n_pts = t.len();
    let mut out = vec![baseline; n_pts];

    for c in 0..n_comp {
        let a = params[3 * c];
        let mu = params[3 * c + 1];
        let s = params[3 * c + 2] + 1e-9; // (s + 1e-9) — py line 638
        let inv_s = 1.0 / s;

        for i in 0..n_pts {
            let z = (t[i] - mu) * inv_s;
            out[i] += a * (-0.5 * z * z).exp();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Python-callable: multi_gauss_func(t, params, n_comp) -> ndarray
///
/// Paired from: lead_fitting_helpers.py:628-639
#[pyfunction]
pub fn multi_gauss_func<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    n_comp: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = multi_gauss_kernel(t.as_slice()?, params.as_slice()?, n_comp);
    Ok(PyArray1::from_vec(py, result))
}
