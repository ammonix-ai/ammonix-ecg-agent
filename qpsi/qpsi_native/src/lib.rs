//! qpsi_native — Rust-accelerated QPSI fitting kernels.
//!
//! Paired translations from Python source files:
//!   gauss.rs       <- gaussian_fitting.py (sum_gaussians_2d, fit_gaussians_1d residual)
//!   residuals.rs   <- wave_fitters.py (K=1/K=2 residuals, _gauss)
//!   multi_gauss.rs <- lead_fitting_helpers.py (_multi_gauss_func)
//!   solver.rs      <- scipy.optimize.least_squares replacement (Levenberg-Marquardt)
//!   parallel.rs    <- lead_fitting.py parallel lead/beat processing
//!   kmeans.rs      <- qrs_complex.py KMeans clustering

mod gauss;
mod residuals;
mod multi_gauss;
pub mod solver;
mod parallel;
mod kmeans;
mod plane;
mod hybrid;
pub mod seeded_fitter;
pub mod qrs_fitter;
mod lead_fitter;
pub mod lapack_svd;
pub mod extrema;
pub mod measure_window;
pub mod measure_window_batch;
pub mod derived_morphology;

use pyo3::prelude::*;

/// Rust-accelerated QPSI fitting kernels.
///
/// Each function is a paired 1:1 translation of its Python equivalent.
/// Zero drift from Python output is enforced by fixture tests.
#[pymodule]
fn qpsi_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize LAPACK DLL for bit-identical SVD with scipy
    lapack_svd::init(m.py());

    // --- gauss.rs exports (paired from gaussian_fitting.py) ---
    m.add_function(wrap_pyfunction!(gauss::sum_gaussians_2d, m)?)?;
    m.add_function(wrap_pyfunction!(gauss::sum_gaussians_1d, m)?)?;
    m.add_function(wrap_pyfunction!(gauss::fit_2d_lm, m)?)?;
    m.add_function(wrap_pyfunction!(gauss::fit_1d_magnitude_lm, m)?)?;
    m.add_function(wrap_pyfunction!(gauss::fit_1d_hybrid_lm, m)?)?;

    // --- residuals.rs exports (paired from wave_fitters.py) ---
    m.add_function(wrap_pyfunction!(residuals::gauss_single, m)?)?;
    m.add_function(wrap_pyfunction!(residuals::gauss_k1_residual, m)?)?;
    m.add_function(wrap_pyfunction!(residuals::gauss_k2_residual, m)?)?;

    // --- multi_gauss.rs exports (paired from lead_fitting_helpers.py) ---
    m.add_function(wrap_pyfunction!(multi_gauss::multi_gauss_func, m)?)?;

    // --- solver.rs exports ---
    m.add_function(wrap_pyfunction!(solver::fit_k1_lm, m)?)?;
    m.add_function(wrap_pyfunction!(solver::fit_k2_lm, m)?)?;

    // --- parallel.rs exports (rayon-parallelized lead/beat processing) ---
    m.add_function(wrap_pyfunction!(parallel::fit_all_leads_beats_parallel, m)?)?;

    // --- kmeans.rs exports (replace sklearn KMeans) ---
    m.add_function(wrap_pyfunction!(kmeans::kmeans_best_of_k, m)?)?;

    // --- plane.rs exports (full Rust plane analysis with rayon) ---
    m.add_function(wrap_pyfunction!(plane::analyse_beats_full_rust, m)?)?;

    // --- lead_fitter.rs exports (full Rust avg-trace fitting with rayon) ---
    m.add_function(wrap_pyfunction!(lead_fitter::fit_avg_traces_all_leads, m)?)?;

    // --- seeded_fitter test/debug exports ---
    m.add_function(wrap_pyfunction!(test_savgol_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_wave_seeded_py, m)?)?;

    // --- extrema.rs export (paired from direct_measurements.py::find_extrema) ---
    m.add_function(wrap_pyfunction!(find_extrema_post_smooth_native, m)?)?;

    // --- measure_window.rs export (paired from direct_measurements.py::measure_window) ---
    m.add_function(wrap_pyfunction!(measure_window_core_native, m)?)?;

    // --- measure_window_batch.rs export (Phase B.5 / Brief 2 — rayon parallel batched dispatcher) ---
    m.add_function(wrap_pyfunction!(measure_window_batch_native, m)?)?;

    // --- derived_morphology.rs export (Phase C — batched morphology over sorted extrema) ---
    m.add_function(wrap_pyfunction!(derived_morphology_batch_native, m)?)?;

    // --- qrs_fitter test exports ---
    m.add_function(wrap_pyfunction!(test_gradient_py, m)?)?;
    m.add_function(wrap_pyfunction!(test_lstsq_py, m)?)?;
    m.add_function(wrap_pyfunction!(test_gauss_smooth_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_qrs_py, m)?)?;

    // --- diagnostic toggle ---
    m.add_function(wrap_pyfunction!(set_solver_diag_py, m)?)?;

    Ok(())
}

// ---------- QRS test helpers ----------

#[pyfunction]
fn test_gauss_smooth_py<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray1<'py, f64>,
    dt_ms: f64,
    sigma_ms: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result = qrs_fitter::gauss_smooth(x.as_slice()?, dt_ms, sigma_ms);
    Ok(numpy::PyArray1::from_vec(py, result))
}

#[pyfunction]
fn test_gradient_py<'py>(
    py: Python<'py>,
    y: numpy::PyReadonlyArray1<'py, f64>,
    t: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result = qrs_fitter::gradient(y.as_slice()?, t.as_slice()?);
    Ok(numpy::PyArray1::from_vec(py, result))
}

#[pyfunction]
fn test_lstsq_py<'py>(
    py: Python<'py>,
    phi_flat: numpy::PyReadonlyArray1<'py, f64>,
    n_rows: usize,
    n_cols: usize,
    b: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result = qrs_fitter::lstsq_svd(phi_flat.as_slice()?, n_rows, n_cols, b.as_slice()?);
    Ok(numpy::PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (trace, time_ms, qrs_lo, qrs_hi, t_r_ms=None, fs=500.0))]
fn fit_qrs_py<'py>(
    py: Python<'py>,
    trace: numpy::PyReadonlyArray1<'py, f64>,
    time_ms: numpy::PyReadonlyArray1<'py, f64>,
    qrs_lo: f64,
    qrs_hi: f64,
    t_r_ms: Option<f64>,
    fs: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let result = qrs_fitter::fit_qrs_unified(
        trace.as_slice()?, time_ms.as_slice()?,
        (qrs_lo, qrs_hi), t_r_ms, fs,
        60.0, 250.0, 10.0, false, (6.0, 60.0), 4, 0.03, 0.05, 8.0,
    );

    let dict = pyo3::types::PyDict::new(py);
    match result {
        Some(r) => {
            dict.set_item("r_peak_time", r.r_peak_time)?;
            dict.set_item("bounds", (r.bounds.0, r.bounds.1))?;
            let comps: Vec<Bound<'py, pyo3::types::PyDict>> = r.components.iter().map(|c| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("label", c.label.as_str()).unwrap();
                d.set_item("amp_mv", c.amp_mv).unwrap();
                d.set_item("center_ms", c.center_ms).unwrap();
                d.set_item("sigma_ms", c.sigma_ms).unwrap();
                d
            }).collect();
            dict.set_item("components", comps)?;
            dict.set_item("n_components", r.components.len())?;
        }
        None => {
            dict.set_item("components", pyo3::types::PyList::empty(py))?;
            dict.set_item("n_components", 0)?;
        }
    }
    Ok(dict)
}

// ---------- Test helper: expose savgol to Python for validation ----------

#[pyfunction]
fn test_savgol_py<'py>(
    py: Python<'py>,
    y: numpy::PyReadonlyArray1<'py, f64>,
    window_len: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let y_slice = y.as_slice()?;
    let result = seeded_fitter::savgol_filter_quadratic_pub(y_slice, window_len);
    Ok(numpy::PyArray1::from_vec(py, result))
}

// ---------- extrema.rs PyO3 binding ----------
//
// Steps 3–8 of `qpsi.direct_measurements.find_extrema` (gradient,
// sign-change, parabolic refine, prominence, sort). The Python caller
// pre-computes the savgol-smoothed signal via scipy and the
// edge-mean-detrended signal, then hands both to this kernel; see
// the module-level comment in `extrema.rs` for why smoothing stays
// on the scipy side. Returns a (K, 4) numpy array of
// [time_ms, amp_mv, kind, prominence] rows already sorted by
// descending prominence.
#[pyfunction]
fn find_extrema_post_smooth_native<'py>(
    py: Python<'py>,
    x_d: numpy::PyReadonlyArray1<'py, f64>,
    x_s: numpy::PyReadonlyArray1<'py, f64>,
    t_ms: numpy::PyReadonlyArray1<'py, f64>,
    prominence_frac: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let xd_slice = x_d.as_slice()?;
    let xs_slice = x_s.as_slice()?;
    let t_slice = t_ms.as_slice()?;
    let extrema = extrema::find_extrema_post_smooth_rust(
        xd_slice,
        xs_slice,
        t_slice,
        prominence_frac,
    );

    let rows: Vec<Vec<f64>> = extrema
        .into_iter()
        .map(|(t, a, k, p)| vec![t, a, k, p])
        .collect();
    if rows.is_empty() {
        let arr = numpy::PyArray2::<f64>::zeros(py, [0, 4], false);
        return Ok(arr);
    }
    numpy::PyArray2::from_vec2(py, &rows).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "find_extrema_post_smooth_native: {e}"
        ))
    })
}

// ---------- measure_window.rs PyO3 binding ----------
//
// Returns a tuple (scalars, extrema) where:
//   * scalars is a (11,) f64 numpy array — see measure_window.rs
//     module doc for the field ordering.
//   * extrema is a (K, 4) f64 numpy array of [time, amp, kind, prom]
//     rows in time-insertion order (unsorted by prominence). Python
//     applies np.argsort(-prom) to recover the canonical ordering.
#[pyfunction]
fn measure_window_core_native<'py>(
    py: Python<'py>,
    x_d: numpy::PyReadonlyArray1<'py, f64>,
    x_s: numpy::PyReadonlyArray1<'py, f64>,
    t_ms: numpy::PyReadonlyArray1<'py, f64>,
    prominence_frac: f64,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray2<f64>>)> {
    let xd_slice = x_d.as_slice()?;
    let xs_slice = x_s.as_slice()?;
    let t_slice = t_ms.as_slice()?;

    let (scalars, extrema) = measure_window::measure_window_core_rust(
        xd_slice,
        xs_slice,
        t_slice,
        prominence_frac,
    );

    let scalars_arr = numpy::PyArray1::from_vec(py, scalars);
    let rows: Vec<Vec<f64>> = extrema
        .into_iter()
        .map(|(t, a, k, p)| vec![t, a, k, p])
        .collect();
    let extrema_arr = if rows.is_empty() {
        numpy::PyArray2::<f64>::zeros(py, [0, 4], false)
    } else {
        numpy::PyArray2::from_vec2(py, &rows).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "measure_window_core_native: {e}"
            ))
        })?
    };
    Ok((scalars_arr, extrema_arr))
}

// ---------- measure_window_batch.rs PyO3 binding ----------
//
// Brief 2 / Phase B.5 — rayon-parallel batched dispatcher. Takes flat
// concatenated x_d / x_s / t_ms arrays + (offsets, lengths) metadata
// describing N windows; returns:
//   * scalars: (N, 11) f64 — per-window 11 base scalars (same order
//     as measure_window_core_native)
//   * extrema_flat: (total_extrema, 4) f64 — concatenated extrema lists
//     in time-insertion order, one window's contribution after another
//   * extrema_counts: (N,) i64 — number of extrema per window
//     (caller uses cumulative-sum to slice extrema_flat per window)
// The flat-array encoding for extrema is the cheapest PyO3 marshalling
// shape for a heterogeneous list-of-lists across N ~= 504 windows.
#[pyfunction]
fn measure_window_batch_native<'py>(
    py: Python<'py>,
    x_d_flat: numpy::PyReadonlyArray1<'py, f64>,
    x_s_flat: numpy::PyReadonlyArray1<'py, f64>,
    t_ms_flat: numpy::PyReadonlyArray1<'py, f64>,
    offsets: numpy::PyReadonlyArray1<'py, i64>,
    lengths: numpy::PyReadonlyArray1<'py, i64>,
    prominence_frac: f64,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray1<i64>>,
)> {
    let xd = x_d_flat.as_slice()?;
    let xs = x_s_flat.as_slice()?;
    let tm = t_ms_flat.as_slice()?;
    let off_i64 = offsets.as_slice()?;
    let len_i64 = lengths.as_slice()?;
    if off_i64.len() != len_i64.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "measure_window_batch_native: offsets len {} != lengths len {}",
            off_i64.len(),
            len_i64.len()
        )));
    }
    // Convert i64 → usize. Negative values would be a caller bug; clamp
    // to 0 (the n<5 short-circuit returns empty results).
    let offsets_us: Vec<usize> = off_i64.iter().map(|&v| v.max(0) as usize).collect();
    let lengths_us: Vec<usize> = len_i64.iter().map(|&v| v.max(0) as usize).collect();

    // GIL is held here; release it via `allow_threads` so rayon
    // workers don't contend on it. Rust-side state is independent
    // per worker (each indexes into the shared read-only flat arrays
    // and writes its own slot in `results`).
    let results = py.allow_threads(|| {
        measure_window_batch::measure_window_batch_rust(
            xd, xs, tm, &offsets_us, &lengths_us, prominence_frac,
        )
    });

    // Pack scalars: (n_windows, 11)
    let n_windows = results.len();
    let mut scalars_flat = Vec::<f64>::with_capacity(n_windows * 11);
    let mut extrema_counts = Vec::<i64>::with_capacity(n_windows);
    let mut total_extrema = 0_usize;
    for (sc, ext) in &results {
        for &v in sc {
            scalars_flat.push(v);
        }
        extrema_counts.push(ext.len() as i64);
        total_extrema += ext.len();
    }
    let scalars_arr = if n_windows == 0 {
        numpy::PyArray2::<f64>::zeros(py, [0, 11], false)
    } else {
        numpy::PyArray2::from_vec2(
            py,
            &scalars_flat.chunks(11).map(|c| c.to_vec()).collect::<Vec<_>>(),
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "measure_window_batch_native scalars pack: {e}"
            ))
        })?
    };

    // Pack extrema flat: (total_extrema, 4)
    let extrema_arr = if total_extrema == 0 {
        numpy::PyArray2::<f64>::zeros(py, [0, 4], false)
    } else {
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(total_extrema);
        for (_, ext) in &results {
            for (t, a, k, p) in ext {
                rows.push(vec![*t, *a, *k, *p]);
            }
        }
        numpy::PyArray2::from_vec2(py, &rows).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "measure_window_batch_native extrema pack: {e}"
            ))
        })?
    };

    let counts_arr = numpy::PyArray1::from_vec(py, extrema_counts);
    Ok((scalars_arr, extrema_arr, counts_arr))
}

// ---------- derived_morphology.rs PyO3 binding ----------
//
// Phase C — batched morphology over N windows in one PyO3 call. Replaces
// the per-window Python `derived_morphology` call inside Brief 2's
// `_measure_all_beats_via_batch` (qpsi/direct_measurements.py).
//
// Inputs:
//   * extrema_flat: (total_extrema, 4) f64 — concatenated prominence-
//     sorted extrema lists; one window's contribution after another
//     in the SAME order the Python caller produced via
//     `np.argsort(-prom)` per window (the Python hybrid contract).
//   * extrema_counts: (n_windows,) i64 — number of extrema per window
//     (caller's cumulative sum slices `extrema_flat`).
//   * window_sizes_ms: (n_windows,) f64 — time span of each window.
//
// Output: (n_windows, 7) f64 — per-window morphology row:
//   [n_significant_extrema, slope_reversal_count, biphasic_ratio,
//    notch_depth_mv, notch_position_frac, saddle_depth_relative,
//    monotonic_fraction].
#[pyfunction]
fn derived_morphology_batch_native<'py>(
    py: Python<'py>,
    extrema_flat: numpy::PyReadonlyArray2<'py, f64>,
    extrema_counts: numpy::PyReadonlyArray1<'py, i64>,
    window_sizes_ms: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let counts_slice = extrema_counts.as_slice()?;
    let sizes_slice = window_sizes_ms.as_slice()?;
    if counts_slice.len() != sizes_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "derived_morphology_batch_native: extrema_counts len {} != window_sizes_ms len {}",
            counts_slice.len(),
            sizes_slice.len()
        )));
    }

    // Flatten the (total_extrema, 4) array to a single &[f64] for the kernel.
    // PyReadonlyArray2 may be non-contiguous; copy via as_array().
    let extrema_view = extrema_flat.as_array();
    let total_rows = extrema_view.shape()[0];
    let mut extrema_buf: Vec<f64> = Vec::with_capacity(total_rows * 4);
    for r in 0..total_rows {
        extrema_buf.push(extrema_view[[r, 0]]);
        extrema_buf.push(extrema_view[[r, 1]]);
        extrema_buf.push(extrema_view[[r, 2]]);
        extrema_buf.push(extrema_view[[r, 3]]);
    }

    let n_windows = counts_slice.len();
    let flat_out = py.allow_threads(|| {
        derived_morphology::derived_morphology_batch_rust(
            &extrema_buf,
            counts_slice,
            sizes_slice,
        )
    });

    // Reshape to (n_windows, 7)
    if n_windows == 0 {
        return Ok(numpy::PyArray2::<f64>::zeros(py, [0, 7], false));
    }
    let rows: Vec<Vec<f64>> = flat_out.chunks(7).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "derived_morphology_batch_native: {e}"
        ))
    })
}

// ---------- Direct seeded fitter access for Python testing ----------

#[pyfunction]
#[pyo3(signature = (x, y, wave_type, residual_hint=None, center_hint_ms=None))]
fn fit_wave_seeded_py<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray1<'py, f64>,
    y: numpy::PyReadonlyArray1<'py, f64>,
    wave_type: &str,
    residual_hint: Option<numpy::PyReadonlyArray1<'py, f64>>,
    center_hint_ms: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let x_s = x.as_slice()?;
    let y_s = y.as_slice()?;
    let hint: Option<Vec<f64>> = residual_hint.map(|h| h.as_slice().unwrap().to_vec());
    let hint_ref = hint.as_deref();

    let (sigma_bounds, min_sep, max_sep, prefer_left, min_snr, hint_sigma) = match wave_type {
        "P" => ((8.0, 60.0), 18.0, 140.0, true, -3.0, 25.0),
        _ => ((30.0, 140.0), 40.0, 180.0, false, -4.0, 35.0),
    };

    let result = seeded_fitter::fit_wave_components_core_rust(
        x_s, y_s,
        sigma_bounds, min_sep, max_sep,
        true, true, 4.0, true,
        center_hint_ms, hint_sigma,
        prefer_left, true, 5, min_snr,
        hint_ref,
        false, 20.0, // aicc_early_term disabled
    );

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model", &result.model)?;
    dict.set_item("present", result.present)?;
    dict.set_item("snr_db", result.snr_db)?;
    dict.set_item("notched", result.notched)?;
    dict.set_item("n_components", result.components.len())?;
    let comps: Vec<Bound<'py, pyo3::types::PyDict>> = result.components.iter().map(|c| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("amp_mv", c.amp_mv).unwrap();
        d.set_item("center_ms", c.center_ms).unwrap();
        d.set_item("sigma_ms", c.sigma_ms).unwrap();
        d
    }).collect();
    dict.set_item("components", comps)?;
    Ok(dict)
}

#[pyfunction]
fn set_solver_diag_py(enabled: bool) {
    solver::set_solver_diagnostics(enabled);
}
