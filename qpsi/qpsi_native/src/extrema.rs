//! Post-smoothing helper for ``qpsi.direct_measurements.find_extrema``.
//!
//! See `qpsi/direct_measurements.py:42-167` for the full reference
//! algorithm. This kernel handles steps 3–8 only (everything AFTER
//! the Savitzky–Golay smoothing pass), taking ``x_d`` (detrended) +
//! ``x_s`` (smoothed) as already-prepared inputs from the Python
//! caller. The split is deliberate: scipy's ``savgol_filter`` uses a
//! Vandermonde + ``np.linalg.lstsq`` (SVD) coefficient computation
//! that differs from the closed-form quadratic SG coefficients in
//! ``seeded_fitter::savgol_filter_quadratic_pub`` by ~1 ULP per tap,
//! so a Rust-side smoothing pass would inherit pre-existing
//! [[project-qpsi-direct-drift-free-vs-apr28-gt]]-violating drift.
//! Keeping smoothing on the scipy side preserves the direct-namespace
//! bit-clean property; the loop-heavy work (sign-change scan,
//! parabolic refine, prominence loop, sort) still moves to Rust.
//!
//! Pipeline (steps the Rust kernel covers):
//!   3. numpy.gradient (uniform-spacing branch matches the
//!      ``(y[i+1] - y[i-1]) / (2h)`` evaluation order)
//!   4. Sign-change scan over the derivative (zeros treated as +1)
//!   5. 3-point parabolic refinement of each extremum
//!   6. Prominence relative to flanking opposite-kind extremum or
//!      ``x_d`` window edge value
//!   7. Threshold by ``prominence_frac * (x_d max - x_d min)``
//!
//! NOT covered (intentionally delegated back to the Python caller):
//!   8. Sort by descending prominence — Python uses ``np.argsort
//!      (-proms)`` (quicksort, default), whose tie-order is
//!      implementation-dependent. Replicating that exactly from
//!      Rust is brittle; ~14 % of real Apr28 beat slices contain
//!      tied prominences. Returning extrema in their **natural
//!      insertion order** (time-ascending — the same order the
//!      Python loop builds ``extrema_raw``) lets the Python wrapper
//!      run the identical numpy.argsort downstream. Per-call cost
//!      of that sort on a ~4-10 element float64 array is
//!      sub-microsecond, so the perf hit is negligible.

/// One refined extremum: ``(time_ms, amp_mv, kind, prominence)``.
/// ``kind`` is ``+1.0`` for local maxima, ``-1.0`` for minima.
pub type Extremum = (f64, f64, f64, f64);

/// numpy.gradient with ``edge_order=1`` (default).
///
/// Mirrors numpy/lib/function_base.py::gradient exactly, including the
/// uniform-spacing fast path (which numpy enters when every diff of
/// ``x`` equals the first diff). Distinguishing the two paths is
/// load-bearing for bit-identicality: under uniform spacing numpy
/// evaluates ``(y[i+1] - y[i-1]) / (2*h)`` directly, NOT the
/// algebraically-equivalent abc form below, and the two differ by
/// 1 ULP for typical inputs.
fn numpy_gradient(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0_f64; n];
    if n < 2 {
        return g;
    }

    // Detect uniform spacing — numpy treats ``np.gradient(y, t)`` with a
    // uniform ``t`` array as the scalar-spacing case ``h = diffs[0]``.
    let h0 = x[1] - x[0];
    let mut uniform = true;
    for i in 1..n - 1 {
        if x[i + 1] - x[i] != h0 {
            uniform = false;
            break;
        }
    }

    if uniform {
        g[0] = (y[1] - y[0]) / h0;
        let two_h = 2.0 * h0;
        for i in 1..n - 1 {
            g[i] = (y[i + 1] - y[i - 1]) / two_h;
        }
        g[n - 1] = (y[n - 1] - y[n - 2]) / h0;
    } else {
        g[0] = (y[1] - y[0]) / (x[1] - x[0]);
        for i in 1..n - 1 {
            let dx1 = x[i] - x[i - 1];
            let dx2 = x[i + 1] - x[i];
            let a = -dx2 / (dx1 * (dx1 + dx2));
            let b = (dx2 - dx1) / (dx1 * dx2);
            let c = dx1 / (dx2 * (dx1 + dx2));
            g[i] = a * y[i - 1] + b * y[i] + c * y[i + 1];
        }
        g[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    }

    g
}

/// Find extrema given pre-smoothed inputs (steps 3–8 of `find_extrema`).
///
/// ``x_d`` — detrended (NOT smoothed) signal; used for edge-flanking
/// prominence references and the peak-to-peak threshold normaliser.
/// ``x_s`` — smoothed signal (scipy.signal.savgol_filter output) used
/// for derivative + parabolic-refine. Caller is responsible for
/// computing both in exactly the same numerical environment that the
/// reference Python implementation uses (scipy savgol, numpy detrend).
pub fn find_extrema_post_smooth_rust(
    x_d: &[f64],
    x_s: &[f64],
    t_ms: &[f64],
    prominence_frac: f64,
) -> Vec<Extremum> {
    let n = x_d.len();
    if n < 7 || x_s.len() != n || t_ms.len() != n {
        return Vec::new();
    }

    // ---- First derivative via numpy.gradient ----
    let dxdt = numpy_gradient(x_s, t_ms);

    // ---- Sign array (zero -> +1 to match np.sign post-fixup) ----
    let signs: Vec<f64> = dxdt
        .iter()
        .map(|v| {
            if *v > 0.0 {
                1.0
            } else if *v < 0.0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();

    // ---- Sign-change indices: position just before the change ----
    let mut sign_changes: Vec<usize> = Vec::new();
    if !signs.is_empty() {
        for i in 0..signs.len() - 1 {
            if signs[i] != signs[i + 1] {
                sign_changes.push(i);
            }
        }
    }
    if sign_changes.is_empty() {
        return Vec::new();
    }

    // ---- Sub-sample parabolic refinement ----
    let dx_step = t_ms[1] - t_ms[0];
    let mut extrema_raw: Vec<(f64, f64, i32)> = Vec::new();
    for &ix in &sign_changes {
        let i = ix + 1; // the actual local extremum sample
        if i < 1 || i > n - 2 {
            continue;
        }
        let a = x_s[i - 1];
        let b = x_s[i];
        let c = x_s[i + 1];
        let denom = a - 2.0 * b + c;
        let (t_e, amp_e) = if denom.abs() > 1e-12 {
            let offset = 0.5 * (a - c) / denom;
            (t_ms[i] + offset * dx_step, b - 0.25 * (a - c) * offset)
        } else {
            (t_ms[i], b)
        };
        let kind = if signs[ix] > 0.0 && signs[ix + 1] <= 0.0 {
            1
        } else {
            -1
        };
        extrema_raw.push((t_e, amp_e, kind));
    }

    if extrema_raw.is_empty() {
        return Vec::new();
    }

    // ---- Prominence per extremum ----
    let n_extr = extrema_raw.len();
    let mut proms = vec![0.0_f64; n_extr];
    for i in 0..n_extr {
        let (_, ae, ke) = extrema_raw[i];
        // Default to window edge of x_d when no opposite-kind flank exists.
        let mut left_flank = x_d[0];
        for j in (0..i).rev() {
            if extrema_raw[j].2 == -ke {
                left_flank = extrema_raw[j].1;
                break;
            }
        }
        let mut right_flank = x_d[n - 1];
        for j in (i + 1)..n_extr {
            if extrema_raw[j].2 == -ke {
                right_flank = extrema_raw[j].1;
                break;
            }
        }
        proms[i] = if ke > 0 {
            (ae - left_flank.max(right_flank)).max(0.0)
        } else {
            (left_flank.min(right_flank) - ae).max(0.0)
        };
    }

    // ---- Prominence threshold filter ----
    let mut max_x = f64::NEG_INFINITY;
    let mut min_x = f64::INFINITY;
    for &v in x_d {
        if v > max_x {
            max_x = v;
        }
        if v < min_x {
            min_x = v;
        }
    }
    let p2p = max_x - min_x;
    let thr = (prominence_frac * p2p).max(1e-6);
    // Return prominence-filtered extrema in natural insertion order
    // (time-ascending). The Python caller applies ``np.argsort
    // (-proms)`` to recover the descending-prominence ordering — see
    // module doc for why the final sort is delegated.
    extrema_raw
        .iter()
        .zip(proms.iter())
        .filter_map(|(&(t, a, k), &p)| {
            if p >= thr {
                Some((t, a, k as f64, p))
            } else {
                None
            }
        })
        .collect()
}
