//! Paired translation of `qpsi.direct_measurements.measure_window`.
//!
//! Covers the 11 base scalars (positive/negative peak + parabolic
//! refine, peak-to-peak, centroid, duration-rms, duration-fwhm,
//! max-abs-derivative, rms amplitude, snr_db) plus the
//! prominence-filtered extrema list (in time-insertion order, just
//! like `extrema::find_extrema_post_smooth_rust`). Python keeps:
//!
//!   * detrend + scipy savgol (same as Item #1 — Rust savgol differs
//!     from scipy by ~1 ULP per tap, so smoothing stays on the
//!     scipy side to preserve direct-namespace bit-cleanness against
//!     Apr28 GT [[project-qpsi-direct-drift-free-vs-apr28-gt]]).
//!   * `np.argsort(-prom)` to recover the descending-prominence
//!     ordering Apr28 GT was generated with (numpy quicksort tie
//!     behaviour, ~14 % of beat slices have tied prominences).
//!   * `derived_morphology` — depends on the prominence-sorted
//!     extrema and is small enough that the Python ↔ Rust dict
//!     marshalling cost would dominate any Rust port gain.
//!
//! The kernel returns:
//!   * ``scalars`` — (11,) f64 array, ordered [pos_t, pos_a, neg_t,
//!     neg_a, p2p, centroid, duration_rms, duration_fwhm,
//!     max_abs_deriv, rms_amp, snr_db].
//!   * ``extrema`` — (K, 4) f64 array of [time, amp, kind, prom]
//!     rows. Empty (0, 4) if ``n < 7`` or no extremum passes the
//!     prominence threshold.

use crate::extrema::find_extrema_post_smooth_rust;

/// Block size for the unrolled-8 pairwise summation path. Mirrors
/// numpy's ``PW_BLOCKSIZE`` (`numpy/_core/src/umath/loops_utils.h.src`).
const PW_BLOCKSIZE: usize = 128;

/// Bit-identical port of numpy's pairwise summation reduce (the
/// algorithm behind ``arr.sum()`` / ``np.add.reduce``). Matters
/// because float64 addition isn't associative — linear summation
/// drifts from numpy's pairwise tree by ~1 ULP for arrays >= 8
/// elements, which propagates into centroid / duration_rms / rms /
/// snr scalars in measure_window. Three branches:
///   * ``n < 8`` — linear sum (matches numpy's small-n fallback).
///   * ``8 <= n <= PW_BLOCKSIZE`` — 8 unrolled accumulators with the
///     specific combine tree ``((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))``.
///   * ``n > PW_BLOCKSIZE`` — recursive split at ``n/2`` rounded down
///     to a multiple of 8.
fn numpy_sum_pairwise(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 8 {
        let mut s = 0.0_f64;
        for &v in xs {
            s += v;
        }
        return s;
    }
    if n <= PW_BLOCKSIZE {
        let mut r: [f64; 8] = [
            xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7],
        ];
        let tail_start = n - (n % 8);
        let mut i = 8;
        while i < tail_start {
            r[0] += xs[i];
            r[1] += xs[i + 1];
            r[2] += xs[i + 2];
            r[3] += xs[i + 3];
            r[4] += xs[i + 4];
            r[5] += xs[i + 5];
            r[6] += xs[i + 6];
            r[7] += xs[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3]))
            + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += xs[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    numpy_sum_pairwise(&xs[..n2]) + numpy_sum_pairwise(&xs[n2..])
}

/// (scalars_vec, extrema_vec).
pub type MeasureWindowOutput = (Vec<f64>, Vec<(f64, f64, f64, f64)>);

/// 3-point parabolic refinement at index ``i`` of the detrended
/// signal. Mirrors `measure_window._refine` exactly (linear
/// arithmetic; same evaluation order).
fn refine_extremum(x_d: &[f64], t_ms: &[f64], i: usize) -> (f64, f64) {
    let n = x_d.len();
    if i > 0 && i < n - 1 {
        let a = x_d[i - 1];
        let b = x_d[i];
        let c = x_d[i + 1];
        let denom = a - 2.0 * b + c;
        if denom.abs() > 1e-12 {
            let offset = 0.5 * (a - c) / denom;
            let t_peak = t_ms[i] + offset * (t_ms[1] - t_ms[0]);
            let amp = b - 0.25 * (a - c) * offset;
            return (t_peak, amp);
        }
    }
    (t_ms[i], x_d[i])
}

/// numpy.gradient with uniform-spacing fast path. Local copy mirrored
/// from `extrema.rs` so this module doesn't take a cross-module
/// dependency on a private helper.
fn numpy_gradient(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0_f64; n];
    if n < 2 {
        return g;
    }

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

/// numpy.std with ``ddof=0``: ``sqrt(mean((x - mean(x))**2))``.
/// Uses pairwise summation for both reductions to match numpy
/// bit-identically even when the edge window grows past 7 samples
/// (per Python: ``edge = max(3, int(0.10 * n))`` — n=200 → edge=20).
fn numpy_std_ddof0(slice: &[f64]) -> f64 {
    let n = slice.len();
    if n == 0 {
        return 0.0;
    }
    let mean = numpy_sum_pairwise(slice) / (n as f64);
    let mut diffs2 = Vec::with_capacity(n);
    for &v in slice {
        let d = v - mean;
        diffs2.push(d * d);
    }
    (numpy_sum_pairwise(&diffs2) / (n as f64)).sqrt()
}

/// Compute the 11 base scalars + prominence-filtered extrema array
/// for a single (lead, region, beat) window. Mirrors
/// `qpsi.direct_measurements.measure_window` lines 313-372 exactly.
pub fn measure_window_core_rust(
    x_d: &[f64],
    x_s: &[f64],
    t_ms: &[f64],
    prominence_frac: f64,
) -> MeasureWindowOutput {
    let n = x_d.len();
    if n < 5 || x_s.len() != n || t_ms.len() != n {
        return (vec![0.0; 11], Vec::new());
    }

    // ---- Local-edge length (Python: max(3, int(0.10 * n))) ----
    let edge = (3_usize).max((0.10_f64 * n as f64) as usize);

    // ---- Signed extrema indices ----
    // Match numpy.argmax / argmin tie behaviour: return the FIRST
    // occurrence of the maximum / minimum value.
    let mut i_pos: usize = 0;
    let mut i_neg: usize = 0;
    let mut x_max = x_d[0];
    let mut x_min = x_d[0];
    for i in 1..n {
        if x_d[i] > x_max {
            x_max = x_d[i];
            i_pos = i;
        }
        if x_d[i] < x_min {
            x_min = x_d[i];
            i_neg = i;
        }
    }

    // ---- Parabolic refinement of both peaks ----
    let (pos_t, pos_a) = refine_extremum(x_d, t_ms, i_pos);
    let (neg_t, neg_a) = refine_extremum(x_d, t_ms, i_neg);

    // ---- Peak-to-peak (baseline-invariant) ----
    let p2p = pos_a - neg_a;

    // ---- Centroid + second moment of |x_d| ----
    // Match the Python evaluation order: ``(t * abs_x).sum() / norm``
    // followed by ``sqrt(((t - centroid)**2 * abs_x).sum() / norm)``.
    // All reductions use numpy's pairwise summation tree.
    let abs_x: Vec<f64> = x_d.iter().map(|v| v.abs()).collect();
    let norm = numpy_sum_pairwise(&abs_x) + 1e-12;

    let mut t_times_abs = Vec::with_capacity(n);
    for i in 0..n {
        t_times_abs.push(t_ms[i] * abs_x[i]);
    }
    let centroid_ms = numpy_sum_pairwise(&t_times_abs) / norm;

    let mut sigma_terms = Vec::with_capacity(n);
    for i in 0..n {
        let dt = t_ms[i] - centroid_ms;
        sigma_terms.push(dt * dt * abs_x[i]);
    }
    let sigma_t_ms = (numpy_sum_pairwise(&sigma_terms) / norm).sqrt();
    let duration_ms_rms = 4.0 * sigma_t_ms;

    // ---- FWHM (relative to dominant peak) ----
    let dom_a = if pos_a.abs() >= neg_a.abs() {
        pos_a
    } else {
        neg_a
    };
    let duration_ms_fwhm = if dom_a.abs() > 1e-9 {
        let thr = 0.5 * dom_a.abs();
        let mut first_above: Option<usize> = None;
        let mut last_above: Option<usize> = None;
        for i in 0..n {
            if abs_x[i] >= thr {
                if first_above.is_none() {
                    first_above = Some(i);
                }
                last_above = Some(i);
            }
        }
        match (first_above, last_above) {
            (Some(lo), Some(hi)) => t_ms[hi] - t_ms[lo],
            _ => 0.0,
        }
    } else {
        0.0
    };

    // ---- Max absolute derivative (uses x_d, not x_s) ----
    let max_abs_deriv = if n >= 3 {
        let dxdt = numpy_gradient(x_d, t_ms);
        let mut m = 0.0_f64;
        for v in dxdt {
            let av = v.abs();
            if av > m {
                m = av;
            }
        }
        m
    } else {
        0.0
    };

    // ---- RMS amplitude after detrend ----
    // ``(x_d ** 2).mean()`` → pairwise sum of squares / n.
    let mut sq: Vec<f64> = Vec::with_capacity(n);
    for &v in x_d {
        sq.push(v * v);
    }
    let rms_amp = (numpy_sum_pairwise(&sq) / n as f64).sqrt();

    // ---- SNR (peak vs edge std) ----
    let edge_std_lo = numpy_std_ddof0(&x_d[..edge]);
    let edge_std_hi = numpy_std_ddof0(&x_d[n - edge..]);
    let edge_std = edge_std_lo.max(edge_std_hi);
    let num = dom_a.abs().max(1e-9);
    let den = edge_std.max(1e-9);
    let snr_db = 20.0 * (num / den).log10();

    // ---- Extrema list (delegate to find_extrema_post_smooth_rust) ----
    // x_d already detrended; x_s already smoothed by scipy. The kernel
    // returns extrema in time-insertion order — same contract as the
    // bare `find_extrema_post_smooth_native` API.
    let extrema = find_extrema_post_smooth_rust(x_d, x_s, t_ms, prominence_frac);

    let scalars = vec![
        pos_t,
        pos_a,
        neg_t,
        neg_a,
        p2p,
        centroid_ms,
        duration_ms_rms,
        duration_ms_fwhm,
        max_abs_deriv,
        rms_amp,
        snr_db,
    ];

    (scalars, extrema)
}
