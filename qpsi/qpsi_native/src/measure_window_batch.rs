//! Rayon-parallel batched dispatcher for measure_window_core_rust.
//!
//! Phase B.5 (Brief 2) — Python pre-batches all per-window inputs
//! (detrended `x_d`, scipy-smoothed `x_s`, time slice `t_ms`), packs
//! them as flat concatenated arrays with `(offsets, lengths)`
//! metadata, hands to this kernel. The kernel runs
//! `measure_window::measure_window_core_rust` on every window in
//! parallel via rayon, then packs results back into flat output
//! arrays for Python to unpack into the per-beat / per-lead /
//! per-region / per-plane / per-region nested dict structure.
//!
//! Why batch instead of per-call: with `qpsi_native::measure_window_
//! core_native` already in place from Item #2a, the dominant residual
//! cost of `measure_all_beats` is two-fold: (a) ~504 Python frames per
//! record × ~30 µs Python overhead = ~15 ms/record; (b) serial work in
//! a Python loop over 504 windows. Batching collapses (a) to a single
//! PyO3 boundary crossing and replaces (b) with rayon parallelism
//! across CPU cores. Savgol stays scipy-side per the Brief 1'
//! decision (see `qpsi/savgol_batch.py` + FIDELITY_REPORT Phase H.C).

use crate::measure_window::measure_window_core_rust;
use rayon::prelude::*;

/// Per-window output of the batched dispatcher.
///
/// `scalars` is an 11-element vec (same ordering as
/// `measure_window_core_rust`). `extrema` is the variable-length
/// `(time, amp, kind, prominence)` list in time-insertion order
/// (Python applies `np.argsort(-prom)` to recover the canonical
/// descending-prominence ordering, matching Item #1's hybrid contract).
pub type BatchOutput = Vec<(Vec<f64>, Vec<(f64, f64, f64, f64)>)>;

/// Process N independent windows in parallel via rayon.
///
/// Inputs:
///   * `x_d_flat` / `x_s_flat` / `t_ms_flat` — concatenated per-window
///     data (caller provides identical layout for all three; one window
///     `i` spans `[offsets[i] .. offsets[i] + lengths[i])` in each).
///   * `offsets` — start index of each window in the flat arrays.
///   * `lengths` — number of samples per window.
///   * `prominence_frac` — same parameter as `find_extrema_post_smooth`.
///
/// Output: one `BatchOutput` entry per window. Always returns exactly
/// `offsets.len()` entries — empty `extrema` indicates either an
/// insufficient-data short-circuit (`n < 5`) or no extremum passed the
/// prominence threshold.
pub fn measure_window_batch_rust(
    x_d_flat: &[f64],
    x_s_flat: &[f64],
    t_ms_flat: &[f64],
    offsets: &[usize],
    lengths: &[usize],
    prominence_frac: f64,
) -> BatchOutput {
    let n_windows = offsets.len();
    assert_eq!(
        lengths.len(),
        n_windows,
        "measure_window_batch: lengths len {} != offsets len {}",
        lengths.len(),
        n_windows
    );

    // Pre-allocate the result vector so rayon can index into it.
    let mut results: Vec<(Vec<f64>, Vec<(f64, f64, f64, f64)>)> =
        (0..n_windows).map(|_| (Vec::new(), Vec::new())).collect();

    // Rayon-parallel across windows. Each worker reads its own slices
    // from the flat arrays — no shared mutable state, no GIL involved
    // (PyO3 boundary is crossed once by the caller, releasing GIL for
    // the duration of this call).
    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            let start = offsets[i];
            let n = lengths[i];
            let end = start + n;
            // Bounds-safe: callers must ensure flat arrays are long
            // enough; if not, treat as empty (matches the n<5 short-
            // circuit in measure_window_core_rust).
            if end > x_d_flat.len() || end > x_s_flat.len() || end > t_ms_flat.len() {
                *slot = (vec![0.0; 11], Vec::new());
                return;
            }
            let x_d = &x_d_flat[start..end];
            let x_s = &x_s_flat[start..end];
            let t_ms = &t_ms_flat[start..end];
            *slot = measure_window_core_rust(x_d, x_s, t_ms, prominence_frac);
        });

    results
}
