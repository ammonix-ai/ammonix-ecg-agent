//! Batched Rust port of `qpsi.direct_measurements.derived_morphology`.
//!
//! See `qpsi/direct_measurements.py:221-314` for the full reference.
//! This kernel processes N windows in a single PyO3 boundary crossing,
//! replacing the per-window Python call inside Brief 2's
//! `_measure_all_beats_via_batch` (qpsi/direct_measurements.py:1006).
//!
//! Inputs are passed as flat arrays per the same encoding Brief 2 uses:
//! the prominence-sorted extrema list per window is concatenated into
//! `extrema_flat` (shape `(total_extrema, 4)` rows of
//! `[time_ms, amp_mv, kind, prominence]`), with `extrema_counts[i]`
//! giving the window's count.
//!
//! Output is a `(n_windows, 7)` matrix; row order matches the 7
//! `derived_morphology` output keys:
//!   0. n_significant_extrema
//!   1. slope_reversal_count
//!   2. biphasic_ratio
//!   3. notch_depth_mv
//!   4. notch_position_frac
//!   5. saddle_depth_relative
//!   6. monotonic_fraction
//!
//! Bit-equality with the Python reference is preserved through:
//!   * `np_sign` helper matching `np.sign(±0.0) == 0.0` (Rust's
//!     `f64::signum` returns ±1.0 for ±0.0 — different).
//!   * `max(iter, default=0.0)` / `min(iter, default=0.0)` via
//!     `reduce(f64::max).unwrap_or(0.0)`.
//!   * `np.argmax(np.abs(amps))` reproduced by a linear scan that
//!     keeps the FIRST occurrence on ties.
//!   * Stable sort by `time_ms` via `sort_by` (Rust's `sort_by` is
//!     stable; matches Python's Timsort).

/// One extremum: `(time_ms, amp_mv, kind, prominence)`.
type Extremum = (f64, f64, f64, f64);

/// `numpy.sign(x)` — returns 0.0 for both +0.0 and -0.0 (unlike
/// Rust's `f64::signum` which returns ±1.0 for signed zeros).
#[inline]
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Compute the 7 morphology scalars for one window's sorted extrema list.
///
/// `extrema` is the prominence-sorted list (the Python caller has already
/// applied `np.argsort(-prom)`); this function chronologically re-sorts
/// internally per the reference.
pub fn derived_morphology_one(
    extrema: &[Extremum],
    window_size_ms: f64,
) -> [f64; 7] {
    let n_extr = extrema.len();

    if n_extr == 0 {
        // Empty-extrema sentinel: monotonic_fraction = 1.0; all else 0.
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    }

    // Chronological sort (stable; matches Python's `sorted(... key=time_ms)`)
    let mut chrono: Vec<Extremum> = extrema.to_vec();
    chrono.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("time_ms must be finite"));

    let n_reversals = (n_extr.saturating_sub(1)) as f64;

    // ----- Biphasic ratio -----
    // dom_max = max amp over kind > 0, default 0.0; dom_min = min over kind < 0, default 0.0
    let dom_max = extrema
        .iter()
        .filter(|e| e.2 > 0.0)
        .map(|e| e.1)
        .reduce(f64::max)
        .unwrap_or(0.0);
    let dom_min = extrema
        .iter()
        .filter(|e| e.2 < 0.0)
        .map(|e| e.1)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let biphasic_ratio = if dom_max.abs() > 1e-9 && dom_min.abs() > 1e-9 {
        let a = dom_max.abs();
        let b = dom_min.abs();
        a.min(b) / a.max(b)
    } else {
        0.0
    };

    // ----- Notch detection (chronological scan) -----
    let mut notch_depth: f64 = 0.0;
    let mut notch_pos: f64 = 0.0;
    if n_extr >= 3 {
        for i in 1..(n_extr - 1) {
            let (t_prev, amp_prev, kind_prev, _) = chrono[i - 1];
            let (t_i, amp_i, kind_i, _) = chrono[i];
            let (t_next, amp_next, kind_next, _) = chrono[i + 1];
            // Pattern 1: max-min-max (positive notch)
            if kind_prev > 0.0 && kind_i < 0.0 && kind_next > 0.0 {
                let depth = amp_prev.min(amp_next) - amp_i;
                if depth > notch_depth {
                    notch_depth = depth;
                    let denom = (t_next - t_prev).max(1e-6);
                    notch_pos = (t_i - t_prev) / denom;
                }
            } else if kind_prev < 0.0 && kind_i > 0.0 && kind_next < 0.0 {
                // Pattern 2: min-max-min (inverted notch)
                let depth = amp_i - amp_prev.max(amp_next);
                if depth > notch_depth {
                    notch_depth = depth;
                    let denom = (t_next - t_prev).max(1e-6);
                    notch_pos = (t_i - t_prev) / denom;
                }
            }
        }
    }

    // ----- Saddle (shoulder on rising side of dominant peak) -----
    let saddle_rel = if n_extr >= 2 {
        // np.argmax(np.abs(amps)) — FIRST occurrence on ties.
        let mut dom_idx: usize = 0;
        let mut dom_abs: f64 = chrono[0].1.abs();
        for i in 1..n_extr {
            let cur = chrono[i].1.abs();
            if cur > dom_abs {
                dom_abs = cur;
                dom_idx = i;
            }
        }
        let dom_amp = chrono[dom_idx].1;
        let dom_t = chrono[dom_idx].0;
        let dom_sign = np_sign(dom_amp);
        let mut best: f64 = 0.0;
        for i in 0..n_extr {
            if i == dom_idx {
                continue;
            }
            let (t_e, amp_e, _, _) = chrono[i];
            if t_e >= dom_t {
                continue; // only earlier shoulders
            }
            if np_sign(amp_e) != dom_sign {
                continue; // same-sign only
            }
            if amp_e.abs() >= dom_amp.abs() {
                continue; // not the dominant
            }
            let denom = dom_amp.abs().max(1e-6);
            let rel = amp_e.abs() / denom;
            if rel > best {
                best = rel;
            }
        }
        best
    } else {
        0.0
    };

    // ----- Monotonic fraction -----
    let monotonic_fraction = if n_extr <= 1 {
        1.0
    } else {
        let total_span = chrono[n_extr - 1].0 - chrono[0].0;
        let denom = window_size_ms.max(1e-6);
        let raw = 1.0 - total_span / denom;
        raw.max(0.0)
    };

    [
        n_extr as f64,
        n_reversals,
        biphasic_ratio,
        notch_depth,
        notch_pos,
        saddle_rel,
        monotonic_fraction,
    ]
}

/// Batched dispatcher — N windows in one call.
///
/// `extrema_flat`: concatenated (time, amp, kind, prom) rows.
/// `extrema_counts[i]`: number of extrema in window `i`.
/// `window_sizes_ms[i]`: time span of window `i`.
///
/// Returns a flat `Vec<f64>` of length `n_windows * 7`, row-major
/// (one window per row; column order matches `derived_morphology_one`'s
/// output array). The caller reshapes to (n_windows, 7).
pub fn derived_morphology_batch_rust(
    extrema_flat: &[f64],
    extrema_counts: &[i64],
    window_sizes_ms: &[f64],
) -> Vec<f64> {
    let n_windows = extrema_counts.len();
    assert_eq!(
        window_sizes_ms.len(),
        n_windows,
        "derived_morphology_batch: window_sizes_ms len {} != extrema_counts len {}",
        window_sizes_ms.len(),
        n_windows
    );

    let mut out = vec![0.0_f64; n_windows * 7];
    let mut row_offset: usize = 0;

    for (wi, &count_i64) in extrema_counts.iter().enumerate() {
        let count = count_i64 as usize;
        let row_end = row_offset + count;

        // Build per-window Extremum vector (length == count)
        let mut window_extrema: Vec<Extremum> = Vec::with_capacity(count);
        for r in row_offset..row_end {
            let base = r * 4;
            if base + 4 > extrema_flat.len() {
                // Defensive: caller passed misaligned data — leave zeros.
                break;
            }
            window_extrema.push((
                extrema_flat[base],
                extrema_flat[base + 1],
                extrema_flat[base + 2],
                extrema_flat[base + 3],
            ));
        }

        let morph = derived_morphology_one(&window_extrema, window_sizes_ms[wi]);
        let out_base = wi * 7;
        out[out_base..out_base + 7].copy_from_slice(&morph);

        row_offset = row_end;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_extrema_returns_monotonic_one() {
        let out = derived_morphology_one(&[], 100.0);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn single_extremum() {
        let extrema = vec![(10.0, 1.0, 1.0, 0.5)];
        let out = derived_morphology_one(&extrema, 100.0);
        assert_eq!(out[0], 1.0); // n_significant_extrema
        assert_eq!(out[1], 0.0); // slope_reversal_count
        assert_eq!(out[6], 1.0); // monotonic_fraction (n_extr <= 1)
    }

    #[test]
    fn np_sign_handles_signed_zeros() {
        assert_eq!(np_sign(0.0), 0.0);
        assert_eq!(np_sign(-0.0), 0.0); // differs from f64::signum
        assert_eq!(np_sign(1.0), 1.0);
        assert_eq!(np_sign(-1.0), -1.0);
    }

    #[test]
    fn notch_detection_max_min_max() {
        // Three peaks chronologically: max-min-max with valley shallower than peaks
        let extrema = vec![
            (10.0, 1.0, 1.0, 0.8),
            (20.0, 0.3, -1.0, 0.4),
            (30.0, 0.9, 1.0, 0.5),
        ];
        let out = derived_morphology_one(&extrema, 100.0);
        // depth = min(peak1, peak2) - valley = min(1.0, 0.9) - 0.3 = 0.6
        assert!((out[3] - 0.6).abs() < 1e-12, "notch_depth = {}", out[3]);
        // position = (20 - 10) / (30 - 10) = 0.5
        assert!((out[4] - 0.5).abs() < 1e-12, "notch_pos = {}", out[4]);
    }

    #[test]
    fn batch_dispatcher() {
        // Two windows: one empty, one with a notch
        let extrema_flat = vec![
            10.0, 1.0, 1.0, 0.8,
            20.0, 0.3, -1.0, 0.4,
            30.0, 0.9, 1.0, 0.5,
        ];
        let counts = vec![0_i64, 3_i64];
        let window_sizes = vec![100.0, 100.0];
        let out = derived_morphology_batch_rust(&extrema_flat, &counts, &window_sizes);
        assert_eq!(out.len(), 14); // 2 windows × 7 = 14
        // Window 0 (empty): [0, 0, 0, 0, 0, 0, 1.0]
        assert_eq!(&out[0..7], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        // Window 1: n_extrema=3, notch_depth=0.6
        assert_eq!(out[7], 3.0);
        assert!((out[10] - 0.6).abs() < 1e-12);
    }
}
