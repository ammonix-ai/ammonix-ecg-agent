//! Simple KMeans implementation to replace sklearn.cluster.KMeans.
//!
//! Used for QRS morphology clustering in semantic feature extraction.
//! Implements Lloyd's algorithm with k-means++ initialization.

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// K-means++ initialization: pick initial centroids spread apart.
fn kmeans_plus_plus(data: &[Vec<f64>], k: usize, seed: u64) -> Vec<Vec<f64>> {
    let n = data.len();
    let d = if n > 0 { data[0].len() } else { return vec![] };
    if k == 0 || n == 0 { return vec![] }

    let mut rng_state = seed;
    let mut next_rand = || -> f64 {
        // Simple xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64) / (u64::MAX as f64)
    };

    let mut centroids = Vec::with_capacity(k);
    // First centroid: random point
    let first_idx = ((next_rand() * n as f64) as usize).min(n - 1);
    centroids.push(data[first_idx].clone());

    for _ in 1..k {
        // Compute min distance to existing centroids for each point
        let mut distances: Vec<f64> = data.iter().map(|point| {
            centroids.iter().map(|c| {
                point.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
            }).fold(f64::MAX, f64::min)
        }).collect();

        // Cumulative distribution
        let total: f64 = distances.iter().sum();
        if total < 1e-15 { break; }
        for i in 1..n { distances[i] += distances[i - 1]; }

        let target = next_rand() * total;
        let idx = distances.iter().position(|&d| d >= target).unwrap_or(n - 1);
        centroids.push(data[idx].clone());
    }

    centroids
}

/// Run Lloyd's k-means algorithm.
/// Returns (labels, centroids, inertia).
fn kmeans_lloyd(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<usize>, Vec<Vec<f64>>, f64) {
    let n = data.len();
    let d = if n > 0 { data[0].len() } else { return (vec![], vec![], 0.0) };
    if k == 0 || n == 0 { return (vec![0; n], vec![], 0.0); }
    if k >= n { return ((0..n).collect(), data.to_vec(), 0.0); }

    let mut centroids = kmeans_plus_plus(data, k, seed);
    let mut labels = vec![0usize; n];
    let mut inertia = f64::MAX;

    for _ in 0..max_iter {
        // Assign each point to nearest centroid
        let mut new_inertia = 0.0;
        for i in 0..n {
            let mut best_k = 0;
            let mut best_dist = f64::MAX;
            for j in 0..k {
                let dist: f64 = (0..d).map(|dim| {
                    let diff = data[i][dim] - centroids[j][dim];
                    diff * diff
                }).sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = j;
                }
            }
            labels[i] = best_k;
            new_inertia += best_dist;
        }

        // Check convergence
        if (inertia - new_inertia).abs() < 1e-10 * inertia.abs().max(1e-15) {
            inertia = new_inertia;
            break;
        }
        inertia = new_inertia;

        // Update centroids
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![vec![0.0f64; d]; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for dim in 0..d {
                new_centroids[c][dim] += data[i][dim];
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for dim in 0..d {
                    new_centroids[j][dim] /= counts[j] as f64;
                }
            } else {
                new_centroids[j] = centroids[j].clone();
            }
        }
        centroids = new_centroids;
    }

    (labels, centroids, inertia)
}

/// Silhouette score using centroid distances (O(n*k) instead of O(n²)).
/// For small datasets this is nearly identical to full silhouette.
fn silhouette_score(data: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = data.len();
    if n < 2 || k < 2 { return -1.0; }
    let d = data[0].len();

    // Pre-compute centroids and cluster sizes
    let mut centroids = vec![vec![0.0f64; d]; k];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let c = labels[i];
        counts[c] += 1;
        for dim in 0..d { centroids[c][dim] += data[i][dim]; }
    }
    for j in 0..k {
        if counts[j] > 0 {
            for dim in 0..d { centroids[j][dim] /= counts[j] as f64; }
        }
    }

    let mut total = 0.0;
    for i in 0..n {
        let ci = labels[i];
        // a(i) = distance to own centroid
        let a: f64 = (0..d).map(|dim| {
            let diff = data[i][dim] - centroids[ci][dim];
            diff * diff
        }).sum::<f64>().sqrt();

        // b(i) = min distance to other centroid
        let mut b = f64::MAX;
        for ck in 0..k {
            if ck == ci { continue; }
            let dist: f64 = (0..d).map(|dim| {
                let diff = data[i][dim] - centroids[ck][dim];
                diff * diff
            }).sum::<f64>().sqrt();
            b = b.min(dist);
        }

        let s = if a.max(b) > 0.0 { (b - a) / a.max(b) } else { 0.0 };
        total += s;
    }
    total / n as f64
}

/// Best-of-K KMeans: try k=1..k_max, pick best by silhouette.
///
/// Paired from: qrs_complex.py _kmeans_best_of_k, lines 88-119
#[pyfunction]
pub fn kmeans_best_of_k<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    n_rows: usize,
    n_cols: usize,
    k_min: usize,
    k_max: usize,
    n_init: usize,
    random_state: u64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, usize)> {
    let n = n_rows;
    let d = n_cols;
    let raw = data.as_slice()?;

    // Convert to Vec<Vec<f64>>
    let points: Vec<Vec<f64>> = (0..n)
        .map(|i| raw[i * d..(i + 1) * d].to_vec())
        .collect();

    if n < 3 {
        let labels: Vec<i64> = vec![0; n];
        return Ok((PyArray1::from_vec(py, labels), 1));
    }

    let mut best_k = 1usize;
    let mut best_labels = vec![0i64; n];
    let mut best_score = -2.0f64;

    for k in k_min..=k_max.min(n) {
        if k < 1 { continue; }

        // Run n_init times, pick best inertia
        let mut best_inertia = f64::MAX;
        let mut best_run_labels = vec![0usize; n];

        for init in 0..n_init {
            let seed = random_state.wrapping_add(init as u64 * 1000003);
            let (labels, _, inertia) = kmeans_lloyd(&points, k, 300, seed);
            if inertia < best_inertia {
                best_inertia = inertia;
                best_run_labels = labels;
            }
        }

        let score = if k < 2 {
            -1.0
        } else {
            let n_unique: std::collections::HashSet<usize> = best_run_labels.iter().copied().collect();
            if n_unique.len() < 2 || n_unique.len() >= n {
                -1.0
            } else {
                silhouette_score(&points, &best_run_labels, k)
            }
        };

        if score > best_score {
            best_score = score;
            best_k = k;
            best_labels = best_run_labels.iter().map(|&v| v as i64).collect();
        }
    }

    Ok((PyArray1::from_vec(py, best_labels), best_k))
}
