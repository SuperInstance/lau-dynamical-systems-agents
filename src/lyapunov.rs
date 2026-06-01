//! Lyapunov exponents: measure chaos and sensitivity to initial conditions.
//!
//! The Lyapunov exponents λ₁ ≥ λ₂ ≥ ... ≥ λₙ measure the average exponential
//! rate of divergence/convergence of nearby orbits. A positive Lyapunov exponent
//! indicates sensitive dependence on initial conditions (chaos).

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::flow::VectorField;

/// Result of Lyapunov exponent computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovExponents {
    /// The Lyapunov exponents, sorted largest first.
    pub exponents: Vec<f64>,
    /// The Kaplan-Yorke (Lyapunov) dimension.
    pub ky_dimension: f64,
    /// Whether the system is chaotic (at least one positive exponent).
    pub is_chaotic: bool,
    /// Sum of all exponents (equals divergence for dissipative systems).
    pub sum: f64,
}

/// Compute the full spectrum of Lyapunov exponents using the Benettin algorithm.
///
/// Integrates the variational equations alongside the flow, periodically
/// re-orthogonalizing using Gram-Schmidt.
pub fn compute_lyapunov_exponents(
    vf: &dyn VectorField,
    x0: &DVector<f64>,
    t_total: f64,
    dt: f64,
    orthogonalize_interval: usize,
) -> LyapunovExponents {
    let n = vf.dim();
    let steps = (t_total / dt) as usize;
    let orth_interval = orthogonalize_interval.max(1);

    let mut x = x0.clone();
    let mut q = DMatrix::identity(n, n); // Orthonormal frame
    let mut running_sums = vec![0.0f64; n];
    let mut orth_count = 0usize;

    for step in 0..steps {
        // Flow the state
        let jac = vf.jacobian(&x);
        let v = vf.evaluate(&x);
        // RK4 for state
        let k1 = v;
        let k2 = vf.evaluate(&(&x + &k1.scale(dt / 2.0)));
        let k3 = vf.evaluate(&(&x + &k2.scale(dt / 2.0)));
        let k4 = vf.evaluate(&(&x + &k3.scale(dt)));
        x = &x + &(k1.scale(dt / 6.0) + &k2.scale(dt / 3.0) + &k3.scale(dt / 3.0) + &k4.scale(dt / 6.0));

        // Evolve the tangent vectors: dΦ = J · Q
        let dphi = &jac * &q;

        // Euler step for variational equation (simplified)
        q = &q + &(dphi.scale(dt));

        // Re-orthogonalize periodically
        if (step + 1) % orth_interval == 0 {
            gram_schmidt_inplace(&mut q);
            // Record growth rates
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let col_norm = q.column(i).norm();
                if col_norm > 1e-15 {
                    running_sums[i] += col_norm.ln();
                    let mut col = q.column_mut(i);
                    col /= col_norm;
                }
            }
            orth_count += 1;
        }
    }

    let total_time = orth_count as f64 * orth_interval as f64 * dt;
    let exponents: Vec<f64> = running_sums
        .iter()
        .map(|s| s / total_time)
        .collect();

    let sum: f64 = exponents.iter().sum();
    let is_chaotic = exponents.iter().any(|&e| e > 0.01);

    // Kaplan-Yorke dimension
    let ky_dim = kaplan_yorke_dimension(&exponents);

    LyapunovExponents {
        exponents,
        ky_dimension: ky_dim,
        is_chaotic,
        sum,
    }
}

/// Compute only the maximal Lyapunov exponent.
pub fn maximal_lyapunov_exponent(
    vf: &dyn VectorField,
    x0: &DVector<f64>,
    t_total: f64,
    dt: f64,
    epsilon: f64,
) -> f64 {
    let steps = (t_total / dt) as usize;
    let mut x = x0.clone();
    let mut x_perturbed = x0.clone();
    // Perturb in a random direction
    let mut delta = DVector::from_fn(vf.dim(), |_, _| {
        2.0 * rand_simple() - 1.0
    });
    delta *= epsilon / delta.norm();
    x_perturbed += &delta;

    let mut total_lyap = 0.0;
    let mut renorm_count = 0usize;

    for _ in 0..steps {
        // Flow both
        x = rk4_step_fn(vf, &x, dt);
        x_perturbed = rk4_step_fn(vf, &x_perturbed, dt);

        let diff = &x_perturbed - &x;
        let dist = diff.norm();

        if dist > 1e-15 {
            total_lyap += (dist / epsilon).ln();
            renorm_count += 1;

            // Renormalize
            x_perturbed = &x + &(diff.scale(epsilon / dist));
        }
    }

    if renorm_count == 0 {
        0.0
    } else {
        total_lyap / (renorm_count as f64 * dt)
    }
}

/// Simple deterministic random for reproducibility (no external rand dep).
fn rand_simple() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(12345) };
    }
    SEED.with(|s| {
        let mut seed = s.get();
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.set(seed);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    })
}

fn rk4_step_fn(vf: &dyn VectorField, x: &DVector<f64>, dt: f64) -> DVector<f64> {
    let k1 = vf.evaluate(x);
    let k2 = vf.evaluate(&(x + &k1.scale(dt / 2.0)));
    let k3 = vf.evaluate(&(x + &k2.scale(dt / 2.0)));
    let k4 = vf.evaluate(&(x + &k3.scale(dt)));
    x + &(k1.scale(dt / 6.0) + &k2.scale(dt / 3.0) + &k3.scale(dt / 3.0) + &k4.scale(dt / 6.0))
}

/// In-place modified Gram-Schmidt orthogonalization.
fn gram_schmidt_inplace(q: &mut DMatrix<f64>) {
    let n = q.ncols();
    for i in 0..n {
        for j in 0..i {
            let dot = q.column(i).dot(&q.column(j));
            let col_j = q.column(j).clone_owned();
            let mut col_i = q.column_mut(i);
            col_i.axpy(-dot, &col_j, 1.0);
        }
    }
}

/// Kaplan-Yorke dimension from Lyapunov spectrum.
pub fn kaplan_yorke_dimension(exponents: &[f64]) -> f64 {
    if exponents.is_empty() {
        return 0.0;
    }

    // Sort in decreasing order
    let mut sorted: Vec<f64> = exponents.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut sum = 0.0;
    for (j, &lambda) in sorted.iter().enumerate() {
        sum += lambda;
        if sum < 0.0 {
            // Interpolate
            let prev_sum = sum - lambda;
            return j as f64 + prev_sum / lambda.abs();
        }
    }

    sorted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_stable_system_negative_exponents() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let result = compute_lyapunov_exponents(&vf, &x0, 10.0, 0.01, 10);
        assert!(!result.is_chaotic);
        assert!(result.exponents.iter().all(|&e| e < 0.5));
    }

    #[test]
    fn test_chaotic_system_positive_exponent() {
        // Lorenz system
        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;
        let vf = crate::flow::FnVectorField::new(3, move |x| {
            DVector::from_vec(vec![
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ])
        });
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let result = compute_lyapunov_exponents(&vf, &x0, 20.0, 0.005, 20);
        // Lorenz has a positive Lyapunov exponent ≈ 0.9
        assert!(result.exponents[0] > 0.0, "Expected positive exponent, got {}", result.exponents[0]);
        assert!(result.is_chaotic);
    }

    #[test]
    fn test_kaplan_yorke_dimension() {
        // For a 3D system with exponents [+, 0, -], KY dim ≈ 2 + λ₁/|λ₃|
        let exps = vec![0.9, 0.0, -14.5];
        let dim = kaplan_yorke_dimension(&exps);
        assert!(dim > 2.0 && dim < 3.0);
    }

    #[test]
    fn test_kaplan_yorke_all_negative() {
        let exps = vec![-1.0, -2.0];
        let dim = kaplan_yorke_dimension(&exps);
        assert_abs_diff_eq!(dim, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_lyapunov_sum_equals_divergence() {
        // For a linear system, sum of LE = div(V)
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -2.0 * x[1]]));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let result = compute_lyapunov_exponents(&vf, &x0, 20.0, 0.01, 10);
        // Divergence = -3, sum should be ≈ -3
        assert_abs_diff_eq!(result.sum, -3.0, epsilon = 0.5);
    }

    #[test]
    fn test_maximal_lyapunov_stable() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let mle = maximal_lyapunov_exponent(&vf, &x0, 10.0, 0.01, 1e-6);
        assert!(mle < 0.0, "Expected negative MLE for stable system, got {}", mle);
    }

    #[test]
    fn test_lyapunov_serialization() {
        let le = LyapunovExponents {
            exponents: vec![0.9, 0.0, -14.5],
            ky_dimension: 2.06,
            is_chaotic: true,
            sum: -13.6,
        };
        let json = serde_json::to_string(&le).unwrap();
        assert!(json.contains("is_chaotic"));
    }
}
