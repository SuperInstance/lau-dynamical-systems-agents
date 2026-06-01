//! Linearization: Jacobian at fixed point → eigenvalue stability analysis.
//!
//! Near a fixed point x*, the dynamics are approximately dx = J(x*) · dx,
//! where J is the Jacobian. Eigenvalues of J determine local behavior.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::flow::VectorField;

/// Result of linearization analysis at a point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearizationResult {
    /// The Jacobian matrix at the fixed point.
    pub jacobian: Vec<Vec<f64>>,
    /// Eigenvalues (complex).
    pub eigenvalues: Vec<(f64, f64)>,
    /// Eigenvectors (real parts of complex eigenvectors).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Whether the fixed point is hyperbolic (no eigenvalue has zero real part).
    pub is_hyperbolic: bool,
    /// Trace of the Jacobian.
    pub trace: f64,
    /// Determinant of the Jacobian.
    pub determinant: f64,
}

/// Perform full linearization analysis at a point.
pub fn linearize(vf: &dyn VectorField, point: &DVector<f64>) -> LinearizationResult {
    let jac = vf.jacobian(point);
    let eigen = jac.complex_eigenvalues();

    let trace = jac.trace();
    let det = jac.determinant();

    let eigenvalues: Vec<(f64, f64)> = eigen.iter().map(|e| (e.re, e.im)).collect();

    let is_hyperbolic = eigenvalues.iter().all(|(re, _)| re.abs() > 1e-8);

    let eigensystem = jac.clone().try_symmetric_eigen(1e-10, 100).unwrap_or_else(|| {
        nalgebra::linalg::SymmetricEigen::new(jac.clone())
    });
    let eigenvectors: Vec<Vec<f64>> = (0..vf.dim())
        .map(|i| {
            if i < eigensystem.eigenvectors.ncols() {
                eigensystem.eigenvectors.column(i).iter().copied().collect()
            } else {
                vec![0.0; vf.dim()]
            }
        })
        .collect();

    LinearizationResult {
        jacobian: jac.row_iter().map(|r| r.iter().copied().collect()).collect(),
        eigenvalues,
        eigenvectors,
        is_hyperbolic,
        trace,
        determinant: det,
    }
}

/// Compute the Hartman-Grobman linearized flow map.
/// Given the linearization at x*, compute φ₁(x) ≈ x* + exp(J·t)·(x - x*) for small deviations.
pub fn linearized_flow(
    vf: &dyn VectorField,
    fixed_point: &DVector<f64>,
    x: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    let jac = vf.jacobian(fixed_point);
    let delta = x - fixed_point;
    // Matrix exponential via eigendecomposition (for small matrices)
    let exp_jt = matrix_exp(&jac, t);
    fixed_point + &exp_jt * &delta
}

/// Compute matrix exponential via eigendecomposition.
/// For a diagonalizable matrix A = PDP⁻¹, exp(At) = P·exp(Dt)·P⁻¹.
pub fn matrix_exp(a: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    let n = a.nrows();
    if n == 1 {
        return DMatrix::from_vec(1, 1, vec![a[(0, 0)].exp()]);
    }

    // Use Padé approximation via scaling and squaring
    // For simplicity, use eigendecomposition approach
    let _eigen = a.complex_eigenvalues();

    // For each eigenvalue λ, compute exp(λt)
    // Build exp(At) using the formula with projections
    // Fallback: Taylor series with scaling and squaring
    let nsteps = 20;
    let scale = 2.0_f64.powi(nsteps);
    let scaled_t = t / scale;

    // Taylor: exp(A·scaled_t) ≈ I + A·scaled_t + (A·scaled_t)²/2! + ...
    let mut result = DMatrix::identity(n, n);
    let mut term = DMatrix::identity(n, n);
    for k in 1..=30 {
        term = term * &(a.clone() * scaled_t) / k as f64;
        result += &term;
    }

    // Squaring: exp(A·t) = [exp(A·t/2^n)]^{2^n}
    for _ in 0..nsteps {
        result = &result * &result;
    }

    result
}

/// Classify a 2D fixed point from trace/determinant of Jacobian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedPointType2D {
    StableNode,
    UnstableNode,
    StableSpiral,
    UnstableSpiral,
    Center,
    Saddle,
    Degenerate,
}

impl std::fmt::Display for FixedPointType2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Classify a 2D fixed point using trace τ and determinant Δ.
///
/// - Δ < 0: saddle
/// - Δ > 0, τ < 0, τ² > 4Δ: stable node
/// - Δ > 0, τ < 0, τ² < 4Δ: stable spiral
/// - Δ > 0, τ > 0, τ² > 4Δ: unstable node
/// - Δ > 0, τ > 0, τ² < 4Δ: unstable spiral
/// - Δ > 0, τ = 0: center
pub fn classify_2d(trace: f64, determinant: f64) -> FixedPointType2D {
    let tau = trace;
    let delta = determinant;
    if delta < 0.0 {
        FixedPointType2D::Saddle
    } else if delta > 0.0 {
        if tau.abs() < 1e-10 {
            FixedPointType2D::Center
        } else if tau < 0.0 {
            if tau * tau > 4.0 * delta {
                FixedPointType2D::StableNode
            } else {
                FixedPointType2D::StableSpiral
            }
        } else {
            if tau * tau > 4.0 * delta {
                FixedPointType2D::UnstableNode
            } else {
                FixedPointType2D::UnstableSpiral
            }
        }
    } else {
        FixedPointType2D::Degenerate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linearize_stable_node() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-2.0 * x[0], -3.0 * x[1]]));
        let pt = DVector::from_vec(vec![0.0, 0.0]);
        let result = linearize(&vf, &pt);
        assert!(result.is_hyperbolic);
        assert_abs_diff_eq!(result.trace, -5.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.determinant, 6.0, epsilon = 1e-5);
        // Eigenvalues should be -2 and -3
        let real_parts: Vec<f64> = result.eigenvalues.iter().map(|(re, _)| *re).collect();
        assert!(real_parts.iter().all(|&r| r < 0.0));
    }

    #[test]
    fn test_linearize_spiral() {
        // dx/dt = -x + y, dy/dt = -x - y → spiral
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0] + x[1], -x[0] - x[1]]));
        let pt = DVector::from_vec(vec![0.0, 0.0]);
        let result = linearize(&vf, &pt);
        assert!(result.eigenvalues.iter().any(|(_, im)| im.abs() > 0.1));
    }

    #[test]
    fn test_classify_2d_saddle() {
        assert_eq!(classify_2d(0.0, -1.0), FixedPointType2D::Saddle);
    }

    #[test]
    fn test_classify_2d_stable_node() {
        assert_eq!(classify_2d(-5.0, 6.0), FixedPointType2D::StableNode);
    }

    #[test]
    fn test_classify_2d_stable_spiral() {
        // τ² < 4Δ: τ=-2, Δ=2 → 4 < 8 ✓
        assert_eq!(classify_2d(-2.0, 2.0), FixedPointType2D::StableSpiral);
    }

    #[test]
    fn test_classify_2d_unstable_node() {
        assert_eq!(classify_2d(5.0, 6.0), FixedPointType2D::UnstableNode);
    }

    #[test]
    fn test_classify_2d_unstable_spiral() {
        assert_eq!(classify_2d(2.0, 2.0), FixedPointType2D::UnstableSpiral);
    }

    #[test]
    fn test_classify_2d_center() {
        assert_eq!(classify_2d(0.0, 1.0), FixedPointType2D::Center);
    }

    #[test]
    fn test_linearized_flow() {
        let vf = crate::flow::FnVectorField::new(1, |x| DVector::from_vec(vec![-x[0]]));
        let fp = DVector::from_vec(vec![0.0]);
        let x = DVector::from_vec(vec![1.0]);
        let result = linearized_flow(&vf, &fp, &x, 1.0);
        assert_abs_diff_eq!(result[0], (-1.0f64).exp(), epsilon = 1e-4);
    }

    #[test]
    fn test_matrix_exp_identity() {
        let a = DMatrix::identity(2, 2);
        let exp = matrix_exp(&a, 1.0);
        assert_abs_diff_eq!(exp[(0, 0)], 1.0_f64.exp(), epsilon = 1e-6);
        assert_abs_diff_eq!(exp[(1, 1)], 1.0_f64.exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_non_hyperbolic() {
        // dx/dt = -y, dy/dt = x → eigenvalues ±i
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[1], x[0]]));
        let pt = DVector::from_vec(vec![0.0, 0.0]);
        let result = linearize(&vf, &pt);
        assert!(!result.is_hyperbolic);
    }
}
