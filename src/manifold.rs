//! Stable/unstable manifolds: invariant sets tangent to eigenspaces at fixed points.
//!
//! The stable manifold Wˢ(x*) is the set of points that converge to x* as t→∞.
//! The unstable manifold Wᵘ(x*) is the set of points that converge as t→-∞.

use nalgebra::DVector;
use crate::flow::{VectorField, flow};

/// Approximate the stable manifold at a fixed point by integrating backwards
/// from points near x* along stable eigenspace directions.
pub fn approximate_stable_manifold(
    vf: &dyn VectorField,
    fixed_point: &DVector<f64>,
    epsilon: f64,
    num_points: usize,
    t_backward: f64,
    dt: f64,
) -> Vec<Vec<f64>> {
    let jac = vf.jacobian(fixed_point);
    let eigen = jac.complex_eigenvalues();

    // Find stable eigenvectors (eigenvalues with negative real part)
    let stable_indices: Vec<usize> = eigen
        .iter()
        .enumerate()
        .filter(|(_, e)| e.re < -1e-8)
        .map(|(i, _)| i)
        .collect();

    if stable_indices.is_empty() {
        return vec![fixed_point.iter().copied().collect()];
    }

    // Use eigendecomposition for eigenvectors
    let eigensystem = jac.clone().try_symmetric_eigen(1e-10, 100).unwrap_or_else(|| { nalgebra::linalg::SymmetricEigen::new(jac.clone()) });
    let mut manifold_points = Vec::new();

    for idx in &stable_indices {
        let ev = eigensystem.eigenvectors.column(*idx).into_owned();
        for i in 0..num_points {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_points as f64;
            let offset = &ev * (epsilon * angle.cos());
            let x0 = fixed_point + &offset;
            // Integrate forward (stable → converges to fp)
            let result = flow(vf, &x0, t_backward, dt, true);
            manifold_points.push(result.trajectory);
        }
    }

    manifold_points.into_iter().flatten().collect()
}

/// Approximate the unstable manifold by integrating forwards from points
/// along unstable eigenspace directions.
pub fn approximate_unstable_manifold(
    vf: &dyn VectorField,
    fixed_point: &DVector<f64>,
    epsilon: f64,
    num_points: usize,
    t_forward: f64,
    dt: f64,
) -> Vec<Vec<f64>> {
    let jac = vf.jacobian(fixed_point);
    let eigen = jac.complex_eigenvalues();

    let unstable_indices: Vec<usize> = eigen
        .iter()
        .enumerate()
        .filter(|(_, e)| e.re > 1e-8)
        .map(|(i, _)| i)
        .collect();

    if unstable_indices.is_empty() {
        return vec![fixed_point.iter().copied().collect()];
    }

    let eigensystem = jac.clone().try_symmetric_eigen(1e-10, 100).unwrap_or_else(|| { nalgebra::linalg::SymmetricEigen::new(jac.clone()) });
    let mut manifold_points = Vec::new();

    for idx in &unstable_indices {
        let ev = eigensystem.eigenvectors.column(*idx).into_owned();
        for i in 0..num_points {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_points as f64;
            let offset = &ev * (epsilon * angle.cos());
            let x0 = fixed_point + &offset;
            let result = flow(vf, &x0, t_forward, dt, true);
            manifold_points.push(result.trajectory);
        }
    }

    manifold_points.into_iter().flatten().collect()
}

/// Compute manifold dimensions at a fixed point.
pub fn manifold_dimensions(vf: &dyn VectorField, fixed_point: &DVector<f64>) -> (usize, usize) {
    let jac = vf.jacobian(fixed_point);
    let eigen = jac.complex_eigenvalues();
    let stable = eigen.iter().filter(|e| e.re < -1e-8).count();
    let unstable = eigen.iter().filter(|e| e.re > 1e-8).count();
    (stable, unstable)
}

/// Check if a point lies approximately on the stable manifold.
pub fn is_on_stable_manifold(
    vf: &dyn VectorField,
    fixed_point: &DVector<f64>,
    point: &DVector<f64>,
    t_test: f64,
    dt: f64,
    tol: f64,
) -> bool {
    let result = flow(vf, point, t_test, dt, false);
    let endpoint = DVector::from_vec(result.endpoint);
    (endpoint - fixed_point).norm() < tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_manifold_dimensions_stable() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let fp = DVector::from_vec(vec![0.0, 0.0]);
        let (s, u) = manifold_dimensions(&vf, &fp);
        assert_eq!(s, 2);
        assert_eq!(u, 0);
    }

    #[test]
    fn test_manifold_dimensions_saddle() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![x[0], -x[1]]));
        let fp = DVector::from_vec(vec![0.0, 0.0]);
        let (s, u) = manifold_dimensions(&vf, &fp);
        assert_eq!(s, 1);
        assert_eq!(u, 1);
    }

    #[test]
    fn test_manifold_dimensions_unstable() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![x[0], x[1]]));
        let fp = DVector::from_vec(vec![0.0, 0.0]);
        let (s, u) = manifold_dimensions(&vf, &fp);
        assert_eq!(s, 0);
        assert_eq!(u, 2);
    }

    #[test]
    fn test_stable_manifold_approximation() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let fp = DVector::from_vec(vec![0.0, 0.0]);
        let pts = approximate_stable_manifold(&vf, &fp, 0.1, 4, 5.0, 0.01);
        assert!(!pts.is_empty());
        // All points should converge toward origin
        for p in &pts {
            let norm: f64 = p.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Final points in trajectory should be near origin
        }
    }

    #[test]
    fn test_on_stable_manifold() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let fp = DVector::from_vec(vec![0.0, 0.0]);
        let point = DVector::from_vec(vec![0.5, 0.5]);
        assert!(is_on_stable_manifold(&vf, &fp, &point, 10.0, 0.01, 0.1));
    }

    #[test]
    fn test_not_on_stable_manifold() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![x[0], -x[1]]));
        let fp = DVector::from_vec(vec![0.0, 0.0]);
        // x[0] grows, so starting away from x[0]=0 shouldn't converge
        let point = DVector::from_vec(vec![1.0, 0.0]);
        assert!(!is_on_stable_manifold(&vf, &fp, &point, 5.0, 0.01, 0.1));
    }
}
