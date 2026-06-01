//! Fixed points: equilibria where agent stops changing.
//!
//! A fixed point x* satisfies V(x*) = 0. Classification via eigenvalues
//! of the Jacobian: all negative real parts → stable, any positive → unstable.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::flow::VectorField;

/// Stability classification of a fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stability {
    /// All eigenvalues have negative real parts.
    Stable,
    /// At least one eigenvalue has positive real part.
    Unstable,
    /// Some eigenvalues have zero real part, rest negative.
    Center,
    /// Mix of positive and negative real parts (saddle).
    Saddle,
}

impl std::fmt::Display for Stability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stability::Stable => write!(f, "Stable"),
            Stability::Unstable => write!(f, "Unstable"),
            Stability::Center => write!(f, "Center"),
            Stability::Saddle => write!(f, "Saddle"),
        }
    }
}

/// A fixed point of a dynamical system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint {
    /// Location of the fixed point.
    pub point: Vec<f64>,
    /// Stability classification.
    pub stability: Stability,
    /// Eigenvalues of the Jacobian at this point.
    pub eigenvalues: Vec<Complex64>,
    /// Dimension of stable manifold.
    pub stable_manifold_dim: usize,
    /// Dimension of unstable manifold.
    pub unstable_manifold_dim: usize,
}

/// Complex number (real + imaginary).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

/// Find a fixed point using Newton's method: x_{n+1} = x_n - J^{-1} V(x_n).
pub fn find_fixed_point(
    vf: &dyn VectorField,
    x0: &DVector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<DVector<f64>> {
    let mut x = x0.clone();
    for _ in 0..max_iter {
        let v = vf.evaluate(&x);
        if v.norm() < tol {
            return Some(x);
        }
        let jac = vf.jacobian(&x);
        if let Some(delta) = jac.clone().lu().solve(&v) {
            x -= &delta;
        } else {
            let perturbed = jac + DMatrix::identity(vf.dim(), vf.dim()) * 1e-8;
            if let Some(delta) = perturbed.lu().solve(&v) {
                x -= &delta;
            } else {
                return None;
            }
        }
    }
    let v = vf.evaluate(&x);
    if v.norm() < tol * 10.0 {
        Some(x)
    } else {
        None
    }
}

/// Classify the stability of a fixed point using eigenvalue analysis.
pub fn classify_stability(vf: &dyn VectorField, point: &DVector<f64>) -> FixedPoint {
    let jac = vf.jacobian(point);
    let eigen = jac.complex_eigenvalues();

    let eigenvalues: Vec<Complex64> = eigen.iter().map(|e| Complex64::new(e.re, e.im)).collect();

    let mut has_positive = false;
    let mut has_negative = false;
    let mut has_zero = false;

    for ev in &eigenvalues {
        if ev.re > 1e-8 {
            has_positive = true;
        } else if ev.re < -1e-8 {
            has_negative = true;
        } else {
            has_zero = true;
        }
    }

    let stability = match (has_positive, has_negative, has_zero) {
        (false, true, false) => Stability::Stable,
        (true, _, _) if has_negative => Stability::Saddle,
        (true, _, false) => Stability::Unstable,
        (false, _, true) => Stability::Center,
        _ => Stability::Center,
    };

    let stable_manifold_dim = eigenvalues.iter().filter(|e| e.re < -1e-8).count();
    let unstable_manifold_dim = eigenvalues.iter().filter(|e| e.re > 1e-8).count();

    FixedPoint {
        point: point.iter().copied().collect(),
        stability,
        eigenvalues,
        stable_manifold_dim,
        unstable_manifold_dim,
    }
}

/// Scan a region for fixed points by trying Newton's method from a grid of initial points.
pub fn scan_fixed_points(
    vf: &dyn VectorField,
    bounds: &[(f64, f64)],
    grid_res: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<FixedPoint> {
    let dim = vf.dim();
    let mut found = Vec::new();

    // Generate grid points
    let mut grid_points = vec![DVector::zeros(dim)];
    for d in 0..dim {
        let (lo, hi) = bounds.get(d).copied().unwrap_or((-1.0, 1.0));
        let step = (hi - lo) / grid_res as f64;
        let mut new_points = Vec::new();
        for pt in &grid_points {
            for i in 0..=grid_res {
                let mut p = pt.clone();
                p[d] = lo + i as f64 * step;
                new_points.push(p);
            }
        }
        grid_points = new_points;
    }

    let mut dedup: Vec<DVector<f64>> = Vec::new();

    for x0 in &grid_points {
        if let Some(fp) = find_fixed_point(vf, x0, max_iter, tol) {
            // Deduplicate
            let is_dup = dedup.iter().any(|p| (p - &fp).norm() < tol * 100.0);
            if !is_dup {
                dedup.push(fp.clone());
                found.push(classify_stability(vf, &fp));
            }
        }
    }

    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_find_origin_fixed_point() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let x0 = DVector::from_vec(vec![0.1, 0.1]);
        let fp = find_fixed_point(&vf, &x0, 50, 1e-10).unwrap();
        assert_abs_diff_eq!(fp[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(fp[1], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_stable_classification() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let pt = DVector::from_vec(vec![0.0, 0.0]);
        let fp = classify_stability(&vf, &pt);
        assert_eq!(fp.stability, Stability::Stable);
        assert_eq!(fp.stable_manifold_dim, 2);
        assert_eq!(fp.unstable_manifold_dim, 0);
    }

    #[test]
    fn test_unstable_classification() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![x[0], x[1]]));
        let pt = DVector::from_vec(vec![0.0, 0.0]);
        let fp = classify_stability(&vf, &pt);
        assert_eq!(fp.stability, Stability::Unstable);
    }

    #[test]
    fn test_saddle_classification() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![x[0], -x[1]]));
        let pt = DVector::from_vec(vec![0.0, 0.0]);
        let fp = classify_stability(&vf, &pt);
        assert_eq!(fp.stability, Stability::Saddle);
        assert_eq!(fp.stable_manifold_dim, 1);
        assert_eq!(fp.unstable_manifold_dim, 1);
    }

    #[test]
    fn test_find_nontrivial_fixed_point() {
        // dx/dt = x - x³ → fixed points at x = 0, ±1
        let vf = crate::flow::FnVectorField::new(1, |x| DVector::from_vec(vec![x[0] - x[0].powi(3)]));
        let fp = find_fixed_point(&vf, &DVector::from_vec(vec![0.5]), 50, 1e-10).unwrap();
        assert_abs_diff_eq!(fp[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_scan_fixed_points() {
        let vf = crate::flow::FnVectorField::new(1, |x| DVector::from_vec(vec![x[0] - x[0].powi(3)]));
        let fps = scan_fixed_points(&vf, &[(-2.0, 2.0)], 10, 50, 1e-8);
        assert!(fps.len() >= 2); // Should find at least 0 and ±1
    }

    #[test]
    fn test_stability_display() {
        assert_eq!(format!("{}", Stability::Stable), "Stable");
        assert_eq!(format!("{}", Stability::Saddle), "Saddle");
    }

    #[test]
    fn test_center_classification() {
        // dx/dt = -y, dy/dt = x → eigenvalues ±i (pure imaginary)
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[1], x[0]]));
        let pt = DVector::from_vec(vec![0.0, 0.0]);
        let fp = classify_stability(&vf, &pt);
        assert_eq!(fp.stability, Stability::Center);
    }
}
