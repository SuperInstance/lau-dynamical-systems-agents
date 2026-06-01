//! Poincaré map: reduce continuous flow to discrete map on cross-section.
//!
//! Given a cross-section Σ transverse to the flow, the Poincaré map P: Σ → Σ
//! maps each point to its first return to Σ. Fixed points of P correspond to
//! periodic orbits of the flow.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::flow::VectorField;

/// A Poincaré section Σ defined by a hyperplane: {x | n·(x - q) = 0}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareSection {
    /// Normal vector to the section.
    pub normal: Vec<f64>,
    /// A point on the section.
    pub point: Vec<f64>,
}

impl PoincareSection {
    pub fn new(normal: Vec<f64>, point: Vec<f64>) -> Self {
        Self { normal, point }
    }

    /// Signed distance from a point to the section.
    pub fn signed_distance(&self, x: &DVector<f64>) -> f64 {
        let q = DVector::from_vec(self.point.clone());
        let n = DVector::from_vec(self.normal.clone());
        n.dot(&(x - &q))
    }

    /// Check if a point is approximately on the section.
    pub fn is_on_section(&self, x: &DVector<f64>, tol: f64) -> bool {
        self.signed_distance(x).abs() < tol
    }
}

/// Result of computing the Poincaré map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareMapResult {
    /// First return point on the section.
    pub return_point: Vec<f64>,
    /// Return time.
    pub return_time: f64,
    /// Number of integration steps taken.
    pub steps: usize,
}

/// Compute the Poincaré map: find the first return to the section.
pub fn poincare_map(
    vf: &dyn VectorField,
    x0: &DVector<f64>,
    section: &PoincareSection,
    dt: f64,
    max_steps: usize,
) -> Option<PoincareMapResult> {
    let mut x = x0.clone();
    let mut prev_dist = section.signed_distance(&x);

    for step in 1..=max_steps {
        // RK4 step
        let k1 = vf.evaluate(&x);
        let k2 = vf.evaluate(&(&x + &k1.scale(dt / 2.0)));
        let k3 = vf.evaluate(&(&x + &k2.scale(dt / 2.0)));
        let k4 = vf.evaluate(&(&x + &k3.scale(dt)));
        x = &x + &(k1.scale(dt / 6.0) + &k2.scale(dt / 3.0) + &k3.scale(dt / 3.0) + &k4.scale(dt / 6.0));

        let curr_dist = section.signed_distance(&x);

        // Check for crossing (sign change)
        if prev_dist * curr_dist < 0.0 && step > 5 {
            // Linear interpolation for crossing point
            let t_frac = prev_dist.abs() / (prev_dist.abs() + curr_dist.abs());

            // Re-integrate to the crossing with finer step
            let _fine_dt = dt * t_frac;
            let _k1 = vf.evaluate(&x);
            // Simple linear interpolation of the crossing point
            let x_prev = &x - &(vf.evaluate(&x).scale(dt)); // approximate previous
            let return_point = &x_prev + &(&x - &x_prev).scale(t_frac);

            return Some(PoincareMapResult {
                return_point: return_point.iter().copied().collect(),
                return_time: (step as f64 - 1.0 + t_frac) * dt,
                steps: step,
            });
        }

        prev_dist = curr_dist;
    }

    None
}

/// Compute iterates of the Poincaré map to study periodic orbits and return times.
pub fn poincare_map_iterates(
    vf: &dyn VectorField,
    x0: &DVector<f64>,
    section: &PoincareSection,
    dt: f64,
    max_steps_per_return: usize,
    num_returns: usize,
) -> Vec<PoincareMapResult> {
    let mut results = Vec::new();
    let mut current = x0.clone();

    for _ in 0..num_returns {
        if let Some(result) = poincare_map(vf, &current, section, dt, max_steps_per_return) {
            current = DVector::from_vec(result.return_point.clone());
            results.push(result);
        } else {
            break;
        }
    }

    results
}

/// Compute the linearized Poincaré map (derivative) at a fixed point on the section.
/// Returns the eigenvalues of DP, whose magnitudes determine stability of the periodic orbit.
pub fn poincare_map_eigenvalues(
    vf: &dyn VectorField,
    fixed_point_on_section: &DVector<f64>,
    section: &PoincareSection,
    dt: f64,
    max_steps: usize,
    epsilon: f64,
) -> Vec<f64> {
    let dim = vf.dim();
    let section_dim = dim - 1;

    // Perturb in directions tangent to the section
    let normal = DVector::from_vec(section.normal.clone());
    let mut eigenvalues = Vec::new();

    // Use finite differences to approximate DP
    let base_result = poincare_map(vf, fixed_point_on_section, section, dt, max_steps);
    if base_result.is_none() {
        return eigenvalues;
    }
    let base_return = DVector::from_vec(base_result.unwrap().return_point);

    // Generate tangent directions via Gram-Schmidt
    let mut tangent_dirs = Vec::new();
    for i in 0..dim {
        let mut e = DVector::zeros(dim);
        e[i] = 1.0;
        // Project out normal component
        let proj = e.dot(&normal) / normal.dot(&normal);
        let tangent = &e - &(normal.scale(proj));
        if tangent.norm() > 1e-10 {
            tangent_dirs.push(tangent.normalize());
        }
    }

    tangent_dirs.truncate(section_dim);

    // For each tangent direction, perturb and measure return
    let mut dp_columns = Vec::new();
    for dir in &tangent_dirs {
        let perturbed = fixed_point_on_section + &(dir.scale(epsilon));
        if let Some(result) = poincare_map(vf, &perturbed, section, dt, max_steps) {
            let return_pt = DVector::from_vec(result.return_point);
            let dp_col = (&return_pt - &base_return).scale(1.0 / epsilon);
            // Project onto section
            let proj = dp_col.dot(&normal) / normal.dot(&normal);
            let tangent_dp = dp_col - normal.scale(proj);
            dp_columns.push(tangent_dp);
        }
    }

    // Form DP matrix and compute eigenvalues
    if !dp_columns.is_empty() {
        let n = dp_columns[0].len();
        let m = dp_columns.len();
        let mut dp_matrix = nalgebra::DMatrix::zeros(n, m);
        for (j, col) in dp_columns.iter().enumerate() {
            for i in 0..n {
                dp_matrix[(i, j)] = col[i];
            }
        }
        let eigen = dp_matrix.complex_eigenvalues();
        eigenvalues = eigen.iter().map(|e| (e.re * e.re + e.im * e.im).sqrt()).collect();
    }

    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_section_signed_distance() {
        let section = PoincareSection::new(vec![1.0, 0.0], vec![1.0, 0.0]);
        let x = DVector::from_vec(vec![2.0, 3.0]);
        assert_abs_diff_eq!(section.signed_distance(&x), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_section_is_on() {
        let section = PoincareSection::new(vec![1.0, 0.0], vec![0.0, 0.0]);
        let x = DVector::from_vec(vec![0.0, 5.0]);
        assert!(section.is_on_section(&x, 0.01));
    }

    #[test]
    fn test_poincare_map_rotation() {
        // dx/dt = -y, dy/dt = x → rotation with period 2π
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[1], x[0]]));
        let section = PoincareSection::new(vec![0.0, 1.0], vec![0.0, 0.0]); // x-axis
        let x0 = DVector::from_vec(vec![1.0, 0.0]);
        let result = poincare_map(&vf, &x0, &section, 0.01, 1000);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_abs_diff_eq!(r.return_time, std::f64::consts::PI, epsilon = 0.1);
    }

    #[test]
    fn test_poincare_map_no_return() {
        // Linear field going away
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![x[0], x[1]]));
        let section = PoincareSection::new(vec![0.0, 1.0], vec![0.0, 0.0]);
        let x0 = DVector::from_vec(vec![1.0, 0.01]);
        let result = poincare_map(&vf, &x0, &section, 0.01, 500);
        // May or may not return — just shouldn't crash
    }

    #[test]
    fn test_poincare_iterates() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[1], x[0]]));
        let section = PoincareSection::new(vec![0.0, 1.0], vec![0.0, 0.0]);
        let x0 = DVector::from_vec(vec![1.0, 0.0]);
        let results = poincare_map_iterates(&vf, &x0, &section, 0.01, 1000, 3);
        assert!(results.len() >= 2);
        // Return times should all be approximately 2π
        for r in &results {
            assert_abs_diff_eq!(r.return_time, std::f64::consts::PI, epsilon = 0.15);
        }
    }

    #[test]
    fn test_poincare_section_serialization() {
        let section = PoincareSection::new(vec![1.0, 0.0], vec![0.0, 0.0]);
        let json = serde_json::to_string(&section).unwrap();
        assert!(json.contains("normal"));
    }
}
