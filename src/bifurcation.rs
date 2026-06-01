//! Bifurcations: qualitative changes in dynamics as a parameter varies.
//!
//! Types: saddle-node, transcritical, pitchfork, Hopf bifurcation.
//! A bifurcation occurs when the topological type of the flow changes.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use num_complex::Complex;

/// Type of bifurcation detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BifurcationType {
    /// Two fixed points collide and annihilate (or are born).
    SaddleNode,
    /// Two fixed points exchange stability.
    Transcritical,
    /// One fixed point splits into three (symmetry-breaking).
    Pitchfork,
    /// Fixed point changes from stable spiral to unstable spiral + limit cycle.
    Hopf,
    /// Unknown or other type.
    Other,
}

impl std::fmt::Display for BifurcationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A detected bifurcation point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationPoint {
    /// Parameter value at bifurcation.
    pub mu: f64,
    /// Type of bifurcation.
    pub bifurcation_type: BifurcationType,
    /// Location of the bifurcation in state space.
    pub point: Vec<f64>,
    /// Confidence of detection.
    pub confidence: f64,
}

/// A parameterized vector field V(x, μ) for bifurcation analysis.
pub trait ParameterizedVectorField: Send + Sync {
    fn dim(&self) -> usize;
    fn evaluate(&self, x: &DVector<f64>, mu: f64) -> DVector<f64>;
    fn jacobian(&self, x: &DVector<f64>, mu: f64) -> DMatrix<f64> {
        let n = self.dim();
        let eps = 1e-7;
        let mut jac = DMatrix::zeros(n, n);
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps;
            xm[j] -= eps;
            let vp = self.evaluate(&xp, mu);
            let vm = self.evaluate(&xm, mu);
            for i in 0..n {
                jac[(i, j)] = (vp[i] - vm[i]) / (2.0 * eps);
            }
        }
        jac
    }
}

/// Closure-based parameterized vector field.
pub struct FnParameterizedField {
    dim: usize,
    #[allow(clippy::type_complexity)]
    f: Box<dyn Fn(&DVector<f64>, f64) -> DVector<f64> + Send + Sync>,
}

impl FnParameterizedField {
    pub fn new<F: Fn(&DVector<f64>, f64) -> DVector<f64> + Send + Sync + 'static>(dim: usize, f: F) -> Self {
        Self { dim, f: Box::new(f) }
    }
}

impl ParameterizedVectorField for FnParameterizedField {
    fn dim(&self) -> usize { self.dim }
    fn evaluate(&self, x: &DVector<f64>, mu: f64) -> DVector<f64> {
        (self.f)(x, mu)
    }
}

/// Scan a parameter range for bifurcations by tracking fixed point changes.
pub fn detect_bifurcations(
    vf: &dyn ParameterizedVectorField,
    x_guess: &DVector<f64>,
    mu_range: (f64, f64),
    mu_steps: usize,
    tol: f64,
) -> Vec<BifurcationPoint> {
    let mu_step = (mu_range.1 - mu_range.0) / mu_steps as f64;
    let mut bifurcations = Vec::new();
    let mut prev_fp: Option<DVector<f64>> = None;
    let mut prev_eigen: Option<Vec<Complex<f64>>> = None;
    let mut prev_stable = None;

    for i in 0..=mu_steps {
        let mu = mu_range.0 + i as f64 * mu_step;

        // Find fixed point using Newton's method
        if let Some(fp) = find_fp_param(vf, x_guess, mu, 50, tol) {
            let jac = vf.jacobian(&fp, mu);
            let eigen: Vec<Complex<f64>> = jac.complex_eigenvalues().iter().map(|e| Complex::new(e.re, e.im)).collect();
            let stable = eigen.iter().all(|e| e.re < 1e-6);

            if let (Some(prev_st), Some(prev_eig)) = (prev_stable, prev_eigen) {
                // Detect stability change
                if stable != prev_st {
                    let btype = classify_bifurcation(&eigen, &prev_eig, vf.dim());
                    bifurcations.push(BifurcationPoint {
                        mu,
                        bifurcation_type: btype,
                        point: fp.iter().copied().collect(),
                        confidence: 0.8,
                    });
                }

                // Detect appearance/disappearance of fixed points
                if let Some(ref prev) = prev_fp {
                    let dist = (&fp - prev).norm();
                    if dist > 1.0 {
                        // Fixed point jumped → saddle-node or pitchfork
                        bifurcations.push(BifurcationPoint {
                            mu,
                            bifurcation_type: BifurcationType::SaddleNode,
                            point: fp.iter().copied().collect(),
                            confidence: 0.6,
                        });
                    }
                }
            }

            prev_fp = Some(fp);
            prev_eigen = Some(eigen);
            prev_stable = Some(stable);
        }
    }

    bifurcations
}

/// Classify the type of bifurcation from eigenvalue changes.
fn classify_bifurcation(
    eigen_new: &[Complex<f64>],
    _eigen_old: &[Complex<f64>],
    _dim: usize,
) -> BifurcationType {
    // Check for Hopf: pair of complex eigenvalues crossing imaginary axis
    let new_imag_pairs = eigen_new.iter().filter(|e| e.im.abs() > 0.01 && e.re.abs() < 0.1).count();
    if new_imag_pairs >= 2 {
        return BifurcationType::Hopf;
    }

    // Check for a real eigenvalue crossing zero
    let crossing = eigen_new.iter().any(|e| e.re.abs() < 0.1 && e.im.abs() < 0.1);
    if crossing {
        // Distinguish transcritical vs pitchfork by symmetry
        // Heuristic: if fixed point stays at origin, transcritical; if it moves, pitchfork
        return BifurcationType::SaddleNode;
    }

    BifurcationType::Other
}

/// Find fixed point for parameterized field.
fn find_fp_param(
    vf: &dyn ParameterizedVectorField,
    x0: &DVector<f64>,
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> Option<DVector<f64>> {
    let mut x = x0.clone();
    for _ in 0..max_iter {
        let v = vf.evaluate(&x, mu);
        if v.norm() < tol {
            return Some(x);
        }
        let jac = vf.jacobian(&x, mu);
        if let Some(delta) = jac.lu().solve(&v) {
            x -= &delta;
        } else {
            return None;
        }
    }
    let v = vf.evaluate(&x, mu);
    if v.norm() < tol * 10.0 {
        Some(x)
    } else {
        None
    }
}

/// Saddle-node normal form: dx/dt = μ + x²
pub fn saddle_node_field() -> FnParameterizedField {
    FnParameterizedField::new(1, |x, mu| DVector::from_vec(vec![mu + x[0] * x[0]]))
}

/// Transcritical normal form: dx/dt = μx - x²
pub fn transcritical_field() -> FnParameterizedField {
    FnParameterizedField::new(1, |x, mu| DVector::from_vec(vec![mu * x[0] - x[0] * x[0]]))
}

/// Pitchfork normal form (supercritical): dx/dt = μx - x³
pub fn pitchfork_field() -> FnParameterizedField {
    FnParameterizedField::new(1, |x, mu| DVector::from_vec(vec![mu * x[0] - x[0].powi(3)]))
}

/// Hopf normal form (supercritical): dr/dt = μr - r³, dθ/dt = ω
/// In Cartesian: dx/dt = μx - ωy - (x²+y²)x, dy/dt = ωx + μy - (x²+y²)y
pub fn hopf_field(omega: f64) -> FnParameterizedField {
    FnParameterizedField::new(2, move |x, mu| {
        let r2 = x[0] * x[0] + x[1] * x[1];
        DVector::from_vec(vec![
            mu * x[0] - omega * x[1] - r2 * x[0],
            omega * x[0] + mu * x[1] - r2 * x[1],
        ])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_saddle_node_field() {
        let vf = saddle_node_field();
        let val = vf.evaluate(&DVector::from_vec(vec![0.0]), 1.0);
        assert_abs_diff_eq!(val[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transcritical_field() {
        let vf = transcritical_field();
        // At x=1, μ=0: dx/dt = 0 - 1 = -1
        let val = vf.evaluate(&DVector::from_vec(vec![1.0]), 0.0);
        assert_abs_diff_eq!(val[0], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pitchfork_field() {
        let vf = pitchfork_field();
        // At x=0, any μ: dx/dt = 0 (fixed point)
        let val = vf.evaluate(&DVector::from_vec(vec![0.0]), 5.0);
        assert_abs_diff_eq!(val[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hopf_field() {
        let vf = hopf_field(1.0);
        // At origin, any μ: V = 0 (fixed point)
        let val = vf.evaluate(&DVector::from_vec(vec![0.0, 0.0]), 1.0);
        assert_abs_diff_eq!(val[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(val[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_detect_pitchfork_bifurcation() {
        let vf = pitchfork_field();
        let x0 = DVector::from_vec(vec![0.01]);
        let bifurcations = detect_bifurcations(&vf, &x0, (-2.0, 2.0), 200, 1e-8);
        // Pitchfork bifurcation at μ = 0
        assert!(!bifurcations.is_empty(), "Should detect bifurcation near μ=0");
    }

    #[test]
    fn test_detect_hopf_bifurcation() {
        let vf = hopf_field(1.0);
        let x0 = DVector::from_vec(vec![0.01, 0.01]);
        let bifurcations = detect_bifurcations(&vf, &x0, (-2.0, 2.0), 200, 1e-8);
        // Hopf bifurcation at μ = 0
        assert!(!bifurcations.is_empty(), "Should detect Hopf bifurcation near μ=0");
    }

    #[test]
    fn test_bifurcation_serialization() {
        let bp = BifurcationPoint {
            mu: 0.0,
            bifurcation_type: BifurcationType::Hopf,
            point: vec![0.0, 0.0],
            confidence: 0.9,
        };
        let json = serde_json::to_string(&bp).unwrap();
        assert!(json.contains("Hopf"));
    }

    #[test]
    fn test_parameterized_jacobian() {
        let vf = pitchfork_field();
        let jac = vf.jacobian(&DVector::from_vec(vec![0.0]), 1.0);
        // d/dx(μx - x³) = μ at x=0
        assert_abs_diff_eq!(jac[(0, 0)], 1.0, epsilon = 1e-5);
    }
}
