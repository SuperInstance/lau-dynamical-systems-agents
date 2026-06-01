//! Melnikov method: detect chaos in perturbed Hamiltonian systems.
//!
//! For a system ẋ = f(x) + εg(x,t), the Melnikov function M(t₀) measures
//! the distance between stable and unstable manifolds. Simple zeros of M
//! imply transverse intersection → Smale horseshoe → chaos.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::flow::VectorField;

/// Result of Melnikov analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelnikovResult {
    /// The Melnikov function values M(t₀) for a range of t₀.
    pub melnikov_values: Vec<f64>,
    /// t₀ values.
    pub t0_values: Vec<f64>,
    /// Whether simple zeros were detected (implies chaos).
    pub has_simple_zeros: bool,
    /// Estimated measure of chaos.
    pub chaos_indicator: f64,
}

/// Compute the Melnikov function for a periodically perturbed system.
///
/// System: ẋ = f(x) + ε·g(x, t)
/// where f is the unperturbed Hamiltonian vector field and g is the perturbation.
///
/// The Melnikov function is:
/// M(t₀) = ∫_{-∞}^{∞} f(x⁰(t)) ∧ g(x⁰(t), t+t₀) · exp(-∫₀ᵗ div(f)(x⁰(s)) ds) dt
///
/// Simplified for 2D: M(t₀) = ∫ f₁(x⁰(t))·g₂(x⁰(t), t+t₀) - f₂(x⁰(t))·g₁(x⁰(t), t+t₀) dt
#[allow(clippy::too_many_arguments)]
pub fn melnikov_function(
    f: &dyn VectorField,
    g: &dyn Fn(&DVector<f64>, f64) -> DVector<f64>,
    heteroclinic_orbit: &[Vec<f64>],
    orbit_times: &[f64],
    t0_range: (f64, f64),
    t0_steps: usize,
    _epsilon: f64,
    _omega: f64, // perturbation frequency
) -> MelnikovResult {
    let dt0 = (t0_range.1 - t0_range.0) / t0_steps as f64;
    let mut melnikov_values = Vec::new();
    let mut t0_values = Vec::new();

    for i in 0..=t0_steps {
        let t0 = t0_range.0 + i as f64 * dt0;
        let mut integral = 0.0;

        for (k, (point, &time)) in heteroclinic_orbit.iter().zip(orbit_times.iter()).enumerate() {
            let x = DVector::from_vec(point.clone());
            let f_val = f.evaluate(&x);
            let g_val = g(&x, time + t0);

            // Wedge product in 2D: f₁g₂ - f₂g₁
            let integrand = f_val[0] * g_val[1] - f_val[1] * g_val[0];

            if k > 0 && k < heteroclinic_orbit.len() - 1 {
                let dt = if k < orbit_times.len() - 1 {
                    orbit_times[k + 1] - orbit_times[k]
                } else {
                    orbit_times[k] - orbit_times[k - 1]
                };
                integral += integrand * dt;
            }
        }

        melnikov_values.push(integral);
        t0_values.push(t0);
    }

    // Detect simple zeros: sign changes in M(t₀)
    let mut has_simple_zeros = false;
    for i in 1..melnikov_values.len() {
        if melnikov_values[i] * melnikov_values[i - 1] < 0.0 {
            has_simple_zeros = true;
            break;
        }
    }

    let chaos_indicator = melnikov_values.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));

    MelnikovResult {
        melnikov_values,
        t0_values,
        has_simple_zeros,
        chaos_indicator,
    }
}

/// Generate a heteroclinic orbit for the undamped pendulum: ẍ = sin(x).
/// The separatrix connects the saddle points at x = ±π.
/// Parametric form: x(t) = 4·arctan(eᵗ) - π, ẋ(t) = 2/cosh(t)
pub fn pendulum_separatrix(t_range: (f64, f64), steps: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let dt = (t_range.1 - t_range.0) / steps as f64;
    let mut orbit = Vec::new();
    let mut times = Vec::new();

    for i in 0..=steps {
        let t = t_range.0 + i as f64 * dt;
        let x = 4.0 * (t.exp().atan()) - std::f64::consts::PI;
        let v = 2.0 / t.cosh();
        orbit.push(vec![x, v]);
        times.push(t);
    }

    (orbit, times)
}

/// Compute the Melnikov integral analytically for the damped driven pendulum.
/// ẍ + δẋ + sin(x) = γcos(ωt)
///
/// M(t₀) = -δ·I₁ + γ·I₂·sin(ωt₀)  where I₁, I₂ are known integrals.
pub fn melnikov_analytic_damped_driven(delta: f64, gamma: f64, omega: f64, t0: f64) -> f64 {
    // I₁ = ∫ ẋ⁰(t)² dt = ∫ 4/cosh²(t) dt = 8
    let i1 = 8.0;
    // I₂ = ∫ ẋ⁰(t)·cos(x⁰(t))·cos(ωt) dt = πω/cosh(πω/2)
    let i2 = std::f64::consts::PI * omega / (std::f64::consts::PI * omega / 2.0).cosh();

    -delta * i1 + gamma * i2 * (omega * t0).sin()
}

/// Check if Melnikov function has simple zeros (chaos condition).
pub fn chaos_condition_damped_driven(delta: f64, gamma: f64, omega: f64) -> bool {
    let i1 = 8.0;
    let i2 = std::f64::consts::PI * omega / (std::f64::consts::PI * omega / 2.0).cosh();
    // Simple zeros exist iff |γ·I₂| > |δ·I₁|
    (gamma * i2).abs() > (delta * i1).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pendulum_separatrix() {
        let (orbit, times) = pendulum_separatrix((-5.0, 5.0), 100);
        assert_eq!(orbit.len(), 101);
        // At t=0: x ≈ 0, v = 2
        assert_abs_diff_eq!(orbit[50][0], 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(orbit[50][1], 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_melnikov_analytic() {
        // When γ is large enough, should have chaos
        let m = melnikov_analytic_damped_driven(0.1, 1.0, 1.0, 0.0);
        assert!(m.is_finite());
    }

    #[test]
    fn test_chaos_condition_true() {
        // Large driving, small damping → chaos
        assert!(chaos_condition_damped_driven(0.01, 1.0, 1.0));
    }

    #[test]
    fn test_chaos_condition_false() {
        // Small driving, large damping → no chaos
        assert!(!chaos_condition_damped_driven(10.0, 0.001, 1.0));
    }

    #[test]
    fn test_melnikov_numerical() {
        let f = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![x[1], x[0].sin()]));
        let g = |x: &DVector<f64>, t: f64| DVector::from_vec(vec![0.0, (1.0 * t).cos()]);
        let (orbit, times) = pendulum_separatrix((-5.0, 5.0), 200);
        let result = melnikov_function(&f, &g, &orbit, &times, (0.0, 2.0 * std::f64::consts::PI), 50, 0.1, 1.0);
        assert_eq!(result.melnikov_values.len(), 51);
    }

    #[test]
    fn test_melnikov_serialization() {
        let result = MelnikovResult {
            melnikov_values: vec![1.0, -1.0],
            t0_values: vec![0.0, 1.0],
            has_simple_zeros: true,
            chaos_indicator: 1.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("has_simple_zeros"));
    }

    #[test]
    fn test_separatrix_boundary() {
        let (orbit, _) = pendulum_separatrix((-10.0, 10.0), 100);
        // At large t: x → π
        assert!(orbit.last().unwrap()[0] > 2.5);
        // At large -t: x → -π
        assert!(orbit.first().unwrap()[0] < -2.5);
    }
}
