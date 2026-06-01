//! Application: predict agent behavioral regimes.
//!
//! Models agent behavior as a dynamical system, predicting transitions
//! between stable, oscillating, and chaotic behavioral regimes.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::flow::VectorField;
use crate::lyapunov::{LyapunovExponents, compute_lyapunov_exponents};
use crate::bifurcation::{detect_bifurcations, ParameterizedVectorField};
use crate::attractor::{AttractorType, detect_attractor};

/// Behavioral regime of an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BehavioralRegime {
    /// Agent converges to a stable state.
    Stable,
    /// Agent oscillates between states.
    Oscillating,
    /// Agent behavior is unpredictable / chaotic.
    Chaotic,
    /// Agent is at a bifurcation point.
    Transitional,
}

impl std::fmt::Display for BehavioralRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A behavioral parameter (e.g., "stress_level", "motivation").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralParameter {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
}

impl BehavioralParameter {
    pub fn new(name: &str, value: f64, min: f64, max: f64) -> Self {
        Self { name: name.to_string(), value, min, max }
    }
}

/// Agent state in behavioral space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// State vector (e.g., [activity, mood, engagement]).
    pub state: Vec<f64>,
    /// Current regime.
    pub regime: BehavioralRegime,
    /// Confidence in regime classification.
    pub confidence: f64,
    /// Timestamp.
    pub timestamp: f64,
}

/// Prediction of agent behavioral evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPrediction {
    /// Current regime.
    pub current_regime: BehavioralRegime,
    /// Predicted regime after parameter change.
    pub predicted_regime: BehavioralRegime,
    /// Lyapunov exponents.
    pub lyapunov_exponents: Vec<f64>,
    /// Detected attractors.
    pub attractors: Vec<String>,
    /// Upcoming bifurcations.
    pub bifurcations: Vec<String>,
    /// Risk of chaotic transition (0-1).
    pub chaos_risk: f64,
}

/// Analyze agent behavioral dynamics and predict regime transitions.
pub fn analyze_agent_behavior(
    vf: &dyn VectorField,
    current_state: &DVector<f64>,
    parameterized_vf: &dyn ParameterizedVectorField,
    param_range: (f64, f64),
    t_analysis: f64,
    dt: f64,
) -> BehaviorPrediction {
    // 1. Determine current regime via Lyapunov exponents
    let le = compute_lyapunov_exponents(vf, current_state, t_analysis, dt, 10);
    let current_regime = classify_regime(&le);

    // 2. Detect attractors
    let attractor = detect_attractor(vf, current_state, t_analysis, dt, t_analysis / 2.0);
    let attractor_desc = match &attractor.attractor_type {
        AttractorType::FixedPoint(p) => format!("Fixed point at ({:.2}, ...)", p[0]),
        AttractorType::LimitCycle { center: _, approximate_radius } => {
            format!("Limit cycle (r≈{:.2})", approximate_radius)
        }
        AttractorType::Strange { estimated_dimension } => {
            format!("Strange attractor (dim≈{:.2})", estimated_dimension)
        }
        AttractorType::Torus { radii } => format!("Torus ({:?})", radii),
    };

    // 3. Scan for bifurcations
    let bifurcations = detect_bifurcations(parameterized_vf, current_state, param_range, 100, 1e-6);
    let bif_descriptions: Vec<String> = bifurcations.iter().map(|b| format!("{:?} at μ={:.3}", b.bifurcation_type, b.mu)).collect();

    // 4. Predict regime at end of parameter range
    let end_state = current_state.clone(); // simplified
    let le_end = compute_lyapunov_exponents(vf, &end_state, t_analysis.min(5.0), dt, 10);
    let predicted_regime = if !bifurcations.is_empty() {
        BehavioralRegime::Transitional
    } else {
        classify_regime(&le_end)
    };

    // 5. Estimate chaos risk
    let chaos_risk = if le.is_chaotic {
        1.0
    } else if !bifurcations.is_empty() {
        0.5
    } else {
        let max_le = le.exponents.first().copied().unwrap_or(-1.0);
        (max_le + 1.0).max(0.0).min(1.0)
    };

    BehaviorPrediction {
        current_regime,
        predicted_regime,
        lyapunov_exponents: le.exponents,
        attractors: vec![attractor_desc],
        bifurcations: bif_descriptions,
        chaos_risk,
    }
}

/// Classify behavioral regime from Lyapunov exponents.
pub fn classify_regime(le: &LyapunovExponents) -> BehavioralRegime {
    if le.exponents.is_empty() {
        return BehavioralRegime::Stable;
    }

    let max_le = le.exponents.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max_le > 0.01 {
        BehavioralRegime::Chaotic
    } else if max_le > -0.5 {
        // Near-zero exponent suggests oscillation (neither strongly stable nor chaotic)
        BehavioralRegime::Oscillating
    } else {
        BehavioralRegime::Stable
    }
}

/// Simple agent dynamics model: coupled activity-mood system.
/// Uses a struct to implement VectorField for parameterized dynamics.
pub struct CoupledActivityMood {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
}

impl CoupledActivityMood {
    pub fn new(alpha: f64, beta: f64, gamma: f64, delta: f64) -> Self {
        Self { alpha, beta, gamma, delta }
    }
}

impl crate::flow::VectorField for CoupledActivityMood {
    fn dim(&self) -> usize { 2 }
    fn evaluate(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![
            self.alpha * x[0] - self.beta * x[0] * x[1],
            -self.gamma * x[1] + self.delta * x[0] * x[1],
        ])
    }
}

/// Compute the regime diagram over a 2D parameter grid.
pub fn regime_diagram<VF: VectorField + 'static>(
    vf_factory: &dyn Fn(f64, f64) -> Box<dyn VectorField>,
    param1_range: (f64, f64),
    param2_range: (f64, f64),
    grid_res: usize,
    x0: &DVector<f64>,
    t_analysis: f64,
    dt: f64,
) -> Vec<Vec<BehavioralRegime>> {
    let mut diagram = Vec::new();
    let dp1 = (param1_range.1 - param1_range.0) / grid_res as f64;
    let dp2 = (param2_range.1 - param2_range.0) / grid_res as f64;

    for i in 0..=grid_res {
        let mut row = Vec::new();
        for j in 0..=grid_res {
            let p1 = param1_range.0 + i as f64 * dp1;
            let p2 = param2_range.0 + j as f64 * dp2;
            let vf = vf_factory(p1, p2);
            let le = compute_lyapunov_exponents(vf.as_ref(), x0, t_analysis.min(5.0), dt, 10);
            row.push(classify_regime(&le));
        }
        diagram.push(row);
    }

    diagram
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_regime_stable() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let le = compute_lyapunov_exponents(&vf, &x0, 10.0, 0.01, 10);
        let regime = classify_regime(&le);
        assert_eq!(regime, BehavioralRegime::Stable);
    }

    #[test]
    fn test_regime_display() {
        assert_eq!(format!("{}", BehavioralRegime::Stable), "Stable");
        assert_eq!(format!("{}", BehavioralRegime::Chaotic), "Chaotic");
    }

    #[test]
    fn test_coupled_activity_mood() {
        let vf = CoupledActivityMood::new(1.0, 0.1, 1.0, 0.1);
        let val = vf.evaluate(&DVector::from_vec(vec![1.0, 1.0]));
        assert_abs_diff_eq!(val[0], 0.9, epsilon = 1e-10);
        assert_abs_diff_eq!(val[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_behavioral_parameter() {
        let p = BehavioralParameter::new("stress", 0.5, 0.0, 1.0);
        assert_eq!(p.name, "stress");
        assert_eq!(p.value, 0.5);
    }

    #[test]
    fn test_agent_state_serialization() {
        let state = AgentState {
            state: vec![1.0, 2.0],
            regime: BehavioralRegime::Stable,
            confidence: 0.9,
            timestamp: 1.0,
        };
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("Stable"));
    }

    #[test]
    fn test_chaos_risk_stable() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);

        struct SimpleParam;
        impl ParameterizedVectorField for SimpleParam {
            fn dim(&self) -> usize { 2 }
            fn evaluate(&self, x: &DVector<f64>, _mu: f64) -> DVector<f64> {
                DVector::from_vec(vec![-x[0], -x[1]])
            }
        }

        let pred = analyze_agent_behavior(&vf, &x0, &SimpleParam, (-1.0, 1.0), 5.0, 0.01);
        assert!(pred.chaos_risk < 0.5);
        assert_eq!(pred.current_regime, BehavioralRegime::Stable);
    }

    #[test]
    fn test_prediction_serialization() {
        let pred = BehaviorPrediction {
            current_regime: BehavioralRegime::Oscillating,
            predicted_regime: BehavioralRegime::Chaotic,
            lyapunov_exponents: vec![0.5, -1.0],
            attractors: vec!["Limit cycle".to_string()],
            bifurcations: vec![],
            chaos_risk: 0.7,
        };
        let json = serde_json::to_string(&pred).unwrap();
        assert!(json.contains("Oscillating"));
    }

    #[test]
    fn test_regime_diagram() {
        let diagram = regime_diagram(
            &|p1, _p2| -> Box<dyn VectorField> {
                Box::new(crate::flow::FnVectorField::new(2, move |x| {
                    DVector::from_vec(vec![-p1.abs().max(0.1) * x[0], -x[1]])
                }))
            },
            (0.1, 1.0),
            (0.1, 1.0),
            3,
            &DVector::from_vec(vec![1.0, 1.0]),
            5.0,
            0.05,
        );
        assert!(!diagram.is_empty());
        assert!(diagram[0].len() == 4);
    }
}
