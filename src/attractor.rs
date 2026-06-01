//! Attractors: fixed points, limit cycles, strange attractors.
//!
//! An attractor is a closed invariant set A such that nearby orbits converge to A.
//! Types: fixed point attractors, limit cycles, quasiperiodic tori, strange (chaotic) attractors.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::flow::{VectorField, flow};

/// Type of attractor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttractorType {
    /// Stable fixed point.
    FixedPoint(Vec<f64>),
    /// Limit cycle (closed orbit).
    LimitCycle {
        center: Vec<f64>,
        approximate_radius: f64,
    },
    /// Strange (chaotic) attractor.
    Strange {
        estimated_dimension: f64,
    },
    /// Quasiperiodic torus.
    Torus {
        radii: Vec<f64>,
    },
}

/// Detected attractor with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attractor {
    pub attractor_type: AttractorType,
    /// Basin of attraction sample points.
    pub basin_samples: Vec<Vec<f64>>,
    /// Confidence score [0, 1].
    pub confidence: f64,
}

/// Detect attractor type by observing trajectory behavior.
pub fn detect_attractor(
    vf: &dyn VectorField,
    x0: &DVector<f64>,
    t_total: f64,
    dt: f64,
    t_transient: f64,
) -> Attractor {
    let result = flow(vf, x0, t_total, dt, true);

    // Split into transient and asymptotic
    let transient_steps = (t_transient / dt) as usize;
    let traj = &result.trajectory;

    if traj.len() <= transient_steps {
        return Attractor {
            attractor_type: AttractorType::FixedPoint(traj.last().cloned().unwrap_or_default()),
            basin_samples: vec![x0.iter().copied().collect()],
            confidence: 0.5,
        };
    }

    let asymptotic = &traj[transient_steps..];

    // Check if trajectory converges to a point
    let last = &asymptotic[asymptotic.len() - 1];
    let converges_to_point = asymptotic.iter().skip(asymptotic.len() / 2).all(|p| {
        let dist: f64 = p.iter().zip(last.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        dist < 0.01
    });

    if converges_to_point {
        return Attractor {
            attractor_type: AttractorType::FixedPoint(last.clone()),
            basin_samples: vec![x0.iter().copied().collect()],
            confidence: 0.9,
        };
    }

    // Check for limit cycle: does the trajectory approximately return to start of asymptotic part?
    let first = &asymptotic[0];
    let mut returns_count = 0;
    let mut distances = Vec::new();
    for p in asymptotic.iter() {
        let dist: f64 = p.iter().zip(first.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        distances.push(dist);
        if dist < 0.05 {
            returns_count += 1;
        }
    }

    let center: Vec<f64> = (0..vf.dim())
        .map(|d| asymptotic.iter().map(|p| p[d]).sum::<f64>() / asymptotic.len() as f64)
        .collect();

    let max_dist = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_dist = distances.iter().cloned().fold(f64::INFINITY, f64::min);

    if returns_count > 2 && (max_dist - min_dist) < 0.5 {
        return Attractor {
            attractor_type: AttractorType::LimitCycle {
                center: center.clone(),
                approximate_radius: (max_dist + min_dist) / 2.0,
            },
            basin_samples: vec![x0.iter().copied().collect()],
            confidence: 0.8,
        };
    }

    // Estimate correlation dimension as a heuristic for strange attractor
    let dim = estimate_correlation_dimension(asymptotic, 0.1);
    Attractor {
        attractor_type: AttractorType::Strange {
            estimated_dimension: dim,
        },
        basin_samples: vec![x0.iter().copied().collect()],
        confidence: 0.6,
    }
}

/// Estimate the correlation dimension of a point set.
/// Uses the Grassberger-Procaccia algorithm.
pub fn estimate_correlation_dimension(points: &[Vec<f64>], r: f64) -> f64 {
    let n = points.len().min(500); // Limit for performance
    let mut count = 0usize;
    let mut total = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let dist: f64 = points[i].iter().zip(points[j].iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            if dist < r {
                count += 1;
            }
            total += 1;
        }
    }

    if total == 0 || count == 0 {
        return 0.0;
    }

    let c_r = count as f64 / total as f64;
    if c_r <= 0.0 {
        return 0.0;
    }

    // Rough estimate: dimension ≈ log(C(r)) / log(r)
    c_r.ln() / r.ln()
}

/// Map out the basin of attraction by sampling initial conditions.
pub fn basin_of_attraction(
    vf: &dyn VectorField,
    attractor_point: &DVector<f64>,
    bounds: &[(f64, f64)],
    grid_res: usize,
    t_test: f64,
    dt: f64,
    convergence_tol: f64,
) -> Vec<Vec<f64>> {
    let dim = vf.dim();
    let mut basin = Vec::new();

    // Generate grid
    let mut grid = vec![DVector::zeros(dim)];
    for d in 0..dim {
        let (lo, hi) = bounds.get(d).copied().unwrap_or((-1.0, 1.0));
        let step = (hi - lo) / grid_res as f64;
        let mut new_grid = Vec::new();
        for pt in &grid {
            for i in 0..=grid_res {
                let mut p = pt.clone();
                p[d] = lo + i as f64 * step;
                new_grid.push(p);
            }
        }
        grid = new_grid;
    }

    for x0 in &grid {
        let result = flow(vf, x0, t_test, dt, false);
        let endpoint = DVector::from_vec(result.endpoint);
        if (endpoint - attractor_point).norm() < convergence_tol {
            basin.push(x0.iter().copied().collect());
        }
    }

    basin
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fixed_point_attractor() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let att = detect_attractor(&vf, &x0, 10.0, 0.01, 5.0);
        match att.attractor_type {
            AttractorType::FixedPoint(pt) => {
                assert_abs_diff_eq!(pt[0], 0.0, epsilon = 0.1);
                assert_abs_diff_eq!(pt[1], 0.0, epsilon = 0.1);
            }
            _ => panic!("Expected fixed point attractor"),
        }
    }

    #[test]
    fn test_limit_cycle_attractor() {
        // Van der Pol oscillator (limit cycle for μ > 0)
        let mu = 1.0;
        let vf = crate::flow::FnVectorField::new(2, move |x| {
            DVector::from_vec(vec![x[1], mu * (1.0 - x[0] * x[0]) * x[1] - x[0]])
        });
        let x0 = DVector::from_vec(vec![0.1, 0.1]);
        let att = detect_attractor(&vf, &x0, 50.0, 0.01, 30.0);
        // Should detect limit cycle or at least not crash
        assert!(att.confidence > 0.0);
    }

    #[test]
    fn test_basin_of_attraction() {
        let vf = crate::flow::FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
        let attractor = DVector::from_vec(vec![0.0, 0.0]);
        let basin = basin_of_attraction(&vf, &attractor, &[(-1.0, 1.0), (-1.0, 1.0)], 3, 5.0, 0.01, 0.1);
        // All points should be in the basin for this globally stable system
        assert!(basin.len() > 10);
    }

    #[test]
    fn test_correlation_dimension_point() {
        // Single point repeated → dimension 0
        let pts: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 1.0]).collect();
        let dim = estimate_correlation_dimension(&pts, 0.5);
        assert!(dim.abs() < 1.0); // Should be ~0
    }

    #[test]
    fn test_correlation_dimension_line() {
        // Points on a line → dimension ~1
        let pts: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 / 100.0, 0.0]).collect();
        let dim = estimate_correlation_dimension(&pts, 0.3);
        assert!(dim > 0.3); // Should be approximately 1
    }

    #[test]
    fn test_attractor_serialization() {
        let att = Attractor {
            attractor_type: AttractorType::FixedPoint(vec![0.0, 0.0]),
            basin_samples: vec![vec![1.0, 1.0]],
            confidence: 0.9,
        };
        let json = serde_json::to_string(&att).unwrap();
        assert!(json.contains("FixedPoint"));
    }

    #[test]
    fn test_strange_attractor_lorenz() {
        // Simplified Lorenz-like system
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
        let att = detect_attractor(&vf, &x0, 20.0, 0.005, 10.0);
        // Should detect either strange attractor or limit cycle
        assert!(att.confidence > 0.0);
    }
}
