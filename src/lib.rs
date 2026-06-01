//! # lau-dynamical-systems-agents
//!
//! Smooth dynamical systems theory applied to agent behavior.
//!
//! This crate models agent dynamics as flows on manifolds, providing tools for:
//! - Fixed point analysis (stable/unstable classification)
//! - Linearization and eigenvalue stability
//! - Attractor identification (fixed points, limit cycles, strange attractors)
//! - Lyapunov exponent computation
//! - Bifurcation detection (saddle-node, transcritical, pitchfork, Hopf)
//! - Poincaré maps for flow reduction
//! - Melnikov method for chaos detection
//! - Smale horseshoe and symbolic dynamics

pub mod flow;
pub mod fixed_point;
pub mod linearization;
pub mod manifold;
pub mod attractor;
pub mod lyapunov;
pub mod bifurcation;
pub mod poincare;
pub mod melnikov;
pub mod horseshoe;
pub mod agent;

pub mod prelude {
    pub use crate::flow::*;
    pub use crate::fixed_point::*;
    pub use crate::linearization::*;
    pub use crate::manifold::*;
    pub use crate::attractor::*;
    pub use crate::lyapunov::*;
    pub use crate::bifurcation::*;
    pub use crate::poincare::*;
    pub use crate::melnikov::*;
    pub use crate::horseshoe::*;
    pub use crate::agent::*;
}
