# lau-dynamical-systems-agents

A Rust library applying **smooth dynamical systems theory to agent behavior modeling** — flows on manifolds, fixed-point analysis, linearization, attractor detection, Lyapunov exponents, bifurcation theory, Poincaré maps, Melnikov chaos detection, Smale horseshoe symbolic dynamics, and behavioral regime prediction.

---

## What This Does

| Module | What you get |
|---|---|
| `flow` | Vector field trait, RK4 integration, divergence, curl, trajectory storage |
| `fixed_point` | Find fixed points (Newton's method), classify stability (Stable/Unstable/Saddle/Center), scan regions |
| `linearization` | Jacobian analysis, eigenvalues/eigenvectors, matrix exponential, 2D fixed-point classification (node/spiral/saddle/center) |
| `manifold` | Stable/unstable manifold approximation, manifold dimension computation |
| `attractor` | Detect attractor type (fixed point, limit cycle, strange, torus), correlation dimension, basin of attraction |
| `lyapunov` | Full Lyapunov spectrum (Benettin algorithm), maximal Lyapunov exponent, Kaplan-Yorke dimension |
| `bifurcation` | Parameterized vector fields, bifurcation scanning, normal forms (saddle-node, transcritical, pitchfork, Hopf) |
| `poincare` | Poincaré section crossing detection, return map iterates, linearized Poincaré map eigenvalues |
| `melnikov` | Melnikov function (numerical + analytic), pendulum separatrix, chaos condition for damped-driven systems |
| `horseshoe` | Symbolic dynamics (subshifts of finite type), topological entropy, Smale horseshoe, periodic orbit counting, dynamical zeta function |
| `agent` | Behavioral regime classification (Stable/Oscillating/Chaotic/Transitional), coupled activity-mood model, regime diagrams, chaos risk assessment |

---

## Key Idea

Agent behavior evolves over time — it can settle into stable patterns, oscillate, or become chaotic. This library models behavioral state as a point on a manifold evolving under a flow. You can then apply the full machinery of dynamical systems theory: find equilibria, classify stability, detect when behavior bifurcates (regime change), measure chaos, and predict transitions.

---

## Install

```toml
[dependencies]
lau-dynamical-systems-agents = { git = "https://github.com/SuperInstance/lau-dynamical-systems-agents" }
```

Requires Rust 2021 edition. Dependencies: `nalgebra` 0.33, `serde` 1, `num-traits` 0.2, `num-complex` 0.4.

---

## Quick Start

### Define a behavioral vector field and integrate

```rust
use lau_dynamical_systems_agents::flow::{FnVectorField, flow};
use nalgebra::DVector;

// dx/dt = -x, dy/dt = -y (stable equilibrium at origin)
let vf = FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
let x0 = DVector::from_vec(vec![1.0, 1.0]);
let result = flow(&vf, &x0, 5.0, 0.01, true);

println!("Endpoint: {:?}", result.endpoint); // ≈ [0.007, 0.007]
println!("Trajectory length: {}", result.trajectory.len());
```

### Find and classify fixed points

```rust
use lau_dynamical_systems_agents::fixed_point::{find_fixed_point, classify_stability, Stability};

let vf = FnVectorField::new(2, |x| DVector::from_vec(vec![-x[0], -x[1]]));
let fp = find_fixed_point(&vf, &DVector::from_vec(vec![0.5, 0.5]), 50, 1e-10).unwrap();
let result = classify_stability(&vf, &fp);

assert_eq!(result.stability, Stability::Stable);
assert_eq!(result.stable_manifold_dim, 2);
```

### Lyapunov exponents and chaos detection

```rust
use lau_dynamical_systems_agents::lyapunov::compute_lyapunov_exponents;

// Lorenz system (chaotic at canonical parameters)
let sigma = 10.0; let rho = 28.0; let beta = 8.0/3.0;
let vf = FnVectorField::new(3, move |x| DVector::from_vec(vec![
    sigma * (x[1] - x[0]),
    x[0] * (rho - x[2]) - x[1],
    x[0] * x[1] - beta * x[2],
]));

let le = compute_lyapunov_exponents(&vf, &DVector::from_vec(vec![1.0, 1.0, 1.0]), 20.0, 0.005, 20);
assert!(le.is_chaotic); // largest exponent > 0
println!("KY dimension: {:.3}", le.ky_dimension); // ≈ 2.06
```

### Detect bifurcations

```rust
use lau_dynamical_systems_agents::bifurcation::{pitchfork_field, detect_bifurcations};

let vf = pitchfork_field(); // dx/dt = μx - x³
let x0 = DVector::from_vec(vec![0.01]);
let bifs = detect_bifurcations(&vf, &x0, (-2.0, 2.0), 200, 1e-8);
// Detects pitchfork bifurcation near μ=0
```

### Agent behavioral regime prediction

```rust
use lau_dynamical_systems_agents::agent::{
    analyze_agent_behavior, CoupledActivityMood, BehavioralRegime
};
use lau_dynamical_systems_agents::bifurcation::FnParameterizedField;

let vf = CoupledActivityMood::new(1.0, 0.1, 1.0, 0.1);
let x0 = DVector::from_vec(vec![1.0, 1.0]);

struct SimpleParam;
impl lau_dynamical_systems_agents::bifurcation::ParameterizedVectorField for SimpleParam {
    fn dim(&self) -> usize { 2 }
    fn evaluate(&self, x: &DVector<f64>, _mu: f64) -> DVector<f64> {
        DVector::from_vec(vec![-x[0], -x[1]])
    }
}

let prediction = analyze_agent_behavior(&vf, &x0, &SimpleParam, (-1.0, 1.0), 5.0, 0.01);
println!("Regime: {} (chaos risk: {:.1}%)", prediction.current_regime, prediction.chaos_risk * 100.0);
```

### Symbolic dynamics and topological entropy

```rust
use lau_dynamical_systems_agents::horseshoe::{SymbolicDynamics, HorseshoeMap};

// Full 2-shift (Smale horseshoe)
let sd = SymbolicDynamics::full_shift(2);
println!("Topological entropy: {:.4} (= ln 2)", sd.topological_entropy());
println!("Sequences of length 10: {}", sd.count_sequences(10));

let hs = HorseshoeMap::new(3.0, 2);
println!("Periodic orbits of period 5: {}", hs.count_periodic_orbits(5));
```

### Melnikov method for chaos

```rust
use lau_dynamical_systems_agents::melnikov::{
    melnikov_analytic_damped_driven, chaos_condition_damped_driven
};

// Damped driven pendulum: ẍ + δẋ + sin(x) = γcos(ωt)
let has_chaos = chaos_condition_damped_driven(0.1, 1.0, 1.0);
assert!(has_chaos); // driving overcomes damping → chaos
```

---

## API Reference

### `flow`

| Type / Function | Description |
|---|---|
| `VectorField` (trait) | `dim()`, `evaluate(x)`, `jacobian(x)` (central differences default). |
| `FnVectorField` | Closure-based vector field. |
| `flow(vf, x0, t, dt, store)` | RK4 integration. Returns `FlowResult { endpoint, trajectory, times }`. |
| `flow_steps(vf, x0, steps, dt)` | Fixed-step integration, returns final state. |
| `divergence(vf, x)` | ∇·V = Σ ∂Vᵢ/∂xᵢ. |
| `curl_2d(vf, x)` | ∂V₂/∂x₁ − ∂V₁/∂x₂. |

### `fixed_point`

| Type / Function | Description |
|---|---|
| `Stability` | `Stable`, `Unstable`, `Center`, `Saddle`. |
| `FixedPoint` | Point + stability + eigenvalues + manifold dimensions. |
| `find_fixed_point(vf, x0, max_iter, tol)` | Newton's method. |
| `classify_stability(vf, point)` | Jacobian → eigenvalues → classification. |
| `scan_fixed_points(vf, bounds, grid_res, ...)` | Grid search with deduplication. |

### `linearization`

| Type / Function | Description |
|---|---|
| `LinearizationResult` | Jacobian, eigenvalues, eigenvectors, hyperbolicity, trace, determinant. |
| `linearize(vf, point)` | Full linearization analysis. |
| `linearized_flow(vf, fp, x, t)` | Hartman-Grobman approximation: x* + exp(Jt)·(x−x*). |
| `matrix_exp(a, t)` | Matrix exponential via Padé (scaling + squaring). |
| `classify_2d(trace, det)` | 2D fixed-point classification from τ and Δ. |
| `FixedPointType2D` | `StableNode`, `UnstableNode`, `StableSpiral`, `UnstableSpiral`, `Center`, `Saddle`, `Degenerate`. |

### `manifold`

| Type / Function | Description |
|---|---|
| `approximate_stable_manifold(vf, fp, ε, n, t, dt)` | Integrate from eigenspace directions. |
| `approximate_unstable_manifold(vf, fp, ε, n, t, dt)` | Same, forward direction. |
| `manifold_dimensions(vf, fp)` | Returns (stable_dim, unstable_dim). |
| `is_on_stable_manifold(vf, fp, point, t, dt, tol)` | Check if a point converges to the fixed point. |

### `attractor`

| Type / Function | Description |
|---|---|
| `AttractorType` | `FixedPoint`, `LimitCycle { center, radius }`, `Strange { dimension }`, `Torus { radii }`. |
| `detect_attractor(vf, x0, t_total, dt, t_transient)` | Classify attractor by trajectory analysis. |
| `estimate_correlation_dimension(points, r)` | Grassberger-Procaccia algorithm. |
| `basin_of_attraction(vf, attractor, bounds, grid, t, dt, tol)` | Grid sweep for basin boundary. |

### `lyapunov`

| Type / Function | Description |
|---|---|
| `LyapunovExponents` | Exponents (sorted), KY dimension, `is_chaotic`, sum. |
| `compute_lyapunov_exponents(vf, x0, t, dt, orth_interval)` | Full Benettin algorithm. |
| `maximal_lyapunov_exponent(vf, x0, t, dt, ε)` | MLE via perturbation-renormalization. |
| `kaplan_yorke_dimension(exponents)` | D_KY from spectrum. |

### `bifurcation`

| Type / Function | Description |
|---|---|
| `BifurcationType` | `SaddleNode`, `Transcritical`, `Pitchfork`, `Hopf`, `Other`. |
| `BifurcationPoint` | μ value + type + location + confidence. |
| `ParameterizedVectorField` (trait) | `evaluate(x, μ)`, `jacobian(x, μ)`. |
| `FnParameterizedField` | Closure-based parameterized field. |
| `detect_bifurcations(vf, x_guess, μ_range, steps, tol)` | Sweep parameter range for bifurcations. |
| `saddle_node_field()`, `transcritical_field()`, `pitchfork_field()`, `hopf_field(ω)` | Normal forms. |

### `poincare`

| Type / Function | Description |
|---|---|
| `PoincareSection` | Hyperplane: normal + point. `signed_distance()`, `is_on_section()`. |
| `PoincareMapResult` | Return point + return time + steps. |
| `poincare_map(vf, x0, section, dt, max_steps)` | First return to section. |
| `poincare_map_iterates(vf, x0, section, dt, max, n)` | Multiple returns. |
| `poincare_map_eigenvalues(vf, fp, section, dt, max, ε)` | Linearized Poincaré map → Floquet multiplier magnitudes. |

### `melnikov`

| Type / Function | Description |
|---|---|
| `MelnikovResult` | M(t₀) values, t₀ values, `has_simple_zeros`, `chaos_indicator`. |
| `melnikov_function(f, g, orbit, times, t₀_range, ...)` | Numerical Melnikov integral. |
| `pendulum_separatrix(t_range, steps)` | Analytic heteroclinic orbit for undamped pendulum. |
| `melnikov_analytic_damped_driven(δ, γ, ω, t₀)` | Closed-form Melnikov for damped driven pendulum. |
| `chaos_condition_damped_driven(δ, γ, ω)` | Check if |γI₂| > |δI₁| (simple zeros exist → chaos). |

### `horseshoe`

| Type / Function | Description |
|---|---|
| `SymbolSequence` | Symbol vector with `shift()`, `push()`. |
| `SymbolicDynamics` | Alphabet + allowed transitions. `is_valid_sequence()`, `count_sequences(n)`, `topological_entropy()`. |
| `HorseshoeMap` | Stretch factor + strips. `entropy()`, `symbolic_dynamics()`, `count_periodic_orbits(n)`. |
| `dynamical_zeta(symbols, z, max_n)` | Dynamical zeta function computation. |

### `agent`

| Type / Function | Description |
|---|---|
| `BehavioralRegime` | `Stable`, `Oscillating`, `Chaotic`, `Transitional`. |
| `AgentState` | State + regime + confidence + timestamp. |
| `BehaviorPrediction` | Current/predicted regime, Lyapunov exponents, attractors, bifurcations, chaos risk. |
| `CoupledActivityMood` | 2D behavioral model (activity × mood). Implements `VectorField`. |
| `analyze_agent_behavior(vf, state, param_vf, μ_range, t, dt)` | Full behavioral analysis pipeline. |
| `classify_regime(le)` | Lyapunov exponents → behavioral regime. |
| `regime_diagram(factory, p1_range, p2_range, grid, x0, t, dt)` | 2D parameter grid → regime map. |

---

## How It Works

The library builds a pipeline from primitives to behavioral prediction:

1. **Flows** define how agent state evolves: dx/dt = V(x). The `VectorField` trait captures this, with RK4 integration for stepping forward (or backward) in time. Divergence and curl provide scalar summaries of the field.

2. **Fixed points** are equilibria where V(x*) = 0. Newton's method finds them iteratively. The Jacobian's eigenvalues at x* classify the equilibrium: all negative real parts → stable (agent settles), mixed → saddle (some directions attract, some repel), all positive → unstable (agent diverges), pure imaginary → center (agent oscillates).

3. **Linearization** approximates dynamics near a fixed point: dx ≈ J·dx. The Hartman-Grobman theorem guarantees this is topologically correct for hyperbolic fixed points. The matrix exponential exp(Jt) gives the linearized flow. For 2D systems, trace τ and determinant Δ of the Jacobian fully classify the fixed point type.

4. **Manifolds** extend fixed-point analysis: the stable manifold Wˢ is the set of all points converging to x* (t→∞), the unstable manifold Wᵘ converges backwards (t→−∞). These are approximated by integrating along eigenspace directions.

5. **Attractors** are what the system settles into long-term: fixed points, limit cycles (periodic orbits), strange attractors (fractal, chaotic), or tori (quasiperiodic). Detection works by discarding transients and analyzing the asymptotic trajectory — convergence to a point, periodicity, or fractal dimension via the Grassberger-Procaccia algorithm.

6. **Lyapunov exponents** measure exponential divergence/convergence rates. The Benettin algorithm evolves perturbation vectors alongside the trajectory, periodically reorthogonalizing via Gram-Schmidt. A positive largest Lyapunov exponent means chaos. The Kaplan-Yorke dimension interpolates between exponents to estimate the attractor's fractal dimension.

7. **Bifurcations** occur when a parameter change causes a qualitative change in dynamics. The library scans parameter ranges, tracking fixed points and their stability. When stability changes or fixed points appear/disappear, a bifurcation is detected and classified. Normal forms (saddle-node, transcritical, pitchfork, Hopf) provide canonical examples.

8. **Poincaré maps** reduce a continuous flow to a discrete map on a cross-section. Each return to the section gives one point; the return map's fixed points correspond to periodic orbits. The linearized Poincaré map's eigenvalues (Floquet multipliers) determine periodic orbit stability.

9. **Melnikov method** detects chaos in perturbed Hamiltonian systems by measuring the distance between stable and unstable manifolds. If the Melnikov function has simple zeros, the manifolds intersect transversely, implying a Smale horseshoe and hence chaos. The damped driven pendulum is the canonical example.

10. **Symbolic dynamics** discretizes orbits into symbol sequences. The Smale horseshoe creates a full shift on N symbols; restricted transitions give subshifts of finite type. Topological entropy (log of largest eigenvalue of transition matrix) quantifies chaos. Periodic orbits are counted via Möbius inversion.

11. **Agent modeling** ties it all together: behavioral state evolves under a flow, Lyapunov exponents classify the regime (stable/oscillating/chaotic), bifurcation detection warns of transitions, and chaos risk is estimated. The regime diagram maps behavioral regimes across a 2D parameter space.

---

## The Math

### Linear Stability (Hartman-Grobman)
Near a hyperbolic fixed point x* (no eigenvalue with zero real part), the nonlinear flow is topologically conjugate to its linearization: the phase portrait near x* looks like the linear system ẋ = J(x*)·x. Classification depends on the eigenvalues of J.

### 2D Classification (Trace-Determinant Plane)
For a 2D system with Jacobian having trace τ and determinant Δ:
- **Saddle:** Δ < 0 (eigenvalues real, opposite signs)
- **Stable node:** Δ > 0, τ < 0, τ² > 4Δ (real negative eigenvalues)
- **Stable spiral:** Δ > 0, τ < 0, τ² < 4Δ (complex eigenvalues, negative real part)
- **Unstable node:** Δ > 0, τ > 0, τ² > 4Δ
- **Unstable spiral:** Δ > 0, τ > 0, τ² < 4Δ
- **Center:** Δ > 0, τ = 0 (pure imaginary eigenvalues)

### Benettin Algorithm (Lyapunov Spectrum)
For an n-dimensional system, evolve n orthonormal tangent vectors w₁,...,wₙ alongside the trajectory by the variational equation dw/dt = J(x(t))·w. Every k steps, apply Gram-Schmidt reorthonormalization. The i-th Lyapunov exponent is:

λᵢ = lim(T→∞) (1/T) Σₖ ln ‖wᵢ⁽ᵏ⁾‖

### Kaplan-Yorke Dimension
Given sorted exponents λ₁ ≥ λ₂ ≥ ... ≥ λₙ, find the largest j with Σᵢ₌₁ʲ λᵢ ≥ 0. Then:

D_KY = j + (Σᵢ₌₁ʲ λᵢ) / |λ_{j+1}|

This bounds the information dimension of the attractor.

### Bifurcation Normal Forms
- **Saddle-node:** ẋ = μ + x². Two fixed points collide at μ = 0.
- **Transcritical:** ẋ = μx − x². Fixed points exchange stability at μ = 0.
- **Pitchfork:** ẋ = μx − x³. One fixed point splits into three (symmetry breaking).
- **Hopf:** A stable spiral becomes unstable + limit cycle. In polar: ṙ = μr − r³, θ̇ = ω.

### Melnikov Method
For ẋ = f(x) + εg(x,t), the Melnikov function M(t₀) = ∫ f(x⁰(t)) ∧ g(x⁰(t), t+t₀) dt measures the distance between stable and unstable manifolds. Simple zeros of M ⟹ transverse homoclinic intersections ⟹ Smale horseshoe ⟹ chaos.

For the damped driven pendulum: M(t₀) = −δ·8 + γ·(πω/cosh(πω/2))·sin(ωt₀). Simple zeros exist when |γ·πω/cosh(πω/2)| > |8δ|.

### Topological Entropy
For a subshift of finite type with transition matrix A, the topological entropy is h_top = ln(λ_max) where λ_max is the largest eigenvalue of A. For the full N-shift, h_top = ln(N).

### Periodic Orbit Counting (Möbius Inversion)
The number of primitive periodic orbits of period n under a full N-shift is:

P(n) = (1/n) Σ_{d|n} μ(n/d) · N^d

where μ is the Möbius function.

---

## License

MIT
