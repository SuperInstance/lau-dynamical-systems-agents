//! Smale horseshoe: symbolic dynamics, topological entropy.
//!
//! The Smale horseshoe map stretches and folds the phase space, creating
//! a fractal invariant set with chaotic dynamics described by symbol sequences.
//! Topological entropy measures the exponential growth rate of distinguishable orbits.

use serde::{Serialize, Deserialize};

/// A symbol sequence representing an orbit in symbolic dynamics.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolSequence {
    pub symbols: Vec<u8>,
}

impl SymbolSequence {
    pub fn new(symbols: Vec<u8>) -> Self {
        Self { symbols }
    }

    /// Length of the symbol sequence.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Shift the sequence left by one position, dropping the first symbol.
    pub fn shift(&self) -> SymbolSequence {
        if self.symbols.is_empty() {
            return Self::new(vec![]);
        }
        Self::new(self.symbols[1..].to_vec())
    }

    /// Append a symbol.
    pub fn push(&mut self, s: u8) {
        self.symbols.push(s);
    }
}

/// A symbolic dynamics system with forbidden transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicDynamics {
    /// Number of symbols (alphabet size).
    pub alphabet_size: u8,
    /// Allowed transitions: (from, to) pairs.
    pub allowed_transitions: Vec<(u8, u8)>,
}

impl SymbolicDynamics {
    pub fn full_shift(alphabet_size: u8) -> Self {
        let mut transitions = Vec::new();
        for i in 0..alphabet_size {
            for j in 0..alphabet_size {
                transitions.push((i, j));
            }
        }
        Self {
            alphabet_size,
            allowed_transitions: transitions,
        }
    }

    pub fn new(alphabet_size: u8, allowed_transitions: Vec<(u8, u8)>) -> Self {
        Self { alphabet_size, allowed_transitions }
    }

    /// Check if a transition is allowed.
    pub fn is_allowed(&self, from: u8, to: u8) -> bool {
        self.allowed_transitions.contains(&(from, to))
    }

    /// Check if a symbol sequence is valid.
    pub fn is_valid_sequence(&self, seq: &SymbolSequence) -> bool {
        for i in 0..seq.symbols.len().saturating_sub(1) {
            if !self.is_allowed(seq.symbols[i], seq.symbols[i + 1]) {
                return false;
            }
        }
        true
    }

    /// Count the number of valid sequences of length n.
    pub fn count_sequences(&self, n: usize) -> u64 {
        if n == 0 {
            return 1;
        }
        if n == 1 {
            return self.alphabet_size as u64;
        }

        // Use transition matrix power
        let m = self.transition_matrix();
        let mp = matrix_power(&m, n - 1);
        let mut total = 0u64;
        for row in mp.iter() {
            for &val in row.iter() {
                total += val;
            }
        }
        total
    }

    /// Build the transition matrix.
    pub fn transition_matrix(&self) -> Vec<Vec<u64>> {
        let n = self.alphabet_size as usize;
        let mut m = vec![vec![0u64; n]; n];
        for &(from, to) in &self.allowed_transitions {
            m[from as usize][to as usize] = 1;
        }
        m
    }

    /// Compute topological entropy: h_top = lim_{n→∞} (1/n) log(|Words(n)|).
    /// For a subshift of finite type, h_top = log(λ_max) where λ_max is the
    /// largest eigenvalue of the transition matrix.
    pub fn topological_entropy(&self) -> f64 {
        let m = self.transition_matrix();
        let lambda = largest_eigenvalue(&m);
        if lambda > 0.0 {
            lambda.ln()
        } else {
            0.0
        }
    }
}

/// Compute matrix power.
fn matrix_power(m: &[Vec<u64>], p: usize) -> Vec<Vec<u64>> {
    let n = m.len();
    if p == 0 {
        let mut id = vec![vec![0u64; n]; n];
        for (i, row) in id.iter_mut().enumerate() {
            row[i] = 1;
        }
        return id;
    }
    if p == 1 {
        return m.to_vec();
    }

    let half = matrix_power(m, p / 2);
    let mut result = mat_mul(&half, &half);
    if !p.is_multiple_of(2) {
        result = mat_mul(&result, m);
    }
    result
}

fn mat_mul(a: &[Vec<u64>], b: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let n = a.len();
    let mut c = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Estimate largest eigenvalue using power iteration.
fn largest_eigenvalue(m: &[Vec<u64>]) -> f64 {
    let n = m.len();
    if n == 0 {
        return 0.0;
    }

    let m_f64: Vec<Vec<f64>> = m.iter().map(|r| r.iter().map(|&v| v as f64).collect()).collect();

    let mut v = vec![1.0 / n as f64; n];
    let mut lambda = 1.0;

    for _ in 0..100 {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += m_f64[i][j] * v[j];
            }
        }
        lambda = new_v.iter().cloned().fold(0.0f64, f64::max);
        if lambda > 1e-15 {
            for x in &mut new_v {
                *x /= lambda;
            }
        }
        v = new_v;
    }

    lambda
}

/// Smale horseshoe map: models the stretch-and-fold dynamics.
/// Maps a square S into a horseshoe shape: S → H(S).
#[derive(Debug, Clone)]
pub struct HorseshoeMap {
    /// Stretch factor (> 1).
    pub stretch: f64,
    /// Number of horizontal strips that map back into S.
    pub num_strips: u8,
}

impl HorseshoeMap {
    pub fn new(stretch: f64, num_strips: u8) -> Self {
        Self { stretch, num_strips }
    }

    /// Compute topological entropy: h = log(num_strips) for full shift.
    pub fn entropy(&self) -> f64 {
        (self.num_strips as f64).ln()
    }

    /// Generate the associated symbolic dynamics (full shift on N symbols).
    pub fn symbolic_dynamics(&self) -> SymbolicDynamics {
        SymbolicDynamics::full_shift(self.num_strips)
    }

    /// Count periodic orbits of period n.
    pub fn count_periodic_orbits(&self, n: usize) -> u64 {
        // For full shift: N^n orbits total, but we want primitive ones
        // Use Möbius inversion: primitive(n) = Σ_{d|n} μ(n/d) * N^d
        let mut total = 0i64;
        for d in 1..=n {
            if n.is_multiple_of(d) {
                let mobius = mobius_function(n / d);
                total += mobius * (self.num_strips as i64).pow(d as u32);
            }
        }
        total.max(0) as u64
    }
}

/// Möbius function μ(n).
fn mobius_function(n: usize) -> i64 {
    if n == 1 {
        return 1;
    }
    let mut m = n;
    let mut count = 0i32;
    let mut d = 2;
    while d * d <= m {
        if m.is_multiple_of(d) {
            m /= d;
            count += 1;
            if m.is_multiple_of(d) {
                return 0; // squared prime factor
            }
        }
        d += 1;
    }
    if m > 1 {
        count += 1;
    }
    if count % 2 == 0 { 1 } else { -1 }
}

/// Compute the zeta function ζ(z) = exp(Σ_{n≥1} |Fix(fⁿ)| zⁿ / n)
/// for the horseshoe map. |Fix(fⁿ)| = N^n for full shift.
pub fn dynamical_zeta(num_symbols: u8, z: f64, max_n: usize) -> f64 {
    let mut sum = 0.0;
    for n in 1..=max_n {
        sum += (num_symbols as f64).powi(n as i32) * z.powi(n as i32) / n as f64;
    }
    sum.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_shift_entropy() {
        let sd = SymbolicDynamics::full_shift(2);
        assert!((sd.topological_entropy() - 2.0_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_full_shift_sequences() {
        let sd = SymbolicDynamics::full_shift(2);
        assert_eq!(sd.count_sequences(1), 2);
        assert_eq!(sd.count_sequences(2), 4);
        assert_eq!(sd.count_sequences(3), 8);
    }

    #[test]
    fn test_subshift_sequences() {
        // Only transitions: 0→1, 1→0
        let sd = SymbolicDynamics::new(2, vec![(0, 1), (1, 0)]);
        assert_eq!(sd.count_sequences(1), 2);
        assert_eq!(sd.count_sequences(2), 2); // 01, 10
        assert_eq!(sd.count_sequences(3), 2); // 010, 101
    }

    #[test]
    fn test_valid_sequence() {
        let sd = SymbolicDynamics::new(2, vec![(0, 1), (1, 0)]);
        assert!(sd.is_valid_sequence(&SymbolSequence::new(vec![0, 1, 0, 1])));
        assert!(!sd.is_valid_sequence(&SymbolSequence::new(vec![0, 0])));
    }

    #[test]
    fn test_symbol_shift() {
        let seq = SymbolSequence::new(vec![0, 1, 2]);
        let shifted = seq.shift();
        assert_eq!(shifted.symbols, vec![1, 2]);
    }

    #[test]
    fn test_horseshoe_entropy() {
        let hs = HorseshoeMap::new(3.0, 2);
        assert!((hs.entropy() - 2.0_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_horseshoe_periodic_orbits() {
        let hs = HorseshoeMap::new(3.0, 2);
        // Period 1: 2 (0, 1)
        assert_eq!(hs.count_periodic_orbits(1), 2);
        // Period 2: 2^2 - 2 = 2 primitive (00→0 period 1, 11→1 period 1, 01, 10)
        assert_eq!(hs.count_periodic_orbits(2), 2);
    }

    #[test]
    fn test_mobius() {
        assert_eq!(mobius_function(1), 1);
        assert_eq!(mobius_function(2), -1);
        assert_eq!(mobius_function(3), -1);
        assert_eq!(mobius_function(4), 0);
        assert_eq!(mobius_function(6), 1);
    }

    #[test]
    fn test_dynamical_zeta() {
        let zeta = dynamical_zeta(2, 0.1, 10);
        assert!(zeta > 1.0);
    }

    #[test]
    fn test_symbolic_dynamics_serialization() {
        let sd = SymbolicDynamics::full_shift(2);
        let json = serde_json::to_string(&sd).unwrap();
        assert!(json.contains("alphabet_size"));
    }

    #[test]
    fn test_subshift_entropy() {
        // Golden mean shift: 0→0, 0→1, 1→0 → entropy = log(φ) where φ is golden ratio
        let sd = SymbolicDynamics::new(2, vec![(0, 0), (0, 1), (1, 0)]);
        let entropy = sd.topological_entropy();
        // Largest eigenvalue of [[1,1],[1,0]] is golden ratio ≈ 1.618
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((entropy - golden.ln()).abs() < 0.01);
    }
}
