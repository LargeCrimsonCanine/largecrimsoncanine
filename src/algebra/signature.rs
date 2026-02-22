//! Metric signature definitions for Clifford algebras Cl(p,q,r).
//!
//! The signature determines how basis vectors square:
//! - `p` basis vectors square to +1 (positive/spacelike)
//! - `q` basis vectors square to -1 (negative/timelike)
//! - `r` basis vectors square to 0 (degenerate/null)
//!
//! Common algebras:
//! - Euclidean R^n: Cl(n,0,0)
//! - Projective GA (PGA): Cl(n,0,1)
//! - Conformal GA (CGA): Cl(n+1,1,0)
//! - Spacetime Algebra (STA): Cl(1,3,0) or Cl(3,1,0)

/// Metric signature for a Clifford algebra Cl(p,q,r).
///
/// The total dimension is p + q + r, and the algebra has 2^(p+q+r) basis blades.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    /// Number of basis vectors squaring to +1
    pub p: usize,
    /// Number of basis vectors squaring to -1
    pub q: usize,
    /// Number of basis vectors squaring to 0 (degenerate)
    pub r: usize,
}

impl Signature {
    /// Create a new signature Cl(p,q,r).
    ///
    /// # Arguments
    /// * `p` - Number of basis vectors with positive square (+1)
    /// * `q` - Number of basis vectors with negative square (-1)
    /// * `r` - Number of degenerate basis vectors (square to 0)
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// let sig = Signature::new(3, 1, 0);  // Spacetime algebra Cl(3,1,0)
    /// assert_eq!(sig.dimension(), 4);
    /// ```
    pub fn new(p: usize, q: usize, r: usize) -> Self {
        Self { p, q, r }
    }

    /// Create a Euclidean signature Cl(n,0,0).
    ///
    /// All basis vectors square to +1.
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// let r3 = Signature::euclidean(3);
    /// assert_eq!(r3.p, 3);
    /// assert_eq!(r3.q, 0);
    /// assert_eq!(r3.r, 0);
    /// ```
    pub fn euclidean(n: usize) -> Self {
        Self { p: n, q: 0, r: 0 }
    }

    /// Create a Projective Geometric Algebra signature Cl(n,0,1).
    ///
    /// PGA adds one degenerate basis vector (e0) for representing
    /// points at infinity and enabling translations as rotations.
    ///
    /// - PGA2D: Cl(2,0,1) for 2D Euclidean geometry
    /// - PGA3D: Cl(3,0,1) for 3D Euclidean geometry
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// let pga3d = Signature::pga(3);
    /// assert_eq!(pga3d, Signature::new(3, 0, 1));
    /// assert_eq!(pga3d.dimension(), 4);
    /// ```
    pub fn pga(n: usize) -> Self {
        Self { p: n, q: 0, r: 1 }
    }

    /// Create a Conformal Geometric Algebra signature Cl(n+1,1,0).
    ///
    /// CGA adds two extra dimensions: one positive (e+) and one negative (e-).
    /// These are often combined into origin (eo) and infinity (ei) null vectors.
    ///
    /// - CGA2D: Cl(3,1,0) for 2D conformal geometry
    /// - CGA3D: Cl(4,1,0) for 3D conformal geometry
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// let cga3d = Signature::cga(3);
    /// assert_eq!(cga3d, Signature::new(4, 1, 0));
    /// assert_eq!(cga3d.dimension(), 5);
    /// ```
    pub fn cga(n: usize) -> Self {
        Self { p: n + 1, q: 1, r: 0 }
    }

    /// Create a Spacetime Algebra signature Cl(1,3,0).
    ///
    /// This is the "mostly minus" convention used in particle physics,
    /// where the timelike basis vector squares to +1 and spatial vectors to -1.
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// let sta = Signature::sta();
    /// assert_eq!(sta.basis_square(0), 1.0);  // Timelike
    /// assert_eq!(sta.basis_square(1), -1.0); // Spacelike
    /// ```
    pub fn sta() -> Self {
        Self { p: 1, q: 3, r: 0 }
    }

    /// Create an Algebra of Physical Space signature Cl(3,0,0).
    ///
    /// This is standard 3D Euclidean space, also known as APS.
    /// The even subalgebra is isomorphic to quaternions.
    pub fn aps() -> Self {
        Self::euclidean(3)
    }

    /// Total dimension of the vector space (number of basis vectors).
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// assert_eq!(Signature::euclidean(3).dimension(), 3);
    /// assert_eq!(Signature::pga(3).dimension(), 4);
    /// assert_eq!(Signature::cga(3).dimension(), 5);
    /// ```
    #[inline]
    pub fn dimension(&self) -> usize {
        self.p + self.q + self.r
    }

    /// Total number of basis blades in the algebra.
    ///
    /// This is 2^dimension, including scalar, vectors, bivectors, etc.
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// assert_eq!(Signature::euclidean(3).num_blades(), 8);   // 1 + 3 + 3 + 1
    /// assert_eq!(Signature::pga(3).num_blades(), 16);        // 2^4
    /// assert_eq!(Signature::cga(3).num_blades(), 32);        // 2^5
    /// ```
    #[inline]
    pub fn num_blades(&self) -> usize {
        1 << self.dimension()
    }

    /// Returns what basis vector i squares to: +1.0, -1.0, or 0.0.
    ///
    /// Basis vectors are ordered as:
    /// - Indices 0..p square to +1
    /// - Indices p..p+q square to -1
    /// - Indices p+q..p+q+r square to 0
    ///
    /// # Panics
    /// Panics if `i >= dimension()`.
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Signature;
    /// let pga = Signature::pga(3);  // Cl(3,0,1)
    /// assert_eq!(pga.basis_square(0), 1.0);  // e1
    /// assert_eq!(pga.basis_square(1), 1.0);  // e2
    /// assert_eq!(pga.basis_square(2), 1.0);  // e3
    /// assert_eq!(pga.basis_square(3), 0.0);  // e0 (degenerate)
    /// ```
    #[inline]
    pub fn basis_square(&self, i: usize) -> f64 {
        debug_assert!(i < self.dimension(), "Basis index {} out of bounds for dimension {}", i, self.dimension());
        if i < self.p {
            1.0
        } else if i < self.p + self.q {
            -1.0
        } else {
            0.0
        }
    }

    /// Check if this is a Euclidean signature (no negative or degenerate dimensions).
    #[inline]
    pub fn is_euclidean(&self) -> bool {
        self.q == 0 && self.r == 0
    }

    /// Check if this is a PGA signature (one degenerate dimension).
    #[inline]
    pub fn is_pga(&self) -> bool {
        self.q == 0 && self.r == 1
    }

    /// Check if this algebra has any degenerate dimensions.
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.r > 0
    }
}

impl std::fmt::Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cl({},{},{})", self.p, self.q, self.r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_signature() {
        let r3 = Signature::euclidean(3);
        assert_eq!(r3.p, 3);
        assert_eq!(r3.q, 0);
        assert_eq!(r3.r, 0);
        assert_eq!(r3.dimension(), 3);
        assert_eq!(r3.num_blades(), 8);
        assert!(r3.is_euclidean());
        assert!(!r3.is_pga());
        assert!(!r3.is_degenerate());
    }

    #[test]
    fn test_pga_signature() {
        let pga3d = Signature::pga(3);
        assert_eq!(pga3d.p, 3);
        assert_eq!(pga3d.q, 0);
        assert_eq!(pga3d.r, 1);
        assert_eq!(pga3d.dimension(), 4);
        assert_eq!(pga3d.num_blades(), 16);
        assert!(!pga3d.is_euclidean());
        assert!(pga3d.is_pga());
        assert!(pga3d.is_degenerate());
    }

    #[test]
    fn test_cga_signature() {
        let cga3d = Signature::cga(3);
        assert_eq!(cga3d.p, 4);
        assert_eq!(cga3d.q, 1);
        assert_eq!(cga3d.r, 0);
        assert_eq!(cga3d.dimension(), 5);
        assert_eq!(cga3d.num_blades(), 32);
        assert!(!cga3d.is_euclidean());
        assert!(!cga3d.is_degenerate());
    }

    #[test]
    fn test_sta_signature() {
        let sta = Signature::sta();
        assert_eq!(sta.p, 1);
        assert_eq!(sta.q, 3);
        assert_eq!(sta.r, 0);
        assert_eq!(sta.dimension(), 4);
    }

    #[test]
    fn test_basis_square_euclidean() {
        let r3 = Signature::euclidean(3);
        assert_eq!(r3.basis_square(0), 1.0);
        assert_eq!(r3.basis_square(1), 1.0);
        assert_eq!(r3.basis_square(2), 1.0);
    }

    #[test]
    fn test_basis_square_pga() {
        // In PGA(3), the degenerate vector is the last one
        let pga = Signature::pga(3);
        assert_eq!(pga.basis_square(0), 1.0);  // e1
        assert_eq!(pga.basis_square(1), 1.0);  // e2
        assert_eq!(pga.basis_square(2), 1.0);  // e3
        assert_eq!(pga.basis_square(3), 0.0);  // e0 (degenerate)
    }

    #[test]
    fn test_basis_square_sta() {
        let sta = Signature::sta();  // Cl(1,3,0)
        assert_eq!(sta.basis_square(0), 1.0);   // Timelike
        assert_eq!(sta.basis_square(1), -1.0);  // Spacelike x
        assert_eq!(sta.basis_square(2), -1.0);  // Spacelike y
        assert_eq!(sta.basis_square(3), -1.0);  // Spacelike z
    }

    #[test]
    fn test_basis_square_cga() {
        let cga = Signature::cga(3);  // Cl(4,1,0)
        // First 4 basis vectors square to +1
        assert_eq!(cga.basis_square(0), 1.0);
        assert_eq!(cga.basis_square(1), 1.0);
        assert_eq!(cga.basis_square(2), 1.0);
        assert_eq!(cga.basis_square(3), 1.0);
        // Last one squares to -1
        assert_eq!(cga.basis_square(4), -1.0);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Signature::euclidean(3)), "Cl(3,0,0)");
        assert_eq!(format!("{}", Signature::pga(3)), "Cl(3,0,1)");
        assert_eq!(format!("{}", Signature::cga(3)), "Cl(4,1,0)");
        assert_eq!(format!("{}", Signature::sta()), "Cl(1,3,0)");
    }

    #[test]
    fn test_equality() {
        assert_eq!(Signature::euclidean(3), Signature::new(3, 0, 0));
        assert_eq!(Signature::pga(3), Signature::new(3, 0, 1));
        assert_eq!(Signature::cga(3), Signature::new(4, 1, 0));
    }
}
