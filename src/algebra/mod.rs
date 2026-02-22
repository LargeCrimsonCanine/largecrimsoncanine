//! Algebra module for LargeCrimsonCanine.
//!
//! This module provides support for Clifford algebras Cl(p,q,r) with arbitrary
//! metric signatures:
//!
//! - `p` basis vectors square to +1 (Euclidean/spacelike)
//! - `q` basis vectors square to -1 (anti-Euclidean/timelike)
//! - `r` basis vectors square to 0 (degenerate/null)
//!
//! # Common Algebras
//!
//! | Name | Signature | Use Case |
//! |------|-----------|----------|
//! | R^n | Cl(n,0,0) | Euclidean geometry |
//! | PGA(n) | Cl(n,0,1) | Projective geometry (translations) |
//! | CGA(n) | Cl(n+1,1,0) | Conformal geometry (circles, spheres) |
//! | STA | Cl(1,3,0) | Spacetime algebra (relativity) |
//!
//! # Example
//!
//! ```
//! use largecrimsoncanine::algebra::{Algebra, Signature};
//!
//! // Create a 3D Euclidean algebra
//! let r3 = Algebra::euclidean(3);
//!
//! // Basis vector squares to +1
//! let (blade, sign) = r3.product(1, 1);  // e1 * e1
//! assert_eq!(blade, 0);  // scalar
//! assert_eq!(sign, 1.0); // +1
//!
//! // Create PGA for 3D projective geometry
//! let pga3d = Algebra::pga(3);
//!
//! // Degenerate basis vector squares to 0
//! let e0 = 1 << 3;  // 4th basis vector (index 8)
//! let (blade, sign) = pga3d.product(e0, e0);
//! assert_eq!(blade, 0);
//! assert_eq!(sign, 0.0);
//! ```

pub mod blades;
pub mod cayley;
mod registry;
mod signature;

// Re-export main types
pub use registry::{get_euclidean, Algebra};
pub use signature::Signature;

// Re-export commonly used blade functions at module level
pub use blades::{
    blade_grade, blade_name, clifford_conjugate_sign, grade_involution_sign, reverse_sign,
};

/// Compute the geometric product of two basis blades assuming Euclidean metric.
///
/// This is a backward-compatible function that assumes all basis vectors
/// square to +1. For non-Euclidean metrics, use `cayley::blade_product_signed`
/// or an `Algebra` instance.
///
/// Returns (result_blade, sign) where `e_a * e_b = sign * e_result`.
#[inline]
pub fn blade_product(a: usize, b: usize) -> (usize, f64) {
    // In Euclidean metric, all basis vectors square to +1,
    // so we only need the reordering sign (no metric contractions affect the sign)
    cayley::blade_product_signed(a, b, &Signature::euclidean(64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_module_exports() {
        // Verify we can use the main types
        let sig = Signature::euclidean(3);
        assert_eq!(sig.dimension(), 3);

        let alg = Algebra::new(sig);
        assert_eq!(alg.num_blades(), 8);
    }

    #[test]
    fn test_convenience_constructors() {
        let r3 = Algebra::euclidean(3);
        let pga3d = Algebra::pga(3);
        let cga3d = Algebra::cga(3);
        let sta = Algebra::sta();

        assert_eq!(r3.dimension(), 3);
        assert_eq!(pga3d.dimension(), 4);
        assert_eq!(cga3d.dimension(), 5);
        assert_eq!(sta.dimension(), 4);
    }

    #[test]
    fn test_re_exported_blade_functions() {
        // These should be available at module level
        assert_eq!(blade_grade(7), 3);
        assert_eq!(blade_name(3, 3), "e12");
        assert_relative_eq!(reverse_sign(2), -1.0);
        assert_relative_eq!(grade_involution_sign(1), -1.0);
        assert_relative_eq!(clifford_conjugate_sign(1), -1.0);
    }
}
