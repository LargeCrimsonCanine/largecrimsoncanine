//! Algebra registry and cached algebra structures.
//!
//! This module provides the `Algebra` struct, which combines a metric signature
//! with pre-computed Cayley tables for efficient geometric algebra operations.
//!
//! Algebras are typically wrapped in `Arc<Algebra>` for shared ownership across
//! multiple multivectors.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use super::blades::{all_blade_names, all_grade_masks};
use super::cayley::{compute_cayley_table, CayleyTable};
use super::Signature;

/// A pre-computed algebra structure for efficient geometric algebra operations.
///
/// Contains the metric signature and cached Cayley table for looking up
/// products of basis blades in O(1) time.
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::Algebra;
///
/// let r3 = Algebra::euclidean(3);
/// assert_eq!(r3.signature.dimension(), 3);
/// assert_eq!(r3.num_blades(), 8);
///
/// // Product lookup
/// let (blade, sign) = r3.product(1, 2);  // e1 * e2 = e12
/// assert_eq!(blade, 3);
/// assert_eq!(sign, 1.0);
/// ```
#[derive(Debug)]
pub struct Algebra {
    /// The metric signature (p, q, r)
    pub signature: Signature,

    /// Pre-computed Cayley table for product lookups
    cayley: CayleyTable,

    /// Human-readable names for each blade
    pub blade_names: Vec<String>,

    /// Bitmasks for each grade (grade_masks[k] has bits set for all grade-k blades)
    pub grade_masks: Vec<usize>,
}

impl Algebra {
    /// Create a new algebra from a signature.
    ///
    /// Pre-computes the Cayley table and blade names.
    ///
    /// # Arguments
    /// * `sig` - The metric signature Cl(p,q,r)
    ///
    /// # Returns
    /// An `Arc<Algebra>` for shared ownership.
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::{Algebra, Signature};
    ///
    /// let minkowski = Algebra::new(Signature::new(1, 3, 0));
    /// assert_eq!(minkowski.signature.p, 1);
    /// assert_eq!(minkowski.signature.q, 3);
    /// ```
    pub fn new(sig: Signature) -> Arc<Self> {
        let cayley = compute_cayley_table(&sig);
        let blade_names = all_blade_names(&sig);
        let grade_masks = all_grade_masks(sig.dimension());

        Arc::new(Self {
            signature: sig,
            cayley,
            blade_names,
            grade_masks,
        })
    }

    /// Create a Euclidean algebra Cl(n,0,0).
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Algebra;
    ///
    /// let r3 = Algebra::euclidean(3);
    /// assert!(r3.signature.is_euclidean());
    /// ```
    pub fn euclidean(n: usize) -> Arc<Self> {
        Self::new(Signature::euclidean(n))
    }

    /// Create a Projective Geometric Algebra Cl(n,0,1).
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Algebra;
    ///
    /// let pga3d = Algebra::pga(3);
    /// assert!(pga3d.signature.is_pga());
    /// assert_eq!(pga3d.signature.dimension(), 4);
    /// ```
    pub fn pga(n: usize) -> Arc<Self> {
        Self::new(Signature::pga(n))
    }

    /// Create a Conformal Geometric Algebra Cl(n+1,1,0).
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Algebra;
    ///
    /// let cga3d = Algebra::cga(3);
    /// assert_eq!(cga3d.signature.dimension(), 5);
    /// assert_eq!(cga3d.num_blades(), 32);
    /// ```
    pub fn cga(n: usize) -> Arc<Self> {
        Self::new(Signature::cga(n))
    }

    /// Create a Spacetime Algebra Cl(1,3,0).
    ///
    /// Uses the "mostly minus" convention common in particle physics.
    pub fn sta() -> Arc<Self> {
        Self::new(Signature::sta())
    }

    /// Get the total number of basis blades.
    #[inline]
    pub fn num_blades(&self) -> usize {
        self.cayley.num_blades
    }

    /// Get the dimension of the underlying vector space.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.signature.dimension()
    }

    /// Look up the geometric product of two basis blades.
    ///
    /// Returns (result_blade_index, sign) where `e_a * e_b = sign * e_result`.
    ///
    /// This is an O(1) table lookup.
    ///
    /// # Arguments
    /// * `a` - Binary index of first blade
    /// * `b` - Binary index of second blade
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Algebra;
    ///
    /// let r3 = Algebra::euclidean(3);
    ///
    /// // e1 * e2 = +e12
    /// let (blade, sign) = r3.product(1, 2);
    /// assert_eq!(blade, 3);
    /// assert_eq!(sign, 1.0);
    ///
    /// // e2 * e1 = -e12
    /// let (blade, sign) = r3.product(2, 1);
    /// assert_eq!(blade, 3);
    /// assert_eq!(sign, -1.0);
    ///
    /// // e1 * e1 = +1 (scalar)
    /// let (blade, sign) = r3.product(1, 1);
    /// assert_eq!(blade, 0);
    /// assert_eq!(sign, 1.0);
    /// ```
    #[inline]
    pub fn product(&self, a: usize, b: usize) -> (usize, f64) {
        self.cayley.product(a, b)
    }

    /// Get the sign of the product of two basis blades.
    ///
    /// This is slightly faster than `product()` when you only need the sign
    /// and already know the result blade (which is always `a ^ b`).
    #[inline]
    pub fn product_sign(&self, a: usize, b: usize) -> f64 {
        self.cayley.product(a, b).1
    }

    /// Get direct access to the Cayley table signs array.
    ///
    /// The array is flattened as `signs[a * num_blades + b]`.
    #[inline]
    pub fn cayley_signs(&self) -> &[f64] {
        &self.cayley.signs
    }

    /// Get direct access to the Cayley table blades array.
    ///
    /// The array is flattened as `blades[a * num_blades + b]`.
    #[inline]
    pub fn cayley_blades(&self) -> &[usize] {
        &self.cayley.blades
    }

    /// Get the human-readable name for a blade index.
    ///
    /// # Example
    /// ```
    /// use largecrimsoncanine::algebra::Algebra;
    ///
    /// let r3 = Algebra::euclidean(3);
    /// assert_eq!(r3.blade_name(0), "1");
    /// assert_eq!(r3.blade_name(1), "e1");
    /// assert_eq!(r3.blade_name(7), "e123");
    /// ```
    #[inline]
    pub fn blade_name(&self, index: usize) -> &str {
        &self.blade_names[index]
    }

    /// Get the grade mask for a specific grade.
    ///
    /// Returns a bitmask where bit i is set if blade i has the given grade.
    #[inline]
    pub fn grade_mask(&self, grade: usize) -> usize {
        if grade <= self.dimension() {
            self.grade_masks[grade]
        } else {
            0
        }
    }

    /// Check if a blade index has a specific grade.
    #[inline]
    pub fn blade_has_grade(&self, blade: usize, grade: usize) -> bool {
        if grade <= self.dimension() {
            (self.grade_masks[grade] >> blade) & 1 == 1
        } else {
            false
        }
    }

    /// Get the grade of a blade by its index.
    #[inline]
    pub fn blade_grade(&self, index: usize) -> usize {
        index.count_ones() as usize
    }

    /// Return indices of all blades of a specific grade.
    pub fn blades_of_grade(&self, grade: usize) -> Vec<usize> {
        if grade > self.dimension() {
            return vec![];
        }
        (0..self.num_blades())
            .filter(|&i| self.blade_grade(i) == grade)
            .collect()
    }

    /// Get the pseudoscalar (highest-grade blade) index.
    ///
    /// For dimension n, this is 2^n - 1 (all bits set).
    #[inline]
    pub fn pseudoscalar_index(&self) -> usize {
        self.num_blades() - 1
    }

    /// Get the sign of the pseudoscalar squared.
    ///
    /// This depends on the signature and dimension.
    pub fn pseudoscalar_squared_sign(&self) -> f64 {
        let ps = self.pseudoscalar_index();
        self.product(ps, ps).1
    }
}

static EUCLIDEAN_ALGEBRAS: Lazy<RwLock<HashMap<usize, Arc<Algebra>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Get or create a cached Euclidean algebra Cl(n,0,0).
///
/// This function caches algebras by dimension to avoid recomputing
/// Cayley tables for the same dimension multiple times.
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::get_euclidean;
///
/// let r3 = get_euclidean(3);  // Creates and caches Cl(3,0,0)
/// let r3_again = get_euclidean(3);  // Returns cached version
/// assert!(std::sync::Arc::ptr_eq(&r3, &r3_again));
/// ```
pub fn get_euclidean(n: usize) -> Arc<Algebra> {
    // Fast path: check if already cached
    {
        let cache = EUCLIDEAN_ALGEBRAS.read().unwrap();
        if let Some(alg) = cache.get(&n) {
            return alg.clone();
        }
    }
    // Slow path: create and cache
    let alg = Algebra::euclidean(n);
    EUCLIDEAN_ALGEBRAS.write().unwrap().insert(n, alg.clone());
    alg
}

impl std::fmt::Display for Algebra {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Algebra({})", self.signature)
    }
}

impl PartialEq for Algebra {
    fn eq(&self, other: &Self) -> bool {
        self.signature == other.signature
    }
}

impl Eq for Algebra {}

impl std::hash::Hash for Algebra {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==========================================================================
    // Construction tests
    // ==========================================================================

    #[test]
    fn test_euclidean_construction() {
        let r3 = Algebra::euclidean(3);
        assert_eq!(r3.signature, Signature::euclidean(3));
        assert_eq!(r3.dimension(), 3);
        assert_eq!(r3.num_blades(), 8);
        assert!(r3.signature.is_euclidean());
    }

    #[test]
    fn test_pga_construction() {
        let pga = Algebra::pga(3);
        assert_eq!(pga.signature, Signature::pga(3));
        assert_eq!(pga.dimension(), 4);
        assert_eq!(pga.num_blades(), 16);
        assert!(pga.signature.is_pga());
    }

    #[test]
    fn test_cga_construction() {
        let cga = Algebra::cga(3);
        assert_eq!(cga.signature, Signature::cga(3));
        assert_eq!(cga.dimension(), 5);
        assert_eq!(cga.num_blades(), 32);
    }

    #[test]
    fn test_sta_construction() {
        let sta = Algebra::sta();
        assert_eq!(sta.signature, Signature::sta());
        assert_eq!(sta.dimension(), 4);
        assert_eq!(sta.num_blades(), 16);
    }

    // ==========================================================================
    // Product tests
    // ==========================================================================

    #[test]
    fn test_euclidean_products() {
        let r3 = Algebra::euclidean(3);

        // Basis vectors square to +1
        assert_eq!(r3.product(1, 1), (0, 1.0));
        assert_eq!(r3.product(2, 2), (0, 1.0));
        assert_eq!(r3.product(4, 4), (0, 1.0));

        // Anticommutativity
        let (b1, s1) = r3.product(1, 2);
        let (b2, s2) = r3.product(2, 1);
        assert_eq!(b1, b2);
        assert_relative_eq!(s1, -s2);
    }

    #[test]
    fn test_pga_products() {
        let pga = Algebra::pga(3);

        // Euclidean basis vectors square to +1
        assert_eq!(pga.product(1, 1), (0, 1.0));
        assert_eq!(pga.product(2, 2), (0, 1.0));
        assert_eq!(pga.product(4, 4), (0, 1.0));

        // Degenerate vector (e0 at index 8) squares to 0
        assert_eq!(pga.product(8, 8), (0, 0.0));
    }

    #[test]
    fn test_sta_products() {
        let sta = Algebra::sta();

        // Timelike (index 1) squares to +1
        assert_eq!(sta.product(1, 1), (0, 1.0));

        // Spacelike (indices 2, 4, 8) square to -1
        assert_eq!(sta.product(2, 2), (0, -1.0));
        assert_eq!(sta.product(4, 4), (0, -1.0));
        assert_eq!(sta.product(8, 8), (0, -1.0));
    }

    #[test]
    fn test_cga_products() {
        let cga = Algebra::cga(3);

        // First 4 basis vectors square to +1
        for i in 0..4 {
            let e_i = 1 << i;
            assert_eq!(cga.product(e_i, e_i), (0, 1.0),
                "e{} should square to +1", i);
        }

        // Last basis vector (e-) squares to -1
        let e_minus = 1 << 4;
        assert_eq!(cga.product(e_minus, e_minus), (0, -1.0));
    }

    // ==========================================================================
    // Blade naming tests
    // ==========================================================================

    #[test]
    fn test_blade_names() {
        let r3 = Algebra::euclidean(3);

        assert_eq!(r3.blade_name(0), "1");
        assert_eq!(r3.blade_name(1), "e1");
        assert_eq!(r3.blade_name(2), "e2");
        assert_eq!(r3.blade_name(3), "e12");
        assert_eq!(r3.blade_name(4), "e3");
        assert_eq!(r3.blade_name(5), "e13");
        assert_eq!(r3.blade_name(6), "e23");
        assert_eq!(r3.blade_name(7), "e123");
    }

    #[test]
    fn test_blade_names_pga() {
        let pga = Algebra::pga(2);

        // 8 blades in PGA2D
        assert_eq!(pga.blade_names.len(), 8);
        assert_eq!(pga.blade_name(0), "1");
        // The degenerate vector is at bit 2 (index 4)
        assert_eq!(pga.blade_name(4), "e3");
    }

    // ==========================================================================
    // Grade mask tests
    // ==========================================================================

    #[test]
    fn test_grade_masks() {
        let r3 = Algebra::euclidean(3);

        // Grade 0: scalar only (index 0)
        assert_eq!(r3.grade_mask(0), 0b00000001);

        // Grade 1: vectors (indices 1, 2, 4)
        assert_eq!(r3.grade_mask(1), 0b00010110);

        // Grade 2: bivectors (indices 3, 5, 6)
        assert_eq!(r3.grade_mask(2), 0b01101000);

        // Grade 3: trivector (index 7)
        assert_eq!(r3.grade_mask(3), 0b10000000);

        // Grade 4: doesn't exist
        assert_eq!(r3.grade_mask(4), 0);
    }

    #[test]
    fn test_blade_has_grade() {
        let r3 = Algebra::euclidean(3);

        // Scalar is grade 0
        assert!(r3.blade_has_grade(0, 0));
        assert!(!r3.blade_has_grade(0, 1));

        // e1 is grade 1
        assert!(r3.blade_has_grade(1, 1));
        assert!(!r3.blade_has_grade(1, 0));
        assert!(!r3.blade_has_grade(1, 2));

        // e12 is grade 2
        assert!(r3.blade_has_grade(3, 2));

        // e123 is grade 3
        assert!(r3.blade_has_grade(7, 3));
    }

    #[test]
    fn test_blades_of_grade() {
        let r3 = Algebra::euclidean(3);

        assert_eq!(r3.blades_of_grade(0), vec![0]);
        assert_eq!(r3.blades_of_grade(1), vec![1, 2, 4]);
        assert_eq!(r3.blades_of_grade(2), vec![3, 5, 6]);
        assert_eq!(r3.blades_of_grade(3), vec![7]);
        assert_eq!(r3.blades_of_grade(4), Vec::<usize>::new());
    }

    // ==========================================================================
    // Pseudoscalar tests
    // ==========================================================================

    #[test]
    fn test_pseudoscalar() {
        let r3 = Algebra::euclidean(3);
        assert_eq!(r3.pseudoscalar_index(), 7);

        let r4 = Algebra::euclidean(4);
        assert_eq!(r4.pseudoscalar_index(), 15);

        let pga = Algebra::pga(3);
        assert_eq!(pga.pseudoscalar_index(), 15);
    }

    #[test]
    fn test_pseudoscalar_squared_euclidean() {
        // In R^n, I^2 = (-1)^(n(n-1)/2)
        // R^2: I^2 = (-1)^1 = -1
        let r2 = Algebra::euclidean(2);
        assert_relative_eq!(r2.pseudoscalar_squared_sign(), -1.0);

        // R^3: I^2 = (-1)^3 = -1
        let r3 = Algebra::euclidean(3);
        assert_relative_eq!(r3.pseudoscalar_squared_sign(), -1.0);

        // R^4: I^2 = (-1)^6 = 1
        let r4 = Algebra::euclidean(4);
        assert_relative_eq!(r4.pseudoscalar_squared_sign(), 1.0);
    }

    // ==========================================================================
    // Display and equality tests
    // ==========================================================================

    #[test]
    fn test_display() {
        let r3 = Algebra::euclidean(3);
        assert_eq!(format!("{}", r3), "Algebra(Cl(3,0,0))");

        let pga = Algebra::pga(3);
        assert_eq!(format!("{}", pga), "Algebra(Cl(3,0,1))");
    }

    #[test]
    fn test_equality() {
        let a1 = Algebra::euclidean(3);
        let a2 = Algebra::euclidean(3);
        let a3 = Algebra::euclidean(4);

        assert_eq!(*a1, *a2);
        assert_ne!(*a1, *a3);
    }

    // ==========================================================================
    // Comprehensive metric verification
    // ==========================================================================

    #[test]
    fn test_all_signatures_basis_squares() {
        let test_cases = vec![
            (Signature::euclidean(4), vec![1.0, 1.0, 1.0, 1.0]),
            (Signature::pga(3), vec![1.0, 1.0, 1.0, 0.0]),
            (Signature::sta(), vec![1.0, -1.0, -1.0, -1.0]),
            (Signature::cga(2), vec![1.0, 1.0, 1.0, -1.0]),
            (Signature::new(2, 1, 1), vec![1.0, 1.0, -1.0, 0.0]),
        ];

        for (sig, expected_squares) in test_cases {
            let algebra = Algebra::new(sig);

            for (i, &expected) in expected_squares.iter().enumerate() {
                let e_i = 1 << i;
                let (blade, sign) = algebra.product(e_i, e_i);

                assert_eq!(blade, 0,
                    "e{} * e{} should produce scalar in {:?}", i, i, sig);
                assert_relative_eq!(sign, expected,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_anticommutativity_all_signatures() {
        let signatures = vec![
            Signature::euclidean(3),
            Signature::pga(2),
            Signature::sta(),
            Signature::cga(2),
        ];

        for sig in signatures {
            let algebra = Algebra::new(sig);

            for i in 0..sig.dimension() {
                for j in (i + 1)..sig.dimension() {
                    let e_i = 1 << i;
                    let e_j = 1 << j;

                    let (blade_ij, sign_ij) = algebra.product(e_i, e_j);
                    let (blade_ji, sign_ji) = algebra.product(e_j, e_i);

                    assert_eq!(blade_ij, blade_ji,
                        "e{} * e{} and e{} * e{} should produce same blade in {:?}",
                        i, j, j, i, sig);
                    assert_relative_eq!(sign_ij, -sign_ji,
                        epsilon = 1e-10,
                        max_relative = 1e-10
                    );
                }
            }
        }
    }
}
