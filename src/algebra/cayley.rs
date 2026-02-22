//! Cayley table generation for Clifford algebras.
//!
//! The Cayley table defines the geometric product of all pairs of basis blades.
//! For a Cl(p,q,r) algebra with n = p+q+r dimensions, there are 2^n basis blades,
//! and the Cayley table has 2^(2n) entries.
//!
//! Each entry contains:
//! - The result blade index (which basis blade the product produces)
//! - The sign factor (+1, -1, or 0)

use super::Signature;

/// Represents a pre-computed Cayley table for an algebra.
///
/// The table is stored as two parallel flattened arrays for cache efficiency:
/// - `blades[a * num_blades + b]` = result blade index for `e_a * e_b`
/// - `signs[a * num_blades + b]` = sign factor for `e_a * e_b`
#[derive(Clone, Debug)]
pub struct CayleyTable {
    /// Result blade indices, flattened [a * num_blades + b]
    pub blades: Vec<usize>,
    /// Sign factors, flattened [a * num_blades + b]
    pub signs: Vec<f64>,
    /// Number of basis blades (2^dimension)
    pub num_blades: usize,
}

impl CayleyTable {
    /// Look up the product of two basis blades.
    ///
    /// Returns (result_blade, sign) where `e_a * e_b = sign * e_result`.
    #[inline]
    pub fn product(&self, a: usize, b: usize) -> (usize, f64) {
        let idx = a * self.num_blades + b;
        (self.blades[idx], self.signs[idx])
    }
}

/// Computes the geometric product of two basis blades under a given metric signature.
///
/// Returns (result_blade, sign) where sign accounts for:
/// 1. Reordering transpositions (anticommutativity of orthogonal vectors)
/// 2. Metric contractions (when the same basis vector appears in both blades)
///
/// # Arguments
/// * `a` - Binary index of first blade
/// * `b` - Binary index of second blade
/// * `sig` - Metric signature of the algebra
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::{Signature, cayley::blade_product_signed};
///
/// let euclidean = Signature::euclidean(3);
/// // e1 * e1 = +1 (scalar)
/// assert_eq!(blade_product_signed(1, 1, &euclidean), (0, 1.0));
///
/// let pga = Signature::pga(3);  // Cl(3,0,1)
/// // e0 * e0 = 0 (degenerate basis vector)
/// let e0 = 1 << 3;  // Index 8 for 4th basis vector
/// assert_eq!(blade_product_signed(e0, e0, &pga), (0, 0.0));
/// ```
pub fn blade_product_signed(a: usize, b: usize, sig: &Signature) -> (usize, f64) {
    // Start with the reordering sign
    let mut sign = compute_reorder_sign(a, b);

    // The result blade is XOR of the inputs (symmetric difference)
    let result_blade = a ^ b;

    // Apply metric: for each basis vector appearing in both a and b,
    // multiply by that basis vector's square (the contraction)
    let common = a & b;
    for i in 0..sig.dimension() {
        if (common >> i) & 1 == 1 {
            sign *= sig.basis_square(i);
        }
    }

    (result_blade, sign)
}

/// Compute the sign from reordering basis vectors to canonical form.
///
/// When computing e_a * e_b, we need to reorder the basis vectors from the
/// combined sequence to canonical order. Each transposition of adjacent
/// vectors contributes a factor of -1.
///
/// This counts, for each bit set in b, how many bits in a are greater than it.
fn compute_reorder_sign(a: usize, b: usize) -> f64 {
    let mut sign = 1.0f64;
    let mut b_remaining = b;

    while b_remaining != 0 {
        // Get the lowest set bit in b
        let lowest_b_bit = b_remaining & b_remaining.wrapping_neg();
        let b_position = lowest_b_bit.trailing_zeros() as usize;

        // Count bits in a that are strictly greater than b_position.
        // Each such bit requires one transposition past the current b bit.
        let higher_bits_in_a = a >> (b_position + 1);
        let transpositions = higher_bits_in_a.count_ones();

        if transpositions % 2 == 1 {
            sign = -sign;
        }

        b_remaining &= !lowest_b_bit;
    }

    sign
}

/// Compute the full Cayley table for an algebra.
///
/// This pre-computes all 2^(2n) products for efficient lookup during
/// multivector operations.
///
/// # Arguments
/// * `sig` - Metric signature of the algebra
///
/// # Returns
/// A CayleyTable containing all pre-computed products.
pub fn compute_cayley_table(sig: &Signature) -> CayleyTable {
    let num_blades = sig.num_blades();
    let table_size = num_blades * num_blades;

    let mut blades = vec![0usize; table_size];
    let mut signs = vec![0.0f64; table_size];

    for a in 0..num_blades {
        for b in 0..num_blades {
            let (blade, sign) = blade_product_signed(a, b, sig);
            let idx = a * num_blades + b;
            blades[idx] = blade;
            signs[idx] = sign;
        }
    }

    CayleyTable {
        blades,
        signs,
        num_blades,
    }
}

/// Compute only the sign portion of the Cayley table (for when we don't need blade indices).
///
/// Since blade indices are always a XOR b, we sometimes only need the signs.
pub fn compute_sign_table(sig: &Signature) -> Vec<f64> {
    let num_blades = sig.num_blades();
    let table_size = num_blades * num_blades;

    let mut signs = vec![0.0f64; table_size];

    for a in 0..num_blades {
        for b in 0..num_blades {
            let (_, sign) = blade_product_signed(a, b, sig);
            signs[a * num_blades + b] = sign;
        }
    }

    signs
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==========================================================================
    // Basic blade product tests
    // ==========================================================================

    #[test]
    fn test_scalar_products() {
        let sig = Signature::euclidean(3);
        // Scalar * anything = that thing
        assert_eq!(blade_product_signed(0, 0, &sig), (0, 1.0));
        assert_eq!(blade_product_signed(0, 1, &sig), (1, 1.0));
        assert_eq!(blade_product_signed(0, 7, &sig), (7, 1.0));
        assert_eq!(blade_product_signed(1, 0, &sig), (1, 1.0));
    }

    #[test]
    fn test_euclidean_basis_squares() {
        let sig = Signature::euclidean(3);
        // e_i * e_i = +1 for Euclidean
        assert_eq!(blade_product_signed(1, 1, &sig), (0, 1.0));
        assert_eq!(blade_product_signed(2, 2, &sig), (0, 1.0));
        assert_eq!(blade_product_signed(4, 4, &sig), (0, 1.0));
    }

    #[test]
    fn test_anticommutativity() {
        let sig = Signature::euclidean(3);
        // e_i * e_j = -e_j * e_i for i != j
        let (blade1, sign1) = blade_product_signed(1, 2, &sig);
        let (blade2, sign2) = blade_product_signed(2, 1, &sig);
        assert_eq!(blade1, blade2);
        assert_relative_eq!(sign1, -sign2);

        let (blade1, sign1) = blade_product_signed(1, 4, &sig);
        let (blade2, sign2) = blade_product_signed(4, 1, &sig);
        assert_eq!(blade1, blade2);
        assert_relative_eq!(sign1, -sign2);

        let (blade1, sign1) = blade_product_signed(2, 4, &sig);
        let (blade2, sign2) = blade_product_signed(4, 2, &sig);
        assert_eq!(blade1, blade2);
        assert_relative_eq!(sign1, -sign2);
    }

    #[test]
    fn test_orthogonal_product() {
        let sig = Signature::euclidean(3);
        // e1 * e2 = e12
        assert_eq!(blade_product_signed(1, 2, &sig), (3, 1.0));
        // e2 * e1 = -e12
        assert_eq!(blade_product_signed(2, 1, &sig), (3, -1.0));
    }

    // ==========================================================================
    // PGA signature tests
    // ==========================================================================

    #[test]
    fn test_pga_degenerate_squares() {
        let pga = Signature::pga(3);  // Cl(3,0,1)
        // e0 (the degenerate vector) squares to 0
        let e0 = 1 << 3;  // Index 8 for 4th basis vector
        let (blade, sign) = blade_product_signed(e0, e0, &pga);
        assert_eq!(blade, 0);
        assert_relative_eq!(sign, 0.0);

        // Euclidean vectors still square to +1
        assert_eq!(blade_product_signed(1, 1, &pga), (0, 1.0));
        assert_eq!(blade_product_signed(2, 2, &pga), (0, 1.0));
        assert_eq!(blade_product_signed(4, 4, &pga), (0, 1.0));
    }

    #[test]
    fn test_pga_mixed_products() {
        let pga = Signature::pga(3);  // Cl(3,0,1)
        let e0 = 1 << 3;  // Degenerate vector

        // e1 * e0 should give e10 (mixed blade)
        let (blade, sign) = blade_product_signed(1, e0, &pga);
        assert_eq!(blade, 1 | e0);  // e10
        // Sign depends on reordering only (no contraction)
        assert_relative_eq!(sign, 1.0);

        // e0 * e1 should give -e10 (anticommutes)
        let (blade2, sign2) = blade_product_signed(e0, 1, &pga);
        assert_eq!(blade2, blade);
        assert_relative_eq!(sign2, -sign);
    }

    // ==========================================================================
    // STA (Spacetime Algebra) tests
    // ==========================================================================

    #[test]
    fn test_sta_basis_squares() {
        let sta = Signature::sta();  // Cl(1,3,0)

        // e0 (timelike) squares to +1
        assert_eq!(blade_product_signed(1, 1, &sta), (0, 1.0));

        // e1, e2, e3 (spacelike) square to -1
        assert_eq!(blade_product_signed(2, 2, &sta), (0, -1.0));
        assert_eq!(blade_product_signed(4, 4, &sta), (0, -1.0));
        assert_eq!(blade_product_signed(8, 8, &sta), (0, -1.0));
    }

    #[test]
    fn test_sta_anticommutativity() {
        let sta = Signature::sta();

        // Mixed timelike-spacelike products
        // e0 * e1 = e01
        let (blade1, sign1) = blade_product_signed(1, 2, &sta);
        // e1 * e0 = -e01
        let (blade2, sign2) = blade_product_signed(2, 1, &sta);
        assert_eq!(blade1, blade2);
        assert_relative_eq!(sign1, -sign2);
    }

    // ==========================================================================
    // CGA (Conformal GA) tests
    // ==========================================================================

    #[test]
    fn test_cga_basis_squares() {
        let cga = Signature::cga(3);  // Cl(4,1,0)

        // First 4 basis vectors square to +1
        assert_eq!(blade_product_signed(1, 1, &cga), (0, 1.0));
        assert_eq!(blade_product_signed(2, 2, &cga), (0, 1.0));
        assert_eq!(blade_product_signed(4, 4, &cga), (0, 1.0));
        assert_eq!(blade_product_signed(8, 8, &cga), (0, 1.0));

        // Last basis vector (e-) squares to -1
        let e_minus = 1 << 4;  // Index 16
        assert_eq!(blade_product_signed(e_minus, e_minus, &cga), (0, -1.0));
    }

    // ==========================================================================
    // Cayley table tests
    // ==========================================================================

    #[test]
    fn test_cayley_table_euclidean() {
        let sig = Signature::euclidean(2);  // Small for fast test
        let table = compute_cayley_table(&sig);

        assert_eq!(table.num_blades, 4);
        assert_eq!(table.blades.len(), 16);
        assert_eq!(table.signs.len(), 16);

        // Verify a few products
        // e1 * e1 = 1
        assert_eq!(table.product(1, 1), (0, 1.0));
        // e1 * e2 = e12
        assert_eq!(table.product(1, 2), (3, 1.0));
        // e2 * e1 = -e12
        assert_eq!(table.product(2, 1), (3, -1.0));
        // e12 * e12 = -1 (bivector squares to -1 in Euclidean)
        assert_eq!(table.product(3, 3), (0, -1.0));
    }

    #[test]
    fn test_cayley_table_pga() {
        let sig = Signature::pga(2);  // Cl(2,0,1), 3D algebra
        let table = compute_cayley_table(&sig);

        assert_eq!(table.num_blades, 8);

        // e0 (degenerate) is at index 4 (bit 2)
        let e0 = 4;
        // e0 * e0 = 0
        let (blade, sign) = table.product(e0, e0);
        assert_eq!(blade, 0);
        assert_relative_eq!(sign, 0.0);

        // e1 * e1 = +1
        assert_eq!(table.product(1, 1), (0, 1.0));
        // e2 * e2 = +1
        assert_eq!(table.product(2, 2), (0, 1.0));
    }

    #[test]
    fn test_cayley_table_sta() {
        let sig = Signature::sta();  // Cl(1,3,0)
        let table = compute_cayley_table(&sig);

        assert_eq!(table.num_blades, 16);

        // e0 (timelike, index 1) squares to +1
        assert_eq!(table.product(1, 1), (0, 1.0));

        // e1, e2, e3 (spacelike, indices 2, 4, 8) square to -1
        assert_eq!(table.product(2, 2), (0, -1.0));
        assert_eq!(table.product(4, 4), (0, -1.0));
        assert_eq!(table.product(8, 8), (0, -1.0));
    }

    // ==========================================================================
    // Associativity tests
    // ==========================================================================

    #[test]
    fn test_associativity_euclidean() {
        let sig = Signature::euclidean(3);
        let table = compute_cayley_table(&sig);

        // Test (e1 * e2) * e3 == e1 * (e2 * e3)
        let (ab, sign_ab) = table.product(1, 2);
        let (abc_left, sign_abc_left) = table.product(ab, 4);
        let final_sign_left = sign_ab * sign_abc_left;

        let (bc, sign_bc) = table.product(2, 4);
        let (abc_right, sign_abc_right) = table.product(1, bc);
        let final_sign_right = sign_bc * sign_abc_right;

        assert_eq!(abc_left, abc_right);
        assert_relative_eq!(final_sign_left, final_sign_right);
    }

    #[test]
    fn test_associativity_sta() {
        let sig = Signature::sta();
        let table = compute_cayley_table(&sig);

        // Test (e0 * e1) * e2 == e0 * (e1 * e2)
        let (ab, sign_ab) = table.product(1, 2);
        let (abc_left, sign_abc_left) = table.product(ab, 4);
        let final_sign_left = sign_ab * sign_abc_left;

        let (bc, sign_bc) = table.product(2, 4);
        let (abc_right, sign_abc_right) = table.product(1, bc);
        let final_sign_right = sign_bc * sign_abc_right;

        assert_eq!(abc_left, abc_right);
        assert_relative_eq!(final_sign_left, final_sign_right);
    }

    // ==========================================================================
    // Comprehensive signature verification
    // ==========================================================================

    #[test]
    fn test_all_basis_squares() {
        // Test several signatures
        let signatures = vec![
            Signature::euclidean(4),
            Signature::pga(3),
            Signature::cga(2),
            Signature::sta(),
            Signature::new(2, 2, 1),  // Custom signature
        ];

        for sig in signatures {
            let table = compute_cayley_table(&sig);

            // Verify each basis vector squares correctly
            for i in 0..sig.dimension() {
                let e_i = 1 << i;
                let (blade, sign) = table.product(e_i, e_i);

                // Result should be scalar
                assert_eq!(blade, 0, "e{} * e{} should produce scalar in {}", i, i, sig);

                // Sign should match signature
                let expected = sig.basis_square(i);
                assert_relative_eq!(sign, expected,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_all_anticommutativity() {
        let signatures = vec![
            Signature::euclidean(3),
            Signature::pga(2),
            Signature::sta(),
        ];

        for sig in signatures {
            let table = compute_cayley_table(&sig);

            for i in 0..sig.dimension() {
                for j in (i + 1)..sig.dimension() {
                    let e_i = 1 << i;
                    let e_j = 1 << j;

                    let (blade_ij, sign_ij) = table.product(e_i, e_j);
                    let (blade_ji, sign_ji) = table.product(e_j, e_i);

                    // Same result blade
                    assert_eq!(blade_ij, blade_ji,
                        "e{} * e{} and e{} * e{} should produce same blade in {}",
                        i, j, j, i, sig);

                    // Opposite signs (anticommute)
                    assert_relative_eq!(sign_ij, -sign_ji,
                        epsilon = 1e-10,
                        max_relative = 1e-10
                    );
                }
            }
        }
    }
}
