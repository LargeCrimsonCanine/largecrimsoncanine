/// Returns the grade of a basis blade given its binary index.
///
/// The grade equals the number of set bits (popcount) in the index.
/// Example: index 5 = 0b101 = e1^e3, grade 2.
///
/// Reference: Dorst et al. ch.2 [VERIFY]
pub fn blade_grade(index: usize) -> usize {
    index.count_ones() as usize
}

/// Computes the geometric product of two basis blades.
///
/// Returns (result_blade, sign) where sign is +1 or -1.
///
/// The result blade is the symmetric difference of the two input blades (XOR).
/// The sign is determined by counting the number of adjacent transpositions
/// needed to sort the combined sequence of basis vectors, following the
/// canonical ordering e1 < e2 < e3 < ...
///
/// Assumes Euclidean metric: all basis vectors square to +1.
///
/// Reference: Dorst et al. ch.3 [VERIFY]
pub fn blade_product(a: usize, b: usize) -> (usize, f64) {
    let result_blade = a ^ b;
    let sign = compute_sign(a, b);
    (result_blade, sign)
}

/// Returns the sign factor for the reverse operation on a blade of given grade.
///
/// The reverse of a grade-k blade picks up a sign of (-1)^(k(k-1)/2).
/// This equals +1 for grades 0, 1, 4, 5, 8, 9, ... (k mod 4 in {0, 1})
/// and -1 for grades 2, 3, 6, 7, 10, 11, ... (k mod 4 in {2, 3})
///
/// Reference: Dorst et al. ch.2 [VERIFY]
pub fn reverse_sign(grade: usize) -> f64 {
    // (-1)^(k(k-1)/2) simplifies to checking k mod 4
    if grade % 4 < 2 { 1.0 } else { -1.0 }
}

/// Counts the number of transpositions needed to merge two sorted blade sequences.
///
/// For each bit set in b, counts the number of bits in a that are greater than it.
/// Each such pair requires a transposition to restore canonical ordering,
/// contributing a factor of -1 to the sign.
fn compute_sign(a: usize, b: usize) -> f64 {
    let mut sign = 1.0f64;
    let mut b_remaining = b;

    while b_remaining != 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_blade_grade() {
        assert_eq!(blade_grade(0), 0); // scalar
        assert_eq!(blade_grade(1), 1); // e1
        assert_eq!(blade_grade(2), 1); // e2
        assert_eq!(blade_grade(3), 2); // e12
        assert_eq!(blade_grade(7), 3); // e123
    }

    #[test]
    fn test_blade_product_orthogonal() {
        // e1 * e2 = e12, sign +1
        let (blade, sign) = blade_product(1, 2);
        assert_eq!(blade, 3);
        assert_relative_eq!(sign, 1.0);
    }

    #[test]
    fn test_blade_product_anticommutes() {
        // e2 * e1 = -e12, sign -1
        let (blade, sign) = blade_product(2, 1);
        assert_eq!(blade, 3);
        assert_relative_eq!(sign, -1.0);
    }

    #[test]
    fn test_blade_product_squares_to_scalar() {
        // e1 * e1 = scalar (index 0), sign +1 (Euclidean metric)
        let (blade, sign) = blade_product(1, 1);
        assert_eq!(blade, 0);
        assert_relative_eq!(sign, 1.0);
    }

    #[test]
    fn test_reverse_sign() {
        // Formula: (-1)^(k(k-1)/2)
        // Grade 0: (-1)^(0*-1/2) = (-1)^0 = 1
        assert_relative_eq!(reverse_sign(0), 1.0);
        // Grade 1: (-1)^(1*0/2) = (-1)^0 = 1
        assert_relative_eq!(reverse_sign(1), 1.0);
        // Grade 2: (-1)^(2*1/2) = (-1)^1 = -1
        assert_relative_eq!(reverse_sign(2), -1.0);
        // Grade 3: (-1)^(3*2/2) = (-1)^3 = -1
        assert_relative_eq!(reverse_sign(3), -1.0);
        // Grade 4: (-1)^(4*3/2) = (-1)^6 = 1
        assert_relative_eq!(reverse_sign(4), 1.0);
        // Grade 5: (-1)^(5*4/2) = (-1)^10 = 1
        assert_relative_eq!(reverse_sign(5), 1.0);
    }
}

