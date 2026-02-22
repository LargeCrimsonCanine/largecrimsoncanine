//! Blade utilities for geometric algebra.
//!
//! This module provides functions for working with basis blades:
//! - Grade computation (how many basis vectors in a blade)
//! - Blade naming (human-readable names like "e12", "e123")
//! - Sign factors for involutions (reverse, grade involution, conjugate)

use super::Signature;

/// Returns the grade of a basis blade given its binary index.
///
/// The grade equals the number of set bits (popcount) in the index.
/// Grade 0 is scalar, grade 1 is vectors, grade 2 is bivectors, etc.
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::blades::blade_grade;
/// assert_eq!(blade_grade(0), 0);  // scalar
/// assert_eq!(blade_grade(1), 1);  // e1
/// assert_eq!(blade_grade(3), 2);  // e12 = e1^e2
/// assert_eq!(blade_grade(7), 3);  // e123 = e1^e2^e3
/// ```
#[inline]
pub fn blade_grade(index: usize) -> usize {
    index.count_ones() as usize
}

/// Returns a human-readable name for a basis blade.
///
/// Blade names follow the convention:
/// - Index 0: "1" (scalar)
/// - Index 1: "e1", Index 2: "e2", etc.
/// - Index 3: "e12" (e1 wedge e2)
/// - Index 7: "e123" (e1 wedge e2 wedge e3)
///
/// # Arguments
/// * `index` - The blade's binary index
/// * `dimension` - Total dimension of the algebra
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::blades::blade_name;
/// assert_eq!(blade_name(0, 3), "1");
/// assert_eq!(blade_name(1, 3), "e1");
/// assert_eq!(blade_name(3, 3), "e12");
/// assert_eq!(blade_name(7, 3), "e123");
/// ```
pub fn blade_name(index: usize, dimension: usize) -> String {
    if index == 0 {
        return "1".to_string();
    }

    let mut name = String::from("e");
    for i in 0..dimension {
        if (index >> i) & 1 == 1 {
            // Use 1-indexed basis vectors (e1, e2, e3...)
            name.push_str(&(i + 1).to_string());
        }
    }
    name
}

/// Returns a human-readable name for a basis blade, using signature-aware naming.
///
/// For PGA, the degenerate vector is typically called e0.
/// For CGA, special naming conventions may apply.
///
/// # Arguments
/// * `index` - The blade's binary index
/// * `sig` - The algebra's metric signature
pub fn blade_name_with_signature(index: usize, sig: &Signature) -> String {
    if index == 0 {
        return "1".to_string();
    }

    let dimension = sig.dimension();
    let mut name = String::from("e");

    for i in 0..dimension {
        if (index >> i) & 1 == 1 {
            if sig.is_pga() && i == dimension - 1 {
                // In PGA, the last (degenerate) vector is e0
                name.push('0');
            } else if sig.is_pga() {
                // Other PGA vectors are 1-indexed
                name.push_str(&(i + 1).to_string());
            } else {
                // Standard 1-indexed naming
                name.push_str(&(i + 1).to_string());
            }
        }
    }
    name
}

/// Generate all blade names for an algebra.
///
/// Returns a vector where index i contains the name of blade i.
pub fn all_blade_names(sig: &Signature) -> Vec<String> {
    let num_blades = sig.num_blades();
    let dimension = sig.dimension();
    (0..num_blades)
        .map(|i| blade_name(i, dimension))
        .collect()
}

/// Returns the sign factor for the reverse operation on a blade of given grade.
///
/// The reverse of a grade-k blade picks up a sign of (-1)^(k(k-1)/2).
///
/// Pattern by grade mod 4:
/// - Grade 0, 1: +1
/// - Grade 2, 3: -1
/// - Grade 4, 5: +1
/// - Grade 6, 7: -1
/// ...
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::blades::reverse_sign;
/// assert_eq!(reverse_sign(0), 1.0);   // Scalars unchanged
/// assert_eq!(reverse_sign(1), 1.0);   // Vectors unchanged
/// assert_eq!(reverse_sign(2), -1.0);  // Bivectors negated
/// assert_eq!(reverse_sign(3), -1.0);  // Trivectors negated
/// assert_eq!(reverse_sign(4), 1.0);   // 4-vectors unchanged
/// ```
#[inline]
pub fn reverse_sign(grade: usize) -> f64 {
    // (-1)^(k(k-1)/2) simplifies to checking k mod 4
    if grade % 4 < 2 {
        1.0
    } else {
        -1.0
    }
}

/// Returns the sign factor for grade involution on a blade of given grade.
///
/// Grade involution (main involution) is (-1)^k:
/// - Even grades: +1
/// - Odd grades: -1
///
/// This operation negates vectors while leaving scalars unchanged.
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::blades::grade_involution_sign;
/// assert_eq!(grade_involution_sign(0), 1.0);   // Even
/// assert_eq!(grade_involution_sign(1), -1.0);  // Odd
/// assert_eq!(grade_involution_sign(2), 1.0);   // Even
/// assert_eq!(grade_involution_sign(3), -1.0);  // Odd
/// ```
#[inline]
pub fn grade_involution_sign(grade: usize) -> f64 {
    if grade % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Returns the sign factor for Clifford conjugation on a blade of given grade.
///
/// Clifford conjugation combines reverse and grade involution:
/// (-1)^(k(k+1)/2) where k is the grade.
///
/// Pattern by grade mod 4:
/// - Grade 0: +1 (scalars unchanged)
/// - Grade 1: -1 (vectors negated)
/// - Grade 2: -1 (bivectors negated)
/// - Grade 3: +1 (trivectors unchanged)
///
/// # Example
/// ```
/// use largecrimsoncanine::algebra::blades::clifford_conjugate_sign;
/// assert_eq!(clifford_conjugate_sign(0), 1.0);
/// assert_eq!(clifford_conjugate_sign(1), -1.0);
/// assert_eq!(clifford_conjugate_sign(2), -1.0);
/// assert_eq!(clifford_conjugate_sign(3), 1.0);
/// ```
#[inline]
pub fn clifford_conjugate_sign(grade: usize) -> f64 {
    // (-1)^(k(k+1)/2): k mod 4 in {0, 3} gives +1, k mod 4 in {1, 2} gives -1
    match grade % 4 {
        0 | 3 => 1.0,
        _ => -1.0,
    }
}

/// Computes the reverse sign for a specific blade index.
#[inline]
pub fn reverse_sign_for_blade(index: usize) -> f64 {
    reverse_sign(blade_grade(index))
}

/// Computes the grade involution sign for a specific blade index.
#[inline]
pub fn grade_involution_sign_for_blade(index: usize) -> f64 {
    grade_involution_sign(blade_grade(index))
}

/// Computes the Clifford conjugate sign for a specific blade index.
#[inline]
pub fn clifford_conjugate_sign_for_blade(index: usize) -> f64 {
    clifford_conjugate_sign(blade_grade(index))
}

/// Generate a mask for all blades of a specific grade.
///
/// Returns a bitmask where bit i is set if blade i has the given grade.
pub fn grade_mask(dimension: usize, grade: usize) -> usize {
    let num_blades = 1usize << dimension;
    let mut mask = 0usize;
    for i in 0..num_blades {
        if blade_grade(i) == grade {
            mask |= 1 << i;
        }
    }
    mask
}

/// Generate masks for all grades in an algebra.
///
/// Returns a vector where index k contains the mask for grade k blades.
pub fn all_grade_masks(dimension: usize) -> Vec<usize> {
    (0..=dimension)
        .map(|k| grade_mask(dimension, k))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blade_grade() {
        assert_eq!(blade_grade(0), 0);  // scalar
        assert_eq!(blade_grade(1), 1);  // e1
        assert_eq!(blade_grade(2), 1);  // e2
        assert_eq!(blade_grade(3), 2);  // e12
        assert_eq!(blade_grade(4), 1);  // e3
        assert_eq!(blade_grade(5), 2);  // e13
        assert_eq!(blade_grade(6), 2);  // e23
        assert_eq!(blade_grade(7), 3);  // e123
        assert_eq!(blade_grade(15), 4); // e1234
    }

    #[test]
    fn test_blade_name() {
        // 3D algebra
        assert_eq!(blade_name(0, 3), "1");
        assert_eq!(blade_name(1, 3), "e1");
        assert_eq!(blade_name(2, 3), "e2");
        assert_eq!(blade_name(3, 3), "e12");
        assert_eq!(blade_name(4, 3), "e3");
        assert_eq!(blade_name(5, 3), "e13");
        assert_eq!(blade_name(6, 3), "e23");
        assert_eq!(blade_name(7, 3), "e123");
    }

    #[test]
    fn test_blade_name_4d() {
        // 4D algebra (like PGA3D)
        assert_eq!(blade_name(0, 4), "1");
        assert_eq!(blade_name(15, 4), "e1234");
        assert_eq!(blade_name(8, 4), "e4");
    }

    #[test]
    fn test_all_blade_names() {
        let sig = Signature::euclidean(2);
        let names = all_blade_names(&sig);
        assert_eq!(names, vec!["1", "e1", "e2", "e12"]);
    }

    #[test]
    fn test_reverse_sign() {
        // Pattern: +1 for grades 0,1, -1 for grades 2,3, +1 for grades 4,5, etc.
        assert_eq!(reverse_sign(0), 1.0);
        assert_eq!(reverse_sign(1), 1.0);
        assert_eq!(reverse_sign(2), -1.0);
        assert_eq!(reverse_sign(3), -1.0);
        assert_eq!(reverse_sign(4), 1.0);
        assert_eq!(reverse_sign(5), 1.0);
        assert_eq!(reverse_sign(6), -1.0);
        assert_eq!(reverse_sign(7), -1.0);
    }

    #[test]
    fn test_grade_involution_sign() {
        // (-1)^k: +1 for even, -1 for odd
        assert_eq!(grade_involution_sign(0), 1.0);
        assert_eq!(grade_involution_sign(1), -1.0);
        assert_eq!(grade_involution_sign(2), 1.0);
        assert_eq!(grade_involution_sign(3), -1.0);
        assert_eq!(grade_involution_sign(4), 1.0);
        assert_eq!(grade_involution_sign(5), -1.0);
    }

    #[test]
    fn test_clifford_conjugate_sign() {
        // Pattern: +1 for grades 0,3, -1 for grades 1,2
        assert_eq!(clifford_conjugate_sign(0), 1.0);
        assert_eq!(clifford_conjugate_sign(1), -1.0);
        assert_eq!(clifford_conjugate_sign(2), -1.0);
        assert_eq!(clifford_conjugate_sign(3), 1.0);
        assert_eq!(clifford_conjugate_sign(4), 1.0);
        assert_eq!(clifford_conjugate_sign(5), -1.0);
    }

    #[test]
    fn test_grade_mask() {
        // 3D algebra: 8 blades
        // Grade 0: blade 0 (scalar)
        assert_eq!(grade_mask(3, 0), 0b00000001);
        // Grade 1: blades 1, 2, 4 (e1, e2, e3)
        assert_eq!(grade_mask(3, 1), 0b00010110);
        // Grade 2: blades 3, 5, 6 (e12, e13, e23)
        assert_eq!(grade_mask(3, 2), 0b01101000);
        // Grade 3: blade 7 (e123)
        assert_eq!(grade_mask(3, 3), 0b10000000);
    }

    #[test]
    fn test_all_grade_masks() {
        let masks = all_grade_masks(3);
        assert_eq!(masks.len(), 4);  // Grades 0, 1, 2, 3
        assert_eq!(masks[0], 0b00000001);  // Grade 0
        assert_eq!(masks[1], 0b00010110);  // Grade 1
        assert_eq!(masks[2], 0b01101000);  // Grade 2
        assert_eq!(masks[3], 0b10000000);  // Grade 3
    }
}
