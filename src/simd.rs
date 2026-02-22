//! SIMD-accelerated operations for geometric algebra.
//!
//! Uses the `wide` crate for portable SIMD that works on stable Rust.
//! The `wide` crate provides SIMD types that compile to efficient vector
//! instructions on x86_64 (SSE/AVX) and ARM (NEON).
//!
//! # Performance Considerations
//!
//! SIMD acceleration is most beneficial for larger algebras:
//! - R3 (8 blades): SIMD overhead often exceeds benefit
//! - R4/PGA3D (16 blades): Marginal benefit
//! - CGA3D (32 blades): Good speedup
//! - Larger algebras: Significant speedup
//!
//! The threshold is tunable via `SIMD_THRESHOLD`.

use wide::f64x4;

/// Minimum number of blades for SIMD to be beneficial.
/// Below this, scalar code is typically faster due to SIMD setup overhead.
pub const SIMD_THRESHOLD: usize = 16;

/// Check if SIMD is beneficial for this algebra size.
///
/// SIMD overhead (loading vectors, horizontal operations) makes it slower
/// for small algebras. The crossover point is typically around 16 blades.
#[inline]
pub fn should_use_simd(num_blades: usize) -> bool {
    num_blades >= SIMD_THRESHOLD
}

/// SIMD-accelerated geometric product accumulation.
///
/// For each blade `i` in the left multivector with non-zero coefficient `a[i]`,
/// this function processes all products `a[i] * b[j]` in SIMD chunks of 4,
/// accumulating into the result array.
///
/// # Arguments
///
/// * `result` - Mutable slice to accumulate results into (must be pre-zeroed)
/// * `a_coeffs` - Coefficients of the left multivector
/// * `b_coeffs` - Coefficients of the right multivector
/// * `cayley_blades` - Flattened Cayley table blade indices [i * num_blades + j]
/// * `cayley_signs` - Flattened Cayley table signs [i * num_blades + j]
/// * `num_blades` - Number of basis blades (must be 2^n)
///
/// # Safety
///
/// All slices must have appropriate lengths:
/// - `result.len() == num_blades`
/// - `a_coeffs.len() == num_blades`
/// - `b_coeffs.len() == num_blades`
/// - `cayley_blades.len() == num_blades * num_blades`
/// - `cayley_signs.len() == num_blades * num_blades`
pub fn simd_geometric_product(
    result: &mut [f64],
    a_coeffs: &[f64],
    b_coeffs: &[f64],
    cayley_blades: &[usize],
    cayley_signs: &[f64],
    num_blades: usize,
) {
    debug_assert_eq!(result.len(), num_blades);
    debug_assert_eq!(a_coeffs.len(), num_blades);
    debug_assert_eq!(b_coeffs.len(), num_blades);
    debug_assert_eq!(cayley_blades.len(), num_blades * num_blades);
    debug_assert_eq!(cayley_signs.len(), num_blades * num_blades);

    // Process each non-zero coefficient in 'a'
    for i in 0..num_blades {
        let a_i = a_coeffs[i];
        if a_i == 0.0 {
            continue;
        }

        let row_offset = i * num_blades;
        let a_vec = f64x4::splat(a_i);

        // Process 'b' coefficients in chunks of 4 (f64x4)
        let chunks = num_blades / 4;
        let remainder = num_blades % 4;

        for chunk in 0..chunks {
            let j_base = chunk * 4;
            let idx_base = row_offset + j_base;

            // Load 4 b coefficients
            let b_vec = f64x4::new([
                b_coeffs[j_base],
                b_coeffs[j_base + 1],
                b_coeffs[j_base + 2],
                b_coeffs[j_base + 3],
            ]);

            // Load 4 signs from Cayley table
            let signs_vec = f64x4::new([
                cayley_signs[idx_base],
                cayley_signs[idx_base + 1],
                cayley_signs[idx_base + 2],
                cayley_signs[idx_base + 3],
            ]);

            // Compute products: a * b * sign
            let products = a_vec * b_vec * signs_vec;

            // Extract and accumulate - we need to scatter to different result indices
            let products_arr = products.to_array();

            // Get blade indices for this chunk
            let blade0 = cayley_blades[idx_base];
            let blade1 = cayley_blades[idx_base + 1];
            let blade2 = cayley_blades[idx_base + 2];
            let blade3 = cayley_blades[idx_base + 3];

            // Accumulate results (scatter operation)
            result[blade0] += products_arr[0];
            result[blade1] += products_arr[1];
            result[blade2] += products_arr[2];
            result[blade3] += products_arr[3];
        }

        // Handle remaining elements (when num_blades is not divisible by 4)
        let j_start = chunks * 4;
        for j in j_start..(j_start + remainder) {
            let b_j = b_coeffs[j];
            if b_j == 0.0 {
                continue;
            }
            let idx = row_offset + j;
            let blade = cayley_blades[idx];
            let sign = cayley_signs[idx];
            result[blade] += sign * a_i * b_j;
        }
    }
}

/// SIMD-accelerated geometric product with sparsity check.
///
/// Same as `simd_geometric_product` but first checks sparsity and falls back
/// to scalar if the multivectors are mostly zero.
///
/// Returns true if SIMD path was used, false if scalar fallback was used.
pub fn simd_geometric_product_adaptive(
    result: &mut [f64],
    a_coeffs: &[f64],
    b_coeffs: &[f64],
    cayley_blades: &[usize],
    cayley_signs: &[f64],
    num_blades: usize,
) -> bool {
    // Count non-zeros in both operands
    let a_nonzeros = a_coeffs.iter().filter(|&&x| x != 0.0).count();
    let b_nonzeros = b_coeffs.iter().filter(|&&x| x != 0.0).count();

    // Estimate work: scalar does a_nz * b_nz, SIMD does a_nz * num_blades
    // SIMD wins when b is dense enough
    let scalar_work = a_nonzeros * b_nonzeros;
    let simd_work = a_nonzeros * (num_blades / 4 + 1);

    // Use SIMD if it does less work (roughly), accounting for SIMD being ~2-4x faster per op
    if simd_work * 2 < scalar_work || b_nonzeros > num_blades / 2 {
        simd_geometric_product(result, a_coeffs, b_coeffs, cayley_blades, cayley_signs, num_blades);
        true
    } else {
        // Scalar fallback for sparse case
        for (i, &a) in a_coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let row_offset = i * num_blades;
            for (j, &b) in b_coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let idx = row_offset + j;
                let blade = cayley_blades[idx];
                let sign = cayley_signs[idx];
                result[blade] += sign * a * b;
            }
        }
        false
    }
}

/// SIMD-accelerated dot product for dense coefficient arrays.
///
/// Computes sum(a[i] * b[i]) using SIMD.
pub fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = f64x4::ZERO;

    for chunk in 0..chunks {
        let i = chunk * 4;
        let a_vec = f64x4::new([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let b_vec = f64x4::new([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        sum_vec += a_vec * b_vec;
    }

    // Horizontal sum of SIMD vector
    let arr = sum_vec.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    // Handle remainder
    let start = chunks * 4;
    for i in start..(start + remainder) {
        sum += a[i] * b[i];
    }

    sum
}

/// SIMD-accelerated coefficient scaling (in-place).
///
/// Multiplies all coefficients by a scalar.
pub fn simd_scale(coeffs: &mut [f64], scalar: f64) {
    let n = coeffs.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let scalar_vec = f64x4::splat(scalar);

    for chunk in 0..chunks {
        let i = chunk * 4;
        let vec = f64x4::new([coeffs[i], coeffs[i + 1], coeffs[i + 2], coeffs[i + 3]]);
        let result = vec * scalar_vec;
        let arr = result.to_array();
        coeffs[i] = arr[0];
        coeffs[i + 1] = arr[1];
        coeffs[i + 2] = arr[2];
        coeffs[i + 3] = arr[3];
    }

    // Handle remainder
    let start = chunks * 4;
    for i in start..(start + remainder) {
        coeffs[i] *= scalar;
    }
}

/// SIMD-accelerated coefficient addition (in-place).
///
/// Adds b to a: a[i] += b[i]
pub fn simd_add(a: &mut [f64], b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for chunk in 0..chunks {
        let i = chunk * 4;
        let a_vec = f64x4::new([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let b_vec = f64x4::new([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let result = a_vec + b_vec;
        let arr = result.to_array();
        a[i] = arr[0];
        a[i + 1] = arr[1];
        a[i + 2] = arr[2];
        a[i + 3] = arr[3];
    }

    // Handle remainder
    let start = chunks * 4;
    for i in start..(start + remainder) {
        a[i] += b[i];
    }
}

/// SIMD-accelerated coefficient subtraction (in-place).
///
/// Subtracts b from a: a[i] -= b[i]
pub fn simd_sub(a: &mut [f64], b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for chunk in 0..chunks {
        let i = chunk * 4;
        let a_vec = f64x4::new([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let b_vec = f64x4::new([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let result = a_vec - b_vec;
        let arr = result.to_array();
        a[i] = arr[0];
        a[i + 1] = arr[1];
        a[i + 2] = arr[2];
        a[i + 3] = arr[3];
    }

    // Handle remainder
    let start = chunks * 4;
    for i in start..(start + remainder) {
        a[i] -= b[i];
    }
}

/// SIMD-accelerated sum of squares.
///
/// Computes sum(coeffs[i]^2).
pub fn simd_sum_of_squares(coeffs: &[f64]) -> f64 {
    let n = coeffs.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = f64x4::ZERO;

    for chunk in 0..chunks {
        let i = chunk * 4;
        let vec = f64x4::new([coeffs[i], coeffs[i + 1], coeffs[i + 2], coeffs[i + 3]]);
        sum_vec += vec * vec;
    }

    // Horizontal sum
    let arr = sum_vec.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    // Handle remainder
    let start = chunks * 4;
    for i in start..(start + remainder) {
        sum += coeffs[i] * coeffs[i];
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_should_use_simd() {
        assert!(!should_use_simd(8));   // R3
        assert!(should_use_simd(16));   // R4/PGA3D
        assert!(should_use_simd(32));   // CGA3D
        assert!(should_use_simd(64));   // R6
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0 + 5.0 * 6.0;
        assert_relative_eq!(simd_dot_product(&a, &b), expected);
    }

    #[test]
    fn test_simd_dot_product_exact_chunks() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_relative_eq!(simd_dot_product(&a, &b), 36.0);
    }

    #[test]
    fn test_simd_scale() {
        let mut coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        simd_scale(&mut coeffs, 2.0);
        assert_eq!(coeffs, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_simd_add() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        simd_add(&mut a, &b);
        assert_eq!(a, vec![6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_simd_sub() {
        let mut a = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        simd_sub(&mut a, &b);
        assert_eq!(a, vec![4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_simd_sum_of_squares() {
        let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 1.0 + 4.0 + 9.0 + 16.0 + 25.0;
        assert_relative_eq!(simd_sum_of_squares(&coeffs), expected);
    }

    #[test]
    fn test_simd_geometric_product_simple() {
        // Test with a simple 4-blade algebra (R2)
        // Cayley table for R2:
        // Index: 0=1, 1=e1, 2=e2, 3=e12
        let num_blades = 4;

        // Simplified Cayley table for R2 Euclidean
        // product(a, b) -> (blade, sign)
        let cayley_blades = vec![
            0, 1, 2, 3,  // 1 * {1, e1, e2, e12}
            1, 0, 3, 2,  // e1 * {1, e1, e2, e12}
            2, 3, 0, 1,  // e2 * {1, e1, e2, e12}
            3, 2, 1, 0,  // e12 * {1, e1, e2, e12}
        ];
        let cayley_signs = vec![
            1.0, 1.0, 1.0, 1.0,   // 1 * anything
            1.0, 1.0, 1.0, 1.0,   // e1 * {1, e1, e2, e12}
            1.0, -1.0, 1.0, -1.0, // e2 * {1, e1, e2, e12}
            1.0, -1.0, 1.0, -1.0, // e12 * {1, e1, e2, e12}
        ];

        // Test: e1 * e2 = e12
        let a = vec![0.0, 1.0, 0.0, 0.0];  // e1
        let b = vec![0.0, 0.0, 1.0, 0.0];  // e2
        let mut result = vec![0.0; 4];

        simd_geometric_product(&mut result, &a, &b, &cayley_blades, &cayley_signs, num_blades);

        assert_relative_eq!(result[0], 0.0);  // scalar
        assert_relative_eq!(result[1], 0.0);  // e1
        assert_relative_eq!(result[2], 0.0);  // e2
        assert_relative_eq!(result[3], 1.0);  // e12 = e1 * e2
    }

    #[test]
    fn test_simd_geometric_product_scalar_mult() {
        // Test scalar multiplication: 2 * e1 = 2*e1
        let num_blades = 4;

        let cayley_blades = vec![
            0, 1, 2, 3,
            1, 0, 3, 2,
            2, 3, 0, 1,
            3, 2, 1, 0,
        ];
        let cayley_signs = vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, -1.0, 1.0, -1.0,
            1.0, -1.0, 1.0, -1.0,
        ];

        let a = vec![2.0, 0.0, 0.0, 0.0];  // 2 (scalar)
        let b = vec![0.0, 3.0, 0.0, 0.0];  // 3*e1
        let mut result = vec![0.0; 4];

        simd_geometric_product(&mut result, &a, &b, &cayley_blades, &cayley_signs, num_blades);

        assert_relative_eq!(result[0], 0.0);  // scalar
        assert_relative_eq!(result[1], 6.0);  // 2 * 3*e1 = 6*e1
        assert_relative_eq!(result[2], 0.0);  // e2
        assert_relative_eq!(result[3], 0.0);  // e12
    }
}
