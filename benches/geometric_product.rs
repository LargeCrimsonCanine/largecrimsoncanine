//! Comprehensive benchmarks for LargeCrimsonCanine geometric algebra operations.
//!
//! Run with: `cargo bench`
//!
//! This benchmark suite tests performance across multiple algebras:
//! - R3: 3D Euclidean space Cl(3,0,0)
//! - R4: 4D Euclidean space Cl(4,0,0)
//! - PGA3D: 3D Projective Geometric Algebra Cl(3,0,1)
//! - CGA3D: 3D Conformal Geometric Algebra Cl(4,1,0)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use largecrimsoncanine::algebra::Algebra;
use largecrimsoncanine::Multivector;
use largecrimsoncanine::simd::{simd_geometric_product, should_use_simd};

// =============================================================================
// Helper functions for creating test multivectors
// =============================================================================

/// Create a random-ish multivector for benchmarking.
/// Uses deterministic "random" values based on index for reproducibility.
fn create_test_multivector(dims: usize, seed: usize) -> Multivector {
    let num_blades = 1 << dims;
    let coeffs: Vec<f64> = (0..num_blades)
        .map(|i| {
            // Deterministic pseudo-random values
            let x = ((i + seed) * 2654435761) as f64;
            (x % 100.0) / 50.0 - 1.0
        })
        .collect();
    Multivector::new(coeffs).unwrap()
}

/// Create a simple rotor (rotation in e12 plane).
fn create_test_rotor(dims: usize, angle: f64) -> Multivector {
    let half_angle = angle / 2.0;
    let num_blades = 1 << dims;
    let mut coeffs = vec![0.0; num_blades];

    // Rotor = cos(theta/2) + sin(theta/2) * B
    // where B is unit bivector (e12)
    coeffs[0] = half_angle.cos();  // scalar part
    if dims >= 2 {
        coeffs[3] = half_angle.sin();  // e12 part
    }

    Multivector::new(coeffs).unwrap()
}

/// Create a test vector.
fn create_test_vector(dims: usize, seed: usize) -> Multivector {
    let num_blades = 1 << dims;
    let mut coeffs = vec![0.0; num_blades];

    // Set vector components (grade 1) at indices 1, 2, 4, 8, ...
    for i in 0..dims {
        let x = ((i + seed) * 2654435761) as f64;
        coeffs[1 << i] = (x % 100.0) / 50.0 - 1.0;
    }

    Multivector::new(coeffs).unwrap()
}

// =============================================================================
// Geometric Product Benchmarks
// =============================================================================

fn bench_geometric_product_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_product/single");

    // Test different algebra dimensions
    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let a = create_test_multivector(dims, 42);
        let b = create_test_multivector(dims, 137);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &(a, b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(a.geometric_product(b).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_geometric_product_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_product/batched");

    // Batch sizes to test
    for batch_size in [10, 100, 1000] {
        let dims = 3;

        // Pre-create multivectors
        let pairs: Vec<_> = (0..batch_size)
            .map(|i| {
                (
                    create_test_multivector(dims, i * 2),
                    create_test_multivector(dims, i * 2 + 1),
                )
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("R3", batch_size),
            &pairs,
            |bench, pairs| {
                bench.iter(|| {
                    for (a, b) in pairs {
                        black_box(a.geometric_product(b).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Outer Product Benchmarks
// =============================================================================

fn bench_outer_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("outer_product");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let a = create_test_vector(dims, 42);
        let b = create_test_vector(dims, 137);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &(a, b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(a.outer_product(b).unwrap())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Inner Product Benchmarks
// =============================================================================

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let a = create_test_multivector(dims, 42);
        let b = create_test_multivector(dims, 137);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &(a, b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(a.inner(b).unwrap())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Sandwich Product Benchmarks (Rotor Application)
// =============================================================================

fn bench_sandwich_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("sandwich/single");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let rotor = create_test_rotor(dims, std::f64::consts::PI / 4.0);
        let vector = create_test_vector(dims, 42);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &(rotor, vector),
            |bench, (rotor, vector)| {
                bench.iter(|| {
                    black_box(rotor.sandwich(vector).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_sandwich_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("sandwich/batched");

    for batch_size in [10, 100, 1000] {
        let dims = 3;
        let rotor = create_test_rotor(dims, std::f64::consts::PI / 4.0);

        // Pre-create vectors to transform
        let vectors: Vec<_> = (0..batch_size)
            .map(|i| create_test_vector(dims, i))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("R3", batch_size),
            &(rotor, vectors),
            |bench, (rotor, vectors)| {
                bench.iter(|| {
                    for v in vectors {
                        black_box(rotor.sandwich(v).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Exponential and Logarithm Benchmarks
// =============================================================================

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        // Create a scaled bivector (angle * unit_bivector)
        let angle = std::f64::consts::PI / 6.0;
        let num_blades = 1 << dims;
        let mut coeffs = vec![0.0; num_blades];
        if dims >= 2 {
            coeffs[3] = angle;  // e12 component
        }
        let bivector = Multivector::new(coeffs).unwrap();

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &bivector,
            |bench, biv| {
                bench.iter(|| {
                    black_box(biv.exp().unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("log");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let rotor = create_test_rotor(dims, std::f64::consts::PI / 4.0);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &rotor,
            |bench, r| {
                bench.iter(|| {
                    black_box(r.log().unwrap())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Norm Benchmarks
// =============================================================================

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let mv = create_test_multivector(dims, 42);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &mv,
            |bench, mv| {
                bench.iter(|| {
                    black_box(mv.norm())
                });
            },
        );
    }

    group.finish();
}

fn bench_norm_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_squared");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let mv = create_test_multivector(dims, 42);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &mv,
            |bench, mv| {
                bench.iter(|| {
                    black_box(mv.norm_squared())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Inverse Benchmarks
// =============================================================================

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("inverse");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4/PGA3D",
            5 => "CGA3D",
            _ => "Unknown",
        };

        // Use a rotor which is guaranteed to be invertible
        let rotor = create_test_rotor(dims, std::f64::consts::PI / 4.0);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &rotor,
            |bench, r| {
                bench.iter(|| {
                    black_box(r.inverse().unwrap())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Algebra/Cayley Table Benchmarks
// =============================================================================

fn bench_blade_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("blade_product");

    // Test blade product lookup speed for different algebras
    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let algebra = Algebra::euclidean(dims);
        let num_blades = 1 << dims;

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &(algebra, num_blades),
            |bench, (algebra, num_blades)| {
                bench.iter(|| {
                    // Benchmark all blade-blade products
                    for a in 0..*num_blades {
                        for b in 0..*num_blades {
                            black_box(algebra.product(a, b));
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_algebra_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("algebra_creation");

    // Benchmark Cayley table computation for different sizes
    for dims in [3, 4, 5, 6] {
        let name = format!("Cl({},0,0)", dims);

        group.bench_with_input(
            BenchmarkId::new(&name, dims),
            &dims,
            |bench, &dims| {
                bench.iter(|| {
                    black_box(Algebra::euclidean(dims))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Grade Projection Benchmarks
// =============================================================================

fn bench_grade_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("grade_projection");

    let dims = 4;
    let mv = create_test_multivector(dims, 42);

    for grade in 0..=dims {
        group.bench_with_input(
            BenchmarkId::new("R4", grade),
            &(mv.clone(), grade),
            |bench, (mv, grade)| {
                bench.iter(|| {
                    black_box(mv.grade(*grade).unwrap())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Reverse Benchmarks
// =============================================================================

fn bench_reverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse");

    for dims in [3, 4, 5] {
        let name = match dims {
            3 => "R3",
            4 => "R4",
            5 => "CGA3D",
            _ => "Unknown",
        };

        let mv = create_test_multivector(dims, 42);

        group.bench_with_input(
            BenchmarkId::new(name, dims),
            &mv,
            |bench, mv| {
                bench.iter(|| {
                    black_box(mv.reverse())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SIMD vs Scalar Benchmarks
// =============================================================================

/// Scalar implementation of geometric product for comparison.
fn scalar_geometric_product(
    result: &mut [f64],
    a_coeffs: &[f64],
    b_coeffs: &[f64],
    cayley_blades: &[usize],
    cayley_signs: &[f64],
    num_blades: usize,
) {
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
}

fn bench_gp_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("gp_simd_vs_scalar");

    // Test algebras where SIMD should help: R4 (16 blades), CGA3D (32 blades), R6 (64 blades)
    for dims in [4, 5, 6] {
        let algebra = Algebra::euclidean(dims);
        let num_blades = 1 << dims;

        // Create dense test multivectors
        let a_coeffs: Vec<f64> = (0..num_blades)
            .map(|i| ((i * 2654435761) as f64 % 100.0) / 50.0 - 1.0)
            .collect();
        let b_coeffs: Vec<f64> = (0..num_blades)
            .map(|i| ((i * 1597334677) as f64 % 100.0) / 50.0 - 1.0)
            .collect();

        let cayley_blades = algebra.cayley_blades();
        let cayley_signs = algebra.cayley_signs();

        let name = match dims {
            4 => "R4",
            5 => "CGA3D",
            6 => "R6",
            _ => "Unknown",
        };

        // Benchmark SIMD version
        group.bench_with_input(
            BenchmarkId::new(format!("{}/simd", name), num_blades),
            &(&a_coeffs, &b_coeffs, cayley_blades, cayley_signs, num_blades),
            |bench, (a, b, blades, signs, n)| {
                bench.iter(|| {
                    let mut result = vec![0.0f64; *n];
                    simd_geometric_product(&mut result, a, b, blades, signs, *n);
                    black_box(result)
                });
            },
        );

        // Benchmark scalar version
        group.bench_with_input(
            BenchmarkId::new(format!("{}/scalar", name), num_blades),
            &(&a_coeffs, &b_coeffs, cayley_blades, cayley_signs, num_blades),
            |bench, (a, b, blades, signs, n)| {
                bench.iter(|| {
                    let mut result = vec![0.0f64; *n];
                    scalar_geometric_product(&mut result, a, b, blades, signs, *n);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_gp_simd_dense_vs_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("gp_simd_density");

    let dims = 5;  // CGA3D
    let algebra = Algebra::euclidean(dims);
    let num_blades = 1 << dims;

    let cayley_blades = algebra.cayley_blades();
    let cayley_signs = algebra.cayley_signs();

    // Dense multivector (all coefficients non-zero)
    let dense: Vec<f64> = (0..num_blades)
        .map(|i| ((i * 2654435761) as f64 % 100.0) / 50.0 - 1.0)
        .collect();

    // Sparse multivector (only vectors, ~5 non-zero out of 32)
    let mut sparse = vec![0.0f64; num_blades];
    for i in 0..dims {
        sparse[1 << i] = 1.0 + i as f64;
    }

    // Dense x Dense
    group.bench_function("CGA3D/dense_x_dense", |bench| {
        bench.iter(|| {
            let mut result = vec![0.0f64; num_blades];
            simd_geometric_product(&mut result, &dense, &dense, cayley_blades, cayley_signs, num_blades);
            black_box(result)
        });
    });

    // Dense x Sparse
    group.bench_function("CGA3D/dense_x_sparse", |bench| {
        bench.iter(|| {
            let mut result = vec![0.0f64; num_blades];
            simd_geometric_product(&mut result, &dense, &sparse, cayley_blades, cayley_signs, num_blades);
            black_box(result)
        });
    });

    // Sparse x Sparse
    group.bench_function("CGA3D/sparse_x_sparse", |bench| {
        bench.iter(|| {
            let mut result = vec![0.0f64; num_blades];
            simd_geometric_product(&mut result, &sparse, &sparse, cayley_blades, cayley_signs, num_blades);
            black_box(result)
        });
    });

    group.finish();
}

fn bench_simd_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_threshold");

    // Test around the threshold (16 blades) to validate the decision
    for dims in [3, 4, 5] {
        let num_blades = 1 << dims;
        let a = create_test_multivector(dims, 42);
        let b = create_test_multivector(dims, 137);

        let name = match dims {
            3 => "R3_8blades",
            4 => "R4_16blades",
            5 => "CGA_32blades",
            _ => "Unknown",
        };

        let uses_simd = should_use_simd(num_blades);

        group.bench_with_input(
            BenchmarkId::new(format!("{}/simd={}", name, uses_simd), num_blades),
            &(a, b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(a.geometric_product(b).unwrap())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    // Geometric product
    bench_geometric_product_single,
    bench_geometric_product_batched,
    // SIMD comparisons
    bench_gp_simd_vs_scalar,
    bench_gp_simd_dense_vs_sparse,
    bench_simd_threshold,
    // Other products
    bench_outer_product,
    bench_inner_product,
    // Sandwich (rotor application)
    bench_sandwich_single,
    bench_sandwich_batched,
    // Exponential/logarithm
    bench_exp,
    bench_log,
    // Norms
    bench_norm,
    bench_norm_squared,
    // Inverse
    bench_inverse,
    // Low-level operations
    bench_blade_product,
    bench_algebra_creation,
    // Grade operations
    bench_grade_projection,
    bench_reverse,
);

criterion_main!(benches);
