# LargeCrimsonCanine Benchmarks

Comprehensive benchmark suite for comparing LargeCrimsonCanine performance against competing geometric algebra libraries.

## Quick Start

### Run All Benchmarks

```bash
# From project root
python benches/run_all.py
```

### Run Rust Benchmarks Only

```bash
cargo bench
```

### Run Python Comparison Only

```bash
python benches/vs_competition.py
```

## Benchmark Types

### Rust Benchmarks (Criterion)

The `geometric_product.rs` benchmark uses [Criterion](https://github.com/bheisler/criterion.rs) for statistically rigorous Rust benchmarks.

**Operations tested:**
- Geometric product (single and batched)
- Outer product (wedge)
- Inner product (left contraction)
- Sandwich product (rotor application)
- Exponential (bivector to rotor)
- Logarithm (rotor to bivector)
- Norm computation
- Inverse computation
- Grade projection
- Reverse operation
- Blade product lookup
- Algebra/Cayley table creation

**Algebras tested:**
- R3: Cl(3,0,0) - 3D Euclidean, 8 blades
- R4: Cl(4,0,0) - 4D Euclidean, 16 blades
- CGA3D: Cl(4,1,0) - 3D Conformal, 32 blades

### Python Benchmarks

The `vs_competition.py` script compares LargeCrimsonCanine against:

- **clifford**: Pure Python/NumPy GA library (`pip install clifford`)
- **kingdon**: NumPy-based GA with symbolic capabilities (`pip install kingdon`)

**Operations compared:**
- Geometric product
- Outer product
- Inner product
- Sandwich product (rotor application)
- Batched vector rotations (1000 vectors)
- Exponential
- Logarithm
- Norm
- Inverse

## Installation

### Prerequisites

```bash
# Rust benchmarks require criterion
cargo build --release

# Python benchmarks require numpy
pip install numpy

# Optional: competing libraries for comparison
pip install clifford kingdon
```

### Building LargeCrimsonCanine for Python

```bash
# Using maturin
pip install maturin
maturin develop --release
```

## Running Benchmarks

### Full Suite

```bash
python benches/run_all.py
```

### Options

```bash
python benches/run_all.py --rust-only    # Only Rust benchmarks
python benches/run_all.py --python-only  # Only Python benchmarks
python benches/run_all.py --quick        # Fewer iterations (faster)
python benches/run_all.py --output report.md  # Save markdown report
python benches/run_all.py --json results.json # Save JSON results
```

### Rust Benchmarks Only

```bash
# Full benchmark suite
cargo bench

# Specific benchmark group
cargo bench -- geometric_product/single

# Quick mode (fewer samples)
cargo bench -- --quick

# Generate HTML report
cargo bench
# Results in: target/criterion/report/index.html
```

### Python Benchmarks Only

```bash
python benches/vs_competition.py
# Results saved to: benches/benchmark_results.json
```

## Output Files

| File | Description |
|------|-------------|
| `benchmark_results.json` | Python benchmark results in JSON |
| `target/criterion/` | Criterion HTML reports and data |
| `target/criterion/report/index.html` | Main Criterion dashboard |

## Comparison Methodology

### Statistical Rigor

- **Warmup**: Discard initial runs to allow JIT/cache warmup
- **Iterations**: Default 1000 iterations per benchmark
- **Timing**: High-resolution `perf_counter` (Python) / `Instant` (Rust)
- **Statistics**: Mean, std, min, max reported

### Fair Comparison

- Same random seed (42) for reproducible test data
- Same mathematical operations across libraries
- Pre-created multivectors (creation time excluded)
- Native API usage for each library (no wrapper overhead)

### Algebras

| Name | Signature | Blades | Use Case |
|------|-----------|--------|----------|
| R3 | Cl(3,0,0) | 8 | 3D Euclidean geometry |
| R4 | Cl(4,0,0) | 16 | 4D Euclidean, rotations |
| PGA3D | Cl(3,0,1) | 16 | Projective geometry, rigid motions |
| CGA3D | Cl(4,1,0) | 32 | Conformal geometry, spheres |

## Interpreting Results

### Rust Criterion Output

```
geometric_product/single/R3
                        time:   [1.234 us 1.256 us 1.278 us]
```

- First value: Lower bound (2.5 percentile)
- Second value: Estimated mean
- Third value: Upper bound (97.5 percentile)

### Python Comparison Table

```
Benchmark                      | Algebra  | largecrimsoncanine   | clifford
===============================================================================
geometric_product              | R3       | *1.23 us* (5.2x)     | 6.40 us
```

- `*` marks the fastest library
- `(Nx)` shows speedup vs slowest

## Expected Performance

Typical results on modern hardware (varies by system):

| Operation | R3 (LCC) | R3 (clifford) | Speedup |
|-----------|----------|---------------|---------|
| Geometric Product | ~1 us | ~5-10 us | 5-10x |
| Sandwich Product | ~2 us | ~15-20 us | 7-10x |
| Exponential | ~1-2 us | ~10-20 us | 5-10x |
| Batched (1000) | ~1-2 ms | ~10-15 ms | 5-10x |

**Note**: Actual results depend on:
- CPU architecture and cache
- Memory bandwidth
- Python version
- Compiler optimization level

## Troubleshooting

### "largecrimsoncanine not found"

Build the library first:
```bash
maturin develop --release
```

### "cargo bench fails"

Ensure criterion is in dev-dependencies:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

### Slow benchmarks

Use quick mode:
```bash
cargo bench -- --quick
python benches/run_all.py --quick
```

### Inconsistent results

- Close other applications
- Use `--quick` for development
- Full benchmarks on idle machine

## Adding New Benchmarks

### Rust

Add to `benches/geometric_product.rs`:

```rust
fn bench_new_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_operation");

    let mv = create_test_multivector(3, 42);

    group.bench_function("R3", |b| {
        b.iter(|| {
            black_box(mv.some_operation())
        });
    });

    group.finish();
}

// Add to criterion_group!
criterion_group!(benches, ..., bench_new_operation);
```

### Python

Add to `vs_competition.py`:

```python
def bench_new_operation_r3(self) -> BenchmarkResult:
    """Benchmark new operation in R3."""
    mean, std, min_, max_ = benchmark(
        lambda: self.r3_mv1.some_operation()
    )
    return BenchmarkResult(
        name="new_operation",
        library="largecrimsoncanine",
        algebra="R3",
        iterations=1000,
        total_time_ms=mean * 1000 / 1e6,
        mean_time_us=mean,
        std_time_us=std,
        min_time_us=min_,
        max_time_us=max_,
    )

# Add to run_all() method
```

## License

Same license as LargeCrimsonCanine (MIT).
