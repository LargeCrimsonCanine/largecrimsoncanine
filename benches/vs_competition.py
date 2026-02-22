#!/usr/bin/env python3
"""
Benchmark comparison: largecrimsoncanine vs competing geometric algebra libraries.

Compares performance against:
- clifford (pip install clifford) - Pure Python/NumPy GA library
- kingdon (pip install kingdon) - NumPy-based GA with JIT compilation

Run with: python benches/vs_competition.py

Results are saved to benches/benchmark_results.json
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# =============================================================================
# Timing utilities
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    library: str
    algebra: str
    iterations: int
    total_time_ms: float
    mean_time_us: float
    std_time_us: float
    min_time_us: float
    max_time_us: float


def benchmark(
    func: Callable[[], Any],
    warmup: int = 10,
    iterations: int = 1000,
    name: str = "benchmark",
) -> tuple[float, float, float, float]:
    """
    Run a benchmark with warmup and multiple iterations.

    Returns (mean_us, std_us, min_us, max_us) timing in microseconds.
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    times = np.array(times)
    return float(times.mean()), float(times.std()), float(times.min()), float(times.max())


def format_time(us: float) -> str:
    """Format time in appropriate units."""
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    elif us >= 1:
        return f"{us:.2f} us"
    else:
        return f"{us * 1000:.2f} ns"


# =============================================================================
# Library imports with availability checking
# =============================================================================

LIBRARIES = {}


def check_largecrimsoncanine():
    """Check if largecrimsoncanine is available."""
    try:
        import largecrimsoncanine

        LIBRARIES["largecrimsoncanine"] = largecrimsoncanine
        return True
    except ImportError:
        print("WARNING: largecrimsoncanine not found. Build with 'maturin develop'")
        return False


def check_clifford():
    """Check if clifford is available."""
    try:
        import clifford

        LIBRARIES["clifford"] = clifford
        return True
    except ImportError:
        print("WARNING: clifford not found. Install with 'pip install clifford'")
        return False


def check_kingdon():
    """Check if kingdon is available."""
    try:
        import kingdon

        LIBRARIES["kingdon"] = kingdon
        return True
    except ImportError:
        print("WARNING: kingdon not found. Install with 'pip install kingdon'")
        return False


# =============================================================================
# Benchmark implementations for each library
# =============================================================================


class LargeCrimsonCanineBenchmarks:
    """Benchmarks using largecrimsoncanine."""

    def __init__(self):
        self.lcc = LIBRARIES["largecrimsoncanine"]
        self._setup_algebras()

    def _setup_algebras(self):
        """Set up test algebras and multivectors."""
        # R3 (8 blades)
        self.r3_coeffs = list(np.random.randn(8))
        self.r3_mv1 = self.lcc.Multivector(self.r3_coeffs)
        self.r3_mv2 = self.lcc.Multivector(list(np.random.randn(8)))

        # R4 (16 blades)
        self.r4_coeffs = list(np.random.randn(16))
        self.r4_mv1 = self.lcc.Multivector(self.r4_coeffs)
        self.r4_mv2 = self.lcc.Multivector(list(np.random.randn(16)))

        # Create rotors (cos(theta/2) + sin(theta/2)*e12)
        angle = np.pi / 4
        r3_rotor_coeffs = [0.0] * 8
        r3_rotor_coeffs[0] = np.cos(angle / 2)
        r3_rotor_coeffs[3] = np.sin(angle / 2)  # e12
        self.r3_rotor = self.lcc.Multivector(r3_rotor_coeffs)

        # Vector to transform
        r3_vec_coeffs = [0.0] * 8
        r3_vec_coeffs[1] = 1.0  # e1
        r3_vec_coeffs[2] = 2.0  # e2
        r3_vec_coeffs[4] = 3.0  # e3
        self.r3_vector = self.lcc.Multivector(r3_vec_coeffs)

        # Bivector for exp
        r3_biv_coeffs = [0.0] * 8
        r3_biv_coeffs[3] = np.pi / 6  # e12 component
        self.r3_bivector = self.lcc.Multivector(r3_biv_coeffs)

    def bench_geometric_product_r3(self) -> BenchmarkResult:
        """Benchmark geometric product in R3."""
        mean, std, min_, max_ = benchmark(
            lambda: self.r3_mv1.geometric_product(self.r3_mv2)
        )
        return BenchmarkResult(
            name="geometric_product",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_geometric_product_r4(self) -> BenchmarkResult:
        """Benchmark geometric product in R4."""
        mean, std, min_, max_ = benchmark(
            lambda: self.r4_mv1.geometric_product(self.r4_mv2)
        )
        return BenchmarkResult(
            name="geometric_product",
            library="largecrimsoncanine",
            algebra="R4",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_outer_product_r3(self) -> BenchmarkResult:
        """Benchmark outer product in R3."""
        mean, std, min_, max_ = benchmark(
            lambda: self.r3_mv1.outer_product(self.r3_mv2)
        )
        return BenchmarkResult(
            name="outer_product",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_inner_product_r3(self) -> BenchmarkResult:
        """Benchmark inner product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1.inner(self.r3_mv2))
        return BenchmarkResult(
            name="inner_product",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_sandwich_r3(self) -> BenchmarkResult:
        """Benchmark sandwich product (rotor application) in R3."""
        mean, std, min_, max_ = benchmark(
            lambda: self.r3_rotor.sandwich(self.r3_vector)
        )
        return BenchmarkResult(
            name="sandwich",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_sandwich_batched_r3(self, batch_size: int = 1000) -> BenchmarkResult:
        """Benchmark rotating many vectors."""
        vectors = [
            self.lcc.Multivector(
                [0.0, float(i), float(i + 1), 0.0, float(i + 2), 0.0, 0.0, 0.0]
            )
            for i in range(batch_size)
        ]

        def rotate_all():
            for v in vectors:
                self.r3_rotor.sandwich(v)

        mean, std, min_, max_ = benchmark(rotate_all, iterations=100)
        return BenchmarkResult(
            name=f"sandwich_batched_{batch_size}",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=100,
            total_time_ms=mean / 1000,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_exp_r3(self) -> BenchmarkResult:
        """Benchmark exponential (bivector -> rotor) in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_bivector.exp())
        return BenchmarkResult(
            name="exp",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_log_r3(self) -> BenchmarkResult:
        """Benchmark logarithm (rotor -> bivector) in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_rotor.log())
        return BenchmarkResult(
            name="log",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_norm_r3(self) -> BenchmarkResult:
        """Benchmark norm computation in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1.norm())
        return BenchmarkResult(
            name="norm",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_inverse_r3(self) -> BenchmarkResult:
        """Benchmark inverse computation in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_rotor.inverse())
        return BenchmarkResult(
            name="inverse",
            library="largecrimsoncanine",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        return [
            self.bench_geometric_product_r3(),
            self.bench_geometric_product_r4(),
            self.bench_outer_product_r3(),
            self.bench_inner_product_r3(),
            self.bench_sandwich_r3(),
            self.bench_sandwich_batched_r3(),
            self.bench_exp_r3(),
            self.bench_log_r3(),
            self.bench_norm_r3(),
            self.bench_inverse_r3(),
        ]


class CliffordBenchmarks:
    """Benchmarks using the clifford library."""

    def __init__(self):
        self.cf = LIBRARIES["clifford"]
        self._setup_algebras()

    def _setup_algebras(self):
        """Set up test algebras and multivectors."""
        # R3
        layout3, blades3 = self.cf.Cl(3)
        self.layout3 = layout3
        self.e1_3, self.e2_3, self.e3_3 = blades3["e1"], blades3["e2"], blades3["e3"]

        # Random multivectors in R3
        np.random.seed(42)
        self.r3_mv1 = layout3.randomMV()
        self.r3_mv2 = layout3.randomMV()

        # R4
        layout4, blades4 = self.cf.Cl(4)
        self.layout4 = layout4
        self.r4_mv1 = layout4.randomMV()
        self.r4_mv2 = layout4.randomMV()

        # Create rotor for R3
        angle = np.pi / 4
        B = blades3["e12"]  # Rotation plane
        self.r3_rotor = np.cos(angle / 2) + np.sin(angle / 2) * B

        # Vector to transform
        self.r3_vector = 1.0 * self.e1_3 + 2.0 * self.e2_3 + 3.0 * self.e3_3

        # Bivector for exp
        self.r3_bivector = (np.pi / 6) * blades3["e12"]

    def bench_geometric_product_r3(self) -> BenchmarkResult:
        """Benchmark geometric product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1 * self.r3_mv2)
        return BenchmarkResult(
            name="geometric_product",
            library="clifford",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_geometric_product_r4(self) -> BenchmarkResult:
        """Benchmark geometric product in R4."""
        mean, std, min_, max_ = benchmark(lambda: self.r4_mv1 * self.r4_mv2)
        return BenchmarkResult(
            name="geometric_product",
            library="clifford",
            algebra="R4",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_outer_product_r3(self) -> BenchmarkResult:
        """Benchmark outer product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1 ^ self.r3_mv2)
        return BenchmarkResult(
            name="outer_product",
            library="clifford",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_inner_product_r3(self) -> BenchmarkResult:
        """Benchmark inner product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1 | self.r3_mv2)
        return BenchmarkResult(
            name="inner_product",
            library="clifford",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_sandwich_r3(self) -> BenchmarkResult:
        """Benchmark sandwich product (rotor application) in R3."""
        rotor_rev = ~self.r3_rotor
        mean, std, min_, max_ = benchmark(
            lambda: self.r3_rotor * self.r3_vector * rotor_rev
        )
        return BenchmarkResult(
            name="sandwich",
            library="clifford",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_sandwich_batched_r3(self, batch_size: int = 1000) -> BenchmarkResult:
        """Benchmark rotating many vectors."""
        vectors = [
            float(i) * self.e1_3 + float(i + 1) * self.e2_3 + float(i + 2) * self.e3_3
            for i in range(batch_size)
        ]
        rotor_rev = ~self.r3_rotor

        def rotate_all():
            for v in vectors:
                self.r3_rotor * v * rotor_rev

        mean, std, min_, max_ = benchmark(rotate_all, iterations=100)
        return BenchmarkResult(
            name=f"sandwich_batched_{batch_size}",
            library="clifford",
            algebra="R3",
            iterations=100,
            total_time_ms=mean / 1000,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_exp_r3(self) -> BenchmarkResult:
        """Benchmark exponential (bivector -> rotor) in R3."""
        # clifford uses math.e ** B or layout.exp()
        import math

        mean, std, min_, max_ = benchmark(lambda: math.e ** self.r3_bivector)
        return BenchmarkResult(
            name="exp",
            library="clifford",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_norm_r3(self) -> BenchmarkResult:
        """Benchmark norm computation in R3."""
        mean, std, min_, max_ = benchmark(lambda: abs(self.r3_mv1))
        return BenchmarkResult(
            name="norm",
            library="clifford",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_inverse_r3(self) -> BenchmarkResult:
        """Benchmark inverse computation in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_rotor.inv())
        return BenchmarkResult(
            name="inverse",
            library="clifford",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        return [
            self.bench_geometric_product_r3(),
            self.bench_geometric_product_r4(),
            self.bench_outer_product_r3(),
            self.bench_inner_product_r3(),
            self.bench_sandwich_r3(),
            self.bench_sandwich_batched_r3(),
            self.bench_exp_r3(),
            self.bench_norm_r3(),
            self.bench_inverse_r3(),
        ]


class KingdonBenchmarks:
    """Benchmarks using the kingdon library."""

    def __init__(self):
        self.kd = LIBRARIES["kingdon"]
        self._setup_algebras()

    def _setup_algebras(self):
        """Set up test algebras and multivectors."""
        # R3
        self.alg3 = self.kd.Algebra(3, 0, 0)

        # Random multivectors
        np.random.seed(42)
        self.r3_mv1 = self.alg3.multivector(np.random.randn(8))
        self.r3_mv2 = self.alg3.multivector(np.random.randn(8))

        # R4
        self.alg4 = self.kd.Algebra(4, 0, 0)
        self.r4_mv1 = self.alg4.multivector(np.random.randn(16))
        self.r4_mv2 = self.alg4.multivector(np.random.randn(16))

        # Create rotor
        angle = np.pi / 4
        e12 = self.alg3.multivector({(1, 2): 1.0})
        self.r3_rotor = self.alg3.multivector({(): np.cos(angle / 2)}) + np.sin(
            angle / 2
        ) * e12

        # Vector
        self.r3_vector = self.alg3.multivector({(1,): 1.0, (2,): 2.0, (3,): 3.0})

        # Bivector for exp
        self.r3_bivector = self.alg3.multivector({(1, 2): np.pi / 6})

    def bench_geometric_product_r3(self) -> BenchmarkResult:
        """Benchmark geometric product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1 * self.r3_mv2)
        return BenchmarkResult(
            name="geometric_product",
            library="kingdon",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_geometric_product_r4(self) -> BenchmarkResult:
        """Benchmark geometric product in R4."""
        mean, std, min_, max_ = benchmark(lambda: self.r4_mv1 * self.r4_mv2)
        return BenchmarkResult(
            name="geometric_product",
            library="kingdon",
            algebra="R4",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_outer_product_r3(self) -> BenchmarkResult:
        """Benchmark outer product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1 ^ self.r3_mv2)
        return BenchmarkResult(
            name="outer_product",
            library="kingdon",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_inner_product_r3(self) -> BenchmarkResult:
        """Benchmark inner product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1 | self.r3_mv2)
        return BenchmarkResult(
            name="inner_product",
            library="kingdon",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_sandwich_r3(self) -> BenchmarkResult:
        """Benchmark sandwich product in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_rotor.sw(self.r3_vector))
        return BenchmarkResult(
            name="sandwich",
            library="kingdon",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_sandwich_batched_r3(self, batch_size: int = 1000) -> BenchmarkResult:
        """Benchmark rotating many vectors."""
        vectors = [
            self.alg3.multivector(
                {(1,): float(i), (2,): float(i + 1), (3,): float(i + 2)}
            )
            for i in range(batch_size)
        ]

        def rotate_all():
            for v in vectors:
                self.r3_rotor.sw(v)

        mean, std, min_, max_ = benchmark(rotate_all, iterations=100)
        return BenchmarkResult(
            name=f"sandwich_batched_{batch_size}",
            library="kingdon",
            algebra="R3",
            iterations=100,
            total_time_ms=mean / 1000,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_exp_r3(self) -> BenchmarkResult:
        """Benchmark exponential in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_bivector.exp())
        return BenchmarkResult(
            name="exp",
            library="kingdon",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_norm_r3(self) -> BenchmarkResult:
        """Benchmark norm computation in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_mv1.normsq())
        return BenchmarkResult(
            name="norm",
            library="kingdon",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def bench_inverse_r3(self) -> BenchmarkResult:
        """Benchmark inverse computation in R3."""
        mean, std, min_, max_ = benchmark(lambda: self.r3_rotor.inv())
        return BenchmarkResult(
            name="inverse",
            library="kingdon",
            algebra="R3",
            iterations=1000,
            total_time_ms=mean * 1000 / 1e6,
            mean_time_us=mean,
            std_time_us=std,
            min_time_us=min_,
            max_time_us=max_,
        )

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        return [
            self.bench_geometric_product_r3(),
            self.bench_geometric_product_r4(),
            self.bench_outer_product_r3(),
            self.bench_inner_product_r3(),
            self.bench_sandwich_r3(),
            self.bench_sandwich_batched_r3(),
            self.bench_exp_r3(),
            self.bench_norm_r3(),
            self.bench_inverse_r3(),
        ]


# =============================================================================
# Results formatting and comparison
# =============================================================================


def print_comparison_table(results: list[BenchmarkResult]):
    """Print a formatted comparison table."""
    # Group results by benchmark name and algebra
    grouped: dict[str, dict[str, BenchmarkResult]] = {}
    for r in results:
        key = f"{r.name}:{r.algebra}"
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.library] = r

    # Print header
    libraries = sorted(set(r.library for r in results))
    header = f"{'Benchmark':<30} | {'Algebra':<8}"
    for lib in libraries:
        header += f" | {lib:<20}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Print rows
    for key in sorted(grouped.keys()):
        name, algebra = key.split(":")
        row = f"{name:<30} | {algebra:<8}"

        # Find the fastest time for this benchmark
        times = {
            lib: grouped[key][lib].mean_time_us
            for lib in libraries
            if lib in grouped[key]
        }
        min_time = min(times.values()) if times else float("inf")

        for lib in libraries:
            if lib in grouped[key]:
                r = grouped[key][lib]
                time_str = format_time(r.mean_time_us)
                # Highlight fastest
                if r.mean_time_us == min_time:
                    time_str = f"*{time_str}*"
                # Show speedup vs slowest
                if times:
                    max_time = max(times.values())
                    if r.mean_time_us < max_time:
                        speedup = max_time / r.mean_time_us
                        time_str += f" ({speedup:.1f}x)"
                row += f" | {time_str:<20}"
            else:
                row += f" | {'N/A':<20}"

        print(row)

    print("=" * len(header))
    print("\n* = fastest for this benchmark")


def save_results(results: list[BenchmarkResult], filepath: Path):
    """Save results to JSON file."""
    data = [
        {
            "name": r.name,
            "library": r.library,
            "algebra": r.algebra,
            "iterations": r.iterations,
            "mean_time_us": r.mean_time_us,
            "std_time_us": r.std_time_us,
            "min_time_us": r.min_time_us,
            "max_time_us": r.max_time_us,
        }
        for r in results
    ]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")


# =============================================================================
# Main entry point
# =============================================================================


def main():
    """Run all benchmarks and print comparison."""
    print("=" * 60)
    print("LargeCrimsonCanine Benchmark Suite - Library Comparison")
    print("=" * 60)
    print()

    # Check available libraries
    print("Checking available libraries...")
    lcc_available = check_largecrimsoncanine()
    clifford_available = check_clifford()
    kingdon_available = check_kingdon()
    print()

    if not any([lcc_available, clifford_available, kingdon_available]):
        print("ERROR: No libraries available for benchmarking!")
        print("Install at least one of:")
        print("  - largecrimsoncanine (maturin develop)")
        print("  - clifford (pip install clifford)")
        print("  - kingdon (pip install kingdon)")
        sys.exit(1)

    all_results: list[BenchmarkResult] = []

    # Run benchmarks for each available library
    if lcc_available:
        print("\nRunning largecrimsoncanine benchmarks...")
        try:
            lcc_bench = LargeCrimsonCanineBenchmarks()
            all_results.extend(lcc_bench.run_all())
            print("  Done!")
        except Exception as e:
            print(f"  Error: {e}")

    if clifford_available:
        print("\nRunning clifford benchmarks...")
        try:
            cf_bench = CliffordBenchmarks()
            all_results.extend(cf_bench.run_all())
            print("  Done!")
        except Exception as e:
            print(f"  Error: {e}")

    if kingdon_available:
        print("\nRunning kingdon benchmarks...")
        try:
            kd_bench = KingdonBenchmarks()
            all_results.extend(kd_bench.run_all())
            print("  Done!")
        except Exception as e:
            print(f"  Error: {e}")

    # Print comparison
    print("\n")
    print_comparison_table(all_results)

    # Save results
    results_path = Path(__file__).parent / "benchmark_results.json"
    save_results(all_results, results_path)


if __name__ == "__main__":
    main()
