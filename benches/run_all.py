#!/usr/bin/env python3
"""
Unified benchmark runner for LargeCrimsonCanine.

This script runs both Rust (criterion) and Python benchmarks and generates
a summary report.

Usage:
    python benches/run_all.py [options]

Options:
    --rust-only     Run only Rust benchmarks
    --python-only   Run only Python benchmarks
    --quick         Run quick benchmarks (fewer iterations)
    --output FILE   Write summary to FILE (default: stdout)
    --json FILE     Write JSON results to FILE
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent


@dataclass
class RustBenchResult:
    """Parsed result from cargo bench."""

    name: str
    mean_ns: float
    std_ns: float
    change_pct: Optional[float] = None


@dataclass
class PythonBenchResult:
    """Result from Python benchmark."""

    name: str
    library: str
    algebra: str
    mean_us: float
    std_us: float


# =============================================================================
# Rust benchmark runner
# =============================================================================


def run_rust_benchmarks(quick: bool = False) -> list[RustBenchResult]:
    """
    Run Rust criterion benchmarks.

    Returns list of parsed benchmark results.
    """
    print("\n" + "=" * 60)
    print("Running Rust Benchmarks (cargo bench)")
    print("=" * 60)

    # Build command
    cmd = ["cargo", "bench"]
    if quick:
        cmd.extend(["--", "--quick"])

    # Run benchmarks
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Rust benchmarks timed out!")
        return []
    except FileNotFoundError:
        print("ERROR: cargo not found. Is Rust installed?")
        return []

    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Parse criterion output
    # Format: "benchmark_name    time:   [123.45 ns 124.56 ns 125.67 ns]"
    results = []
    pattern = r"^(\S+)\s+time:\s+\[[\d.]+ [nmu]s\s+([\d.]+) ([nmu]s)\s+[\d.]+ [nmu]s\]"

    for line in result.stdout.split("\n"):
        match = re.match(pattern, line)
        if match:
            name = match.group(1)
            value = float(match.group(2))
            unit = match.group(3)

            # Convert to nanoseconds
            if unit == "us":
                value *= 1000
            elif unit == "ms":
                value *= 1_000_000

            results.append(
                RustBenchResult(
                    name=name,
                    mean_ns=value,
                    std_ns=0.0,  # Criterion gives range, not std
                )
            )

    return results


# =============================================================================
# Python benchmark runner
# =============================================================================


def run_python_benchmarks(quick: bool = False) -> list[PythonBenchResult]:
    """
    Run Python comparison benchmarks.

    Returns list of benchmark results.
    """
    print("\n" + "=" * 60)
    print("Running Python Benchmarks")
    print("=" * 60)

    # Run the comparison script
    script_path = SCRIPT_DIR / "vs_competition.py"

    if not script_path.exists():
        print(f"ERROR: {script_path} not found!")
        return []

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Python benchmarks timed out!")
        return []

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Load results from JSON if available
    results_file = SCRIPT_DIR / "benchmark_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            return [
                PythonBenchResult(
                    name=r["name"],
                    library=r["library"],
                    algebra=r["algebra"],
                    mean_us=r["mean_time_us"],
                    std_us=r["std_time_us"],
                )
                for r in data
            ]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Could not parse results file: {e}")

    return []


# =============================================================================
# Summary generation
# =============================================================================


def format_time_ns(ns: float) -> str:
    """Format nanoseconds in appropriate unit."""
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    elif ns >= 1000:
        return f"{ns / 1000:.2f} us"
    else:
        return f"{ns:.2f} ns"


def format_time_us(us: float) -> str:
    """Format microseconds in appropriate unit."""
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    elif us >= 1:
        return f"{us:.2f} us"
    else:
        return f"{us * 1000:.2f} ns"


def generate_summary(
    rust_results: list[RustBenchResult],
    python_results: list[PythonBenchResult],
) -> str:
    """Generate a markdown summary of all benchmarks."""
    lines = []

    lines.append("# LargeCrimsonCanine Benchmark Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # System info
    lines.append("## System Information")
    lines.append("")
    lines.append(f"- Platform: {sys.platform}")
    lines.append(f"- Python: {sys.version.split()[0]}")
    try:
        rustc = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=5
        )
        lines.append(f"- Rust: {rustc.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    lines.append("")

    # Rust results
    if rust_results:
        lines.append("## Rust Benchmarks (via Criterion)")
        lines.append("")
        lines.append("| Benchmark | Mean Time |")
        lines.append("|-----------|-----------|")
        for r in rust_results:
            lines.append(f"| {r.name} | {format_time_ns(r.mean_ns)} |")
        lines.append("")

    # Python comparison
    if python_results:
        lines.append("## Python Library Comparison")
        lines.append("")

        # Group by benchmark name
        by_name: dict[str, list[PythonBenchResult]] = {}
        for r in python_results:
            key = f"{r.name} ({r.algebra})"
            if key not in by_name:
                by_name[key] = []
            by_name[key].append(r)

        # Get all libraries
        libraries = sorted(set(r.library for r in python_results))

        # Header
        header = "| Benchmark |"
        for lib in libraries:
            header += f" {lib} |"
        lines.append(header)

        sep = "|-----------|"
        for _ in libraries:
            sep += "------------|"
        lines.append(sep)

        # Data rows
        for name in sorted(by_name.keys()):
            results = {r.library: r for r in by_name[name]}
            row = f"| {name} |"

            # Find fastest
            times = {lib: results[lib].mean_us for lib in results}
            min_time = min(times.values()) if times else float("inf")

            for lib in libraries:
                if lib in results:
                    time_str = format_time_us(results[lib].mean_us)
                    if results[lib].mean_us == min_time and len(times) > 1:
                        time_str = f"**{time_str}**"
                    row += f" {time_str} |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")
        lines.append("**Bold** indicates fastest for that benchmark.")
        lines.append("")

    # Speedup summary
    if python_results:
        lines.append("## Speedup Summary")
        lines.append("")

        lcc_results = [r for r in python_results if r.library == "largecrimsoncanine"]
        other_results = [r for r in python_results if r.library != "largecrimsoncanine"]

        if lcc_results and other_results:
            # Group by benchmark
            for lcc_r in lcc_results:
                key = f"{lcc_r.name}:{lcc_r.algebra}"
                others = [
                    r
                    for r in other_results
                    if f"{r.name}:{r.algebra}" == key
                ]
                for other in others:
                    speedup = other.mean_us / lcc_r.mean_us
                    if speedup > 1:
                        lines.append(
                            f"- {lcc_r.name} ({lcc_r.algebra}): "
                            f"**{speedup:.1f}x faster** than {other.library}"
                        )
                    else:
                        lines.append(
                            f"- {lcc_r.name} ({lcc_r.algebra}): "
                            f"{1/speedup:.1f}x slower than {other.library}"
                        )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run all LargeCrimsonCanine benchmarks"
    )
    parser.add_argument(
        "--rust-only", action="store_true", help="Run only Rust benchmarks"
    )
    parser.add_argument(
        "--python-only", action="store_true", help="Run only Python benchmarks"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmarks (fewer iterations)"
    )
    parser.add_argument(
        "--output", type=str, help="Write summary to file (default: stdout)"
    )
    parser.add_argument(
        "--json", type=str, help="Write JSON results to file"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LargeCrimsonCanine Benchmark Suite")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Working directory: {PROJECT_DIR}")

    rust_results: list[RustBenchResult] = []
    python_results: list[PythonBenchResult] = []

    # Run Rust benchmarks
    if not args.python_only:
        rust_results = run_rust_benchmarks(quick=args.quick)
        print(f"\nRust benchmarks completed: {len(rust_results)} results")

    # Run Python benchmarks
    if not args.rust_only:
        python_results = run_python_benchmarks(quick=args.quick)
        print(f"\nPython benchmarks completed: {len(python_results)} results")

    # Generate summary
    summary = generate_summary(rust_results, python_results)

    # Output summary
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(summary)
        print(f"\nSummary written to {output_path}")
    else:
        print("\n")
        print(summary)

    # Save JSON results
    if args.json:
        json_path = Path(args.json)
        data = {
            "timestamp": datetime.now().isoformat(),
            "rust": [
                {"name": r.name, "mean_ns": r.mean_ns, "std_ns": r.std_ns}
                for r in rust_results
            ],
            "python": [
                {
                    "name": r.name,
                    "library": r.library,
                    "algebra": r.algebra,
                    "mean_us": r.mean_us,
                    "std_us": r.std_us,
                }
                for r in python_results
            ],
        }
        json_path.write_text(json.dumps(data, indent=2))
        print(f"JSON results written to {json_path}")

    print("\nBenchmark run complete!")


if __name__ == "__main__":
    main()
