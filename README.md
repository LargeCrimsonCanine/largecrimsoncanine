# LargeCrimsonCanine

High-performance geometric algebra for Python, built on a Rust backend.

## What is this?

LargeCrimsonCanine is a geometric algebra library for Python. It implements the full Clifford algebra framework — multivectors, the geometric product, outer product, inner product, grade projection, and more — with a Rust backend for performance.

Geometric algebra is a mathematical framework that unifies and extends linear algebra, providing native support for rotations, reflections, and higher-dimensional geometric objects. It is the natural language for problems in physics, robotics, computer graphics, and — we suspect — quite a bit more.

## Why another geometric algebra library?

Existing Python geometric algebra libraries (notably `clifford`) use NumPy as their backend. NumPy was not designed for geometric algebra operations and the mismatch shows: awkward APIs, poor performance at scale, limited expressiveness.

LargeCrimsonCanine implements geometric algebra operations natively in Rust, exposing a clean Python API via PyO3. The result is geometric algebra that is fast, correct, and pleasant to use.

## Status

Early development. Core multivector type and geometric product are the current focus.

## Mathematical Foundation

The implementation follows the Clifford algebra Cl(p,q,r) framework. Primary references:

- Dorst, L., Fontijne, D., & Mann, S. (2007). *Geometric Algebra for Computer Science*. Morgan Kaufmann. [VERIFY]
- Hestenes, D., & Sobczyk, G. (1984). *Clifford Algebra to Geometric Calculus*. Reidel. [VERIFY]

Note: All citations marked [VERIFY] have been flagged for independent verification.

## Installation

```bash
pip install largecrimsoncanine
```

(Not yet available. Coming once core implementation stabilizes.)

## Quick Start

```python
import largecrimsoncanine as lcc

# Create multivectors
a = lcc.Multivector([1.0, 0.0, 0.0])
b = lcc.Multivector([0.0, 1.0, 0.0])

# Geometric product
c = a * b

# Grade projection
scalar_part = c.grade(0)
bivector_part = c.grade(2)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT

