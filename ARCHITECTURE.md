# Architecture

## Core Philosophy

LargeCrimsonCanine is a geometric algebra library for Python with a Rust backend.

The design has two goals that must be held simultaneously:

1. **Fast enough to be useful at scale.** Geometric algebra on large datasets requires performance that Python alone cannot provide.
2. **Feels like a native Python library.** The Rust backend should be invisible to users. If someone has to think about Rust to use LCC, we have failed.

PyO3 is what makes both goals achievable. We define the Python API directly in Rust using `#[pyclass]` and `#[pymethods]`. There is no separate wrapper layer. The Rust structs ARE the Python objects.

## Why Rust, not NumPy

The primary existing geometric algebra library (`clifford`) uses NumPy as its backend. NumPy was not designed for geometric algebra. The result is awkward translation between what geometric algebra wants to do and what NumPy can natively express — performance overhead and API constraints that follow from a mismatched foundation.

Rust lets us implement geometric algebra operations natively. A multivector is a first-class Rust type, not a NumPy array pretending to be one. The geometric product is implemented directly, not decomposed into matrix operations.

## Why not Julia

Julia has a type system that maps naturally onto geometric algebra grade structure and would produce elegant implementation code. However:

- Julia compiles to LLVM, same ceiling as Rust
- Julia has a garbage collector; Rust has deterministic memory management
- Rust compiles to a static library embeddable anywhere; Julia is harder to embed
- PyO3 (Rust-Python bindings) is production quality; equivalent Julia-Python bridges are not

A Julia wrapper around the Rust core may be worth building eventually. The Rust core stays.

## Multivector Representation

Multivectors are stored as flat coefficient arrays indexed by basis blade. The index corresponds to the binary representation of the blade:

- Index 0 = scalar (1)
- Index 1 = e1
- Index 2 = e2
- Index 3 = e12 (= e1 ∧ e2)
- Index 4 = e3
- Index 5 = e13
- etc.

This encoding makes grade computation trivial (`popcount`) and geometric products efficient (XOR for blade combination).

## Metric Signature

Geometric algebra is defined over a vector space with a metric signature Cl(p,q,r) where:
- p basis vectors square to +1
- q basis vectors square to -1
- r basis vectors square to 0 (null vectors)

**Current status:** The implementation assumes Euclidean metric (all basis vectors square to +1). This is a known limitation.

**Planned API:** An `Algebra` context object will carry the metric signature. All multivectors will be created within an algebra context rather than freestanding. This is the correct long-term design. The current freestanding API is a temporary scaffold.

This decision needs to be made before v0.2. Changing the core after adoption is painful.

## Dimension Handling

Dimensions are currently runtime values. Rust const generics would allow compile-time dimension checking (`Multivector<4>` instead of `Multivector`), enabling the compiler to catch dimension mismatches that currently produce runtime errors.

This is a potential future optimization. The API change it would require means it should be decided before v1.0.

## Operator Overloading

We expose geometric algebra operations as Python operators where natural:

- `a * b` — geometric product (fundamental operation)
- `a ^ b` — outer (wedge) product

The goal is that a physicist or roboticist familiar with geometric algebra notation should find LCC immediately readable. We do not invent new notation.

## Equality and Hashing

`Multivector` implements `__eq__` for equality comparison but intentionally does not implement `__hash__`. This means multivectors cannot be used as dictionary keys or set members.

**Rationale:**
1. Multivectors contain floating-point coefficients
2. Floating-point equality is problematic for hashing (equal values might not hash equally due to precision)
3. The common use case for multivectors is computation, not as collection keys
4. Users who need hashable keys can convert to tuples: `tuple(mv.to_list())`

If a future use case requires hashability, we could implement `__hash__` based on a rounded or quantized representation.

## Dual API Pattern: Option vs Exception

Some operations (like `normalize`) have two variants:
- `normalize()` returns `None` on failure (idiomatic for Rust callers, also works in Python)
- `normalized()` raises `ValueError` on failure (idiomatic for Python callers who expect exceptions)

This pattern appears where operations can fail in a predictable, recoverable way.

## Error Messages

Error messages should be specific and actionable. Prefer:

```
"coeffs length must be a power of 2 (got 3, expected 2 or 4)"
```

over:

```
"invalid input"
```

Users encountering errors are often not the people who wrote the calling code. Error messages are documentation.

## Versioning

Semantic versioning. A breaking change is any change that requires existing correct user code to be modified. We prefer adding new APIs over changing existing ones.

## Citation Policy

All mathematical references are marked [VERIFY] until independently confirmed by a human. Do not add citations from memory. See CONTRIBUTING.md.
