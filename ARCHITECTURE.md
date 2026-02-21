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

- `a * b` — geometric product (or scalar multiplication if b is float)
- `a ^ b` — outer (wedge) product
- `a | b` — left contraction (inner product)
- `a + b`, `a - b` — addition, subtraction
- `-a` — negation
- `a / s` — scalar division
- `s * a` — scalar multiplication (commutative)

The goal is that a physicist or roboticist familiar with geometric algebra notation should find LCC immediately readable. We do not invent new notation.

Short method aliases are provided for common operations:
- `gp()` — alias for `geometric_product()`
- `wedge()` — alias for `outer_product()`
- `lc()`, `inner()` — aliases for `left_contraction()`

## Equality and Hashing

`Multivector` implements `__eq__` for equality comparison but intentionally does not implement `__hash__`. This means multivectors cannot be used as dictionary keys or set members.

**Rationale:**
1. Multivectors contain floating-point coefficients
2. Floating-point equality is problematic for hashing (equal values might not hash equally due to precision)
3. The common use case for multivectors is computation, not as collection keys
4. Users who need hashable keys can convert to tuples: `tuple(mv.to_list())`

If a future use case requires hashability, we could implement `__hash__` based on a rounded or quantized representation.

## Dual API Pattern: Option vs Exception

Some operations can fail in predictable ways. Rust and Python have different idioms for handling this:

- **Rust side:** `normalize()` returns `Option<Self>`, returning `None` for zero-norm multivectors
- **Python side:** `normalized()` raises `ValueError` on failure, which Python users expect

Only the exception-raising variant is exposed to Python. Rust callers can use the Option-returning version internally. This keeps Python users in idiomatic Python-land while preserving Rust idioms in the backend.

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

## Visualization Standards

When visualization features are added (Jupyter integration, plotting), they must follow these accessibility standards:

### Colorblind-Safe Palette

The default palette uses colors distinguishable by users with common forms of color blindness (deuteranopia, protanopia, tritanopia). Based on the Wong palette [VERIFY].

The following table defines the 8 standard colors with their hex codes, RGB values, and intended use cases:

| Name    | Hex       | RGB             | Use case           |
|---------|-----------|-----------------|---------------------|
| Blue    | `#0072B2` | (0, 114, 178)   | Primary elements    |
| Orange  | `#E69F00` | (230, 159, 0)   | Secondary elements  |
| Green   | `#009E73` | (0, 158, 115)   | Tertiary elements   |
| Yellow  | `#F0E442` | (240, 228, 66)  | Highlights          |
| Sky     | `#56B4E9` | (86, 180, 233)  | Background accents  |
| Vermillion | `#D55E00` | (213, 94, 0) | Warnings/alerts     |
| Purple  | `#CC79A7` | (204, 121, 167) | Special elements    |
| Black   | `#000000` | (0, 0, 0)       | Text, outlines      |

### Additional Requirements (WCAG 2.1 AA)

- **Never use color alone** — all visual distinctions must also use shape, pattern, or label (WCAG 1.4.1)
- **High contrast** — minimum 4.5:1 contrast ratio for text, 3:1 for graphics (WCAG 1.4.3, 1.4.11)
- **Pattern fills** — provide hatching/stippling options for filled regions

Reference: https://www.w3.org/WAI/WCAG21/quickref/ [VERIFY]

## Versioning

Semantic versioning. A breaking change is any change that requires existing correct user code to be modified. We prefer adding new APIs over changing existing ones.

## Citation Policy

All mathematical references are marked [VERIFY] until independently confirmed by a human. Do not add citations from memory. See CONTRIBUTING.md.
