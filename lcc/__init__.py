"""LargeCrimsonCanine — High-performance geometric algebra for Python.

Core types (Multivector, Algebra, MultivectorBatch) come from the Rust
backend.  This package re-exports them so users can write::

    import lcc
    R3 = lcc.Algebra.euclidean(3)
    v = R3.multivector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

Optional submodules (torch, jax, symbolic, codegen, viz) are imported
on demand and degrade gracefully when their dependencies are missing.
"""

from largecrimsoncanine import Algebra, Multivector, MultivectorBatch

from lcc.viz import show, Graph

__all__ = [
    # Core types from Rust backend
    "Algebra",
    "Multivector",
    "MultivectorBatch",
    # Visualization
    "show",
    "Graph",
]
