"""Symbolic computation for geometric algebra using SymPy.

This module provides symbolic manipulation of geometric algebra expressions,
allowing symbolic coefficients and metric parameters.

Example:
    >>> from lcc.symbolic import SymbolicMultivector, SymbolicAlgebra
    >>> import sympy as sp
    >>>
    >>> # Create a symbolic 3D Euclidean algebra
    >>> alg = SymbolicAlgebra.euclidean(3)
    >>>
    >>> # Create symbolic basis vectors
    >>> e1, e2, e3 = alg.basis_vectors()
    >>>
    >>> # Symbolic coefficients
    >>> a, b = sp.symbols('a b')
    >>> v = a*e1 + b*e2
    >>>
    >>> # Compute products
    >>> v_squared = v.geometric_product(v)
    >>> print(v_squared.simplify())  # a^2 + b^2

The module gracefully handles the case where SymPy is not installed.
"""

# Check if SymPy is available
try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

if SYMPY_AVAILABLE:
    from lcc.symbolic.multivector import SymbolicMultivector, SymbolicAlgebra
    from lcc.symbolic.simplify import (
        simplify_ga,
        collect_blades,
        expand_products,
        apply_metric,
    )

    __all__ = [
        "SymbolicMultivector",
        "SymbolicAlgebra",
        "simplify_ga",
        "collect_blades",
        "expand_products",
        "apply_metric",
        "SYMPY_AVAILABLE",
    ]
else:
    # Provide stub that raises informative error
    def _sympy_not_available(*args, **kwargs):
        raise ImportError(
            "SymPy is required for symbolic computation. "
            "Install it with: pip install sympy"
        )

    class SymbolicMultivector:
        """Stub class when SymPy is not available."""
        def __init__(self, *args, **kwargs):
            _sympy_not_available()

    class SymbolicAlgebra:
        """Stub class when SymPy is not available."""
        def __init__(self, *args, **kwargs):
            _sympy_not_available()

    simplify_ga = _sympy_not_available
    collect_blades = _sympy_not_available
    expand_products = _sympy_not_available
    apply_metric = _sympy_not_available

    __all__ = [
        "SymbolicMultivector",
        "SymbolicAlgebra",
        "simplify_ga",
        "collect_blades",
        "expand_products",
        "apply_metric",
        "SYMPY_AVAILABLE",
    ]
