"""JAX array integration for LargeCrimsonCanine.

This module provides JAX-backed multivector operations for ML and scientific computing.
Operations are JIT-compilable and support vmap for efficient vectorization.

Usage:
    >>> from lcc.jax import JaxMultivector
    >>> import jax.numpy as jnp
    >>> from largecrimsoncanine import Algebra
    >>>
    >>> R3 = Algebra.euclidean(3)
    >>> coeffs = jnp.zeros((100, 8))  # batch of 100 multivectors
    >>> batch = JaxMultivector(R3, coeffs)
    >>>
    >>> # Operations are JIT-compilable
    >>> import jax
    >>> @jax.jit
    >>> def rotate(batch, rotor):
    ...     return batch.sandwich(rotor)
"""

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

if JAX_AVAILABLE:
    from lcc.jax.multivector import JaxMultivector, JaxAlgebra
    __all__ = ["JaxMultivector", "JaxAlgebra", "JAX_AVAILABLE"]
else:
    __all__ = ["JAX_AVAILABLE"]

    def _jax_not_available(*args, **kwargs):
        raise ImportError(
            "JAX is not installed. Install it with: pip install jax jaxlib"
        )

    class JaxMultivector:
        """Placeholder class when JAX is not installed."""
        def __init__(self, *args, **kwargs):
            _jax_not_available()

    class JaxAlgebra:
        """Placeholder class when JAX is not installed."""
        def __init__(self, *args, **kwargs):
            _jax_not_available()
