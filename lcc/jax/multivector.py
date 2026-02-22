"""JAX-backed multivector operations for ML and scientific computing.

This module provides JaxMultivector, a class that stores multivector coefficients
as JAX arrays and implements geometric algebra operations in a way that is:
- JIT-compilable with @jax.jit
- Vectorizable with jax.vmap
- Differentiable with jax.grad

The implementation mirrors the NumPy-based MultivectorBatch but uses JAX's
functional programming model for XLA compilation and automatic differentiation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass
class JaxAlgebra:
    """Pure JAX representation of an algebra's Cayley tables.

    This stores the precomputed multiplication tables as JAX arrays,
    allowing JIT compilation of geometric algebra operations.

    Attributes:
        signature: Tuple (p, q, r) representing the metric signature.
        num_blades: Total number of basis blades (2^dimension).
        dimension: Dimension of the underlying vector space.
        cayley_signs: JAX array of signs for blade products.
        cayley_blades: JAX array of result blade indices for products.
    """
    signature: Tuple[int, int, int]
    num_blades: int
    dimension: int
    cayley_signs: Array  # Shape: (num_blades, num_blades)
    cayley_blades: Array  # Shape: (num_blades, num_blades), dtype=int32

    @classmethod
    def from_rust_algebra(cls, algebra) -> "JaxAlgebra":
        """Create a JaxAlgebra from a Rust-backed Algebra object.

        Args:
            algebra: A largecrimsoncanine.Algebra object.

        Returns:
            A JaxAlgebra with the same structure.

        Example:
            >>> from largecrimsoncanine import Algebra
            >>> R3 = Algebra.euclidean(3)
            >>> jax_alg = JaxAlgebra.from_rust_algebra(R3)
        """
        num_blades = algebra.num_blades()
        dimension = algebra.dimension()
        sig = algebra.signature()

        # Get Cayley table data from algebra
        # The Rust algebra provides flat arrays
        signs_flat = np.array(algebra.cayley_signs())
        blades_flat = np.array(algebra.cayley_blades(), dtype=np.int32)

        # Reshape to 2D for easier indexing
        cayley_signs = jnp.array(signs_flat.reshape(num_blades, num_blades))
        cayley_blades = jnp.array(blades_flat.reshape(num_blades, num_blades))

        return cls(
            signature=(sig.p, sig.q, sig.r),
            num_blades=num_blades,
            dimension=dimension,
            cayley_signs=cayley_signs,
            cayley_blades=cayley_blades,
        )

    @classmethod
    def euclidean(cls, n: int) -> "JaxAlgebra":
        """Create a Euclidean algebra Cl(n,0,0) directly.

        This builds the Cayley table in pure Python/JAX without
        requiring the Rust backend.

        Args:
            n: Dimension of the Euclidean space.

        Returns:
            A JaxAlgebra for Cl(n,0,0).
        """
        num_blades = 2 ** n

        # Build Cayley table
        signs = np.zeros((num_blades, num_blades), dtype=np.float64)
        blades = np.zeros((num_blades, num_blades), dtype=np.int32)

        for a in range(num_blades):
            for b in range(num_blades):
                blade, sign = _blade_product_euclidean(a, b)
                blades[a, b] = blade
                signs[a, b] = sign

        return cls(
            signature=(n, 0, 0),
            num_blades=num_blades,
            dimension=n,
            cayley_signs=jnp.array(signs),
            cayley_blades=jnp.array(blades),
        )


def _blade_product_euclidean(a: int, b: int) -> Tuple[int, float]:
    """Compute the geometric product of two basis blades in Euclidean metric.

    Uses the standard algorithm: count swaps needed to sort basis vectors,
    then contract matching pairs (each contributing +1 in Euclidean metric).

    Args:
        a: Binary index of first blade.
        b: Binary index of second blade.

    Returns:
        Tuple of (result_blade, sign).
    """
    # Result blade is XOR (symmetric difference of basis vectors)
    result = a ^ b

    # Count the sign from reordering
    sign = 1.0

    # For each bit in b, count how many bits in a are to the right
    # This counts the number of swaps needed
    temp_a = a
    temp_b = b

    while temp_b:
        # Isolate lowest bit of b
        lowest_b = temp_b & (-temp_b)

        # Count bits in a that are higher than lowest_b
        # These need to swap past the current b basis vector
        higher_a = temp_a & ~(lowest_b - 1) & ~lowest_b
        swaps = bin(higher_a).count('1')

        if swaps % 2 == 1:
            sign = -sign

        # Clear the processed bit
        temp_b &= temp_b - 1

    return result, sign


def _blade_grade(blade: int) -> int:
    """Get the grade of a blade (number of set bits)."""
    return bin(blade).count('1')


def _reverse_sign(grade: int) -> float:
    """Sign change under reversion: (-1)^(k(k-1)/2)."""
    k = grade
    return 1.0 if (k * (k - 1) // 2) % 2 == 0 else -1.0


def _grade_involution_sign(grade: int) -> float:
    """Sign change under grade involution: (-1)^k."""
    return 1.0 if grade % 2 == 0 else -1.0


class JaxMultivector:
    """A batch of multivectors backed by JAX arrays.

    This class stores multivector coefficients as JAX arrays with shape
    (batch_size, num_blades) and provides geometric algebra operations
    that are JIT-compilable and differentiable.

    Attributes:
        algebra: The JaxAlgebra defining the multiplication structure.
        coeffs: JAX array of shape (batch_size, num_blades) containing
            the coefficients for each blade of each multivector.

    Example:
        >>> from largecrimsoncanine import Algebra
        >>> from lcc.jax import JaxMultivector, JaxAlgebra
        >>> import jax.numpy as jnp
        >>>
        >>> R3 = Algebra.euclidean(3)
        >>> jax_alg = JaxAlgebra.from_rust_algebra(R3)
        >>>
        >>> # Create a batch of 100 random multivectors
        >>> coeffs = jax.random.normal(jax.random.PRNGKey(0), (100, 8))
        >>> batch = JaxMultivector(jax_alg, coeffs)
        >>>
        >>> # Operations are JIT-compilable
        >>> @jax.jit
        >>> def compute(batch):
        ...     return batch.geometric_product(batch).norm()
    """

    def __init__(self, algebra: JaxAlgebra, coeffs: Array):
        """Create a JaxMultivector from algebra and coefficients.

        Args:
            algebra: A JaxAlgebra defining the multiplication tables.
            coeffs: JAX array of shape (batch_size, num_blades) or
                (num_blades,) for a single multivector.
        """
        self.algebra = algebra

        # Ensure coeffs is 2D
        if coeffs.ndim == 1:
            coeffs = coeffs[jnp.newaxis, :]

        if coeffs.shape[-1] != algebra.num_blades:
            raise ValueError(
                f"coeffs must have {algebra.num_blades} columns for "
                f"algebra with signature {algebra.signature}, "
                f"got {coeffs.shape[-1]}"
            )

        self.coeffs = coeffs

    @property
    def batch_size(self) -> int:
        """Number of multivectors in the batch."""
        return self.coeffs.shape[0]

    @property
    def num_blades(self) -> int:
        """Number of basis blades per multivector."""
        return self.algebra.num_blades

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the coefficient array (batch_size, num_blades)."""
        return self.coeffs.shape

    # =========================================================================
    # Construction methods
    # =========================================================================

    @classmethod
    def from_numpy(cls, algebra: JaxAlgebra, coeffs: np.ndarray) -> "JaxMultivector":
        """Create a JaxMultivector from a NumPy array.

        Args:
            algebra: A JaxAlgebra.
            coeffs: NumPy array of shape (batch_size, num_blades).

        Returns:
            A JaxMultivector wrapping the converted array.
        """
        return cls(algebra, jnp.array(coeffs))

    def to_numpy(self) -> np.ndarray:
        """Convert coefficients to a NumPy array.

        Returns:
            NumPy array of shape (batch_size, num_blades).
        """
        return np.array(self.coeffs)

    @classmethod
    def from_vectors(cls, algebra: JaxAlgebra, coords: Array) -> "JaxMultivector":
        """Create a batch of vectors from coordinate array.

        Args:
            algebra: A JaxAlgebra.
            coords: Array of shape (batch_size, dimension).

        Returns:
            A JaxMultivector containing vectors (grade-1 elements).

        Example:
            >>> jax_alg = JaxAlgebra.euclidean(3)
            >>> coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            >>> vectors = JaxMultivector.from_vectors(jax_alg, coords)
        """
        if coords.ndim == 1:
            coords = coords[jnp.newaxis, :]

        batch_size = coords.shape[0]
        dim = coords.shape[1]

        if dim != algebra.dimension:
            raise ValueError(
                f"coords must have {algebra.dimension} columns for "
                f"algebra with signature {algebra.signature}, "
                f"got {dim}"
            )

        # Build coefficient array: vectors have coefficients at indices 1, 2, 4, 8, ...
        coeffs = jnp.zeros((batch_size, algebra.num_blades))

        for i in range(dim):
            blade_idx = 1 << i  # e1=1, e2=2, e3=4, etc.
            coeffs = coeffs.at[:, blade_idx].set(coords[:, i])

        return cls(algebra, coeffs)

    @classmethod
    def from_scalars(cls, algebra: JaxAlgebra, values: Array) -> "JaxMultivector":
        """Create a batch of scalars from a 1D array.

        Args:
            algebra: A JaxAlgebra.
            values: Array of shape (batch_size,).

        Returns:
            A JaxMultivector containing scalars (grade-0 elements).
        """
        if values.ndim == 0:
            values = values[jnp.newaxis]

        batch_size = values.shape[0]
        coeffs = jnp.zeros((batch_size, algebra.num_blades))
        coeffs = coeffs.at[:, 0].set(values)

        return cls(algebra, coeffs)

    @classmethod
    def from_bivectors(cls, algebra: JaxAlgebra, biv_coeffs: Array) -> "JaxMultivector":
        """Create a batch of bivectors from coefficient array.

        Args:
            algebra: A JaxAlgebra.
            biv_coeffs: Array of shape (batch_size, num_bivectors).

        Returns:
            A JaxMultivector containing bivectors (grade-2 elements).
        """
        if biv_coeffs.ndim == 1:
            biv_coeffs = biv_coeffs[jnp.newaxis, :]

        batch_size = biv_coeffs.shape[0]
        dim = algebra.dimension
        num_bivectors = dim * (dim - 1) // 2

        if biv_coeffs.shape[1] != num_bivectors:
            raise ValueError(
                f"biv_coeffs must have {num_bivectors} columns for "
                f"algebra with signature {algebra.signature}, "
                f"got {biv_coeffs.shape[1]}"
            )

        # Get bivector blade indices
        bivector_indices = [i for i in range(algebra.num_blades) if _blade_grade(i) == 2]

        coeffs = jnp.zeros((batch_size, algebra.num_blades))
        for col, blade_idx in enumerate(bivector_indices):
            coeffs = coeffs.at[:, blade_idx].set(biv_coeffs[:, col])

        return cls(algebra, coeffs)

    def to_vector_coords(self) -> Array:
        """Extract vector coordinates from the batch.

        Returns:
            Array of shape (batch_size, dimension) containing vector components.
        """
        dim = self.algebra.dimension
        coords = jnp.zeros((self.batch_size, dim))

        for i in range(dim):
            blade_idx = 1 << i
            coords = coords.at[:, i].set(self.coeffs[:, blade_idx])

        return coords

    def scalar(self) -> Array:
        """Extract scalar (grade-0) parts.

        Returns:
            Array of shape (batch_size,) containing scalar components.
        """
        return self.coeffs[:, 0]

    # =========================================================================
    # Geometric products
    # =========================================================================

    def geometric_product(self, other: "JaxMultivector") -> "JaxMultivector":
        """Compute the geometric product: self * other.

        Supports broadcasting: if one operand has batch_size 1, it is
        broadcast to match the other.

        Args:
            other: Another JaxMultivector.

        Returns:
            The geometric product as a new JaxMultivector.
        """
        return JaxMultivector(
            self.algebra,
            _geometric_product_impl(
                self.coeffs, other.coeffs,
                self.algebra.cayley_signs,
                self.algebra.cayley_blades,
            )
        )

    def gp(self, other: "JaxMultivector") -> "JaxMultivector":
        """Alias for geometric_product."""
        return self.geometric_product(other)

    def outer_product(self, other: "JaxMultivector") -> "JaxMultivector":
        """Compute the outer (wedge) product: self ^ other.

        Args:
            other: Another JaxMultivector.

        Returns:
            The outer product as a new JaxMultivector.
        """
        return JaxMultivector(
            self.algebra,
            _outer_product_impl(
                self.coeffs, other.coeffs,
                self.algebra.cayley_signs,
                self.algebra.cayley_blades,
            )
        )

    def wedge(self, other: "JaxMultivector") -> "JaxMultivector":
        """Alias for outer_product."""
        return self.outer_product(other)

    def inner_product(self, other: "JaxMultivector") -> "JaxMultivector":
        """Compute the left contraction (inner product): self . other.

        Args:
            other: Another JaxMultivector.

        Returns:
            The left contraction as a new JaxMultivector.
        """
        return JaxMultivector(
            self.algebra,
            _inner_product_impl(
                self.coeffs, other.coeffs,
                self.algebra.cayley_signs,
                self.algebra.cayley_blades,
            )
        )

    def lc(self, other: "JaxMultivector") -> "JaxMultivector":
        """Alias for inner_product (left contraction)."""
        return self.inner_product(other)

    # =========================================================================
    # Sandwich product (transformations)
    # =========================================================================

    def sandwich(self, rotor: "JaxMultivector") -> "JaxMultivector":
        """Apply a rotor to all multivectors: rotor * self * ~rotor.

        This is the standard way to apply rotations/transformations in GA.

        Args:
            rotor: A JaxMultivector representing a rotor (should be normalized).

        Returns:
            The transformed multivectors as a new JaxMultivector.

        Example:
            >>> # Create a rotor for rotation around Z axis
            >>> angle = jnp.pi / 4
            >>> rotor_coeffs = jnp.array([[jnp.cos(angle/2), 0, 0, -jnp.sin(angle/2), 0, 0, 0, 0]])
            >>> rotor = JaxMultivector(algebra, rotor_coeffs)
            >>> rotated = vectors.sandwich(rotor)
        """
        return JaxMultivector(
            self.algebra,
            _sandwich_impl(
                self.coeffs, rotor.coeffs,
                self.algebra.cayley_signs,
                self.algebra.cayley_blades,
                self.algebra.num_blades,
            )
        )

    def apply(self, rotor: "JaxMultivector") -> "JaxMultivector":
        """Alias for sandwich (common in robotics literature)."""
        return self.sandwich(rotor)

    # =========================================================================
    # Involutions
    # =========================================================================

    def reverse(self) -> "JaxMultivector":
        """Compute the reverse of each multivector.

        The reverse negates blades of grade k by (-1)^(k(k-1)/2).

        Returns:
            A new JaxMultivector with reversed elements.
        """
        return JaxMultivector(
            self.algebra,
            _reverse_impl(self.coeffs, self.algebra.num_blades)
        )

    def grade_involution(self) -> "JaxMultivector":
        """Compute the grade involution of each multivector.

        Grade involution negates all odd-grade components.

        Returns:
            A new JaxMultivector with grade-inverted elements.
        """
        return JaxMultivector(
            self.algebra,
            _grade_involution_impl(self.coeffs, self.algebra.num_blades)
        )

    # =========================================================================
    # Norms
    # =========================================================================

    def norm_squared(self) -> Array:
        """Compute the squared norm of each multivector.

        The norm squared is the scalar part of self * ~self.

        Returns:
            Array of shape (batch_size,) containing squared norms.
        """
        return _norm_squared_impl(
            self.coeffs,
            self.algebra.cayley_signs,
            self.algebra.cayley_blades,
            self.algebra.num_blades,
        )

    def norm(self) -> Array:
        """Compute the norm of each multivector.

        Returns:
            Array of shape (batch_size,) containing norms.
        """
        return jnp.sqrt(jnp.abs(self.norm_squared()))

    def normalized(self) -> "JaxMultivector":
        """Return normalized copies of all multivectors.

        Multivectors with zero norm are left unchanged.

        Returns:
            A new JaxMultivector with normalized elements.
        """
        norms = self.norm()
        # Avoid division by zero
        safe_norms = jnp.where(norms > 1e-14, norms, 1.0)
        return JaxMultivector(
            self.algebra,
            self.coeffs / safe_norms[:, jnp.newaxis]
        )

    # =========================================================================
    # Grade extraction
    # =========================================================================

    def grade(self, k: int) -> "JaxMultivector":
        """Extract grade-k parts.

        Args:
            k: The grade to extract.

        Returns:
            A new JaxMultivector containing only grade-k components.
        """
        mask = jnp.array([1.0 if _blade_grade(i) == k else 0.0
                          for i in range(self.algebra.num_blades)])
        return JaxMultivector(self.algebra, self.coeffs * mask)

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: "JaxMultivector") -> "JaxMultivector":
        """Add two multivector batches element-wise."""
        return JaxMultivector(self.algebra, self.coeffs + other.coeffs)

    def __sub__(self, other: "JaxMultivector") -> "JaxMultivector":
        """Subtract two multivector batches element-wise."""
        return JaxMultivector(self.algebra, self.coeffs - other.coeffs)

    def __neg__(self) -> "JaxMultivector":
        """Negate all multivectors."""
        return JaxMultivector(self.algebra, -self.coeffs)

    def __mul__(self, scalar: float) -> "JaxMultivector":
        """Scalar multiplication."""
        return JaxMultivector(self.algebra, self.coeffs * scalar)

    def __rmul__(self, scalar: float) -> "JaxMultivector":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "JaxMultivector":
        """Scalar division."""
        return JaxMultivector(self.algebra, self.coeffs / scalar)

    # =========================================================================
    # Indexing and slicing
    # =========================================================================

    def __getitem__(self, idx) -> "JaxMultivector":
        """Index or slice the batch."""
        result = self.coeffs[idx]
        if result.ndim == 1:
            result = result[jnp.newaxis, :]
        return JaxMultivector(self.algebra, result)

    def __len__(self) -> int:
        """Return batch size."""
        return self.batch_size

    # =========================================================================
    # String representation
    # =========================================================================

    def __repr__(self) -> str:
        return f"JaxMultivector(batch_size={self.batch_size}, signature={self.algebra.signature})"


# =============================================================================
# JIT-compiled implementation functions
# =============================================================================

@jax.jit
def _geometric_product_impl(
    a_coeffs: Array,
    b_coeffs: Array,
    cayley_signs: Array,
    cayley_blades: Array,
) -> Array:
    """JIT-compiled geometric product implementation.

    Uses einsum-style operations for efficient GPU execution.
    """
    # Handle broadcasting
    if a_coeffs.shape[0] == 1:
        a_coeffs = jnp.broadcast_to(a_coeffs, b_coeffs.shape)
    elif b_coeffs.shape[0] == 1:
        b_coeffs = jnp.broadcast_to(b_coeffs, a_coeffs.shape)

    batch_size = a_coeffs.shape[0]
    num_blades = a_coeffs.shape[1]

    # Compute all pairwise products: (batch, i, j) -> a[i] * b[j] * sign[i,j]
    # a_expanded: (batch, num_blades, 1)
    # b_expanded: (batch, 1, num_blades)
    a_expanded = a_coeffs[:, :, jnp.newaxis]
    b_expanded = b_coeffs[:, jnp.newaxis, :]

    # Product contributions: (batch, num_blades, num_blades)
    products = a_expanded * b_expanded * cayley_signs

    # Scatter-add to result blades
    # We need to accumulate products[batch, i, j] into result[batch, cayley_blades[i,j]]
    result = jnp.zeros((batch_size, num_blades))

    # Use segment_sum-like operation via scatter_add
    for i in range(num_blades):
        for j in range(num_blades):
            blade_idx = cayley_blades[i, j]
            result = result.at[:, blade_idx].add(products[:, i, j])

    return result


@jax.jit
def _outer_product_impl(
    a_coeffs: Array,
    b_coeffs: Array,
    cayley_signs: Array,
    cayley_blades: Array,
) -> Array:
    """JIT-compiled outer product implementation.

    The outer product is zero when blades share a basis vector (i & j != 0).
    """
    # Handle broadcasting
    if a_coeffs.shape[0] == 1:
        a_coeffs = jnp.broadcast_to(a_coeffs, b_coeffs.shape)
    elif b_coeffs.shape[0] == 1:
        b_coeffs = jnp.broadcast_to(b_coeffs, a_coeffs.shape)

    batch_size = a_coeffs.shape[0]
    num_blades = a_coeffs.shape[1]

    # Create mask for outer product (only where blades don't share vectors)
    outer_mask = jnp.array([[1.0 if (i & j) == 0 else 0.0
                             for j in range(num_blades)]
                            for i in range(num_blades)])

    a_expanded = a_coeffs[:, :, jnp.newaxis]
    b_expanded = b_coeffs[:, jnp.newaxis, :]

    products = a_expanded * b_expanded * cayley_signs * outer_mask

    result = jnp.zeros((batch_size, num_blades))
    for i in range(num_blades):
        for j in range(num_blades):
            blade_idx = cayley_blades[i, j]
            result = result.at[:, blade_idx].add(products[:, i, j])

    return result


@jax.jit
def _inner_product_impl(
    a_coeffs: Array,
    b_coeffs: Array,
    cayley_signs: Array,
    cayley_blades: Array,
) -> Array:
    """JIT-compiled left contraction (inner product) implementation.

    Left contraction A . B is non-zero only when:
    - grade(A) <= grade(B)
    - A is a "subset" of B (all basis vectors of A are in B)
    - result grade = grade(B) - grade(A)
    """
    # Handle broadcasting
    if a_coeffs.shape[0] == 1:
        a_coeffs = jnp.broadcast_to(a_coeffs, b_coeffs.shape)
    elif b_coeffs.shape[0] == 1:
        b_coeffs = jnp.broadcast_to(b_coeffs, a_coeffs.shape)

    batch_size = a_coeffs.shape[0]
    num_blades = a_coeffs.shape[1]

    # Create mask for left contraction
    def lc_mask_entry(i, j):
        grade_i = _blade_grade(i)
        grade_j = _blade_grade(j)
        if grade_i > grade_j:
            return 0.0
        if (i & j) != i:  # i must be subset of j
            return 0.0
        result_blade = i ^ j
        result_grade = _blade_grade(result_blade)
        if result_grade != grade_j - grade_i:
            return 0.0
        return 1.0

    lc_mask = jnp.array([[lc_mask_entry(i, j)
                          for j in range(num_blades)]
                         for i in range(num_blades)])

    a_expanded = a_coeffs[:, :, jnp.newaxis]
    b_expanded = b_coeffs[:, jnp.newaxis, :]

    products = a_expanded * b_expanded * cayley_signs * lc_mask

    result = jnp.zeros((batch_size, num_blades))
    for i in range(num_blades):
        for j in range(num_blades):
            blade_idx = cayley_blades[i, j]
            result = result.at[:, blade_idx].add(products[:, i, j])

    return result


@jax.jit
def _reverse_impl(coeffs: Array, num_blades: int) -> Array:
    """JIT-compiled reverse implementation."""
    signs = jnp.array([_reverse_sign(_blade_grade(i)) for i in range(num_blades)])
    return coeffs * signs


@jax.jit
def _grade_involution_impl(coeffs: Array, num_blades: int) -> Array:
    """JIT-compiled grade involution implementation."""
    signs = jnp.array([_grade_involution_sign(_blade_grade(i)) for i in range(num_blades)])
    return coeffs * signs


@jax.jit
def _sandwich_impl(
    x_coeffs: Array,
    rotor_coeffs: Array,
    cayley_signs: Array,
    cayley_blades: Array,
    num_blades: int,
) -> Array:
    """JIT-compiled sandwich product: rotor * x * ~rotor.

    Computes R * X * R~ for each X in the batch.
    """
    # Broadcast rotor if needed
    if rotor_coeffs.shape[0] == 1:
        rotor_coeffs = jnp.broadcast_to(rotor_coeffs, x_coeffs.shape)

    batch_size = x_coeffs.shape[0]

    # Compute rotor reverse
    rev_signs = jnp.array([_reverse_sign(_blade_grade(i)) for i in range(num_blades)])
    rotor_rev = rotor_coeffs * rev_signs

    # First compute R * X
    rx = _geometric_product_impl(rotor_coeffs, x_coeffs, cayley_signs, cayley_blades)

    # Then compute (R * X) * ~R
    result = _geometric_product_impl(rx, rotor_rev, cayley_signs, cayley_blades)

    return result


@jax.jit
def _norm_squared_impl(
    coeffs: Array,
    cayley_signs: Array,
    cayley_blades: Array,
    num_blades: int,
) -> Array:
    """JIT-compiled norm squared: scalar part of x * ~x."""
    # Compute reverse
    rev_signs = jnp.array([_reverse_sign(_blade_grade(i)) for i in range(num_blades)])
    coeffs_rev = coeffs * rev_signs

    # Compute product and extract scalar
    product = _geometric_product_impl(coeffs, coeffs_rev, cayley_signs, cayley_blades)

    return product[:, 0]


# =============================================================================
# Vectorized versions for vmap
# =============================================================================

def vmap_geometric_product(
    algebra: JaxAlgebra,
    a_batch: Array,
    b_single: Array,
) -> Array:
    """Apply geometric product to batch using vmap.

    This is an alternative to the batched implementation that uses
    vmap for potentially better XLA optimization.

    Args:
        algebra: A JaxAlgebra.
        a_batch: Array of shape (batch_size, num_blades).
        b_single: Array of shape (num_blades,) - single multivector.

    Returns:
        Array of shape (batch_size, num_blades).
    """
    def single_product(a):
        return _geometric_product_impl(
            a[jnp.newaxis, :],
            b_single[jnp.newaxis, :],
            algebra.cayley_signs,
            algebra.cayley_blades,
        )[0]

    return jax.vmap(single_product)(a_batch)
