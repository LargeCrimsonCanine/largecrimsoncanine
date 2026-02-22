"""
Code generation module for LargeCrimsonCanine.

Generates optimized, unrolled code for specific geometric algebras.
Supports Python, Rust, and GLSL output formats.

Example usage:

    from lcc.codegen import Algebra, PythonGenerator

    # Create an algebra
    R3 = Algebra.euclidean(3)
    PGA3D = Algebra.pga(3)

    # Generate Python code
    gen = PythonGenerator()
    code = gen.generate_full_module(R3, "r3_optimized")

    # Generate Rust code
    from lcc.codegen import RustGenerator
    rust_gen = RustGenerator()
    rust_code = rust_gen.generate_full_module(PGA3D, "pga3d")
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Signature:
    """Metric signature for a Clifford algebra Cl(p,q,r)."""

    p: int  # Basis vectors squaring to +1
    q: int  # Basis vectors squaring to -1
    r: int  # Basis vectors squaring to 0 (degenerate)

    @property
    def dimension(self) -> int:
        """Total dimension (number of basis vectors)."""
        return self.p + self.q + self.r

    @property
    def num_blades(self) -> int:
        """Total number of basis blades (2^dimension)."""
        return 1 << self.dimension

    def basis_square(self, i: int) -> float:
        """What basis vector i squares to: +1, -1, or 0."""
        if i < self.p:
            return 1.0
        elif i < self.p + self.q:
            return -1.0
        else:
            return 0.0

    def __str__(self) -> str:
        return f"Cl({self.p},{self.q},{self.r})"


def blade_grade(index: int) -> int:
    """Get the grade of a blade by its binary index."""
    return bin(index).count('1')


def blade_name(index: int, dimension: int) -> str:
    """Get human-readable name for a blade (e.g., 'e12', 'e123')."""
    if index == 0:
        return "1"
    name = "e"
    for i in range(dimension):
        if (index >> i) & 1:
            name += str(i + 1)
    return name


def compute_reorder_sign(a: int, b: int) -> float:
    """
    Compute the sign from reordering basis vectors to canonical form.

    Counts transpositions needed to combine and sort basis vectors.
    """
    sign = 1.0
    b_remaining = b

    while b_remaining != 0:
        # Get lowest set bit
        lowest_b_bit = b_remaining & (-b_remaining)
        b_position = (lowest_b_bit - 1).bit_length()
        if lowest_b_bit == 0:
            b_position = 0
        else:
            b_position = lowest_b_bit.bit_length() - 1

        # Count bits in a greater than b_position
        higher_bits = a >> (b_position + 1)
        transpositions = bin(higher_bits).count('1')

        if transpositions % 2 == 1:
            sign = -sign

        b_remaining &= ~lowest_b_bit

    return sign


def blade_product(a: int, b: int, sig: Signature) -> Tuple[int, float]:
    """
    Compute the geometric product of two basis blades.

    Returns (result_blade, sign) where e_a * e_b = sign * e_result.
    """
    # Start with reordering sign
    sign = compute_reorder_sign(a, b)

    # Result blade is XOR (symmetric difference)
    result_blade = a ^ b

    # Apply metric: for each basis vector in both a and b,
    # multiply by that vector's square
    common = a & b
    for i in range(sig.dimension):
        if (common >> i) & 1:
            sign *= sig.basis_square(i)

    return result_blade, sign


class CayleyTable:
    """Pre-computed Cayley table for an algebra."""

    def __init__(self, sig: Signature):
        self.signature = sig
        self.num_blades = sig.num_blades

        # Compute full table
        size = self.num_blades
        self.blades = [[0] * size for _ in range(size)]
        self.signs = [[0.0] * size for _ in range(size)]

        for a in range(size):
            for b in range(size):
                blade, sign = blade_product(a, b, sig)
                self.blades[a][b] = blade
                self.signs[a][b] = sign

    def product(self, a: int, b: int) -> Tuple[int, float]:
        """Look up product of basis blades a and b."""
        return self.blades[a][b], self.signs[a][b]


class Algebra:
    """
    A geometric algebra Cl(p,q,r) with pre-computed product tables.

    This is the main class for code generation. It provides access to
    the Cayley table and all blade information needed to generate
    optimized code.
    """

    def __init__(self, p: int, q: int = 0, r: int = 0):
        """Create an algebra Cl(p,q,r)."""
        self.signature = Signature(p, q, r)
        self.cayley = CayleyTable(self.signature)

        # Pre-compute blade names
        dim = self.signature.dimension
        self.blade_names = [blade_name(i, dim) for i in range(self.num_blades)]

        # Pre-compute grade masks
        self.grade_masks = self._compute_grade_masks()

    def _compute_grade_masks(self) -> List[int]:
        """Compute bitmask for each grade."""
        masks = []
        for grade in range(self.dimension + 1):
            mask = 0
            for i in range(self.num_blades):
                if blade_grade(i) == grade:
                    mask |= (1 << i)
            masks.append(mask)
        return masks

    @classmethod
    def euclidean(cls, n: int) -> 'Algebra':
        """Create Euclidean algebra Cl(n,0,0)."""
        return cls(n, 0, 0)

    @classmethod
    def pga(cls, n: int) -> 'Algebra':
        """Create Projective Geometric Algebra Cl(n,0,1)."""
        return cls(n, 0, 1)

    @classmethod
    def cga(cls, n: int) -> 'Algebra':
        """Create Conformal Geometric Algebra Cl(n+1,1,0)."""
        return cls(n + 1, 1, 0)

    @classmethod
    def sta(cls) -> 'Algebra':
        """Create Spacetime Algebra Cl(1,3,0)."""
        return cls(1, 3, 0)

    @property
    def dimension(self) -> int:
        """Dimension of the underlying vector space."""
        return self.signature.dimension

    @property
    def num_blades(self) -> int:
        """Total number of basis blades."""
        return self.signature.num_blades

    @property
    def p(self) -> int:
        """Number of positive signature basis vectors."""
        return self.signature.p

    @property
    def q(self) -> int:
        """Number of negative signature basis vectors."""
        return self.signature.q

    @property
    def r(self) -> int:
        """Number of degenerate basis vectors."""
        return self.signature.r

    def product(self, a: int, b: int) -> Tuple[int, float]:
        """Get the product of basis blades a and b."""
        return self.cayley.product(a, b)

    def blade_name(self, index: int) -> str:
        """Get the name of a blade."""
        return self.blade_names[index]

    def blades_of_grade(self, grade: int) -> List[int]:
        """Get all blade indices of a given grade."""
        return [i for i in range(self.num_blades) if blade_grade(i) == grade]

    def reverse_sign(self, index: int) -> float:
        """Get the sign for the reverse operation on a blade."""
        grade = blade_grade(index)
        # (-1)^(k(k-1)/2)
        if grade % 4 < 2:
            return 1.0
        return -1.0

    def grade_involution_sign(self, index: int) -> float:
        """Get the sign for grade involution on a blade."""
        grade = blade_grade(index)
        return 1.0 if grade % 2 == 0 else -1.0

    def __str__(self) -> str:
        return f"Algebra({self.signature})"

    def __repr__(self) -> str:
        return f"Algebra({self.signature})"


# Import generators for convenience
from .python import PythonGenerator
from .rust import RustGenerator
from .glsl import GLSLGenerator

__all__ = [
    'Algebra',
    'Signature',
    'CayleyTable',
    'PythonGenerator',
    'RustGenerator',
    'GLSLGenerator',
    'blade_grade',
    'blade_name',
    'blade_product',
]
