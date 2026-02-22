"""Symbolic multivector implementation using SymPy.

This module provides SymbolicMultivector and SymbolicAlgebra classes for
symbolic manipulation of geometric algebra expressions.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import lru_cache

import sympy as sp
from sympy import Symbol, Expr, S, simplify, expand, latex, diff, sqrt


class SymbolicSignature:
    """Metric signature for a symbolic Clifford algebra Cl(p, q, r).

    The signature supports both concrete and symbolic parameters:
    - p basis vectors square to +1 (positive/spacelike)
    - q basis vectors square to -1 (negative/timelike)
    - r basis vectors square to 0 (degenerate/null)

    Parameters can be SymPy symbols for fully symbolic algebras.
    """

    def __init__(
        self,
        p: Union[int, Expr],
        q: Union[int, Expr] = 0,
        r: Union[int, Expr] = 0,
        metric: Optional[List[Expr]] = None,
    ):
        """Create a metric signature.

        Args:
            p: Number of basis vectors squaring to +1 (or symbolic)
            q: Number of basis vectors squaring to -1 (or symbolic)
            r: Number of basis vectors squaring to 0 (or symbolic)
            metric: Optional explicit metric list [e1^2, e2^2, ...]. If provided,
                    overrides p, q, r for computing basis vector squares.
        """
        self.p = p if isinstance(p, Expr) else S(p)
        self.q = q if isinstance(q, Expr) else S(q)
        self.r = r if isinstance(r, Expr) else S(r)
        self._metric = metric

    @property
    def dimension(self) -> Expr:
        """Total dimension of the vector space."""
        return self.p + self.q + self.r

    @property
    def is_concrete(self) -> bool:
        """Check if the signature has concrete (non-symbolic) dimensions."""
        dim = self.dimension
        return dim.is_Integer and int(dim) >= 0

    @property
    def concrete_dimension(self) -> int:
        """Get concrete dimension. Raises if symbolic."""
        if not self.is_concrete:
            raise ValueError(
                f"Signature {self} has symbolic dimension, cannot get concrete value"
            )
        return int(self.dimension)

    def num_blades(self) -> Expr:
        """Number of basis blades (2^dimension)."""
        return S(2) ** self.dimension

    def basis_square(self, i: int) -> Expr:
        """Get what basis vector i squares to: +1, -1, or 0.

        Uses the explicit metric if provided, otherwise computes from p, q, r.
        """
        if self._metric is not None:
            if i < len(self._metric):
                return self._metric[i]
            raise IndexError(f"Basis index {i} out of range for metric of length {len(self._metric)}")

        # Symbolic case: need concrete p, q values to determine
        if not (self.p.is_Integer and self.q.is_Integer and self.r.is_Integer):
            # Return symbolic representation
            p_val, q_val = Symbol('p'), Symbol('q')
            # This is a simplification - for fully symbolic, we'd need piecewise
            raise ValueError(
                "Cannot determine basis_square for fully symbolic signature. "
                "Provide explicit metric list or use concrete p, q, r."
            )

        p_int = int(self.p)
        q_int = int(self.q)

        if i < p_int:
            return S(1)
        elif i < p_int + q_int:
            return S(-1)
        else:
            return S(0)

    @classmethod
    def euclidean(cls, n: Union[int, Expr]) -> SymbolicSignature:
        """Create Euclidean signature Cl(n, 0, 0)."""
        return cls(n, 0, 0)

    @classmethod
    def pga(cls, n: Union[int, Expr]) -> SymbolicSignature:
        """Create PGA signature Cl(n, 0, 1)."""
        return cls(n, 0, 1)

    @classmethod
    def cga(cls, n: Union[int, Expr]) -> SymbolicSignature:
        """Create CGA signature Cl(n+1, 1, 0)."""
        n_expr = n if isinstance(n, Expr) else S(n)
        return cls(n_expr + 1, 1, 0)

    @classmethod
    def sta(cls) -> SymbolicSignature:
        """Create Spacetime Algebra signature Cl(1, 3, 0)."""
        return cls(1, 3, 0)

    def __repr__(self) -> str:
        return f"Cl({self.p}, {self.q}, {self.r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolicSignature):
            return False
        return (
            sp.simplify(self.p - other.p) == 0 and
            sp.simplify(self.q - other.q) == 0 and
            sp.simplify(self.r - other.r) == 0
        )

    def __hash__(self) -> int:
        # Use string representation for hashing symbolic expressions
        return hash((str(self.p), str(self.q), str(self.r)))


class SymbolicAlgebra:
    """A symbolic Clifford algebra with precomputed Cayley table.

    The algebra caches sign computations for efficient product evaluation.
    """

    def __init__(self, signature: SymbolicSignature):
        """Create an algebra from a signature."""
        self.signature = signature
        self._cayley_cache: Dict[Tuple[int, int], Tuple[int, Expr]] = {}

    @classmethod
    def euclidean(cls, n: int) -> SymbolicAlgebra:
        """Create n-dimensional Euclidean algebra."""
        return cls(SymbolicSignature.euclidean(n))

    @classmethod
    def pga(cls, n: int) -> SymbolicAlgebra:
        """Create n-dimensional Projective Geometric Algebra."""
        return cls(SymbolicSignature.pga(n))

    @classmethod
    def cga(cls, n: int) -> SymbolicAlgebra:
        """Create n-dimensional Conformal Geometric Algebra."""
        return cls(SymbolicSignature.cga(n))

    @classmethod
    def sta(cls) -> SymbolicAlgebra:
        """Create Spacetime Algebra Cl(1,3,0)."""
        return cls(SymbolicSignature.sta())

    @classmethod
    def from_metric(cls, metric: List[Union[int, Expr]]) -> SymbolicAlgebra:
        """Create algebra from explicit metric list.

        Args:
            metric: List of what each basis vector squares to.
                    Example: [1, 1, 1] for 3D Euclidean
                             [1, 1, 1, 0] for PGA3D
                             [1, -1, -1, -1] for STA
        """
        metric_exprs = [m if isinstance(m, Expr) else S(m) for m in metric]
        n = len(metric)
        # Count p, q, r from metric
        p = sum(1 for m in metric_exprs if m == S(1))
        q = sum(1 for m in metric_exprs if m == S(-1))
        r = sum(1 for m in metric_exprs if m == S(0))
        sig = SymbolicSignature(p, q, r, metric=metric_exprs)
        return cls(sig)

    @property
    def dimension(self) -> int:
        """Concrete dimension of the algebra."""
        return self.signature.concrete_dimension

    @property
    def num_blades(self) -> int:
        """Number of basis blades."""
        return 2 ** self.dimension

    def _compute_reorder_sign(self, a: int, b: int) -> Expr:
        """Compute sign from reordering basis vectors to canonical form.

        When computing e_a * e_b, count transpositions needed to reorder.
        """
        sign = S(1)
        b_remaining = b

        while b_remaining != 0:
            # Get lowest set bit
            lowest_bit = b_remaining & (-b_remaining)
            b_position = lowest_bit.bit_length() - 1

            # Count bits in a that are strictly greater than b_position
            higher_bits = a >> (b_position + 1)
            transpositions = bin(higher_bits).count('1')

            if transpositions % 2 == 1:
                sign = -sign

            b_remaining &= ~lowest_bit

        return sign

    def blade_product(self, a: int, b: int) -> Tuple[int, Expr]:
        """Compute product of two basis blades.

        Returns (result_blade, sign) where e_a * e_b = sign * e_result.
        """
        if (a, b) in self._cayley_cache:
            return self._cayley_cache[(a, b)]

        # Result blade is XOR (symmetric difference)
        result_blade = a ^ b

        # Start with reordering sign
        sign = self._compute_reorder_sign(a, b)

        # Apply metric: for each basis vector in both a and b, multiply by its square
        common = a & b
        for i in range(self.dimension):
            if (common >> i) & 1:
                sign = sign * self.signature.basis_square(i)

        self._cayley_cache[(a, b)] = (result_blade, sign)
        return result_blade, sign

    def basis_vectors(self) -> List[SymbolicMultivector]:
        """Create all basis vectors e1, e2, ..., en."""
        return [self.basis(i + 1) for i in range(self.dimension)]

    def basis(self, index: int) -> SymbolicMultivector:
        """Create basis vector e_i (1-indexed)."""
        if index < 1 or index > self.dimension:
            raise ValueError(f"Basis index must be 1-{self.dimension}, got {index}")
        coeffs = {1 << (index - 1): S(1)}
        return SymbolicMultivector(coeffs, self)

    def scalar(self, value: Union[int, float, Expr] = 1) -> SymbolicMultivector:
        """Create scalar multivector."""
        val = value if isinstance(value, Expr) else S(value)
        return SymbolicMultivector({0: val}, self)

    def zero(self) -> SymbolicMultivector:
        """Create zero multivector."""
        return SymbolicMultivector({}, self)

    def pseudoscalar(self) -> SymbolicMultivector:
        """Create unit pseudoscalar."""
        ps_index = self.num_blades - 1
        return SymbolicMultivector({ps_index: S(1)}, self)

    def blade_name(self, index: int) -> str:
        """Get human-readable name for a blade index."""
        if index == 0:
            return "1"
        name = "e"
        for i in range(self.dimension):
            if (index >> i) & 1:
                name += str(i + 1)
        return name

    def blade_grade(self, index: int) -> int:
        """Get grade of a blade from its index."""
        return bin(index).count('1')

    def __repr__(self) -> str:
        return f"SymbolicAlgebra({self.signature})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolicAlgebra):
            return False
        return self.signature == other.signature


class SymbolicMultivector:
    """A symbolic multivector with SymPy expression coefficients.

    Stores coefficients as a sparse dictionary mapping blade indices to
    SymPy expressions. Supports all standard GA operations.
    """

    def __init__(
        self,
        coeffs: Dict[int, Expr],
        algebra: SymbolicAlgebra,
    ):
        """Create a symbolic multivector.

        Args:
            coeffs: Dictionary mapping blade indices to SymPy expressions.
                    Zero coefficients may be omitted.
            algebra: The algebra this multivector belongs to.
        """
        # Filter out zero coefficients
        self._coeffs = {k: v for k, v in coeffs.items() if v != S(0)}
        self._algebra = algebra

    @property
    def algebra(self) -> SymbolicAlgebra:
        """The algebra this multivector belongs to."""
        return self._algebra

    @property
    def coefficients(self) -> Dict[int, Expr]:
        """Dictionary of non-zero coefficients."""
        return self._coeffs.copy()

    def __getitem__(self, blade_index: int) -> Expr:
        """Get coefficient for a blade index."""
        return self._coeffs.get(blade_index, S(0))

    def __setitem__(self, blade_index: int, value: Union[int, float, Expr]):
        """Set coefficient for a blade index."""
        val = value if isinstance(value, Expr) else S(value)
        if val == S(0):
            self._coeffs.pop(blade_index, None)
        else:
            self._coeffs[blade_index] = val

    # =========================================================================
    # Accessors
    # =========================================================================

    def scalar_part(self) -> Expr:
        """Extract scalar (grade-0) coefficient."""
        return self._coeffs.get(0, S(0))

    def grade(self, k: int) -> SymbolicMultivector:
        """Extract grade-k part of the multivector."""
        new_coeffs = {
            idx: coeff
            for idx, coeff in self._coeffs.items()
            if self._algebra.blade_grade(idx) == k
        }
        return SymbolicMultivector(new_coeffs, self._algebra)

    def grades(self) -> List[int]:
        """Get list of grades present in this multivector."""
        return sorted(set(
            self._algebra.blade_grade(idx) for idx in self._coeffs.keys()
        ))

    def is_scalar(self) -> bool:
        """Check if this is a pure scalar."""
        return all(self._algebra.blade_grade(idx) == 0 for idx in self._coeffs)

    def is_vector(self) -> bool:
        """Check if this is a pure vector (grade 1)."""
        return all(self._algebra.blade_grade(idx) == 1 for idx in self._coeffs)

    def is_bivector(self) -> bool:
        """Check if this is a pure bivector (grade 2)."""
        return all(self._algebra.blade_grade(idx) == 2 for idx in self._coeffs)

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: Union[SymbolicMultivector, int, float, Expr]) -> SymbolicMultivector:
        """Add two multivectors or a scalar."""
        if isinstance(other, (int, float, Expr)):
            other = self._algebra.scalar(other)

        if not isinstance(other, SymbolicMultivector):
            return NotImplemented

        if self._algebra != other._algebra:
            raise ValueError("Cannot add multivectors from different algebras")

        new_coeffs = self._coeffs.copy()
        for idx, coeff in other._coeffs.items():
            if idx in new_coeffs:
                new_coeffs[idx] = new_coeffs[idx] + coeff
            else:
                new_coeffs[idx] = coeff

        return SymbolicMultivector(new_coeffs, self._algebra)

    def __radd__(self, other: Union[int, float, Expr]) -> SymbolicMultivector:
        """Right addition with scalar."""
        return self.__add__(other)

    def __sub__(self, other: Union[SymbolicMultivector, int, float, Expr]) -> SymbolicMultivector:
        """Subtract multivectors or a scalar."""
        if isinstance(other, (int, float, Expr)):
            other = self._algebra.scalar(other)

        if not isinstance(other, SymbolicMultivector):
            return NotImplemented

        return self + (-other)

    def __rsub__(self, other: Union[int, float, Expr]) -> SymbolicMultivector:
        """Right subtraction with scalar."""
        return (-self) + other

    def __neg__(self) -> SymbolicMultivector:
        """Negate multivector."""
        new_coeffs = {idx: -coeff for idx, coeff in self._coeffs.items()}
        return SymbolicMultivector(new_coeffs, self._algebra)

    def __mul__(self, other: Union[SymbolicMultivector, int, float, Expr]) -> SymbolicMultivector:
        """Geometric product (default multiplication)."""
        if isinstance(other, (int, float, Expr)):
            # Scalar multiplication
            val = other if isinstance(other, Expr) else S(other)
            new_coeffs = {idx: coeff * val for idx, coeff in self._coeffs.items()}
            return SymbolicMultivector(new_coeffs, self._algebra)

        return self.geometric_product(other)

    def __rmul__(self, other: Union[int, float, Expr]) -> SymbolicMultivector:
        """Right multiplication with scalar."""
        val = other if isinstance(other, Expr) else S(other)
        new_coeffs = {idx: val * coeff for idx, coeff in self._coeffs.items()}
        return SymbolicMultivector(new_coeffs, self._algebra)

    def __truediv__(self, other: Union[int, float, Expr]) -> SymbolicMultivector:
        """Division by scalar."""
        val = other if isinstance(other, Expr) else S(other)
        new_coeffs = {idx: coeff / val for idx, coeff in self._coeffs.items()}
        return SymbolicMultivector(new_coeffs, self._algebra)

    def __pow__(self, n: int) -> SymbolicMultivector:
        """Raise to integer power using geometric product."""
        if n < 0:
            raise ValueError("Negative powers require inverse, not yet implemented")
        if n == 0:
            return self._algebra.scalar(1)
        if n == 1:
            return self

        result = self
        for _ in range(n - 1):
            result = result.geometric_product(self)
        return result

    # =========================================================================
    # GA Products
    # =========================================================================

    def geometric_product(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Compute the geometric product of two multivectors."""
        if self._algebra != other._algebra:
            raise ValueError("Cannot multiply multivectors from different algebras")

        new_coeffs: Dict[int, Expr] = {}

        for i, a in self._coeffs.items():
            for j, b in other._coeffs.items():
                result_blade, sign = self._algebra.blade_product(i, j)
                product = sign * a * b

                if result_blade in new_coeffs:
                    new_coeffs[result_blade] = new_coeffs[result_blade] + product
                else:
                    new_coeffs[result_blade] = product

        return SymbolicMultivector(new_coeffs, self._algebra)

    def outer_product(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Compute the outer (wedge) product.

        The outer product is zero when blades share any basis vector.
        """
        if self._algebra != other._algebra:
            raise ValueError("Cannot wedge multivectors from different algebras")

        new_coeffs: Dict[int, Expr] = {}

        for i, a in self._coeffs.items():
            for j, b in other._coeffs.items():
                # Outer product is zero if blades share a basis vector
                if i & j != 0:
                    continue

                result_blade, sign = self._algebra.blade_product(i, j)
                product = sign * a * b

                if result_blade in new_coeffs:
                    new_coeffs[result_blade] = new_coeffs[result_blade] + product
                else:
                    new_coeffs[result_blade] = product

        return SymbolicMultivector(new_coeffs, self._algebra)

    def wedge(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Alias for outer_product."""
        return self.outer_product(other)

    def __xor__(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Wedge product using ^ operator."""
        return self.outer_product(other)

    def left_contraction(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Compute the left contraction A << B.

        For blades of grade r and s: A_r << B_s = <A_r * B_s>_{s-r} if s >= r, else 0.
        """
        if self._algebra != other._algebra:
            raise ValueError("Cannot contract multivectors from different algebras")

        new_coeffs: Dict[int, Expr] = {}

        for i, a in self._coeffs.items():
            grade_i = self._algebra.blade_grade(i)
            for j, b in other._coeffs.items():
                grade_j = self._algebra.blade_grade(j)

                # Left contraction only keeps grade s-r terms
                if grade_j < grade_i:
                    continue

                result_blade, sign = self._algebra.blade_product(i, j)
                result_grade = self._algebra.blade_grade(result_blade)

                # Only keep if result has expected grade
                if result_grade != grade_j - grade_i:
                    continue

                product = sign * a * b

                if result_blade in new_coeffs:
                    new_coeffs[result_blade] = new_coeffs[result_blade] + product
                else:
                    new_coeffs[result_blade] = product

        return SymbolicMultivector(new_coeffs, self._algebra)

    def __lshift__(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Left contraction using << operator."""
        return self.left_contraction(other)

    def right_contraction(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Compute the right contraction A >> B.

        For blades of grade r and s: A_r >> B_s = <A_r * B_s>_{r-s} if r >= s, else 0.
        """
        if self._algebra != other._algebra:
            raise ValueError("Cannot contract multivectors from different algebras")

        new_coeffs: Dict[int, Expr] = {}

        for i, a in self._coeffs.items():
            grade_i = self._algebra.blade_grade(i)
            for j, b in other._coeffs.items():
                grade_j = self._algebra.blade_grade(j)

                # Right contraction only keeps grade r-s terms
                if grade_i < grade_j:
                    continue

                result_blade, sign = self._algebra.blade_product(i, j)
                result_grade = self._algebra.blade_grade(result_blade)

                # Only keep if result has expected grade
                if result_grade != grade_i - grade_j:
                    continue

                product = sign * a * b

                if result_blade in new_coeffs:
                    new_coeffs[result_blade] = new_coeffs[result_blade] + product
                else:
                    new_coeffs[result_blade] = product

        return SymbolicMultivector(new_coeffs, self._algebra)

    def __rshift__(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Right contraction using >> operator."""
        return self.right_contraction(other)

    def inner_product(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Compute the inner product (left contraction for vectors)."""
        return self.left_contraction(other)

    def scalar_product(self, other: SymbolicMultivector) -> Expr:
        """Compute scalar product <A * B>_0."""
        result = self.geometric_product(other)
        return result.scalar_part()

    def commutator(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Compute commutator [A, B] = (AB - BA) / 2."""
        ab = self.geometric_product(other)
        ba = other.geometric_product(self)
        return (ab - ba) / 2

    def anticommutator(self, other: SymbolicMultivector) -> SymbolicMultivector:
        """Compute anticommutator {A, B} = (AB + BA) / 2."""
        ab = self.geometric_product(other)
        ba = other.geometric_product(self)
        return (ab + ba) / 2

    # =========================================================================
    # Involutions
    # =========================================================================

    def reverse(self) -> SymbolicMultivector:
        """Compute the reverse (dagger) operation.

        Reverses the order of basis vectors in each blade.
        For grade k: sign is (-1)^(k(k-1)/2).
        """
        new_coeffs: Dict[int, Expr] = {}
        for idx, coeff in self._coeffs.items():
            grade = self._algebra.blade_grade(idx)
            sign = S(-1) ** (grade * (grade - 1) // 2)
            new_coeffs[idx] = sign * coeff
        return SymbolicMultivector(new_coeffs, self._algebra)

    def grade_involution(self) -> SymbolicMultivector:
        """Compute grade involution (main involution).

        For grade k: sign is (-1)^k.
        """
        new_coeffs: Dict[int, Expr] = {}
        for idx, coeff in self._coeffs.items():
            grade = self._algebra.blade_grade(idx)
            sign = S(-1) ** grade
            new_coeffs[idx] = sign * coeff
        return SymbolicMultivector(new_coeffs, self._algebra)

    def conjugate(self) -> SymbolicMultivector:
        """Compute Clifford conjugate.

        Combination of reverse and grade involution.
        For grade k: sign is (-1)^(k(k+1)/2).
        """
        new_coeffs: Dict[int, Expr] = {}
        for idx, coeff in self._coeffs.items():
            grade = self._algebra.blade_grade(idx)
            sign = S(-1) ** (grade * (grade + 1) // 2)
            new_coeffs[idx] = sign * coeff
        return SymbolicMultivector(new_coeffs, self._algebra)

    def dual(self) -> SymbolicMultivector:
        """Compute the dual with respect to the pseudoscalar.

        dual(A) = A * I^(-1) where I is the unit pseudoscalar.
        """
        I = self._algebra.pseudoscalar()
        # For many algebras, I^(-1) = +/- I
        # Compute I^2 first
        I_squared = I.geometric_product(I).scalar_part()
        I_inv = I / I_squared
        return self.geometric_product(I_inv)

    # =========================================================================
    # Norms and normalization
    # =========================================================================

    def norm_squared(self) -> Expr:
        """Compute squared norm: <A * reverse(A)>_0."""
        return self.scalar_product(self.reverse())

    def norm(self) -> Expr:
        """Compute norm: sqrt(|norm_squared|)."""
        ns = self.norm_squared()
        return sqrt(sp.Abs(ns))

    def normalize(self) -> SymbolicMultivector:
        """Return normalized version of this multivector."""
        n = self.norm()
        return self / n

    # =========================================================================
    # Simplification and substitution
    # =========================================================================

    def simplify(self) -> SymbolicMultivector:
        """Simplify all coefficients using SymPy."""
        new_coeffs = {
            idx: simplify(coeff)
            for idx, coeff in self._coeffs.items()
        }
        return SymbolicMultivector(new_coeffs, self._algebra)

    def expand(self) -> SymbolicMultivector:
        """Expand all coefficients using SymPy."""
        new_coeffs = {
            idx: expand(coeff)
            for idx, coeff in self._coeffs.items()
        }
        return SymbolicMultivector(new_coeffs, self._algebra)

    def subs(self, substitutions: Dict[Symbol, Any]) -> SymbolicMultivector:
        """Substitute values for symbols.

        Args:
            substitutions: Dictionary mapping symbols to values.
        """
        new_coeffs = {
            idx: coeff.subs(substitutions)
            for idx, coeff in self._coeffs.items()
        }
        return SymbolicMultivector(new_coeffs, self._algebra)

    def evalf(self) -> SymbolicMultivector:
        """Evaluate to floating-point numbers."""
        new_coeffs = {
            idx: coeff.evalf()
            for idx, coeff in self._coeffs.items()
        }
        return SymbolicMultivector(new_coeffs, self._algebra)

    # =========================================================================
    # Differentiation
    # =========================================================================

    def diff(self, symbol: Symbol) -> SymbolicMultivector:
        """Differentiate with respect to a symbol.

        Args:
            symbol: The SymPy symbol to differentiate with respect to.
        """
        new_coeffs = {
            idx: sp.diff(coeff, symbol)
            for idx, coeff in self._coeffs.items()
        }
        return SymbolicMultivector(new_coeffs, self._algebra)

    def gradient(self, symbols: List[Symbol]) -> List[SymbolicMultivector]:
        """Compute gradient with respect to multiple symbols."""
        return [self.diff(s) for s in symbols]

    # =========================================================================
    # Output
    # =========================================================================

    def to_latex(self) -> str:
        """Generate LaTeX representation."""
        if not self._coeffs:
            return "0"

        terms = []
        for idx in sorted(self._coeffs.keys()):
            coeff = self._coeffs[idx]
            blade_name = self._algebra.blade_name(idx)

            # Format coefficient
            coeff_latex = latex(coeff)

            if blade_name == "1":
                # Scalar term
                terms.append(coeff_latex)
            elif coeff == S(1):
                terms.append(blade_name)
            elif coeff == S(-1):
                terms.append(f"-{blade_name}")
            else:
                # Wrap coefficient if it's a sum
                if coeff.is_Add:
                    terms.append(f"({coeff_latex}){blade_name}")
                else:
                    terms.append(f"{coeff_latex}{blade_name}")

        result = " + ".join(terms)
        # Clean up + - to -
        result = result.replace(" + -", " - ")
        return result

    def latex(self) -> str:
        """Alias for to_latex."""
        return self.to_latex()

    def __repr__(self) -> str:
        if not self._coeffs:
            return "0"

        terms = []
        for idx in sorted(self._coeffs.keys()):
            coeff = self._coeffs[idx]
            blade_name = self._algebra.blade_name(idx)

            if blade_name == "1":
                terms.append(str(coeff))
            elif coeff == S(1):
                terms.append(blade_name)
            elif coeff == S(-1):
                terms.append(f"-{blade_name}")
            else:
                terms.append(f"{coeff}*{blade_name}")

        result = " + ".join(terms)
        result = result.replace(" + -", " - ")
        return result

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            other = self._algebra.scalar(other)
        if not isinstance(other, SymbolicMultivector):
            return False
        if self._algebra != other._algebra:
            return False

        # Compare all coefficients
        all_keys = set(self._coeffs.keys()) | set(other._coeffs.keys())
        for key in all_keys:
            if simplify(self[key] - other[key]) != S(0):
                return False
        return True

    def __hash__(self) -> int:
        # Hash based on simplified coefficients
        items = tuple(sorted(
            (k, str(simplify(v))) for k, v in self._coeffs.items()
        ))
        return hash(items)

    # =========================================================================
    # Conversion to numeric
    # =========================================================================

    def to_numeric(self, coeffs_only: bool = False) -> Union[List[float], Any]:
        """Convert to numeric representation.

        Args:
            coeffs_only: If True, return only the coefficient list.
                         Otherwise, attempt to create a numeric Multivector.

        Returns:
            List of float coefficients, or a numeric Multivector if available.
        """
        num_blades = self._algebra.num_blades
        result = [0.0] * num_blades

        for idx, coeff in self._coeffs.items():
            val = complex(coeff.evalf())
            if val.imag != 0:
                raise ValueError(f"Coefficient {coeff} has imaginary part")
            result[idx] = val.real

        if coeffs_only:
            return result

        # Try to create numeric Multivector
        try:
            import largecrimsoncanine as lcc
            return lcc.Multivector(result)
        except ImportError:
            return result
