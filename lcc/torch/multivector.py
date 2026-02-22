"""PyTorch-backed multivector operations for deep learning.

This module provides TorchMultivector, a class that stores multivector coefficients
as PyTorch tensors and implements geometric algebra operations that are:
- Differentiable with PyTorch autograd
- GPU-accelerated with CUDA
- Compatible with torch.compile (PyTorch 2.0+)

The implementation mirrors the JAX-based JaxMultivector but uses PyTorch's
eager execution model with optional compilation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor


@dataclass
class TorchAlgebra:
    """Pure PyTorch representation of an algebra's Cayley tables.

    This stores the precomputed multiplication tables as PyTorch tensors,
    enabling GPU-accelerated geometric algebra operations.

    Attributes:
        signature: Tuple (p, q, r) representing the metric signature.
        num_blades: Total number of basis blades (2^dimension).
        dimension: Dimension of the underlying vector space.
        cayley_signs: Tensor of signs for blade products.
        cayley_blades: Tensor of result blade indices for products.
        device: The device tensors are stored on.
    """
    signature: Tuple[int, int, int]
    num_blades: int
    dimension: int
    cayley_signs: Tensor  # Shape: (num_blades, num_blades)
    cayley_blades: Tensor  # Shape: (num_blades, num_blades), dtype=long
    device: torch.device = torch.device('cpu')

    def to(self, device: Union[str, torch.device]) -> "TorchAlgebra":
        """Move algebra tables to a different device.

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            A new TorchAlgebra on the specified device.
        """
        device = torch.device(device)
        return TorchAlgebra(
            signature=self.signature,
            num_blades=self.num_blades,
            dimension=self.dimension,
            cayley_signs=self.cayley_signs.to(device),
            cayley_blades=self.cayley_blades.to(device),
            device=device,
        )

    def cuda(self, device: Optional[int] = None) -> "TorchAlgebra":
        """Move algebra tables to CUDA."""
        if device is None:
            return self.to('cuda')
        return self.to(f'cuda:{device}')

    def cpu(self) -> "TorchAlgebra":
        """Move algebra tables to CPU."""
        return self.to('cpu')

    @classmethod
    def from_rust_algebra(cls, algebra, device: Union[str, torch.device] = 'cpu') -> "TorchAlgebra":
        """Create a TorchAlgebra from a Rust-backed Algebra object.

        Args:
            algebra: A largecrimsoncanine.Algebra object.
            device: Device to place tensors on.

        Returns:
            A TorchAlgebra with the same structure.

        Example:
            >>> from largecrimsoncanine import Algebra
            >>> R3 = Algebra.euclidean(3)
            >>> torch_alg = TorchAlgebra.from_rust_algebra(R3)
            >>> torch_alg_gpu = TorchAlgebra.from_rust_algebra(R3, device='cuda')
        """
        device = torch.device(device)
        num_blades = algebra.num_blades()
        dimension = algebra.dimension()
        sig = algebra.signature()

        # Get Cayley table data from algebra
        signs_flat = np.array(algebra.cayley_signs())
        blades_flat = np.array(algebra.cayley_blades(), dtype=np.int64)

        # Reshape to 2D for easier indexing
        cayley_signs = torch.tensor(
            signs_flat.reshape(num_blades, num_blades),
            dtype=torch.float64,
            device=device
        )
        cayley_blades = torch.tensor(
            blades_flat.reshape(num_blades, num_blades),
            dtype=torch.long,
            device=device
        )

        return cls(
            signature=(sig.p, sig.q, sig.r),
            num_blades=num_blades,
            dimension=dimension,
            cayley_signs=cayley_signs,
            cayley_blades=cayley_blades,
            device=device,
        )

    @classmethod
    def euclidean(cls, n: int, device: Union[str, torch.device] = 'cpu') -> "TorchAlgebra":
        """Create a Euclidean algebra Cl(n,0,0) directly.

        This builds the Cayley table in pure Python/PyTorch without
        requiring the Rust backend.

        Args:
            n: Dimension of the Euclidean space.
            device: Device to place tensors on.

        Returns:
            A TorchAlgebra for Cl(n,0,0).
        """
        device = torch.device(device)
        num_blades = 2 ** n

        # Build Cayley table
        signs = np.zeros((num_blades, num_blades), dtype=np.float64)
        blades = np.zeros((num_blades, num_blades), dtype=np.int64)

        for a in range(num_blades):
            for b in range(num_blades):
                blade, sign = _blade_product_euclidean(a, b)
                blades[a, b] = blade
                signs[a, b] = sign

        return cls(
            signature=(n, 0, 0),
            num_blades=num_blades,
            dimension=n,
            cayley_signs=torch.tensor(signs, dtype=torch.float64, device=device),
            cayley_blades=torch.tensor(blades, dtype=torch.long, device=device),
            device=device,
        )

    @classmethod
    def pga(cls, n: int, device: Union[str, torch.device] = 'cpu') -> "TorchAlgebra":
        """Create Projective Geometric Algebra Cl(n,0,1).

        Args:
            n: Dimension of the base Euclidean space.
            device: Device to place tensors on.

        Returns:
            A TorchAlgebra for Cl(n,0,1).
        """
        return cls._from_signature(n, 0, 1, device)

    @classmethod
    def sta(cls, device: Union[str, torch.device] = 'cpu') -> "TorchAlgebra":
        """Create Spacetime Algebra Cl(1,3,0).

        Args:
            device: Device to place tensors on.

        Returns:
            A TorchAlgebra for Cl(1,3,0).
        """
        return cls._from_signature(1, 3, 0, device)

    @classmethod
    def _from_signature(cls, p: int, q: int, r: int, device: Union[str, torch.device] = 'cpu') -> "TorchAlgebra":
        """Create algebra from arbitrary signature.

        Args:
            p: Number of basis vectors squaring to +1.
            q: Number of basis vectors squaring to -1.
            r: Number of basis vectors squaring to 0.
            device: Device to place tensors on.

        Returns:
            A TorchAlgebra for Cl(p,q,r).
        """
        device = torch.device(device)
        n = p + q + r
        num_blades = 2 ** n

        # Build metric: first p square to +1, next q to -1, last r to 0
        metric = [1.0] * p + [-1.0] * q + [0.0] * r

        # Build Cayley table
        signs = np.zeros((num_blades, num_blades), dtype=np.float64)
        blades = np.zeros((num_blades, num_blades), dtype=np.int64)

        for a in range(num_blades):
            for b in range(num_blades):
                blade, sign = _blade_product_general(a, b, metric)
                blades[a, b] = blade
                signs[a, b] = sign

        return cls(
            signature=(p, q, r),
            num_blades=num_blades,
            dimension=n,
            cayley_signs=torch.tensor(signs, dtype=torch.float64, device=device),
            cayley_blades=torch.tensor(blades, dtype=torch.long, device=device),
            device=device,
        )


def _blade_product_euclidean(a: int, b: int) -> Tuple[int, float]:
    """Compute the geometric product of two basis blades in Euclidean metric.

    Args:
        a: Binary index of first blade.
        b: Binary index of second blade.

    Returns:
        Tuple of (result_blade, sign).
    """
    result = a ^ b
    sign = 1.0
    temp_a = a
    temp_b = b

    while temp_b:
        lowest_b = temp_b & (-temp_b)
        higher_a = temp_a & ~(lowest_b - 1) & ~lowest_b
        swaps = bin(higher_a).count('1')

        if swaps % 2 == 1:
            sign = -sign

        temp_b &= temp_b - 1

    return result, sign


def _blade_product_general(a: int, b: int, metric: List[float]) -> Tuple[int, float]:
    """Compute the geometric product of two basis blades with arbitrary metric.

    Args:
        a: Binary index of first blade.
        b: Binary index of second blade.
        metric: List of what each basis vector squares to.

    Returns:
        Tuple of (result_blade, sign).
    """
    result = a ^ b
    sign = 1.0
    n = len(metric)

    # Count swaps needed to bring matching vectors together
    temp_a = a
    temp_b = b

    while temp_b:
        lowest_b = temp_b & (-temp_b)
        higher_a = temp_a & ~(lowest_b - 1) & ~lowest_b
        swaps = bin(higher_a).count('1')

        if swaps % 2 == 1:
            sign = -sign

        temp_b &= temp_b - 1

    # Apply metric for contracting pairs
    common = a & b
    for i in range(n):
        if (common >> i) & 1:
            sign *= metric[i]

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


class TorchMultivector:
    """A batch of multivectors backed by PyTorch tensors.

    This class stores multivector coefficients as PyTorch tensors with shape
    (batch_size, num_blades) and provides geometric algebra operations
    that are differentiable and GPU-accelerated.

    Attributes:
        algebra: The TorchAlgebra defining the multiplication structure.
        coeffs: Tensor of shape (batch_size, num_blades) containing
            the coefficients for each blade of each multivector.

    Example:
        >>> from largecrimsoncanine import Algebra
        >>> from lcc.torch import TorchMultivector, TorchAlgebra
        >>> import torch
        >>>
        >>> R3 = Algebra.euclidean(3)
        >>> torch_alg = TorchAlgebra.from_rust_algebra(R3)
        >>>
        >>> # Create a batch of 100 multivectors with gradients
        >>> coeffs = torch.randn(100, 8, requires_grad=True)
        >>> batch = TorchMultivector(torch_alg, coeffs)
        >>>
        >>> # Operations are differentiable
        >>> result = batch.geometric_product(batch)
        >>> loss = result.norm().sum()
        >>> loss.backward()  # Gradients flow through GA operations
    """

    def __init__(self, algebra: TorchAlgebra, coeffs: Tensor):
        """Create a TorchMultivector from algebra and coefficients.

        Args:
            algebra: A TorchAlgebra defining the multiplication tables.
            coeffs: Tensor of shape (batch_size, num_blades) or
                (num_blades,) for a single multivector.
        """
        self.algebra = algebra

        # Ensure coeffs is 2D
        if coeffs.ndim == 1:
            coeffs = coeffs.unsqueeze(0)

        if coeffs.shape[-1] != algebra.num_blades:
            raise ValueError(
                f"coeffs must have {algebra.num_blades} columns for "
                f"algebra with signature {algebra.signature}, "
                f"got {coeffs.shape[-1]}"
            )

        # Move to same device as algebra if needed
        if coeffs.device != algebra.device:
            coeffs = coeffs.to(algebra.device)

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
        return tuple(self.coeffs.shape)

    @property
    def device(self) -> torch.device:
        """Device the tensors are on."""
        return self.coeffs.device

    @property
    def requires_grad(self) -> bool:
        """Whether gradients are being tracked."""
        return self.coeffs.requires_grad

    # =========================================================================
    # Device movement
    # =========================================================================

    def to(self, device: Union[str, torch.device]) -> "TorchMultivector":
        """Move multivector to a different device."""
        device = torch.device(device)
        return TorchMultivector(
            self.algebra.to(device),
            self.coeffs.to(device)
        )

    def cuda(self, device: Optional[int] = None) -> "TorchMultivector":
        """Move to CUDA."""
        if device is None:
            return self.to('cuda')
        return self.to(f'cuda:{device}')

    def cpu(self) -> "TorchMultivector":
        """Move to CPU."""
        return self.to('cpu')

    def detach(self) -> "TorchMultivector":
        """Detach from computation graph."""
        return TorchMultivector(self.algebra, self.coeffs.detach())

    def requires_grad_(self, requires_grad: bool = True) -> "TorchMultivector":
        """Set requires_grad flag in-place."""
        self.coeffs.requires_grad_(requires_grad)
        return self

    # =========================================================================
    # Construction methods
    # =========================================================================

    @classmethod
    def from_numpy(cls, algebra: TorchAlgebra, coeffs: np.ndarray) -> "TorchMultivector":
        """Create a TorchMultivector from a NumPy array.

        Args:
            algebra: A TorchAlgebra.
            coeffs: NumPy array of shape (batch_size, num_blades).

        Returns:
            A TorchMultivector wrapping the converted tensor.
        """
        return cls(algebra, torch.tensor(coeffs, dtype=torch.float64, device=algebra.device))

    def to_numpy(self) -> np.ndarray:
        """Convert coefficients to a NumPy array.

        Returns:
            NumPy array of shape (batch_size, num_blades).
        """
        return self.coeffs.detach().cpu().numpy()

    @classmethod
    def from_vectors(cls, algebra: TorchAlgebra, coords: Tensor) -> "TorchMultivector":
        """Create a batch of vectors from coordinate tensor.

        Args:
            algebra: A TorchAlgebra.
            coords: Tensor of shape (batch_size, dimension).

        Returns:
            A TorchMultivector containing vectors (grade-1 elements).

        Example:
            >>> torch_alg = TorchAlgebra.euclidean(3)
            >>> coords = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            >>> vectors = TorchMultivector.from_vectors(torch_alg, coords)
        """
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)

        batch_size = coords.shape[0]
        dim = coords.shape[1]

        if dim != algebra.dimension:
            raise ValueError(
                f"coords must have {algebra.dimension} columns for "
                f"algebra with signature {algebra.signature}, "
                f"got {dim}"
            )

        # Build coefficient tensor
        coeffs = torch.zeros(batch_size, algebra.num_blades,
                            dtype=coords.dtype, device=algebra.device)

        for i in range(dim):
            blade_idx = 1 << i
            coeffs[:, blade_idx] = coords[:, i]

        return cls(algebra, coeffs)

    @classmethod
    def from_scalars(cls, algebra: TorchAlgebra, values: Tensor) -> "TorchMultivector":
        """Create a batch of scalars from a 1D tensor.

        Args:
            algebra: A TorchAlgebra.
            values: Tensor of shape (batch_size,).

        Returns:
            A TorchMultivector containing scalars (grade-0 elements).
        """
        if values.ndim == 0:
            values = values.unsqueeze(0)

        batch_size = values.shape[0]
        coeffs = torch.zeros(batch_size, algebra.num_blades,
                            dtype=values.dtype, device=algebra.device)
        coeffs[:, 0] = values

        return cls(algebra, coeffs)

    @classmethod
    def from_bivectors(cls, algebra: TorchAlgebra, biv_coeffs: Tensor) -> "TorchMultivector":
        """Create a batch of bivectors from coefficient tensor.

        Args:
            algebra: A TorchAlgebra.
            biv_coeffs: Tensor of shape (batch_size, num_bivectors).

        Returns:
            A TorchMultivector containing bivectors (grade-2 elements).
        """
        if biv_coeffs.ndim == 1:
            biv_coeffs = biv_coeffs.unsqueeze(0)

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

        coeffs = torch.zeros(batch_size, algebra.num_blades,
                            dtype=biv_coeffs.dtype, device=algebra.device)
        for col, blade_idx in enumerate(bivector_indices):
            coeffs[:, blade_idx] = biv_coeffs[:, col]

        return cls(algebra, coeffs)

    def to_vector_coords(self) -> Tensor:
        """Extract vector coordinates from the batch.

        Returns:
            Tensor of shape (batch_size, dimension) containing vector components.
        """
        dim = self.algebra.dimension
        coords = torch.zeros(self.batch_size, dim,
                            dtype=self.coeffs.dtype, device=self.device)

        for i in range(dim):
            blade_idx = 1 << i
            coords[:, i] = self.coeffs[:, blade_idx]

        return coords

    def scalar(self) -> Tensor:
        """Extract scalar (grade-0) parts.

        Returns:
            Tensor of shape (batch_size,) containing scalar components.
        """
        return self.coeffs[:, 0]

    # =========================================================================
    # Geometric products
    # =========================================================================

    def geometric_product(self, other: "TorchMultivector") -> "TorchMultivector":
        """Compute the geometric product: self * other.

        Supports broadcasting: if one operand has batch_size 1, it is
        broadcast to match the other.

        Args:
            other: Another TorchMultivector.

        Returns:
            The geometric product as a new TorchMultivector.
        """
        result = _geometric_product_impl(
            self.coeffs, other.coeffs,
            self.algebra.cayley_signs,
            self.algebra.cayley_blades,
        )
        return TorchMultivector(self.algebra, result)

    def gp(self, other: "TorchMultivector") -> "TorchMultivector":
        """Alias for geometric_product."""
        return self.geometric_product(other)

    def outer_product(self, other: "TorchMultivector") -> "TorchMultivector":
        """Compute the outer (wedge) product: self ^ other.

        Args:
            other: Another TorchMultivector.

        Returns:
            The outer product as a new TorchMultivector.
        """
        result = _outer_product_impl(
            self.coeffs, other.coeffs,
            self.algebra.cayley_signs,
            self.algebra.cayley_blades,
            self.algebra.num_blades,
        )
        return TorchMultivector(self.algebra, result)

    def wedge(self, other: "TorchMultivector") -> "TorchMultivector":
        """Alias for outer_product."""
        return self.outer_product(other)

    def inner_product(self, other: "TorchMultivector") -> "TorchMultivector":
        """Compute the left contraction (inner product): self . other.

        Args:
            other: Another TorchMultivector.

        Returns:
            The left contraction as a new TorchMultivector.
        """
        result = _inner_product_impl(
            self.coeffs, other.coeffs,
            self.algebra.cayley_signs,
            self.algebra.cayley_blades,
            self.algebra.num_blades,
        )
        return TorchMultivector(self.algebra, result)

    def lc(self, other: "TorchMultivector") -> "TorchMultivector":
        """Alias for inner_product (left contraction)."""
        return self.inner_product(other)

    # =========================================================================
    # Sandwich product (transformations)
    # =========================================================================

    def sandwich(self, rotor: "TorchMultivector") -> "TorchMultivector":
        """Apply a rotor to all multivectors: rotor * self * ~rotor.

        This is the standard way to apply rotations/transformations in GA.

        Args:
            rotor: A TorchMultivector representing a rotor (should be normalized).

        Returns:
            The transformed multivectors as a new TorchMultivector.

        Example:
            >>> # Create a rotor for rotation around Z axis
            >>> angle = torch.tensor(torch.pi / 4)
            >>> rotor_coeffs = torch.tensor([[torch.cos(angle/2), 0, 0, -torch.sin(angle/2), 0, 0, 0, 0]])
            >>> rotor = TorchMultivector(algebra, rotor_coeffs)
            >>> rotated = vectors.sandwich(rotor)
        """
        result = _sandwich_impl(
            self.coeffs, rotor.coeffs,
            self.algebra.cayley_signs,
            self.algebra.cayley_blades,
            self.algebra.num_blades,
        )
        return TorchMultivector(self.algebra, result)

    def apply(self, rotor: "TorchMultivector") -> "TorchMultivector":
        """Alias for sandwich (common in robotics literature)."""
        return self.sandwich(rotor)

    # =========================================================================
    # Involutions
    # =========================================================================

    def reverse(self) -> "TorchMultivector":
        """Compute the reverse of each multivector.

        The reverse negates blades of grade k by (-1)^(k(k-1)/2).

        Returns:
            A new TorchMultivector with reversed elements.
        """
        signs = torch.tensor(
            [_reverse_sign(_blade_grade(i)) for i in range(self.algebra.num_blades)],
            dtype=self.coeffs.dtype, device=self.device
        )
        return TorchMultivector(self.algebra, self.coeffs * signs)

    def grade_involution(self) -> "TorchMultivector":
        """Compute the grade involution of each multivector.

        Grade involution negates all odd-grade components.

        Returns:
            A new TorchMultivector with grade-inverted elements.
        """
        signs = torch.tensor(
            [_grade_involution_sign(_blade_grade(i)) for i in range(self.algebra.num_blades)],
            dtype=self.coeffs.dtype, device=self.device
        )
        return TorchMultivector(self.algebra, self.coeffs * signs)

    def conjugate(self) -> "TorchMultivector":
        """Compute the Clifford conjugate.

        Combination of reverse and grade involution.

        Returns:
            A new TorchMultivector with conjugated elements.
        """
        return self.reverse().grade_involution()

    # =========================================================================
    # Norms
    # =========================================================================

    def norm_squared(self) -> Tensor:
        """Compute the squared norm of each multivector.

        The norm squared is the scalar part of self * ~self.

        Returns:
            Tensor of shape (batch_size,) containing squared norms.
        """
        rev = self.reverse()
        product = self.geometric_product(rev)
        return product.scalar()

    def norm(self) -> Tensor:
        """Compute the norm of each multivector.

        Returns:
            Tensor of shape (batch_size,) containing norms.
        """
        return torch.sqrt(torch.abs(self.norm_squared()))

    def normalized(self) -> "TorchMultivector":
        """Return normalized copies of all multivectors.

        Multivectors with zero norm are left unchanged.

        Returns:
            A new TorchMultivector with normalized elements.
        """
        norms = self.norm()
        # Avoid division by zero
        safe_norms = torch.where(norms > 1e-14, norms, torch.ones_like(norms))
        return TorchMultivector(
            self.algebra,
            self.coeffs / safe_norms.unsqueeze(-1)
        )

    # =========================================================================
    # Grade extraction
    # =========================================================================

    def grade(self, k: int) -> "TorchMultivector":
        """Extract grade-k parts.

        Args:
            k: The grade to extract.

        Returns:
            A new TorchMultivector containing only grade-k components.
        """
        mask = torch.tensor(
            [1.0 if _blade_grade(i) == k else 0.0 for i in range(self.algebra.num_blades)],
            dtype=self.coeffs.dtype, device=self.device
        )
        return TorchMultivector(self.algebra, self.coeffs * mask)

    def even(self) -> "TorchMultivector":
        """Extract even-grade parts."""
        mask = torch.tensor(
            [1.0 if _blade_grade(i) % 2 == 0 else 0.0 for i in range(self.algebra.num_blades)],
            dtype=self.coeffs.dtype, device=self.device
        )
        return TorchMultivector(self.algebra, self.coeffs * mask)

    def odd(self) -> "TorchMultivector":
        """Extract odd-grade parts."""
        mask = torch.tensor(
            [1.0 if _blade_grade(i) % 2 == 1 else 0.0 for i in range(self.algebra.num_blades)],
            dtype=self.coeffs.dtype, device=self.device
        )
        return TorchMultivector(self.algebra, self.coeffs * mask)

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: "TorchMultivector") -> "TorchMultivector":
        """Add two multivector batches element-wise."""
        return TorchMultivector(self.algebra, self.coeffs + other.coeffs)

    def __sub__(self, other: "TorchMultivector") -> "TorchMultivector":
        """Subtract two multivector batches element-wise."""
        return TorchMultivector(self.algebra, self.coeffs - other.coeffs)

    def __neg__(self) -> "TorchMultivector":
        """Negate all multivectors."""
        return TorchMultivector(self.algebra, -self.coeffs)

    def __mul__(self, scalar: Union[float, Tensor]) -> "TorchMultivector":
        """Scalar multiplication."""
        return TorchMultivector(self.algebra, self.coeffs * scalar)

    def __rmul__(self, scalar: Union[float, Tensor]) -> "TorchMultivector":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[float, Tensor]) -> "TorchMultivector":
        """Scalar division."""
        return TorchMultivector(self.algebra, self.coeffs / scalar)

    # =========================================================================
    # Indexing and slicing
    # =========================================================================

    def __getitem__(self, idx) -> "TorchMultivector":
        """Index or slice the batch."""
        result = self.coeffs[idx]
        if result.ndim == 1:
            result = result.unsqueeze(0)
        return TorchMultivector(self.algebra, result)

    def __len__(self) -> int:
        """Return batch size."""
        return self.batch_size

    # =========================================================================
    # String representation
    # =========================================================================

    def __repr__(self) -> str:
        device_str = f", device='{self.device}'" if self.device.type != 'cpu' else ""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"TorchMultivector(batch_size={self.batch_size}, signature={self.algebra.signature}{device_str}{grad_str})"


# =============================================================================
# Implementation functions
# =============================================================================

def _geometric_product_impl(
    a_coeffs: Tensor,
    b_coeffs: Tensor,
    cayley_signs: Tensor,
    cayley_blades: Tensor,
) -> Tensor:
    """Geometric product implementation.

    Uses einsum-style operations for efficient GPU execution.
    """
    # Handle broadcasting
    if a_coeffs.shape[0] == 1 and b_coeffs.shape[0] > 1:
        a_coeffs = a_coeffs.expand(b_coeffs.shape[0], -1)
    elif b_coeffs.shape[0] == 1 and a_coeffs.shape[0] > 1:
        b_coeffs = b_coeffs.expand(a_coeffs.shape[0], -1)

    batch_size = a_coeffs.shape[0]
    num_blades = a_coeffs.shape[1]
    dtype = a_coeffs.dtype
    device = a_coeffs.device

    # Ensure consistent dtypes
    signs = cayley_signs.to(dtype=dtype)

    # Compute all pairwise products: (batch, i, j) -> a[i] * b[j] * sign[i,j]
    a_expanded = a_coeffs.unsqueeze(2)  # (batch, num_blades, 1)
    b_expanded = b_coeffs.unsqueeze(1)  # (batch, 1, num_blades)

    # Product contributions: (batch, num_blades, num_blades)
    products = a_expanded * b_expanded * signs.unsqueeze(0)

    # Scatter-add to result blades
    result = torch.zeros(batch_size, num_blades, dtype=dtype, device=device)

    # Flatten for scatter_add
    flat_products = products.reshape(batch_size, -1)
    flat_indices = cayley_blades.reshape(-1).expand(batch_size, -1)

    result.scatter_add_(1, flat_indices, flat_products)

    return result


def _outer_product_impl(
    a_coeffs: Tensor,
    b_coeffs: Tensor,
    cayley_signs: Tensor,
    cayley_blades: Tensor,
    num_blades: int,
) -> Tensor:
    """Outer product implementation.

    The outer product is zero when blades share a basis vector (i & j != 0).
    """
    # Handle broadcasting
    if a_coeffs.shape[0] == 1 and b_coeffs.shape[0] > 1:
        a_coeffs = a_coeffs.expand(b_coeffs.shape[0], -1)
    elif b_coeffs.shape[0] == 1 and a_coeffs.shape[0] > 1:
        b_coeffs = b_coeffs.expand(a_coeffs.shape[0], -1)

    batch_size = a_coeffs.shape[0]
    dtype = a_coeffs.dtype
    device = a_coeffs.device

    # Ensure consistent dtypes
    signs = cayley_signs.to(dtype=dtype)

    # Create mask for outer product (only where blades don't share vectors)
    outer_mask = torch.tensor(
        [[1.0 if (i & j) == 0 else 0.0 for j in range(num_blades)] for i in range(num_blades)],
        dtype=dtype, device=device
    )

    a_expanded = a_coeffs.unsqueeze(2)
    b_expanded = b_coeffs.unsqueeze(1)

    products = a_expanded * b_expanded * signs.unsqueeze(0) * outer_mask.unsqueeze(0)

    result = torch.zeros(batch_size, num_blades, dtype=dtype, device=device)

    flat_products = products.reshape(batch_size, -1)
    flat_indices = cayley_blades.reshape(-1).expand(batch_size, -1)

    result.scatter_add_(1, flat_indices, flat_products)

    return result


def _inner_product_impl(
    a_coeffs: Tensor,
    b_coeffs: Tensor,
    cayley_signs: Tensor,
    cayley_blades: Tensor,
    num_blades: int,
) -> Tensor:
    """Left contraction (inner product) implementation.

    Left contraction A . B is non-zero only when:
    - grade(A) <= grade(B)
    - A is a "subset" of B (all basis vectors of A are in B)
    - result grade = grade(B) - grade(A)
    """
    # Handle broadcasting
    if a_coeffs.shape[0] == 1 and b_coeffs.shape[0] > 1:
        a_coeffs = a_coeffs.expand(b_coeffs.shape[0], -1)
    elif b_coeffs.shape[0] == 1 and a_coeffs.shape[0] > 1:
        b_coeffs = b_coeffs.expand(a_coeffs.shape[0], -1)

    batch_size = a_coeffs.shape[0]
    dtype = a_coeffs.dtype
    device = a_coeffs.device

    # Ensure consistent dtypes
    signs = cayley_signs.to(dtype=dtype)

    # Create mask for left contraction
    def lc_mask_entry(i: int, j: int) -> float:
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

    lc_mask = torch.tensor(
        [[lc_mask_entry(i, j) for j in range(num_blades)] for i in range(num_blades)],
        dtype=dtype, device=device
    )

    a_expanded = a_coeffs.unsqueeze(2)
    b_expanded = b_coeffs.unsqueeze(1)

    products = a_expanded * b_expanded * signs.unsqueeze(0) * lc_mask.unsqueeze(0)

    result = torch.zeros(batch_size, num_blades, dtype=dtype, device=device)

    flat_products = products.reshape(batch_size, -1)
    flat_indices = cayley_blades.reshape(-1).expand(batch_size, -1)

    result.scatter_add_(1, flat_indices, flat_products)

    return result


def _sandwich_impl(
    x_coeffs: Tensor,
    rotor_coeffs: Tensor,
    cayley_signs: Tensor,
    cayley_blades: Tensor,
    num_blades: int,
) -> Tensor:
    """Sandwich product: rotor * x * ~rotor.

    Computes R * X * R~ for each X in the batch.
    """
    dtype = x_coeffs.dtype
    device = x_coeffs.device

    # Broadcast rotor if needed
    if rotor_coeffs.shape[0] == 1 and x_coeffs.shape[0] > 1:
        rotor_coeffs = rotor_coeffs.expand(x_coeffs.shape[0], -1)

    # Compute rotor reverse
    rev_signs = torch.tensor(
        [_reverse_sign(_blade_grade(i)) for i in range(num_blades)],
        dtype=dtype, device=device
    )
    rotor_rev = rotor_coeffs * rev_signs

    # First compute R * X
    rx = _geometric_product_impl(rotor_coeffs, x_coeffs, cayley_signs, cayley_blades)

    # Then compute (R * X) * ~R
    result = _geometric_product_impl(rx, rotor_rev, cayley_signs, cayley_blades)

    return result
