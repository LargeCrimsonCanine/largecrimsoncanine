"""PyTorch tensor integration for LargeCrimsonCanine.

This module provides PyTorch-backed multivector operations for deep learning
and GPU-accelerated geometric algebra computations. All operations are
differentiable and work with PyTorch's autograd.

Usage:
    >>> from lcc.torch import TorchMultivector
    >>> import torch
    >>> from largecrimsoncanine import Algebra
    >>>
    >>> R3 = Algebra.euclidean(3)
    >>> coeffs = torch.zeros((100, 8), requires_grad=True)  # batch of 100 multivectors
    >>> batch = TorchMultivector(R3, coeffs)
    >>>
    >>> # Operations are differentiable
    >>> result = batch.geometric_product(batch)
    >>> loss = result.coeffs.sum()
    >>> loss.backward()  # Gradients flow through GA operations

    >>> # GPU support
    >>> batch_gpu = batch.cuda()
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TORCH_AVAILABLE:
    from lcc.torch.multivector import TorchMultivector, TorchAlgebra
    __all__ = ["TorchMultivector", "TorchAlgebra", "TORCH_AVAILABLE"]
else:
    __all__ = ["TORCH_AVAILABLE"]

    def _torch_not_available(*args, **kwargs):
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install torch"
        )

    class TorchMultivector:
        """Placeholder class when PyTorch is not installed."""
        def __init__(self, *args, **kwargs):
            _torch_not_available()

    class TorchAlgebra:
        """Placeholder class when PyTorch is not installed."""
        def __init__(self, *args, **kwargs):
            _torch_not_available()
