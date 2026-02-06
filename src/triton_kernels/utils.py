"""
Utility helpers for tile sizing, dtype validation, and naive reference implementations.

The naive implementations here serve two purposes:
1. Correctness ground-truth for kernel tests
2. Baseline for benchmark comparisons to quantify speedup
"""

import logging
import math
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Triton requires BLOCK_SIZE to be a power of 2 for efficient vectorized loads.
# These constants bound tile selection.
MIN_BLOCK_SIZE: int = 32
MAX_BLOCK_SIZE: int = 4096

# Minimum sequence length where fused kernels typically outperform PyTorch.
# Below this, kernel launch overhead dominates.
FUSION_BREAKEVEN_SEQ_LEN: int = 256


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n.

    Triton kernels require power-of-two block sizes to enable coalesced vector
    loads via tl.arange(). Using the next-power-of-two for n_cols means we may
    load a few extra (masked) elements, but guarantees aligned DRAM accesses.

    Args:
        n: A positive integer.

    Returns:
        Smallest power of two that is >= n.

    Raises:
        ValueError: If n <= 0.

    Examples:
        >>> next_power_of_two(100)
        128
        >>> next_power_of_two(256)
        256
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return 1 << (n - 1).bit_length()


def select_block_size(n_cols: int) -> int:
    """Select an appropriate Triton block size for a given number of columns.

    Strategy: use the next power of two, clamped to [MIN_BLOCK_SIZE, MAX_BLOCK_SIZE].
    For very wide rows (n_cols > 4096), we tile within the kernel instead of
    attempting a single load — handled by the kernel's inner loop.

    Args:
        n_cols: Number of columns (sequence length for softmax, head_dim for attention).

    Returns:
        Block size for Triton kernel launch.
    """
    block = next_power_of_two(n_cols)
    block = max(MIN_BLOCK_SIZE, min(MAX_BLOCK_SIZE, block))
    logger.debug("Selected block_size=%d for n_cols=%d", block, n_cols)
    return block


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax implemented in pure PyTorch.

    This is the two-pass naive implementation:
    Pass 1: read all values to compute max and denominator
    Pass 2: read all values again to normalize

    Used as correctness baseline in tests. The Triton fused_softmax does the
    equivalent in a single pass by computing max and sum simultaneously.

    Args:
        x: Input tensor of arbitrary shape. Softmax is applied along last dim.

    Returns:
        Softmax probabilities with same shape as x.
    """
    # Subtract max for numerical stability (prevents exp overflow)
    x_max = x.max(dim=-1, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=-1, keepdim=True)


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Standard scaled dot-product attention in PyTorch (O(N^2) memory).

    Materializes the full N×N attention matrix in DRAM. For seq_len=8192 at
    fp32 this costs 8192^2 * 4 bytes = 256 MB per head per batch item.
    Flash Attention avoids this entirely via SRAM tiling.

    Used as correctness baseline and memory-cost comparison in benchmarks.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim).
        k: Key tensor of shape (batch, heads, seq_len, head_dim).
        v: Value tensor of shape (batch, heads, seq_len, head_dim).
        causal: If True, applies causal (autoregressive) masking.
        sm_scale: Scale factor for QK^T. Defaults to 1/sqrt(head_dim).

    Returns:
        Attention output tensor of shape (batch, heads, seq_len, head_dim).
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Full O(N^2) attention matrix — this is what flash attention avoids storing
    scores = torch.einsum("bhmd,bhnd->bhmn", q, k) * sm_scale

    if causal:
        seq_len = q.shape[2]
        # Upper triangle (future positions) set to -inf so softmax gives 0
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", attn_weights, v)


def check_gpu_available() -> bool:
    """Check whether a CUDA GPU is available for kernel execution.

    Returns:
        True if CUDA is available and at least one GPU is visible.
    """
    available = torch.cuda.is_available()
    if available:
        device_name = torch.cuda.get_device_name(0)
        logger.info("GPU available: %s", device_name)
    else:
        logger.warning("No GPU available — kernels will fall back to CPU simulation mode")
    return available


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    name: str = "tensor",
) -> None:
    """Assert two tensors are numerically close, with a descriptive error on failure.

    Args:
        actual: Tensor produced by the kernel under test.
        expected: Reference tensor (e.g., from naive PyTorch implementation).
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        name: Label for error message.

    Raises:
        AssertionError: If tensors differ beyond tolerance.
    """
    max_abs_diff = (actual - expected).abs().max().item()
    if max_abs_diff > atol:
        raise AssertionError(
            f"{name} mismatch: max_abs_diff={max_abs_diff:.2e} > atol={atol:.2e}. "
            f"actual[0,0,:4]={actual.flatten()[:4].tolist()}, "
            f"expected[0,0,:4]={expected.flatten()[:4].tolist()}"
        )
    logger.debug("%s check passed: max_abs_diff=%.2e", name, max_abs_diff)
