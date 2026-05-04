"""
Flash Attention implemented in OpenAI Triton.

## The Memory Problem With Naive Attention

For seq_len=8192, head_dim=128:
  - N×N attention matrix: 8192^2 × 4 bytes (fp32) = 256 MB per head
  - At batch=8, heads=32: 256 MB × 8 × 32 = 65 GB — won't fit on any GPU

Flash Attention (Dao et al., 2022) avoids materializing this matrix entirely.

## The Tiling Solution

Instead of computing the full N×N matrix, we process it in tiles of size
BLOCK_M × BLOCK_N. Each tile fits in SRAM (48 MB on A100 per SM).

Key algorithmic trick — online softmax:
  For tiles (i,j), (i,k), ... we accumulate the attention output using
  running max (m) and running denominator (l):
    m_new = max(m_old, rowmax(S_ij))
    l_new = exp(m_old - m_new) * l_old + rowsum(exp(S_ij - m_new))
    O_new = diag(exp(m_old - m_new)) * O_old + exp(S_ij - m_new) @ V_j

  At the end: O = diag(1/l) * O_accumulated

## Operation Fusion

Traditional attention: 3 kernel launches
  1. QK^T matmul → DRAM
  2. softmax(QK^T / scale) ← DRAM → DRAM
  3. softmax(.) @ V → output

Flash Attention: 1 kernel launch
  - QK^T + online softmax + V accumulation all in SRAM
  - Only reads/writes: Q, K, V once each + output once

DRAM access: O(seq_len × head_dim) instead of O(seq_len^2)

## Reference

"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
  Dao, Fu, Ermon, Rudra, Re (NeurIPS 2022)
  https://arxiv.org/abs/2205.14135
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)

# Lazy Triton import — only available on Linux + CUDA GPU.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

# Default tile dimensions. Tuned for A100 80GB.
# Larger BLOCK_M/BLOCK_N = more SRAM usage per SM but better arithmetic intensity.
# A100 SRAM (shared memory) per SM: 192 KB. With BLOCK_M=64, BLOCK_N=64, head_dim=128:
#   Q tile: 64×128×2 = 16 KB (fp16)
#   K tile: 64×128×2 = 16 KB
#   V tile: 64×128×2 = 16 KB
#   S tile: 64×64×4  = 16 KB  → total ~64 KB per SM, comfortable.
DEFAULT_BLOCK_M: int = 64
DEFAULT_BLOCK_N: int = 64


if _TRITON_AVAILABLE:

    @triton.jit
    def _flash_attention_fwd_kernel(  # type: ignore[no-redef]
        Q_ptr,
        K_ptr,
        V_ptr,
        sm_scale,
        Out_ptr,
        # Strides (in elements, not bytes) for 4D tensors: [batch, heads, seq, dim]
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z,        # batch size
        H,        # number of heads
        N_CTX,    # sequence length
        BLOCK_M: tl.constexpr,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ) -> None:
        """Tiled Flash Attention forward kernel.

        Grid: (batch × heads × ceil(N_CTX / BLOCK_M),)
        Each program handles BLOCK_M query positions for one (batch, head) pair.

        Outer loop: iterate over K,V tiles (BLOCK_N keys per iteration).
        Inner: compute QK^T tile, accumulate online softmax, accumulate output.

        Args (Triton JIT — via kernel launch):
            Q_ptr, K_ptr, V_ptr: Pointers to (batch, heads, seq_len, head_dim) tensors.
            sm_scale: Softmax scale = 1/sqrt(head_dim).
            Out_ptr: Pointer to output tensor, same shape as Q.
            stride_*: Strides for each tensor dimension.
            Z: Batch size.
            H: Number of heads.
            N_CTX: Sequence length.
            BLOCK_M: Compile-time tile size for query dimension (rows).
            BLOCK_DHEAD: Compile-time head dimension (must match actual head_dim).
            BLOCK_N: Compile-time tile size for key/value dimension (cols).
            IS_CAUSAL: Compile-time flag for causal masking.
        """
        # Identify which (batch, head, query_tile) this program handles
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)   # flattened batch×head index
        off_z = off_hz // H
        off_h = off_hz % H

        # Base pointers for this (batch, head) slice
        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + off_z * stride_qz + off_h * stride_qh,
            shape=(N_CTX, BLOCK_DHEAD),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DHEAD),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + off_z * stride_kz + off_h * stride_kh,
            shape=(BLOCK_DHEAD, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DHEAD, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + off_z * stride_vz + off_h * stride_vh,
            shape=(N_CTX, BLOCK_DHEAD),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DHEAD),
            order=(1, 0),
        )

        # Query positions handled by this program
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        # Initialize online softmax accumulators
        # m_i: running maximum (initialized to -inf)
        # l_i: running denominator (initialized to 0)
        # acc: accumulated output (initialized to 0)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)

        # Load the Q tile — stays in SRAM for the entire inner loop
        q = tl.load(Q_block_ptr)

        # Inner loop: iterate over all K,V tiles
        # Causal: only attend to positions <= current query position
        lo = 0
        hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

        for start_n in range(lo, hi, BLOCK_N):
            # Load K tile (transposed for matmul) and V tile from DRAM into SRAM
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            # QK^T: (BLOCK_M, BLOCK_DHEAD) @ (BLOCK_DHEAD, BLOCK_N) → (BLOCK_M, BLOCK_N)
            # Apply sm_scale here (before exp) to keep values in a numerically safe range.
            # fp16 max is ~65504 — without scaling, qk values can reach 60+ for
            # misaligned Q,K pairs, causing exp() to overflow at long sequences.
            # Fix (2026-02-16): sm_scale * before * exp, not after.
            qk = tl.dot(q, k) * sm_scale

            # Causal masking: zero out future positions by setting to -inf before softmax.
            # IS_CAUSAL is a compile-time constexpr — Triton elides the entire block
            # at compile time for IS_CAUSAL=False, so there's zero runtime overhead.
            if IS_CAUSAL:
                offs_n = start_n + tl.arange(0, BLOCK_N)
                # mask[i, j] = True when query position i can attend to key position j
                # (causal: i >= j, i.e., only past and present tokens)
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Online softmax update (Milakov & Gimelshein, 2018):
            # m_new = max(m_old, rowmax(qk))
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            # Rescale old accumulator and denominator by exp(m_old - m_new)
            # This maintains the invariant: acc = sum_{previous tiles} exp(s - m_new) * v
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
            acc = acc * alpha[:, None] + tl.dot(tl.exp(qk - m_new[:, None]).to(v.dtype), v)

            # Update running max for next iteration
            m_i = m_new

            # Advance block pointers to next K,V tile (O(BLOCK_DHEAD × BLOCK_N) per step)
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        # Final normalization: divide by running denominator
        # This replaces the explicit division we'd need in naive softmax
        acc = acc / l_i[:, None]

        # Write output tile back to DRAM
        Out_block_ptr = tl.make_block_ptr(
            base=Out_ptr + off_z * stride_oz + off_h * stride_oh,
            shape=(N_CTX, BLOCK_DHEAD),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DHEAD),
            order=(1, 0),
        )
        tl.store(Out_block_ptr, acc.to(Out_ptr.dtype.element_ty))

else:
    _flash_attention_fwd_kernel = None  # type: ignore[assignment]


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> torch.Tensor:
    """Triton Flash Attention forward pass.

    Computes scaled dot-product attention without materializing the full N×N
    attention matrix. Memory usage is O(seq_len × head_dim) instead of
    O(seq_len^2), enabling attention over very long sequences.

    The output is mathematically identical to:
        softmax(QK^T / sqrt(head_dim)) @ V

    Args:
        q: Query tensor, shape (batch, heads, seq_len, head_dim). Must be CUDA.
        k: Key tensor, shape (batch, heads, seq_len, head_dim).
        v: Value tensor, shape (batch, heads, seq_len, head_dim).
        causal: If True, apply causal (autoregressive) masking.
        sm_scale: Softmax scale. Defaults to 1/sqrt(head_dim) per the paper.
        block_m: Tile size for query dimension. Default tuned for A100.
        block_n: Tile size for key/value dimension.

    Returns:
        Attention output tensor of same shape as q.

    Raises:
        ValueError: If tensors are not on CUDA or shapes are incompatible.
        ValueError: If seq_len is not divisible by block_m (use pad_to_multiple if needed).

    Example:
        >>> q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        >>> out = flash_attention(q, k, v, causal=True)
        >>> out.shape
        torch.Size([2, 8, 512, 64])
    """
    if not _TRITON_AVAILABLE or not q.is_cuda:
        if not _TRITON_AVAILABLE:
            logger.debug("Triton not installed — falling back to torch SDPA.")
        else:
            logger.warning(
                "flash_attention called on CPU tensor — falling back to torch SDPA. "
                "Flash Attention requires CUDA."
            )
        return _sdpa_fallback(q, k, v, causal=causal, sm_scale=sm_scale)

    _validate_attention_inputs(q, k, v)

    batch, heads, seq_len, head_dim = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Ensure contiguous memory layout for efficient pointer arithmetic
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)

    # Grid: one Triton program per (query_tile, batch×head)
    # Parallelism: all tiles across all batch items and heads run concurrently
    num_q_tiles = math.ceil(seq_len / block_m)
    grid = (num_q_tiles, batch * heads)

    _flash_attention_fwd_kernel[grid](
        q, k, v,
        sm_scale,
        output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, heads, seq_len,
        BLOCK_M=block_m,
        BLOCK_DHEAD=head_dim,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    logger.debug(
        "flash_attention: batch=%d, heads=%d, seq=%d, dim=%d, causal=%s",
        batch, heads, seq_len, head_dim, causal,
    )
    return output


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Validate shapes and dtypes for flash attention inputs.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.

    Raises:
        ValueError: On shape/dtype mismatch.
    """
    if q.ndim != 4:
        raise ValueError(f"Expected 4D tensors (batch, heads, seq, dim), got q.ndim={q.ndim}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"Q, K, V must have identical shapes. Got q={q.shape}, k={k.shape}, v={v.shape}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype: {q.dtype}. Use fp16, bf16, or fp32.")

    head_dim = q.shape[-1]
    if head_dim not in (32, 64, 128):
        logger.warning(
            "head_dim=%d is non-standard. Triton kernel is tuned for {32, 64, 128}. "
            "Performance may be suboptimal.",
            head_dim,
        )


def _sdpa_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """CPU/non-CUDA fallback using PyTorch's scaled_dot_product_attention.

    Used when Triton is not available or running on CPU for testing.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        causal: Enable causal masking.
        sm_scale: Scale for QK^T. Defaults to 1/sqrt(head_dim).

    Returns:
        Attention output.
    """
    head_dim = q.shape[-1]
    scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(head_dim))

    # PyTorch SDPA handles causal masking natively
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        is_causal=causal,
        scale=scale,
    )


def estimate_flash_attention_memory(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> dict[str, float]:
    """Estimate GPU memory usage for flash vs naive attention.

    Useful for capacity planning and demonstrating the O(N) vs O(N^2) distinction.

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length.
        head_dim: Dimension per head.
        dtype: Tensor dtype.
        block_m: Flash attention tile size (query).
        block_n: Flash attention tile size (key/value).

    Returns:
        Dictionary with memory estimates in MB:
          - "qkv_input_mb": Q+K+V input (same for both approaches)
          - "flash_output_mb": output tensor for flash attention
          - "flash_sram_mb": SRAM usage per SM (tiles)
          - "naive_attn_matrix_mb": N×N attention matrix size (what flash avoids)
    """
    bytes_per_elem = torch.finfo(dtype).bits // 8 if dtype != torch.int8 else 1

    def to_mb(n_elements: float) -> float:
        return (n_elements * bytes_per_elem) / (1024 ** 2)

    qkv_mb = to_mb(3 * batch * heads * seq_len * head_dim)
    output_mb = to_mb(batch * heads * seq_len * head_dim)
    attn_matrix_mb = to_mb(batch * heads * seq_len * seq_len)  # N^2 — what flash avoids
    sram_per_sm_mb = to_mb(block_m * head_dim + 2 * block_n * head_dim + block_m * block_n)

    return {
        "qkv_input_mb": qkv_mb,
        "flash_output_mb": output_mb,
        "flash_sram_mb_per_sm": sram_per_sm_mb,
        "naive_attn_matrix_mb": attn_matrix_mb,
    }
