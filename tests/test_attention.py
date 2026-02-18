"""
Tests for the Triton Flash Attention kernel.

Test strategy:
  - Correctness: compare against naive attention and PyTorch SDPA
  - Causal masking: verify upper triangle is masked out
  - Parametrized seq_len: ensure tiling logic is correct for various lengths
  - Memory estimation: verify O(N) vs O(N^2) memory calculations
  - Error handling: invalid inputs raise expected exceptions
  - CPU fallback: graceful degradation when CUDA unavailable
"""

import logging
import math

import pytest
import torch
import torch.nn.functional as F

from triton_kernels.attention import (
    _sdpa_fallback,
    _validate_attention_inputs,
    estimate_flash_attention_memory,
    flash_attention,
)
from triton_kernels.utils import assert_close, naive_attention

logger = logging.getLogger(__name__)

# Attention outputs are accumulated in float32 then cast to output dtype.
# The tolerance is looser than softmax because of the additional matmul precision.
ATTN_ATOL_FP16: float = 1e-2   # fp16 accumulation
ATTN_ATOL_FP32: float = 1e-4   # fp32 accumulation


class TestFlashAttentionCorrectness:
    """Verify that flash_attention matches reference implementations."""

    @pytest.mark.gpu
    def test_attention_matches_naive_small(
        self,
        attention_inputs_small: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """flash_attention should match naive_attention for small (2,4,128,64) inputs.

        Arrange: Q,K,V tensors (2 batch, 4 heads, 128 seq, 64 dim) on GPU
        Act: Triton flash attention and naive reference
        Assert: max absolute difference < ATTN_ATOL_FP16
        """
        # Arrange
        q, k, v = attention_inputs_small
        q_fp32 = q.float()
        k_fp32 = k.float()
        v_fp32 = v.float()

        # Act
        actual = flash_attention(q, k, v).float()
        expected = naive_attention(q_fp32, k_fp32, v_fp32)

        # Assert
        assert_close(actual, expected, atol=ATTN_ATOL_FP16, name="flash_vs_naive_small")
        assert actual.shape == q.shape

    @pytest.mark.gpu
    def test_attention_matches_pytorch_sdpa(
        self,
        attention_inputs_medium: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """flash_attention should match PyTorch scaled_dot_product_attention.

        SDPA is PyTorch's optimized reference — matching it validates both
        correctness and the softmax scale factor (1/sqrt(head_dim)).
        """
        # Arrange
        q, k, v = attention_inputs_medium
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

        # Act
        actual = flash_attention(q, k, v, sm_scale=sm_scale).float()
        expected = F.scaled_dot_product_attention(q, k, v, scale=sm_scale).float()

        # Assert
        assert_close(actual, expected, atol=ATTN_ATOL_FP16, name="flash_vs_sdpa_medium")

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024, 2048])
    def test_attention_various_seq_lens(self, seq_len: int, device: str) -> None:
        """flash_attention should produce correct output across common seq_lens.

        Each seq_len exercises a different number of tiling iterations.
        For seq_len=128, BLOCK_M=64: 2 query tiles, 2 key tiles = 4 tile pairs.
        For seq_len=2048, BLOCK_M=64: 32 query tiles × 32 key tiles = 1024 pairs.

        Args:
            seq_len: Sequence length to test.
            device: Test device from fixture.
        """
        # Arrange
        batch, heads, head_dim = 2, 4, 64
        dtype = torch.float16 if device == "cuda" else torch.float32
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Act
        actual = flash_attention(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v)

        # Assert
        tol = ATTN_ATOL_FP16 if device == "cuda" else ATTN_ATOL_FP32
        assert_close(actual.float(), expected.float(), atol=tol, name=f"seq_len={seq_len}")


class TestFlashAttentionCausalMask:
    """Verify that causal masking correctly prevents future-token attention."""

    @pytest.mark.gpu
    def test_causal_matches_pytorch_sdpa_causal(
        self,
        attention_inputs_small: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Causal flash attention should match PyTorch SDPA with is_causal=True.

        Causal masking: position i should only attend to positions 0..i.
        The upper triangle of the attention matrix should be zeroed out.

        Arrange: standard small attention inputs on GPU
        Act: flash_attention(causal=True) vs F.scaled_dot_product_attention(is_causal=True)
        Assert: outputs match within tolerance
        """
        # Arrange
        q, k, v = attention_inputs_small

        # Act
        actual = flash_attention(q, k, v, causal=True).float()
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True).float()

        # Assert
        assert_close(actual, expected, atol=ATTN_ATOL_FP16, name="causal_flash_vs_sdpa")

    @pytest.mark.gpu
    def test_causal_first_token_only_attends_to_itself(self, device: str) -> None:
        """First token (position 0) must attend only to itself in causal mode.

        If causal masking is broken, position 0 would have attention bleed from
        future positions, which is physically impossible in autoregressive generation.
        The output of position 0 with causal=True must equal causal=False for pos 0
        because no future positions exist to mask.
        """
        # Arrange
        batch, heads, seq_len, head_dim = 1, 1, 8, 32
        dtype = torch.float16 if device == "cuda" else torch.float32
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Act: causal and non-causal should agree for position 0
        # (position 0 has nothing to mask regardless)
        causal_out = flash_attention(q, k, v, causal=True)
        noncausal_out = flash_attention(q, k, v, causal=False)

        # Assert: first token output should be the same in both cases
        assert_close(
            causal_out[:, :, 0, :].float(),
            noncausal_out[:, :, 0, :].float(),
            atol=ATTN_ATOL_FP16,
            name="first_token_causal_vs_noncausal",
        )

    @pytest.mark.gpu
    def test_causal_last_token_differs_from_noncausal(self, device: str) -> None:
        """Last token should differ between causal and non-causal (sanity check).

        If causal=True produces the same output as causal=False for the last token,
        the causal mask is not being applied at all.
        """
        # Arrange
        batch, heads, seq_len, head_dim = 1, 1, 16, 32
        dtype = torch.float16 if device == "cuda" else torch.float32
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Act
        causal_out = flash_attention(q, k, v, causal=True).float()
        noncausal_out = flash_attention(q, k, v, causal=False).float()

        # Assert: last token output should differ (causal attends to seq_len positions,
        # noncausal also attends to all seq_len positions — actually they can match;
        # the real test is that an intermediate token differs)
        # Test position seq_len//2 instead: in causal it attends to half the keys
        mid = seq_len // 2
        diff = (causal_out[:, :, mid, :] - noncausal_out[:, :, mid, :]).abs().max().item()
        assert diff > 1e-3, (
            f"Causal and non-causal outputs are suspiciously similar at position {mid}: "
            f"max_diff={diff:.2e}. Causal mask may not be applied."
        )


class TestFlashAttentionMemory:
    """Test the memory estimation utility."""

    def test_flash_memory_is_less_than_naive(self) -> None:
        """Flash attention should use O(N) memory while naive uses O(N^2).

        For large seq_len, flash should require substantially less GPU memory.
        """
        # Arrange: large sequence where the difference is pronounced
        params = dict(batch=2, heads=8, seq_len=2048, head_dim=64, dtype=torch.float16)

        # Act
        mem_estimates = estimate_flash_attention_memory(**params)

        # Assert: naive N×N attention matrix should dwarf flash's SRAM usage
        naive_mb = mem_estimates["naive_attn_matrix_mb"]
        sram_mb = mem_estimates["flash_sram_mb_per_sm"]

        assert naive_mb > sram_mb * 10, (
            f"Expected naive ({naive_mb:.1f} MB) >> flash SRAM ({sram_mb:.1f} MB)"
        )
        logger.info(
            "Memory: naive=%.1f MB, flash SRAM=%.1f MB (%.0fx reduction)",
            naive_mb, sram_mb, naive_mb / sram_mb,
        )

    def test_memory_estimate_returns_positive_values(self) -> None:
        """All memory estimates should be positive floats."""
        # Arrange & Act
        estimates = estimate_flash_attention_memory(
            batch=1, heads=4, seq_len=512, head_dim=64
        )

        # Assert
        for key, value in estimates.items():
            assert value > 0, f"{key} should be positive, got {value}"

    def test_memory_scales_quadratically_with_seq_len_for_naive(self) -> None:
        """Naive attention memory should scale as O(N^2).

        Doubling seq_len should quadruple the naive attention matrix size.
        """
        # Arrange
        base_params = dict(batch=1, heads=1, head_dim=64, dtype=torch.float32)

        # Act
        mem_512 = estimate_flash_attention_memory(seq_len=512, **base_params)
        mem_1024 = estimate_flash_attention_memory(seq_len=1024, **base_params)

        # Assert: 2x seq_len → 4x naive memory
        ratio = mem_1024["naive_attn_matrix_mb"] / mem_512["naive_attn_matrix_mb"]
        assert abs(ratio - 4.0) < 0.1, f"Expected 4x scaling, got {ratio:.2f}x"


class TestFlashAttentionErrorHandling:
    """Test that invalid inputs raise clear errors."""

    def test_raises_on_mismatched_shapes(self) -> None:
        """flash_attention should raise ValueError when Q,K,V shapes differ."""
        # Arrange
        q = torch.randn(2, 4, 128, 64)
        k = torch.randn(2, 4, 256, 64)  # Different seq_len
        v = torch.randn_like(q)

        # Act & Assert
        with pytest.raises(ValueError, match="identical shapes"):
            _validate_attention_inputs(q, k, v)

    def test_raises_on_3d_input(self) -> None:
        """flash_attention requires 4D tensors (batch, heads, seq, dim)."""
        # Arrange
        q = torch.randn(4, 128, 64)  # Missing batch dimension

        # Act & Assert
        with pytest.raises(ValueError, match="4D"):
            _validate_attention_inputs(q, q, q)

    def test_cpu_fallback_matches_sdpa(self) -> None:
        """_sdpa_fallback should match F.scaled_dot_product_attention on CPU."""
        # Arrange
        q = torch.randn(2, 4, 64, 32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Act
        actual = _sdpa_fallback(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v)

        # Assert
        assert_close(actual, expected, atol=1e-5, name="sdpa_fallback")


class TestFlashAttentionSMScale:
    """Test custom scale factor behavior."""

    @pytest.mark.gpu
    def test_default_sm_scale_is_one_over_sqrt_head_dim(self, device: str) -> None:
        """Default sm_scale should be 1/sqrt(head_dim), matching the attention paper."""
        # Arrange
        batch, heads, seq_len, head_dim = 1, 1, 128, 64
        dtype = torch.float16 if device == "cuda" else torch.float32
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        expected_scale = 1.0 / math.sqrt(head_dim)

        # Act: implicit default scale vs explicit 1/sqrt(head_dim)
        out_default = flash_attention(q, k, v)
        out_explicit = flash_attention(q, k, v, sm_scale=expected_scale)

        # Assert: both should be identical (same scale)
        assert_close(out_default.float(), out_explicit.float(), atol=1e-5, name="sm_scale_default")

    @pytest.mark.gpu
    def test_custom_sm_scale_changes_output(self, device: str) -> None:
        """Different sm_scale values should produce different outputs.

        This is a sanity check that sm_scale is actually applied.
        """
        # Arrange
        batch, heads, seq_len, head_dim = 1, 1, 128, 64
        dtype = torch.float16 if device == "cuda" else torch.float32
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Act
        out_default = flash_attention(q, k, v, sm_scale=1.0 / math.sqrt(head_dim)).float()
        out_large = flash_attention(q, k, v, sm_scale=1.0).float()  # No scaling

        # Assert: different scales → different outputs
        diff = (out_default - out_large).abs().max().item()
        assert diff > 1e-3, f"Expected different outputs for different sm_scale, got diff={diff:.2e}"
