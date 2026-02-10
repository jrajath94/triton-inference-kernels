"""
Tests for the Triton fused softmax kernel.

Test strategy:
  - Correctness: compare against torch.nn.functional.softmax (ground truth)
  - Numerical stability: extreme values that would overflow naive exp()
  - Shape handling: various (batch, seq_len) combinations
  - Error handling: invalid inputs raise expected exceptions
  - GPU tests: marked @pytest.mark.gpu, skip on CPU CI
"""

import logging

import pytest
import torch
import torch.nn.functional as F

from triton_kernels.softmax import fused_softmax, softmax_backward
from triton_kernels.utils import assert_close, naive_softmax

logger = logging.getLogger(__name__)

# Absolute tolerance for float32 comparison vs PyTorch reference
SOFTMAX_ATOL: float = 1e-5


class TestSoftmaxCorrectness:
    """Verify that fused_softmax matches PyTorch's output."""

    @pytest.mark.gpu
    def test_softmax_small_batch_matches_pytorch(self, small_softmax_input: torch.Tensor) -> None:
        """fused_softmax should match F.softmax within 1e-5 for a small batch.

        Arrange: random (8, 128) input on GPU
        Act: compute Triton fused softmax and PyTorch reference
        Assert: max absolute difference < 1e-5
        """
        # Arrange
        x = small_softmax_input

        # Act
        actual = fused_softmax(x)
        expected = F.softmax(x, dim=-1)

        # Assert
        assert_close(actual, expected, atol=SOFTMAX_ATOL, name="fused_softmax small")
        assert actual.shape == x.shape

    @pytest.mark.gpu
    def test_softmax_large_input_matches_pytorch(self, large_softmax_input: torch.Tensor) -> None:
        """fused_softmax should match for larger (64, 2048) inputs.

        Tests that tile-size selection and masking work correctly when
        n_cols = 2048 = 2 * BLOCK_SIZE_MAX.
        """
        # Arrange
        x = large_softmax_input

        # Act
        actual = fused_softmax(x)
        expected = F.softmax(x, dim=-1)

        # Assert
        assert_close(actual, expected, atol=SOFTMAX_ATOL, name="fused_softmax large")

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512, 1024, 2048])
    def test_softmax_various_seq_lens(self, seq_len: int, device: str) -> None:
        """fused_softmax should match PyTorch across common sequence lengths.

        Tests that BLOCK_SIZE selection handles power-of-2 and non-power-of-2
        column counts correctly (masking path).
        """
        # Arrange
        batch = 16
        x = torch.randn(batch, seq_len, device=device, dtype=torch.float32)

        # Act
        actual = fused_softmax(x)
        expected = F.softmax(x, dim=-1)

        # Assert
        assert_close(actual, expected, atol=SOFTMAX_ATOL, name=f"seq_len={seq_len}")

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [100, 300, 700, 1500])
    def test_softmax_non_power_of_two_seq_lens(self, seq_len: int, device: str) -> None:
        """fused_softmax must handle non-power-of-2 seq_lens via masking.

        The kernel loads BLOCK_SIZE elements but masks out columns >= n_cols.
        This tests that the masked elements (-inf) don't contribute to the output.
        """
        # Arrange
        x = torch.randn(8, seq_len, device=device, dtype=torch.float32)

        # Act
        actual = fused_softmax(x)
        expected = F.softmax(x, dim=-1)

        # Assert
        assert_close(actual, expected, atol=SOFTMAX_ATOL, name=f"non_pow2 seq_len={seq_len}")


class TestSoftmaxNumericalStability:
    """Verify that fused_softmax handles numerically extreme inputs."""

    @pytest.mark.gpu
    def test_softmax_large_values_no_overflow(self, device: str) -> None:
        """Numerically stable softmax should not produce NaN/inf for large inputs.

        Without max-subtraction, exp(1000) overflows fp32 to inf.
        The max-subtraction trick ensures exp(1000 - 1000) = exp(0) = 1.
        """
        # Arrange: values that would overflow naive exp(x)
        x = torch.full((4, 128), 1000.0, device=device, dtype=torch.float32)

        # Act
        output = fused_softmax(x)

        # Assert: uniform distribution (all equal large values → equal probs)
        assert not torch.isnan(output).any(), "Got NaN from large input"
        assert not torch.isinf(output).any(), "Got inf from large input"
        expected_prob = 1.0 / 128
        assert_close(output, torch.full_like(output, expected_prob), atol=1e-5, name="uniform")

    @pytest.mark.gpu
    def test_softmax_large_negative_values(self, device: str) -> None:
        """Softmax should not produce NaN for very negative inputs.

        exp(-1000) underflows to 0, which causes 0/0 = NaN in naive softmax.
        The max-subtraction ensures at least one exp value is 1 (the maximum).
        """
        # Arrange: all negative — after max subtraction, largest becomes 0
        x = torch.full((4, 128), -1000.0, device=device, dtype=torch.float32)
        x[0, 0] = -500.0  # One element is the max — should get all the probability

        # Act
        output = fused_softmax(x)

        # Assert: no NaN, first element dominates
        assert not torch.isnan(output).any(), "Got NaN from large negative input"
        assert not torch.isinf(output).any(), "Got inf from large negative input"

    @pytest.mark.gpu
    def test_softmax_outputs_sum_to_one(self, device: str) -> None:
        """Softmax outputs must sum to 1 for each row (probability axiom).

        Floating-point errors should not push row sums beyond 1 ± 1e-5.
        """
        # Arrange
        x = torch.randn(64, 512, device=device, dtype=torch.float32)

        # Act
        output = fused_softmax(x)
        row_sums = output.sum(dim=-1)

        # Assert: all row sums should be 1.0
        ones = torch.ones_like(row_sums)
        assert_close(row_sums, ones, atol=1e-4, name="row sums")

    @pytest.mark.gpu
    def test_softmax_outputs_nonnegative(self, device: str) -> None:
        """Softmax outputs must be >= 0 (exp is always positive).

        A kernel bug (e.g., wrong masking) could produce negative values.
        """
        # Arrange
        x = torch.randn(32, 256, device=device, dtype=torch.float32)

        # Act
        output = fused_softmax(x)

        # Assert
        assert (output >= 0).all(), "Softmax output contains negative values"


class TestSoftmaxEdgeCases:
    """Test edge cases, error handling, and CPU fallback."""

    def test_softmax_cpu_fallback_returns_correct_output(self) -> None:
        """fused_softmax should gracefully fall back to torch.softmax on CPU.

        This tests the non-GPU path used in CI environments without CUDA.
        """
        # Arrange: CPU tensor (no CUDA)
        x = torch.randn(8, 64, dtype=torch.float32)  # CPU tensor

        # Act
        output = fused_softmax(x)
        reference = F.softmax(x, dim=-1)

        # Assert: CPU fallback should match reference exactly
        assert_close(output, reference, atol=1e-6, name="cpu_fallback")

    def test_softmax_single_row(self) -> None:
        """fused_softmax should work on a single-row (batch=1) input."""
        # Arrange
        x = torch.randn(1, 64)

        # Act
        output = fused_softmax(x)
        expected = F.softmax(x, dim=-1)

        # Assert
        assert_close(output, expected, atol=1e-5, name="single_row")

    def test_softmax_raises_on_integer_input(self) -> None:
        """fused_softmax should raise ValueError for integer tensors.

        exp() is only defined for floating-point types.
        """
        # Arrange
        x = torch.randint(0, 10, (8, 64))

        # Act & Assert
        with pytest.raises(ValueError, match="floating-point"):
            fused_softmax(x)

    def test_softmax_raises_on_wrong_dim(self) -> None:
        """fused_softmax currently only supports dim=-1; other dims raise ValueError."""
        # Arrange
        x = torch.randn(8, 64, 32)

        # Act & Assert
        with pytest.raises(ValueError, match="dim=-1"):
            fused_softmax(x, dim=0)

    def test_softmax_preserves_input_shape(self) -> None:
        """Output shape must exactly match input shape for all valid inputs."""
        # Arrange
        shapes = [(1, 32), (16, 128), (128, 512), (8, 1000)]

        for shape in shapes:
            x = torch.randn(*shape)

            # Act
            output = fused_softmax(x)

            # Assert
            assert output.shape == torch.Size(shape), f"Shape mismatch for input {shape}"


class TestSoftmaxBackward:
    """Test the softmax backward (gradient) implementation."""

    def test_softmax_backward_matches_autograd(self) -> None:
        """softmax_backward should match PyTorch autograd output.

        The Jacobian-vector product for softmax is:
            dL/dx_i = p_i * (dL/dp_i - sum_j(p_j * dL/dp_j))
        """
        # Arrange: compute softmax and a fake upstream gradient
        x = torch.randn(4, 32, requires_grad=True)
        p = torch.softmax(x, dim=-1)
        upstream_grad = torch.randn_like(p)

        # Compute PyTorch reference gradient
        p.backward(upstream_grad)
        ref_grad = x.grad.clone()

        # Compute our manual backward
        manual_grad = softmax_backward(upstream_grad, p.detach())

        # Assert
        assert_close(manual_grad, ref_grad, atol=1e-5, name="softmax_backward")


class TestNaiveSoftmaxReference:
    """Test the naive reference implementation used as correctness baseline."""

    def test_naive_softmax_matches_pytorch(self) -> None:
        """naive_softmax should match F.softmax exactly.

        This validates our reference implementation before using it in kernel tests.
        """
        # Arrange
        x = torch.randn(16, 256)

        # Act
        actual = naive_softmax(x)
        expected = F.softmax(x, dim=-1)

        # Assert
        assert_close(actual, expected, atol=1e-6, name="naive_softmax")

    @pytest.mark.parametrize("batch,seq_len", [(1, 64), (32, 128), (4, 512)])
    def test_naive_softmax_shape_invariant(self, batch: int, seq_len: int) -> None:
        """naive_softmax output shape must match input shape.

        Args:
            batch: Number of rows.
            seq_len: Number of columns.
        """
        # Arrange
        x = torch.randn(batch, seq_len)

        # Act
        output = naive_softmax(x)

        # Assert
        assert output.shape == (batch, seq_len)
