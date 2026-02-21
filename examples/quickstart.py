"""
Quick-start example: using triton-inference-kernels.

Demonstrates:
  1. Fused softmax — correctness verification + timing
  2. Flash attention — shape demo + memory estimate

Run:
    python examples/quickstart.py

Works on CPU (fallback mode) and CUDA GPU.
"""

import logging
import math
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def demo_fused_softmax() -> None:
    """Demonstrate fused softmax kernel."""
    from triton_kernels.softmax import fused_softmax

    print("\n--- Fused Softmax ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create input: batch of 64 sequences, each length 1024
    batch, seq_len = 64, 1024
    x = torch.randn(batch, seq_len, device=device, dtype=torch.float32)

    # Run fused softmax
    output = fused_softmax(x)

    # Verify against PyTorch
    reference = torch.softmax(x, dim=-1)
    max_err = (output - reference).abs().max().item()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Max error vs PyTorch: {max_err:.2e}  {'PASS' if max_err < 1e-4 else 'FAIL'}")

    # Verify probability axioms
    row_sums = output.sum(dim=-1)
    print(f"Row sums in [{row_sums.min().item():.6f}, {row_sums.max().item():.6f}]  (expect: ~1.0)")
    print(f"All non-negative: {(output >= 0).all().item()}")


def demo_flash_attention() -> None:
    """Demonstrate flash attention kernel."""
    from triton_kernels.attention import flash_attention, estimate_flash_attention_memory

    print("\n--- Flash Attention ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Standard transformer attention configuration
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Config: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Forward pass
    output = flash_attention(q, k, v, causal=False)
    print(f"Output shape: {output.shape}")

    # Causal (autoregressive) attention
    output_causal = flash_attention(q, k, v, causal=True)
    print(f"Causal output shape: {output_causal.shape}")

    # Compare vs PyTorch SDPA
    sm_scale = 1.0 / math.sqrt(head_dim)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    max_err = (output.float() - ref.float()).abs().max().item()
    print(f"Max error vs PyTorch SDPA: {max_err:.2e}  {'PASS' if max_err < 1e-2 else 'FAIL'}")

    # Memory comparison
    print("\nMemory scaling (why flash attention matters):")
    for sl in [512, 1024, 2048, 4096]:
        est = estimate_flash_attention_memory(
            batch=batch, heads=heads, seq_len=sl, head_dim=head_dim, dtype=dtype
        )
        print(
            f"  seq_len={sl:5d}: naive N×N matrix = {est['naive_attn_matrix_mb']:6.1f} MB | "
            f"flash SRAM/SM = {est['flash_sram_mb_per_sm']:.2f} MB"
        )


def main() -> None:
    """Run all quickstart demos."""
    import triton_kernels
    print(f"triton-inference-kernels v{triton_kernels.__version__}")
    print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: Running in CPU fallback mode. GPU kernels will use PyTorch fallbacks.")

    demo_fused_softmax()
    demo_flash_attention()
    print("\nAll demos complete.")


if __name__ == "__main__":
    main()
