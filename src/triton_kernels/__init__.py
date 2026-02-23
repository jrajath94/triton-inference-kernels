"""
triton-inference-kernels: GPU inference kernels in OpenAI Triton.

Implements fused softmax and Flash Attention with operation fusion and memory
coalescing. Demonstrates core GPU optimization techniques for LLM inference:
- Single-pass online normalization (avoids redundant global memory reads)
- SRAM-resident tiled computation (DRAM bandwidth reduction)
- Operation fusion (QK^T + softmax + AV in one kernel launch)

Note: Triton GPU kernels require `triton>=2.1.0` and a CUDA-capable GPU.
On CPU-only machines, all functions gracefully fall back to PyTorch equivalents.
"""

from triton_kernels.softmax import fused_softmax  # noqa: F401
from triton_kernels.attention import flash_attention  # noqa: F401

__version__ = "0.1.0"
__author__ = "Rajath John"
__email__ = "jrajath94@gmail.com"

__all__ = [
    "fused_softmax",
    "flash_attention",
]
