"""
triton-inference-kernels: GPU inference kernels in OpenAI Triton.

Implements fused softmax and Flash Attention with operation fusion and memory
coalescing. Demonstrates core GPU optimization techniques for LLM inference:
- Single-pass online normalization (avoids redundant global memory reads)
- SRAM-resident tiled computation (DRAM bandwidth reduction)
- Operation fusion (QK^T + softmax + AV in one kernel launch)
"""

from triton_kernels.softmax import fused_softmax
from triton_kernels.attention import flash_attention

__version__ = "0.1.0"
__author__ = "Rajath John"
__email__ = "jrajath94@gmail.com"

__all__ = [
    "fused_softmax",
    "flash_attention",
]
