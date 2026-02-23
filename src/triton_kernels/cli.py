"""
Command-line interface for triton-inference-kernels benchmarking.

Usage:
    triton-kernels bench --kernel softmax --seq-len 1024 --batch 32
    triton-kernels bench --kernel attention --seq-len 512 --heads 8 --head-dim 64
    triton-kernels info
"""

import argparse
import logging
import sys
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity flag.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def cmd_info(args: argparse.Namespace) -> None:
    """Print GPU info and package version.

    Args:
        args: Parsed CLI arguments (unused).
    """
    from triton_kernels import __version__

    print(f"triton-inference-kernels v{__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        print(f"GPU: {device.name}")
        print(f"  Total memory: {device.total_memory / 1e9:.1f} GB")
        print(f"  SM count: {device.multi_processor_count}")
        print(f"  Compute capability: {device.major}.{device.minor}")

    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("Triton: not installed")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run benchmark for a specific kernel.

    Args:
        args: Parsed CLI arguments with kernel, seq_len, batch, heads, head_dim.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU detected — benchmarks will use CPU fallback (not representative)")

    if args.kernel == "softmax":
        _bench_softmax(args.seq_len, args.batch, device)
    elif args.kernel == "attention":
        _bench_attention(args.seq_len, args.batch, args.heads, args.head_dim, device)
    else:
        print(f"Unknown kernel: {args.kernel}. Choose from: softmax, attention")
        sys.exit(1)


def _bench_softmax(seq_len: int, batch: int, device: str) -> None:
    """Benchmark fused softmax vs PyTorch.

    Args:
        seq_len: Number of columns.
        batch: Number of rows.
        device: 'cuda' or 'cpu'.
    """
    from triton_kernels.softmax import fused_softmax

    x = torch.randn(batch, seq_len, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = fused_softmax(x)
        _ = torch.softmax(x, dim=-1)

    if device == "cuda":
        torch.cuda.synchronize()

    # Time Triton kernel
    t0 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    n_iters = 100
    if device == "cuda":
        t0.record()
        for _ in range(n_iters):
            fused_softmax(x)
        t1.record()
        torch.cuda.synchronize()
        triton_ms = t0.elapsed_time(t1) / n_iters

        t0.record()
        for _ in range(n_iters):
            torch.softmax(x, dim=-1)
        t1.record()
        torch.cuda.synchronize()
        pytorch_ms = t0.elapsed_time(t1) / n_iters
    else:
        import time
        start = time.perf_counter()
        for _ in range(n_iters):
            fused_softmax(x)
        triton_ms = (time.perf_counter() - start) * 1000 / n_iters

        start = time.perf_counter()
        for _ in range(n_iters):
            torch.softmax(x, dim=-1)
        pytorch_ms = (time.perf_counter() - start) * 1000 / n_iters

    speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    print(f"\nSoftmax benchmark (batch={batch}, seq_len={seq_len}):")
    print(f"  Triton fused:   {triton_ms:.3f} ms")
    print(f"  PyTorch:        {pytorch_ms:.3f} ms")
    print(f"  Speedup:        {speedup:.2f}x")


def _bench_attention(
    seq_len: int,
    batch: int,
    heads: int,
    head_dim: int,
    device: str,
) -> None:
    """Benchmark flash attention vs PyTorch SDPA.

    Args:
        seq_len: Sequence length.
        batch: Batch size.
        heads: Number of attention heads.
        head_dim: Dimension per head.
        device: 'cuda' or 'cpu'.
    """
    from triton_kernels.attention import flash_attention

    dtype = torch.float16 if device == "cuda" else torch.float32
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print(f"\nAttention benchmark (batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}):")

    if device == "cuda":
        # Warmup
        for _ in range(3):
            flash_attention(q, k, v)
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        n_iters = 50

        t0.record()
        for _ in range(n_iters):
            flash_attention(q, k, v)
        t1.record()
        torch.cuda.synchronize()
        triton_ms = t0.elapsed_time(t1) / n_iters

        t0.record()
        for _ in range(n_iters):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        t1.record()
        torch.cuda.synchronize()
        sdpa_ms = t0.elapsed_time(t1) / n_iters

        print(f"  Triton Flash:   {triton_ms:.3f} ms")
        print(f"  PyTorch SDPA:   {sdpa_ms:.3f} ms")
        print(f"  Speedup:        {sdpa_ms / triton_ms:.2f}x")
    else:
        print("  (CPU mode — results not representative of GPU performance)")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="triton-kernels",
        description="Benchmark Triton inference kernels",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # info subcommand
    subparsers.add_parser("info", help="Show GPU and package info")

    # bench subcommand
    bench_parser = subparsers.add_parser("bench", help="Run kernel benchmark")
    bench_parser.add_argument(
        "--kernel",
        choices=["softmax", "attention"],
        required=True,
        help="Which kernel to benchmark",
    )
    bench_parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    bench_parser.add_argument("--batch", type=int, default=32, help="Batch size")
    bench_parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    bench_parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the triton-kernels CLI.

    Args:
        argv: Optional list of arguments (defaults to sys.argv).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    dispatch = {
        "info": cmd_info,
        "bench": cmd_bench,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
