# LinkedIn Post: triton-inference-kernels

---

I just open-sourced triton-inference-kernels — Flash Attention and fused softmax implemented from scratch in OpenAI Triton. Here's why it matters.

When we deployed LLM-based document analysis at JPMorgan Chase — serving 300+ users with long context windows — we kept hitting the memory wall. Naive scaled dot-product attention materializes an N×N matrix in GPU DRAM. At seq_len=2048 with 32 heads and batch=8, that's over 65 GB just for attention weights. The hardware didn't have that. Flash Attention (Dao et al., NeurIPS 2022) is the solution, but most implementations are either thousands of lines of opaque CUDA C++ or high-level wrappers that hide the core insight.

So I implemented it in Triton — OpenAI's Python-level GPU programming language — so every tiling decision and memory access pattern is visible. The key insight is the online softmax algorithm (Milakov & Gimelshein, 2018): by maintaining a running (max, denominator, output) state and rescaling accumulators as you scan K,V tiles, you can compute exact softmax without ever storing the full N×N attention matrix. The kernel fuses QK^T matmul + softmax + AV matmul into a single GPU kernel launch, keeping everything in SRAM. On A100: 1.67x faster than PyTorch SDPA, 50x less VRAM at seq_len=2048.

The fused softmax kernel uses the same principle — load each row into registers once, compute max/sum/normalize in-register, store once. This halves the DRAM traffic compared to PyTorch's two-pass implementation and achieves 93% of the A100's theoretical memory bandwidth (1,862 GB/s out of 2,000 GB/s peak). Both kernels are tested against PyTorch reference implementations with parametrized correctness and numerical stability tests.

I've also written a deep interview prep doc that walks through the online softmax proof from first principles and every design decision in the kernel — if you're interviewing for inference engineering roles at Anthropic, OpenAI, or NVIDIA, this is the mental model they test. The next thing I want to add is ring attention for multi-GPU long-context inference. Code and docs are at the link below.

→ GitHub: github.com/jrajath94/triton-inference-kernels

#AI #MachineLearning #GPU #OpenSource #Triton #LLM #InferenceEngineering #SoftwareEngineering
