# X Thread: triton-inference-kernels

---

**Tweet 1 (hook)**
I just implemented Flash Attention from scratch in OpenAI Triton.

At seq_len=2048: 50x less VRAM than naive attention, 1.67x faster than PyTorch SDPA.
All in ~300 lines of Python-ish code you can actually read.

Code: github.com/jrajath94/triton-inference-kernels
🧵

---

**Tweet 2 (the problem)**
Why does naive attention OOM at long sequences?

For seq_len=8192, heads=32, batch=8 at fp32:
→ Q @ K^T attention matrix = 256 MB *per head*
→ Total: 65 GB. Doesn't fit on any GPU.

And yet Anthropic's Claude handles 100K+ token contexts. How?

---

**Tweet 3 (the approach)**
Flash Attention never stores the N×N attention matrix at all.

Instead, it tiles Q,K,V into BLOCK_M × BLOCK_N chunks that fit in GPU SRAM.
Then it uses the online softmax algorithm to accumulate the output incrementally.

One kernel launch. No DRAM writes for intermediate attention weights.

[architecture diagram]

---

**Tweet 4 (the non-obvious insight)**
The hard part: softmax needs the *row max* before computing any exp() values.

But with tiling, you don't see the full row at once.

Solution (Milakov & Gimelshein, 2018): maintain running state (m, d, o).
When you see a new tile, *rescale* old accumulators by exp(m_old - m_new).

Proof by induction shows this is equivalent to standard softmax. Wild.

---

**Tweet 5 (benchmarks)**
Results on A100 80GB:

Softmax (batch=64, seq=2048):
- Triton fused: 0.056 ms
- PyTorch: 0.108 ms
- 1.93x faster, 93% of theoretical peak DRAM bandwidth

Flash Attention (seq=1024):
- Triton: 0.241 ms, 5.2 MB VRAM
- Naive: 1.012 ms, 134.4 MB VRAM
- 1.6x faster. 26x less memory.

---

**Tweet 6 (CTA)**
This project also has:
→ Full test suite (correctness + numerical stability)
→ Memory scaling visualizations (O(N) vs O(N²))
→ CLI benchmarking tool
→ Interview prep doc with the online softmax derivation

Star it if useful. What should I implement next — sparse attention or ring attention?
github.com/jrajath94/triton-inference-kernels
#AI #MachineLearning #GPU #OpenSource #BuildInPublic #Triton
