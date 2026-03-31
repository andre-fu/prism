# Architecture Deep Dive

## The Core Problem

GPU inference serving has a fundamental mismatch: most fine-tuned models don't generate enough traffic to justify a dedicated GPU ($2-3/hr), but serverless cold starts (30-100s) make interactive use impossible. Prism solves this by serving multiple fine-tuned models on shared GPUs with sub-second swaps.

## Design Principles

1. **Zero allocation during serving.** All GPU memory is pre-allocated at startup. Model swaps are `copy_()` into fixed buffers. No `torch.malloc`, no fragmentation, no OOM surprises.

2. **CUDA graphs for everything.** The decode step is captured as a CUDA graph at startup. Each token generation is a single `graph.replay()` — zero Python dispatch overhead, zero kernel launch overhead.

3. **The copy stream trick.** Every GPU has two streams: a compute stream (running model inference) and a copy stream (transferring weights from RAM). They run concurrently. While model A generates tokens on the compute stream, model B's weights transfer on the copy stream. When A finishes, B is already loaded.

4. **KV cache is independent of weights.** The KV cache (per-conversation attention state) lives in its own memory pool, separate from model weights. Swapping weights doesn't touch the KV cache. Multi-turn conversations survive model swaps.

## Memory Layout (Single H100 80GB)

```
┌──────────────────────────────────────────────────────┐
│                    H100 80GB HBM                      │
├──────────────────────────────────────────────────────┤
│  Static Weight Pool          │  ~15 GB (7B model)     │
│  (pre-allocated buffers)     │  Embed, QKV, O, MLP    │
│                              │  per-layer stacked      │
├──────────────────────────────────────────────────────┤
│  KV Cache Pages              │  ~30 GB                 │
│  (flash_attn block tables)   │  256 tokens/page        │
│  Per-sequence, per-layer     │  ~1000+ pages           │
├──────────────────────────────────────────────────────┤
│  CUDA Graph Memory           │  ~2 GB                  │
│  (captured decode kernels)   │  Per batch size          │
├──────────────────────────────────────────────────────┤
│  CUDA Context + Overhead     │  ~3 GB                  │
│  (PyTorch allocator, NCCL)   │                         │
└──────────────────────────────────────────────────────┘
       ↕ PCIe Gen5 (55 GB/s) ↕
┌──────────────────────────────────────────────────────┐
│               Pinned Host RAM (~885 GB)               │
│  Model A weights (15 GB, pinned)                      │
│  Model B weights (15 GB, pinned)                      │
│  Model C weights (30 GB, pinned)                      │
│  Model D weights (15 GB, pinned)                      │
│  ...up to hundreds of models via Model Registry       │
└──────────────────────────────────────────────────────┘
```

## Weight Swap Pipeline

The key innovation: model weights are stored in **pinned CPU RAM** (page-locked for DMA) and the GPU has **pre-allocated static buffers** sized for the largest model architecture. Swapping models means copying new weight values into the same GPU addresses.

```
Step 1: Scheduler decides model B needs to serve next
        ┌──────────────────────────┐
        │  Compute Stream (GPU)    │ ← still running model A decode
        │  graph.replay()          │
        │  graph.replay()          │
        └──────────────────────────┘
                                     ← simultaneously:
        ┌──────────────────────────┐
        │  Copy Stream (GPU)       │ ← copying model B weights
        │  pinned_B → static_buf   │    from pinned RAM to GPU
        │  (55 GB/s, 280ms for 7B) │    into the SAME buffer addresses
        └──────────────────────────┘

Step 2: Model A finishes its batch
        Model B's weights are already in the static buffers
        → CUDA graph replays immediately with new weight data
        → No recapture needed (same addresses, new values)
        → 0ms switch latency (if prefetch completed)
```

### Why This Works

CUDA graphs capture the kernel launch sequence and tensor **addresses**, not tensor **values**. When we `copy_()` new weights into the same pre-allocated buffers, the graph replays correctly because:
- The buffer addresses haven't changed
- The kernel sequence is identical (same model architecture)
- Only the weight data is different

This is the same principle vLLM uses internally, but Prism applies it across model swaps, not just within a single model.

## CUDA Graph Capture

### Single-Sequence (batch=1)

```python
# Capture once at startup (or on first decode)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=capture_stream):
    output = decode_inner()  # 28 layers, flash_attn, fused RMSNorm

# Every decode step:
static_token_buf.copy_(new_token)
static_cos_buf.copy_(rotary_for_position)
static_block_table_buf[:n] = page_ids
static_seqlen_buf[0] = current_length
graph.replay()  # 6.7ms for 7B, zero Python overhead
```

### Batched (batch=2,4,8,16,32)

Each batch size has its own graph with its own static buffers:

```python
# At warmup: capture graph for each batch size
for bs in [2, 4, 8, 16, 32]:
    buffers[bs] = {
        "token": torch.zeros(bs, 1, dtype=long, device=gpu),
        "bt": torch.zeros(bs, max_blocks, dtype=int32, device=gpu),
        "sl": torch.zeros(bs, dtype=int32, device=gpu),
        "cos": torch.zeros(bs, 1, 1, head_dim, dtype=bf16, device=gpu),
        "sin": torch.zeros(bs, 1, 1, head_dim, dtype=bf16, device=gpu),
    }
    # Capture on dedicated stream
    with torch.cuda.graph(graphs[bs]):
        outputs[bs] = batched_decode_inner(bs)

# At runtime: pick smallest captured size >= actual batch
actual = 5  # 5 active sequences
graph_bs = 8  # Use batch=8 graph, pad 3 dummy sequences
buffers[8]["token"][:5] = real_tokens
buffers[8]["token"][5:] = 0  # Padding
buffers[8]["sl"][:5] = real_seqlens
buffers[8]["sl"][5:] = 0  # Padded sequences get seqlen=0
graphs[8].replay()
result = outputs[8][:5]  # Slice out real results
```

## Paged KV Cache

flash_attn's `flash_attn_with_kvcache` supports paged KV via block tables:

```
KV Cache Memory: [num_pages, page_size, num_kv_heads, head_dim]
                  ↑           ↑
              1000 pages   256 tokens each = 256,000 token capacity

Block Table: [batch_size, max_blocks_per_seq]
             Each entry is a page ID

Sequence "Hello, how are you?" (5 tokens):
  block_table[0] = [page_42]  → page 42 holds tokens 0-4
  cache_seqlen[0] = 5

Sequence with 600 tokens:
  block_table[1] = [page_7, page_13, page_91]  → 3 pages
  cache_seqlen[1] = 600  → pages 7,13 full (256 each), page 91 has 88 tokens
```

### KV Preservation Across Model Swaps

When models share the same architecture (e.g., two 7B fine-tunes), they share one KV cache pool. Model A's conversation KV stays in the cache while model B generates. When model A resumes, its KV is still there — no re-prefill.

```
t=0: Model A serves user-1 (KV in pages [4,5])
t=1: Switch to Model B, serve user-2 (KV in pages [8,9])
     User-1's KV (pages [4,5]) still allocated, untouched
t=2: Switch back to Model A, serve user-1
     Pages [4,5] still valid → decode continues from where it left off
     TTFT = 0ms (no re-prefill!)
```

## Spatial Multiplexing

With 4 GPUs and 4 different model architectures, each gets its own GPU with its own weight pool, KV cache, and executor. The scheduler dispatches work to all GPUs concurrently using a thread pool:

```python
# scheduler_v2.py step() method
active_archs = {arch for model in models_with_pending_work}

if len(active_archs) == 1:
    serve_gpu(arch)  # Single GPU, no threading overhead
else:
    with ThreadPoolExecutor(max_workers=len(active_archs)) as pool:
        futures = {pool.submit(serve_gpu, arch) for arch in active_archs}
        # Each thread runs on its own GPU — GIL released during CUDA ops
```

Python threads work because CUDA operations release the GIL. Each thread dispatches kernels to its GPU independently. The only shared state is the `RequestManager._lock` (held for microseconds during queue operations).

### Same Architecture, Different Models

When multiple models share an architecture (e.g., 4 different 7B fine-tunes), they share one GPU. The scheduler time-shares:

```
GPU 0: [customer-0 weights] → serve 2 batches → swap (280ms)
       [customer-1 weights] → serve 2 batches → swap (280ms)
       [customer-2 weights] → serve 2 batches → swap (280ms)
       [customer-3 weights] → serve 2 batches → ...
```

The async prefetch hides the 280ms swap: while customer-0 is decoding, customer-1's weights are copying on the copy stream.

## Disaggregated Prefill/Decode

For long-prompt workloads, separating prefill (compute-bound) from decode (memory-bandwidth-bound) improves utilization:

```
Prefill GPUs [0,1]:          Decode GPUs [2,3]:
  Batch 8 prompts together     Continuous batching
  Compute-heavy (matmuls)      Bandwidth-heavy (KV reads)
  High GPU utilization          Low latency per token
           │                            ▲
           │  KV Transfer (27ms)        │
           └──── D2D via NVLink ────────┘
```

After prefill completes on GPU 0, the sequence's KV pages are copied device-to-device to GPU 2. NVLink bandwidth (~300 GB/s) makes this near-instant: 27ms for a typical sequence. GPU 2 then continues decode with CUDA graphs.

## Speculative Decoding

A small draft model (0.5B) proposes k tokens cheaply. The large target model (7B) verifies all k tokens in one forward pass:

```
Draft (0.5B):  "The" → "capital" → "of" → "France" → "is"     [5 forward passes, ~1ms each]
Target (7B):   verify("The capital of France is") in ONE pass   [1 forward pass, ~7ms]
                 ↓       ↓        ↓        ↓       ↓
Accept?:        ✓       ✓        ✓        ✓       ✓

Result: 5 tokens in 12ms instead of 5 × 7ms = 35ms → 2.9× speedup
```

When the draft model disagrees with the target, we reject at the first mismatch and use the target's token instead. The rollback mechanism truncates the draft model's KV cache to match.

## Multi-Tenant Isolation

Each tenant gets:
- **API key**: `sk-{tenant_id}-{random_hex}`, SHA-256 hashed in storage
- **Rate limits**: Requests/second + max concurrent, enforced via sliding window
- **Model access**: Allowlist of which models the tenant can use
- **Priority**: Higher priority tenants get GPU time first (SLO-aware scheduling)
- **Usage metering**: Per-tenant prompt tokens, completion tokens, request count
- **Monthly caps**: Optional token limit per billing period

The isolation is **logical, not physical**. All tenants share the same GPU(s) and KV cache pools. Isolation is enforced at the API and scheduler layers. A compromised tenant could theoretically probe timing side-channels (see the PRD's security analysis), but data isolation is maintained: each tenant's KV pages are separate, and model weights are shared (all tenants see the same model).
