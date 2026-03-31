# Model Swapping: How It Works

## The Three Swap Paths

### Path 1: Same Architecture, Same GPU (280ms)

When swapping between two 7B fine-tunes on the same GPU:

```
1. Pinned RAM has both models' weights (page-locked for DMA)
2. GPU has ONE StaticWeightPool (pre-allocated buffers)
3. Swap = pool.load_from_pinned(new_model_weights)
   → copy_() from pinned RAM into same GPU buffer addresses
   → 15 GB / 55 GB/s = 280ms
4. CUDA graph replays with new weight values (no recapture)
```

**Why 280ms and not 0ms?** The weights must physically transfer from CPU RAM to GPU HBM over PCIe Gen5. The theoretical minimum for 15GB at 64 GB/s (PCIe Gen5 x16) is 234ms. We achieve 280ms (88% of theoretical).

**Why not keep both models on GPU?** A 7B model needs ~15GB for weights + ~9GB for stacked QKV/gate_up buffers = ~24GB. Two models = 48GB. Add KV cache (30GB) and CUDA graphs (2GB) → 80GB. That's exactly the H100 limit, with zero headroom. Keeping one model's weight pool and swapping values is much more memory-efficient.

### Path 2: Same Architecture, Async Prefetch (0ms perceived)

When the scheduler knows the next model in advance:

```
t=0ms:   Model A decoding on compute stream
         Scheduler starts prefetch: model B weights → copy stream
t=0ms:   [compute stream]  graph.replay() (model A, step N)
         [copy stream]     pinned_B[chunk_0] → static_buf[chunk_0]
t=7ms:   [compute stream]  graph.replay() (model A, step N+1)
         [copy stream]     pinned_B[chunk_1] → static_buf[chunk_1]
...
t=280ms: [copy stream]     transfer complete, event recorded
...
t=350ms: [compute stream]  model A batch done
         Check: prefetch event ready? YES
         → Switch to model B instantly (0ms)
```

The async prefetch overlaps the 280ms transfer with active decode. If model A's batch takes >280ms (e.g., 8 sequences × 40 tokens = 320ms), the swap is completely hidden.

### Path 3: Different Architecture, Cross-GPU (825ms)

When swapping between 7B (GPU 0) and 14B (GPU 1):

```
1. Each architecture has its own GPU (assigned at startup)
2. "Swap" = scheduler stops serving GPU 0, starts serving GPU 1
3. First request to 14B: load weights from pinned RAM → GPU 1 (30GB / 55 GB/s = 545ms)
4. Create executor, capture CUDA graph (~280ms first time)
5. Total cold start: 825ms
6. Subsequent requests: weights already loaded → 0ms swap
```

Cross-architecture swaps are rare in practice. Most multi-tenant workloads use the same base model (e.g., all customers fine-tuned Qwen2.5-7B). Cross-architecture swaps only happen when the workload mixes model sizes.

## KV Cache Behavior During Swaps

### Same Architecture (KV Preserved)

```
Before swap: Model A active, sequences [user-1, user-2] in KV cache
             Pages: user-1 → [4,5,6], user-2 → [12,13]

Swap weights: copy_() model B weights into static buffers
              KV cache UNTOUCHED — pages [4,5,6,12,13] still allocated

After swap: Model B active, but user-1 and user-2's KV still valid
            Model B serves new request user-3 → pages [20,21]
            Later: swap back to A → user-1 resumes from page [6]
```

**Key insight:** KV cache pages are indexed by sequence ID and page ID, not by model name. The page data contains attention keys/values computed by whichever model was active during that sequence's prefill/decode. When we swap back to model A, its KV pages still contain the correct attention state.

**Caveat:** KV pages from model A are meaningless for model B (different weights = different attention patterns). Each sequence "belongs" to one model. The `seq_owner` field in `FlashAttnKVCache` tracks this.

### Different Architecture (KV Destroyed)

When switching architectures, the KV cache has different dimensions (num_kv_heads, head_dim, num_layers). KV pages from the old architecture are freed.

## Benchmarked Swap Times

| Scenario | Model Weights | Transfer | Total Swap |
|---|---|---|---|
| 7B same-arch, async prefetch hit | 15 GB | Overlapped | **0ms** perceived |
| 7B same-arch, sync | 15 GB | 280ms | **280ms** |
| 14B same-arch, sync | 30 GB | 545ms | **545ms** |
| 7B → 14B cross-arch (cold) | 30 GB | 545ms + graph | **825ms** |
| 7B → 14B cross-arch (warm) | 0 GB | 0ms | **0ms** (already loaded) |

## Static Weight Pool Internals

`StaticWeightPool` pre-allocates ALL weight buffers at startup:

```python
class StaticWeightPool:
    def __init__(self, hf_config, device, dtype):
        nl = hf_config.num_hidden_layers  # e.g., 28 for 7B
        hs = hf_config.hidden_size        # e.g., 3584
        
        # These are allocated ONCE and never freed during serving
        self.embed_w = torch.zeros(vocab, hs, dtype=bf16, device=gpu)     # 1.1 GB
        self.lm_head_w = torch.zeros(vocab, hs, dtype=bf16, device=gpu)   # 1.1 GB
        self.qkv_w = [torch.zeros(qkv_size, hs) for _ in range(nl)]      # 28 × 0.05 GB
        self.gu_w = [torch.zeros(2*mlp, hs) for _ in range(nl)]           # 28 × 0.27 GB
        # ... etc
        
    def load_from_pinned(self, pinned_weights, hf_config):
        """Swap model: copy_() new values into same addresses."""
        self.embed_w.copy_(pinned_weights["model.embed_tokens.weight"])
        for i in range(nl):
            self.qkv_w[i][:q].copy_(pinned_weights[f"...q_proj.weight"])
            self.qkv_w[i][q:q+k].copy_(pinned_weights[f"...k_proj.weight"])
            # ...
```

The `load_from_pinned` method stacks QKV and gate+up weights during copy (3 separate weight matrices → 1 contiguous buffer). This enables fused matmul during inference: one kernel launch instead of three.
