# Benchmarks

All benchmarks on 4×H100 80GB, PyTorch 2.10, CUDA 12.8, Qwen2.5 models.

## Single Model Performance (7B)

| Batch Size | ms/step | tok/s | Method |
|---|---|---|---|
| 1 | 6.7 | **150** | CUDA graph |
| 2 | 7.0 | **286** | CUDA graph |
| 4 | 7.1 | **564** | CUDA graph |
| 8 | 7.3 | **1,096** | CUDA graph |
| 16 | 7.5 | **2,133** | CUDA graph |
| 32 | 8.0 | **4,000** | CUDA graph (est.) |

## vs vLLM (7B, enforce_eager mode)

| Metric | Prism | vLLM | Ratio |
|---|---|---|---|
| Single decode tok/s | **150** | 94 | **1.59×** |
| Batch=8 tok/s | **1,096** | 622 | **1.76×** |
| TTFT (5-token prompt) | 31ms | ~50ms | **0.62×** |
| TBT (single seq) | 6.7ms | 10.6ms | **0.63×** |

Prism is faster than vLLM in eager mode because of:
1. CUDA graph eliminates all Python dispatch overhead
2. Fused QKV matmul (3 kernels → 1)
3. Triton fused RMSNorm
4. Pre-computed rotary embeddings
5. flash_attn_with_kvcache combines KV append + attention in one kernel

## Model Swap Overhead

| Operation | Time | Bandwidth |
|---|---|---|
| Load 7B (pinned → GPU) | 280ms | 55 GB/s |
| Load 14B (pinned → GPU) | 545ms | 55 GB/s |
| Evict (del + empty_cache) | 14ms | — |
| Full swap 7B → 14B | 825ms | — |
| Full swap 14B → 7B | 280ms | — |
| Same-arch swap (7B → 7B) | 280ms | 55 GB/s |
| Same-arch with prefetch hit | **0ms** | Overlapped |

## Realistic Workload (7B ↔ 14B)

Long prompts (250-312 tokens), 200 token generation:

| Test | Model | TTFT | TBT | Decode tok/s | Total |
|---|---|---|---|---|---|
| Cold start (272-tok legal) | 7B | 838ms | 7.1ms | 142 | 2.25s |
| Swap + prefill (312-tok medical) | 14B | 603ms | 12.7ms | 79 | 3.14s |
| Swap back (253-tok code) | 7B | 311ms | 7.0ms | 143 | 1.71s |
| 6 interleaved requests | Both | — | — | **133 agg** | 4.51s |

## Production Scenarios

### 1. Four 7B Fine-Tunes (same architecture)
- 12 requests across 4 models, 3 each
- **92.2 tok/s** aggregate
- TTFT p50: 1.6s, p95: 1.9s
- 1 sync load, 6 prefetch hits

### 2. Mixed Architectures (0.5B + 1.5B + 7B)
- 9 requests across 3 architectures
- **148.0 tok/s** aggregate (spatial multiplexing)
- All 3 GPUs utilized concurrently

### 3. Bursty Traffic (10x spike)
- 10 burst + 2 baseline requests
- **224.6 tok/s** aggregate
- TTFT p50: 351ms, p95: 723ms

### 4. Multi-Turn KV Preservation
- 5 interleaved requests across 2 models
- Model-A TTFT: 292ms → 302ms → 312ms (no re-prefill!)
- KV preserved across model swaps

### 5. Sustained Load (15 seconds, 30 requests)
- 30/30 completed
- TTFT p50: 292ms, p95: 295ms
- Memory after cleanup: 0.07 GB (no leak)

## Tensor Parallelism

| Model | TP | Memory/GPU | Decode tok/s |
|---|---|---|---|
| 7B | 1 | 15.2 GB | 150 |
| 7B | 4 | 5.4 GB | 9.2 |
| 14B | 4 | 10.3 GB | 5.1 |
| 72B | 4 | 40.2 GB | 3.3 |

TP decode is slower due to Python-dispatch overhead for cross-GPU all-reduce. The 72B result proves the system handles 145GB models correctly across 4 GPUs.

## Speculative Decoding

| Config | Acceptance Rate | tok/s | Speedup |
|---|---|---|---|
| 7B standard | — | 48 | 1.0× |
| 0.5B draft → 7B target (k=4) | ~60% | 71 | **1.48×** |

## Disaggregated Prefill/Decode

| Metric | Value |
|---|---|
| Prefill GPUs | 0, 1 |
| Decode GPUs | 2, 3 |
| KV transfer (42 tokens, D2D) | 27ms |
| 4 requests, 400 tokens | **205 tok/s** aggregate |

## Optimization Journey

How single-request decode improved from 37 to 150 tok/s:

| Optimization | tok/s | ms/step | Improvement |
|---|---|---|---|
| Naive (transformers + FlashInfer) | 37 | 26.9 | Baseline |
| + Pre-extracted layer references | 50 | 20.0 | 1.35× |
| + Triton fused RMSNorm | 57 | 17.7 | 1.54× |
| + Fused QKV matmul | 72 | 13.9 | 1.94× |
| + Pre-computed rotary | 80 | 12.6 | 2.16× |
| + flash_attn (replace FlashInfer) | 93 | 10.8 | 2.51× |
| + CUDA graph capture | **150** | **6.7** | **4.05×** |
