# Prism

**Multi-tenant GPU inference engine for full fine-tuned models.**

Prism serves multiple fine-tuned LLMs on shared GPUs with sub-second model swaps, CUDA-graph-accelerated decode, and per-tenant isolation. One GPU serves 4+ models instead of each model needing its own dedicated GPU.

## Why

If you have full fine-tuned models (not LoRA adapters), your serving options are bad:

| Option | Cost | Latency | Works for full fine-tunes? |
|---|---|---|---|
| **Dedicated GPU** | $2.49/hr × N models | 0ms swap | Yes, but 95% idle |
| **Serverless** | Pay per second | 30-100s cold start | Yes, but unusable for interactive |
| **LoRA multi-tenant** | Shared base GPU | ~0ms swap | **No** — requires same base weights |
| **Prism** | $2.49/hr shared | **280ms swap** | **Yes** |

Prism turns a $7,200/month cost (4 dedicated GPUs) into $1,800/month (1 shared GPU).

## Performance

Benchmarked on 4×H100 80GB with Qwen2.5 models:

| Metric | Prism | vLLM (dedicated) |
|---|---|---|
| Decode tok/s (7B, single) | **150** | 94 |
| Decode tok/s (7B, batch=8) | **1,096** | ~1,500 |
| Model swap (same arch) | **280ms** | N/A (dedicated) |
| Model swap (cross-arch) | **825ms** | N/A |
| KV cache on swap | **Preserved** | Destroyed |
| Models per GPU | **4+** | 1 |
| Cold start | **838ms** | 30-100s (serverless) |

## Quick Start

```bash
# Start serving 4 models on 1 GPU
python -m engine.serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model Qwen/Qwen2.5-14B-Instruct \
  --port 8000

# Or use a config file
python -m engine.serve --config engine/example_config.yaml
```

```bash
# Send a request (OpenAI-compatible API)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-api-key" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## Configuration

```yaml
# engine.yaml
models:
  - model_id: your-org/contract-review-7b
    name: contract-review
    tp_size: 1
    dtype: bfloat16

  - model_id: your-org/medical-analysis-7b
    name: medical
    tp_size: 1

  - model_id: your-org/code-review-14b
    name: code-review
    tp_size: 1

gpu_ids: [0, 1, 2, 3]
kv_cache_budget_gb: 30
kv_page_size: 256

scheduler:
  max_consecutive_batches: 32

server:
  port: 8000
  max_queue_depth: 64
```

## Multi-Tenant API

```bash
# Create a tenant with API key, rate limits, and model access
curl -X POST http://localhost:8000/v1/tenants \
  -d '{
    "tenant_id": "acme-corp",
    "name": "Acme Corporation",
    "rate_limit_rps": 10,
    "max_concurrent": 16,
    "allowed_models": ["contract-review", "medical"],
    "priority": 10,
    "monthly_token_limit": 1000000
  }'
# Returns: {"api_key": "sk-acme-corp-a1b2c3d4...", "status": "created"}

# Check usage
curl http://localhost:8000/v1/tenants/acme-corp/usage
# Returns: {"total_tokens": 52340, "total_requests": 128, ...}
```

## Features

### Core Engine
- **CUDA Graph Decode**: 150 tok/s single, 1,096 tok/s batched (1.5× vLLM)
- **Static Weight Pool**: Pre-allocated GPU buffers, model swap = `copy_()` (zero allocation)
- **Paged KV Cache**: flash_attn with block tables, memory-efficient variable-length sequences
- **KV Preservation**: Multi-turn conversations survive model swaps (no re-prefill)
- **Fused Kernels**: Triton RMSNorm, fused QKV matmul, fused gate+up projection

### Scheduling
- **Spatial Multiplexing**: Different models on different GPUs run concurrently
- **Async Prefetch**: Load next model on copy stream while current model decodes
- **SLO-Aware**: Priority scheduling, earliest-deadline-first, per-batch preemption
- **Continuous Batching**: New requests join running batches mid-generation

### Scale
- **Batched CUDA Graphs**: Pre-captured for batch sizes 1,2,4,8,16,32
- **Speculative Decoding**: Draft model (0.5B) proposes tokens, target (7B) verifies (1.48× speedup)
- **Disaggregated Prefill/Decode**: Separate GPU pools for compute-bound prefill and bandwidth-bound decode
- **Prefix Caching**: Shared system prompts skip re-prefill (page-level, ref-counted)
- **Model Registry**: Lazy loading from disk, LRU eviction from pinned RAM, hundreds of models
- **Tensor Parallelism**: TP=1 through TP=4 for models up to 72B

### Multi-Tenant
- **API Key Authentication**: Per-tenant `sk-*` keys with SHA-256 hashing
- **Rate Limiting**: Requests/second + max concurrent per tenant
- **Model Access Control**: Tenants only access their allowed models
- **Usage Metering**: Per-tenant prompt/completion token tracking
- **Monthly Token Limits**: Configurable caps per tenant

## Architecture

```
                    HTTP Request
                         │
                         ▼
              ┌─── API Server (FastAPI) ───┐
              │  Authentication            │
              │  Rate Limiting             │
              │  Model Routing             │
              └────────────┬───────────────┘
                           │
              ┌─── Scheduler (per-GPU threads) ───┐
              │                                    │
              │  GPU 0: [model-a weights]          │
              │    → prefill new requests          │
              │    → CUDA graph decode batch       │
              │    → async prefetch model-b        │
              │                                    │
              │  GPU 1: [model-b weights]          │
              │    → (concurrent with GPU 0)       │
              │                                    │
              │  GPU 2: [model-c weights]          │
              │    → (concurrent with GPU 0,1)     │
              └────────────────────────────────────┘
                           │
              ┌─── Static Weight Pool (per-arch) ──┐
              │  Pre-allocated GPU buffers          │
              │  Model swap = copy_() (280ms)       │
              │  CUDA graph replays with new data   │
              └────────────────────────────────────┘
                           │
              ┌─── Paged KV Cache (flash_attn) ────┐
              │  [max_pages, page_size, heads, dim] │
              │  Per-sequence page tracking          │
              │  Prefix cache (shared pages)         │
              │  KV preserved across model swaps     │
              └────────────────────────────────────┘
                           │
              ┌─── Pinned RAM (host memory) ────────┐
              │  All model weights in page-locked RAM│
              │  Model Registry: lazy load, LRU evict│
              │  Async H2D at 55 GB/s (PCIe Gen5)   │
              └────────────────────────────────────┘
```

## Hardware Requirements

- **GPU**: NVIDIA H100 (tested), A100 should work (untested)
- **GPU Memory**: 80GB per GPU recommended
- **Host RAM**: 64GB+ (pinned memory for model weights)
- **NVLink**: Required for TP>1 all-reduce
- **PCIe**: Gen5 for maximum H2D bandwidth (~55 GB/s)

## Dependencies

- PyTorch 2.10+
- flash-attn 2.8+
- transformers 4.57+
- safetensors
- FlashInfer 0.6+ (for legacy FlashInfer path)
- FastAPI + uvicorn
- triton 3.6+

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/v1/completions` | POST | Text completion |
| `/v1/chat/completions` | POST | Chat completion (with streaming) |
| `/v1/tenants` | POST | Create tenant |
| `/v1/tenants` | GET | List all tenants |
| `/v1/tenants/{id}/usage` | GET | Tenant usage stats |
| `/v1/stats` | GET | Scheduler stats |
| `/metrics` | GET | Prometheus metrics |

## Project Structure

```
engine/                    # Core inference engine
├── executor.py            # CUDA graph decode + flash_attn (the fast path)
├── scheduler.py           # Multi-model scheduler (spatial mux + async prefetch)
├── weight_pool.py         # Static GPU weight buffers (zero-alloc swap)
├── kv_cache.py            # Paged KV cache (flash_attn block tables)
├── weight_manager.py      # Model loading, pinned RAM, residency tracking
├── memory_pool.py         # GPU + pinned RAM memory pools
├── fused_kernels.py       # Triton fused RMSNorm
├── config.py              # YAML config dataclasses
├── server.py              # FastAPI HTTP server (OpenAI-compatible)
├── serve.py               # CLI entry point
├── request_manager.py     # Per-model request queues
├── prefetch.py            # Async weight prefetch on copy stream
├── tenant_manager.py      # Multi-tenant auth + rate limiting + metering
├── persistence.py         # SQLite backend (tenants, keys, usage survive restart)
├── model_registry.py      # Lazy loading from disk + LRU eviction
├── model_upload.py        # Model upload validation + storage
├── prefix_cache.py        # Shared KV pages for common prefixes
├── speculative_executor.py # Draft + target speculative decoding
├── disaggregated.py       # Separate prefill/decode GPU pools
├── distributed.py         # TP sharding + NCCL all-reduce
├── tp_executor.py         # Tensor parallel executor
├── vlm_executor.py        # Vision-language model support
├── lifecycle.py           # Graceful shutdown + drain
├── metrics.py             # Histograms, counters, Prometheus export
└── logging.py             # Structured JSON logging

tests/                     # Pytest test suite (37 tests)
benchmarks/                # Performance benchmarks vs vLLM, Modal, RunPod
docs/                      # Architecture docs + benchmark graphics
deploy/                    # nginx TLS config
gpu_swap/                  # Legacy vLLM sleep/wake experiments
```

## License

[TBD]
