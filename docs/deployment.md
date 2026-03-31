# Deployment Guide

## Prerequisites

- NVIDIA GPU with 80GB HBM (H100 tested, A100 should work)
- NVIDIA Driver 580+ (for flash_attn compatibility)
- CUDA 12.8+
- Python 3.10+
- 64GB+ host RAM (pinned memory for model weights)

## Installation

```bash
# Clone
git clone git@github.com:andre-fu/prism.git
cd prism

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install transformers safetensors huggingface-hub
pip install flashinfer  # Optional: for legacy FlashInfer executor
pip install fastapi uvicorn sse-starlette
pip install triton
pip install pyyaml accelerate
```

## Single-GPU Quick Start

```bash
# Serve two models on GPU 0
python -m engine.serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model Qwen/Qwen2.5-14B-Instruct \
  --gpu 0 \
  --port 8000
```

## Multi-GPU with Config File

```yaml
# config.yaml
models:
  - model_id: your-org/legal-7b
    name: legal
  - model_id: your-org/medical-7b
    name: medical
  - model_id: your-org/code-14b
    name: code

gpu_ids: [0, 1, 2, 3]
kv_cache_budget_gb: 30

scheduler:
  max_consecutive_batches: 32

server:
  port: 8000
```

```bash
python -m engine.serve --config config.yaml
```

## Environment Variables

```bash
# Recommended: reduces memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: disable TensorFlow warnings
export TRANSFORMERS_NO_TF=1
export USE_TF=0
```

## Monitoring

### Prometheus

Scrape `http://localhost:8000/metrics` at 15s intervals:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'prism'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
    metrics_path: /metrics
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Scheduler Stats

```bash
curl http://localhost:8000/v1/stats | jq
```

## Production Recommendations

1. **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — reduces fragmentation during model swaps.

2. **Size your KV cache budget carefully.** Each model's concurrent conversations consume KV pages. At 256 tokens/page and 1000 pages: 256K token capacity. Long conversations (4K+ tokens) eat pages fast.

3. **Use same-architecture models when possible.** Same-arch models share CUDA graphs (no recapture on swap), share KV cache pools (KV preserved across swaps), and share weight pool buffers (280ms swap vs 825ms cross-arch).

4. **Set tenant rate limits.** Without limits, one tenant can monopolize the GPU. Start with 10 req/s, 16 max concurrent, and adjust based on monitoring.

5. **Monitor `engine_gpu_memory_allocated_gb`.** If it approaches 80GB, reduce `kv_cache_budget_gb` or add more GPUs.
