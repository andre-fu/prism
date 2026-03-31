# API Reference

Prism exposes an OpenAI-compatible HTTP API. Any client that works with OpenAI's API works with Prism.

## Authentication

```
Authorization: Bearer sk-{tenant_id}-{random_hex}
```

Requests without a key are assigned to the `default` tenant (no rate limits in development).

## Endpoints

### POST /v1/chat/completions

Chat completion with streaming support.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-acme-a1b2c3d4" \
  -d '{
    "model": "contract-review",
    "messages": [
      {"role": "system", "content": "You are a legal assistant."},
      {"role": "user", "content": "Review this NDA clause..."}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": true
  }'
```

**Request Body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Model name (as configured) |
| `messages` | array | required | Chat messages with `role` and `content` |
| `max_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) |
| `stream` | bool | false | Stream tokens via SSE |

**Response (non-streaming):**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "model": "contract-review",
  "choices": [{
    "index": 0,
    "text": "The NDA clause has several issues...",
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 250,
    "total_tokens": 378
  }
}
```

**Response (streaming):**
```
data: {"id":"cmpl-abc123","choices":[{"text":"The","finish_reason":null}]}

data: {"id":"cmpl-abc123","choices":[{"text":" NDA","finish_reason":null}]}

data: {"id":"cmpl-abc123","choices":[{"text":"","finish_reason":"stop"}]}

data: [DONE]
```

### POST /v1/completions

Text completion (non-chat).

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer sk-acme-a1b2c3d4" \
  -d '{"model": "contract-review", "prompt": "The capital of France is", "max_tokens": 50}'
```

### GET /v1/models

List all loaded models.

```json
{
  "object": "list",
  "data": [
    {
      "id": "contract-review",
      "object": "model",
      "model_id": "acme/contract-review-7b",
      "tp_size": 1,
      "dtype": "bfloat16",
      "loaded": true,
      "gpu": 0
    }
  ]
}
```

### POST /v1/tenants

Create a new tenant.

```bash
curl -X POST http://localhost:8000/v1/tenants \
  -d '{
    "tenant_id": "acme-corp",
    "name": "Acme Corporation",
    "rate_limit_rps": 10,
    "max_concurrent": 16,
    "allowed_models": ["contract-review", "case-law"],
    "priority": 10,
    "slo_ttft_ms": 500,
    "monthly_token_limit": 1000000
  }'
```

**Response:**
```json
{
  "tenant_id": "acme-corp",
  "api_key": "sk-acme-corp-72270ca6bbd36152cbf86e8efc18befc",
  "status": "created"
}
```

### GET /v1/tenants

List all tenants with usage stats.

### GET /v1/tenants/{tenant_id}/usage

```json
{
  "tenant_id": "acme-corp",
  "name": "Acme Corporation",
  "total_requests": 1284,
  "total_prompt_tokens": 384200,
  "total_completion_tokens": 192100,
  "total_tokens": 576300,
  "total_errors": 3,
  "current_concurrent": 2,
  "rate_limit_rps": 10,
  "max_concurrent": 16,
  "monthly_token_limit": 1000000
}
```

### GET /v1/stats

Scheduler statistics.

```json
{
  "batches": 12847,
  "evictions": 0,
  "sync_loads": 1,
  "prefetch_triggers": 42,
  "prefetch_hits": 38,
  "completed": 1284,
  "tokens_generated": 192100,
  "pending_requests": 3
}
```

### GET /metrics

Prometheus-format metrics for monitoring.

```
engine_batches_total 12847
engine_completed_total 1284
engine_tokens_generated_total 192100
engine_model_loaded{model="contract-review"} 1
engine_model_pending{model="contract-review"} 0
engine_gpu_memory_allocated_gb{gpu="0"} 16.2
```

### GET /health

```json
{"status": "ok"}
```

## Error Codes

| Code | Meaning |
|---|---|
| 404 | Model not found |
| 403 | Tenant does not have access to model |
| 429 | Rate limit exceeded / queue full |
| 500 | Generation error (GPU OOM, model error) |
| 504 | Request timeout (default: 120s) |
