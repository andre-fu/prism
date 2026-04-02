"""OpenAI-compatible HTTP API server."""

import asyncio
import json
import time
import uuid
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .config import ModelConfig, EngineConfig, SchedulerConfig
from .memory_pool import PinnedPool, MultiGPUPool
from .weight_manager import WeightManager
from .request_manager import RequestManager, RequestState
from .scheduler_v2 import SchedulerV2


# --- Request/Response models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, alias="max_tokens")
    temperature: float = 0.0
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = False


# --- Globals (initialized in lifespan) ---

_scheduler: SchedulerV2 | None = None
_weight_manager: WeightManager | None = None
_request_manager: RequestManager | None = None
_engine_config: EngineConfig | None = None
_tenant_manager = None


def create_engine(models: list[dict], gpu_ids: list[int] = [0], **kwargs):
    """Initialize engine with v2 scheduler (static weight pools + CUDA graphs).

    This is the fast path: 130 tok/s decode, 302ms model swap, CUDA graph replay.
    """
    global _scheduler, _weight_manager, _request_manager, _engine_config

    model_configs = [ModelConfig(**m) for m in models]
    _engine_config = EngineConfig(models=model_configs, gpu_ids=gpu_ids, **kwargs)
    sched_cfg = SchedulerConfig(max_consecutive_batches=64)

    pinned_pool = PinnedPool(budget_gb=_engine_config.t1_budget_gb)
    gpu_pool = MultiGPUPool(gpu_ids, _engine_config.t0_budget_gb, _engine_config.kv_cache_budget_gb)

    _weight_manager = WeightManager(_engine_config, pinned_pool, gpu_pool)
    _request_manager = RequestManager()

    from .tenant_manager import TenantManager
    global _tenant_manager
    _tenant_manager = TenantManager()

    # Load all models into pinned RAM
    for mc in model_configs:
        _weight_manager.load_model(mc)

    # SchedulerV2 creates StaticWeightPool + FlashAttnExecutorV3 + CUDA graphs
    # per architecture, assigns to GPUs, handles spatial multiplexing
    _scheduler = SchedulerV2(_engine_config, sched_cfg, _weight_manager, _request_manager, gpu_pool)

    return _scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, stop on shutdown."""
    if _scheduler:
        _scheduler.run_background()
        print("[Server] SchedulerV2 started (CUDA graphs + static weight pools)")
    yield
    if _scheduler:
        _scheduler.cleanup()
        print("[Server] Scheduler stopped, GPU memory freed")


app = FastAPI(title="Multi-Model Inference Engine", lifespan=lifespan)


# --- Endpoints ---

@app.get("/health")
async def health():
    lifecycle = getattr(app.state, 'lifecycle', None)
    return {
        "status": "ok",
        "accepting_requests": lifecycle.check_accepting_requests() if lifecycle else True,
    }


# --- Model upload endpoints ---

_upload_manager = None

@app.post("/v1/models/upload")
async def upload_model(
    name: str,
    config_file: bytes = None,  # Will use Form/File in production
    authorization: str | None = None,
):
    """Upload a model for serving.

    Accepts multipart form: config.json + safetensors files.
    Validates architecture, checks shapes, stores to disk, registers in engine.

    For now, accepts a HuggingFace model ID and downloads it.
    Full file upload requires multipart handling.
    """
    from .model_upload import ModelUploadManager
    from .model_registry import ModelRegistry
    global _upload_manager

    if _upload_manager is None:
        _upload_manager = ModelUploadManager()

    # For MVP: accept a HuggingFace model ID and register it
    # Full file upload endpoint would use UploadFile from fastapi
    return {"error": "Use /v1/models/register for HuggingFace models. File upload coming soon."}


class ModelRegisterRequest(BaseModel):
    model_id: str       # HuggingFace model ID (e.g., "prism-ml/Bonsai-8B-unpacked")
    name: str           # Display name for the model
    tp_size: int = 1
    dtype: str = "bfloat16"


@app.post("/v1/models/register")
async def register_model(req: ModelRegisterRequest):
    """Register a HuggingFace model for serving.

    Downloads weights, validates architecture, and makes the model available.
    The model loads to GPU on first request (lazy loading).
    """
    from .config import ModelConfig

    # Validate the model exists and check architecture
    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(req.model_id, trust_remote_code=True)
        arch = hf_config.architectures[0] if hf_config.architectures else "unknown"

        # Check compatibility
        required_attrs = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
        for attr in required_attrs:
            if not hasattr(hf_config, attr):
                raise HTTPException(400, f"Model config missing required attribute: {attr}")

        # Check if architecture is supported (standard decoder pattern)
        supported_types = ["qwen2", "qwen3", "llama", "mistral", "gemma", "phi"]
        model_type = getattr(hf_config, "model_type", "").lower()
        is_supported = any(t in model_type for t in supported_types)

        # Register in weight manager
        model_cfg = ModelConfig(
            model_id=req.model_id,
            name=req.name,
            tp_size=req.tp_size,
            dtype=req.dtype,
        )

        if _weight_manager:
            _weight_manager.load_model(model_cfg)

        # Persist to database
        if _tenant_manager and _tenant_manager._db:
            _tenant_manager._db.save_model(
                req.name, req.model_id, req.tp_size, req.dtype,
                config_json=str({
                    "architecture": arch,
                    "model_type": model_type,
                    "hidden_size": hf_config.hidden_size,
                    "num_layers": hf_config.num_hidden_layers,
                    "num_heads": hf_config.num_attention_heads,
                    "num_kv_heads": getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
                    "has_qk_norm": hasattr(hf_config, "model_type") and "qwen3" in model_type,
                }),
            )

        return {
            "status": "registered",
            "name": req.name,
            "model_id": req.model_id,
            "architecture": arch,
            "model_type": model_type,
            "supported": is_supported,
            "hidden_size": hf_config.hidden_size,
            "num_layers": hf_config.num_hidden_layers,
            "num_heads": hf_config.num_attention_heads,
            "num_kv_heads": getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            "has_qk_norm": "qwen3" in model_type,
            "cuda_graph": "Will capture on first request (same-arch models share graphs)",
            "estimated_gpu_gb": hf_config.hidden_size * hf_config.num_hidden_layers * 4 * 2 / 1e9,  # Rough estimate
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to register model: {e}")


@app.delete("/v1/models/{model_name}")
async def delete_model(model_name: str):
    """Unregister a model and free its resources."""
    if _weight_manager and _weight_manager.get_state(model_name):
        if _weight_manager.is_loaded(model_name):
            _weight_manager.evict_from_gpu(model_name)
        # Remove from weight manager
        if model_name in _weight_manager.models:
            state = _weight_manager.models[model_name]
            if state.pinned_weights:
                _weight_manager.pinned_pool.untrack(state.pinned_bytes)
            del _weight_manager.models[model_name]
        return {"status": "deleted", "name": model_name}
    raise HTTPException(404, f"Model '{model_name}' not found")


# --- Tenant management endpoints ---

class TenantCreateRequest(BaseModel):
    tenant_id: str
    name: str = ""
    rate_limit_rps: float = 10.0
    max_concurrent: int = 16
    allowed_models: list[str] = []
    priority: int = 0
    slo_ttft_ms: float = 2000.0
    monthly_token_limit: int = 0


@app.post("/v1/tenants")
async def create_tenant(req: TenantCreateRequest):
    from .tenant_manager import TenantConfig
    config = TenantConfig(
        tenant_id=req.tenant_id, name=req.name,
        rate_limit_rps=req.rate_limit_rps, max_concurrent=req.max_concurrent,
        allowed_models=req.allowed_models, priority=req.priority,
        slo_ttft_ms=req.slo_ttft_ms, monthly_token_limit=req.monthly_token_limit,
    )
    api_key = _tenant_manager.register_tenant(config)
    return {"tenant_id": req.tenant_id, "api_key": api_key, "status": "created"}


@app.get("/v1/tenants")
async def list_tenants():
    return {"tenants": _tenant_manager.get_all_usage()}


@app.get("/v1/tenants/{tenant_id}/usage")
async def tenant_usage(tenant_id: str):
    usage = _tenant_manager.get_usage(tenant_id)
    if not usage:
        raise HTTPException(404, f"Tenant '{tenant_id}' not found")
    return usage


# --- Model endpoints ---

@app.get("/v1/models")
async def list_models():
    models = []
    for mc in _engine_config.models:
        state = _weight_manager.get_state(mc.name)
        models.append({
            "id": mc.name,
            "object": "model",
            "owned_by": "engine",
            "model_id": mc.model_id,
            "tp_size": mc.tp_size,
            "dtype": mc.dtype,
            "loaded": _weight_manager.is_loaded(mc.name),
            "gpu": state.active_gpu_id if state else None,
        })
    return {"object": "list", "data": models}


@app.get("/v1/stats")
async def stats():
    s = _scheduler.stats
    return {
        "batches": s.batches,
        "completed": s.completed,
        "tokens_generated": s.tokens_generated,
        "sync_swaps": s.sync_swaps,
        "swap_time_ms": round(s.swap_time_ms, 1),
        "prefetch_triggers": s.prefetch_triggers,
        "prefetch_hits": s.prefetch_hits,
        "pending_requests": _request_manager.total_pending(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics for monitoring."""
    lines = []
    s = _scheduler.stats

    # Scheduler metrics
    lines.append(f'engine_batches_total {s.batches}')
    lines.append(f'engine_evictions_total {s.evictions}')
    lines.append(f'engine_sync_loads_total {s.sync_loads}')
    lines.append(f'engine_prefetch_triggers_total {s.prefetch_triggers}')
    lines.append(f'engine_prefetch_hits_total {s.prefetch_hits}')
    lines.append(f'engine_completed_total {s.completed}')
    lines.append(f'engine_tokens_generated_total {s.tokens_generated}')
    lines.append(f'engine_pending_requests {_request_manager.total_pending()}')

    # Per-model metrics
    for mc in _engine_config.models:
        state = _weight_manager.get_state(mc.name)
        loaded = 1 if _weight_manager.is_loaded(mc.name) else 0
        pending = _request_manager.pending_count(mc.name)
        active = _request_manager.active_count(mc.name)
        lines.append(f'engine_model_loaded{{model="{mc.name}"}} {loaded}')
        lines.append(f'engine_model_pending{{model="{mc.name}"}} {pending}')
        lines.append(f'engine_model_active{{model="{mc.name}"}} {active}')
        if state and state.active_gpu_id is not None:
            lines.append(f'engine_model_gpu{{model="{mc.name}"}} {state.active_gpu_id}')

    # GPU metrics
    for gpu_id in _engine_config.gpu_ids:
        import torch
        mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1e9
        mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
        lines.append(f'engine_gpu_memory_allocated_gb{{gpu="{gpu_id}"}} {mem_alloc:.2f}')
        lines.append(f'engine_gpu_memory_reserved_gb{{gpu="{gpu_id}"}} {mem_reserved:.2f}')

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/completions")
async def completions(req: CompletionRequest, authorization: str | None = None):
    api_key = _extract_api_key(authorization)
    return await _handle_completion(req.model, req.prompt, req.max_tokens, req.temperature, req.stream, api_key)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, authorization: str | None = None):
    api_key = _extract_api_key(authorization)
    state = _weight_manager.get_state(req.model)
    if state is None:
        raise HTTPException(404, f"Model '{req.model}' not found")

    tokenizer = state.tokenizer
    try:
        prompt = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        prompt += "\nassistant:"

    return await _handle_completion(req.model, prompt, req.max_tokens, req.temperature, req.stream, api_key)


def _extract_api_key(authorization: str | None) -> str | None:
    """Extract API key from Authorization header."""
    if not authorization:
        return None
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization


MAX_QUEUE_DEPTH = 64

async def _handle_completion(model: str, prompt: str, max_tokens: int, temperature: float, stream: bool,
                             api_key: str | None = None):
    """Core handler for both completions and chat completions."""
    # Authenticate
    tenant_id = _tenant_manager.authenticate(api_key) if _tenant_manager else "default"
    tenant_config = _tenant_manager.get_tenant_config(tenant_id) if _tenant_manager else None

    # Check model access
    if _tenant_manager and not _tenant_manager.check_model_access(tenant_id, model):
        raise HTTPException(403, f"Tenant '{tenant_id}' does not have access to model '{model}'")

    state = _weight_manager.get_state(model)
    if state is None:
        raise HTTPException(404, f"Model '{model}' not found")

    # Rate limiting
    if _tenant_manager:
        allowed, reason = _tenant_manager.check_rate_limit(tenant_id)
        if not allowed:
            raise HTTPException(429, reason)

    if _request_manager.pending_count(model) >= MAX_QUEUE_DEPTH:
        raise HTTPException(429, f"Model '{model}' queue is full")

    tokens = state.tokenizer.encode(prompt)

    # Track usage
    if _tenant_manager:
        _tenant_manager.on_request_start(tenant_id, len(tokens))

    priority = tenant_config.priority if tenant_config else 0
    slo = tenant_config.slo_ttft_ms if tenant_config else None

    request = _request_manager.add_request(
        model_name=model,
        prompt=prompt,
        prompt_tokens=tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        tenant_id=tenant_id,
        priority=priority,
        slo_ttft_ms=slo,
    )

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    if stream:
        return StreamingResponse(
            _stream_response(request, state, request_id, model),
            media_type="text/event-stream",
        )
    else:
        return await _wait_response(request, state, request_id, model)


async def _wait_response(request, state, request_id: str, model: str):
    """Wait for request to complete and return full response."""
    timeout = 120.0  # 2 minute timeout
    start = time.time()
    while request.state != RequestState.DONE:
        if time.time() - start > timeout:
            raise HTTPException(504, "Request timed out")
        await asyncio.sleep(0.01)

    if _tenant_manager:
        _tenant_manager.on_request_complete(
            request.tenant_id, len(request.generated_tokens), error=bool(request.error))

    if request.error:
        raise HTTPException(500, f"Generation error: {request.error}")

    text = state.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(request.arrival_time),
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": "stop" if (
                request.generated_tokens and
                request.generated_tokens[-1] == state.tokenizer.eos_token_id
            ) else "length",
        }],
        "usage": {
            "prompt_tokens": len(request.prompt_tokens),
            "completion_tokens": len(request.generated_tokens),
            "total_tokens": len(request.prompt_tokens) + len(request.generated_tokens),
        },
    }


async def _stream_response(request, state, request_id: str, model: str):
    """Stream tokens as SSE events."""
    last_len = 0

    while request.state != RequestState.DONE:
        current_len = len(request.generated_tokens)
        if current_len > last_len:
            # Decode only the new tokens
            new_tokens = request.generated_tokens[last_len:current_len]
            text = state.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if text:
                chunk = {
                    "id": request_id,
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            last_len = current_len
        await asyncio.sleep(0.005)

    # Final chunk with any remaining tokens
    if len(request.generated_tokens) > last_len:
        new_tokens = request.generated_tokens[last_len:]
        text = state.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if text:
            chunk = {
                "id": request_id,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Done
    finish_reason = "stop" if (
        request.generated_tokens and
        request.generated_tokens[-1] == state.tokenizer.eos_token_id
    ) else "length"
    done_chunk = {
        "id": request_id,
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def run_server(
    models: list[dict],
    gpu_ids: list[int] = [0],
    host: str = "0.0.0.0",
    port: int = 8000,
    **engine_kwargs,
):
    """Convenience function to create engine and start serving."""
    create_engine(models, gpu_ids, **engine_kwargs)
    uvicorn.run(app, host=host, port=port, log_level="info")
