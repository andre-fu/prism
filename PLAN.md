# Multi-Tenant GPU Inference Engine — Phase 1 Implementation Plan

**Target:** Multiple models sharing 4×H100, each ≤2× latency of dedicated serving
**Hardware:** 4×H100 80GB, PCIe Gen5, NVLink, 885GB host RAM
**Timeline:** 12 weeks
**Codebase:** `/lambda/nfs/inference-texas/`
**Available models:** Qwen2.5-0.5B, 1.5B, 7B-Instruct, 14B-Instruct, 72B-Instruct (181GB cached)

---

## Guiding Principle: Smallest Provable Unit First

Every milestone produces a runnable script that proves one new capability. We never build two unproven things at once. The sequence is:

1. Can we load weights into pinned RAM and move them to GPU? (Day 1-2 spike)
2. Can we run a forward pass with those weights? (Day 1-2 spike)
3. Can we generate correct tokens? (Week 1-2)
4. Can we do that on multiple GPUs with TP? (Week 3-4)
5. Can we swap between models on one GPU? (Week 5-6)
6. Can we swap with TP=4 across all GPUs? (Week 5-6)
7. Can we overlap the swap with compute? (Week 5-6)
8. Can we schedule multiple models automatically? (Week 7-8)
9. Can we serve this over HTTP? (Week 7-8)
10. Can we beat the 2× latency target? (Week 9-12)

If any step fails, we know exactly what broke and can fix or fall back before building on top of it.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│              API Server (FastAPI)                      │
│  OpenAI-compatible /v1/chat/completions               │
│  Model routing by request `model` field               │
├──────────────────────────────────────────────────────┤
│              Request Manager                          │
│  Per-model request queues                             │
│  Batch formation (prefill vs decode groups)           │
│  Continuous batching                                  │
├──────────────────────────────────────────────────────┤
│         Scheduler + GPU Allocator                     │
│  Round-robin across models with pending reqs          │
│  GPU claim: TP=4 claims all 4, TP=1 claims 1         │
│  Spatial multiplexing: independent models on          │
│    different GPUs can run concurrently                │
│  Queue-depth-triggered prefetch signals               │
│  Batch-level model switching with GPU barriers        │
├──────────────────────────────────────────────────────┤
│            Weight Manager                             │
│  Per-GPU tiered pool: T0 (HBM) ↔ T1 (pinned RAM)   │
│  Per-GPU residency tracking                           │
│  TP-aware: stores full weights in T1, shards on       │
│    T1→T0 transfer                                    │
│  Coordinated multi-GPU async prefetch                 │
│  LRU eviction per GPU                                │
├──────────────────────────────────────────────────────┤
│            Model Executor                             │
│  TP-aware forward pass (NCCL all-reduce)             │
│  Per-GPU paged KV cache (sharded heads)              │
│  Sampling (top-k, top-p, temperature)                │
│  KV cache destruction on model eviction              │
├──────────────────────────────────────────────────────┤
│         GPU Memory Allocator (per-GPU)                │
│  CUDA memory pool for weights + KV cache per device  │
│  Explicit allocation/free (no torch GC)              │
│  4 independent pools, one per H100                   │
└──────────────────────────────────────────────────────┘
```

### GPU Allocation Model

The scheduler treats the 4×H100 cluster as a **GPU resource pool**. Each model declares its TP requirement at registration time:

```
Model A: Qwen2.5-7B-Instruct    TP=1  → claims GPU 0         (7 GB FP8)
Model B: Qwen2.5-7B-Chat        TP=1  → claims GPU 1         (7 GB FP8)
Model C: Qwen2.5-14B-Instruct   TP=2  → claims GPU 2,3       (7 GB/GPU FP8)
Model D: Llama-3.1-70B          TP=4  → claims GPU 0,1,2,3   (17.5 GB/GPU FP8)
```

**Key rules:**
- A TP=4 model needs exclusive access to all 4 GPUs — evict any TP=1/TP=2 models first.
- Multiple TP=1 models can run concurrently on different GPUs (true spatial multiplexing).
- A TP=2 model claims 2 GPUs; the other 2 can serve independent TP=1 models.
- The scheduler treats this as 2D bin-packing: models × GPUs × time.
- When a TP>1 model needs to load, ALL its GPUs must complete their eviction+transfer before compute starts (barrier synchronization).

### Memory Budget Per GPU (80 GB HBM)

| Allocation | Budget | Notes |
|---|---|---|
| Weight T0 pool | 40 GB | Room for multiple small models or 1 large shard |
| KV cache pool | 35 GB | ~70K tokens at 7B, less for larger models |
| Overhead | 5 GB | CUDA context, NCCL buffers, activations, fragmentation |

### Key Design Decisions

1. **Transformers for model execution, not vLLM.** Day 0 spike proved vLLM models can't be used outside their engine (torch.compile coupling). We use `transformers.AutoModelForCausalLM` with meta device + `load_state_dict(assign=True)` for fast weight swaps. ~15-30% slower than vLLM's fused kernels, but correct and fully controllable. Can add flash_attn integration later via `config._attn_implementation = "flash_attention_2"`.

2. **Single process, multi-GPU from day one.** Every data structure is `dict[gpu_id, ...]`. Even when testing on 1 GPU, the code paths are the same — just `gpu_ids=[0]`.

3. **Batch-level switching, not token-level.** Token-level (Aegaeon-style) is Phase 2.

4. **KV cache destruction on switch.** Oneiros-style preservation is Phase 2.

5. **TP-aware weight sharding on transfer.** T1 stores full weights. Sharding happens during T1→T0 copy.

---

## Project Structure

```
inference-texas/
├── gpu_swap/              # Existing code (preserved, not modified)
├── engine/                # New unified engine
│   ├── __init__.py
│   ├── __main__.py        # CLI entry point
│   ├── config.py          # Engine configuration (Pydantic)
│   ├── server.py          # FastAPI OpenAI-compatible API
│   ├── request_manager.py # Per-model queues, batch formation
│   ├── scheduler.py       # Multi-model scheduler + GPU allocator
│   ├── weight_manager.py  # Tiered weight pool (HBM ↔ RAM), per-GPU
│   ├── model_executor.py  # Forward pass, sampling, TP-aware
│   ├── memory_pool.py     # CUDA memory allocator, per-GPU
│   ├── prefetch.py        # Coordinated multi-GPU async prefetch
│   ├── kv_cache.py        # Paged KV cache management, per-GPU
│   ├── distributed.py     # NCCL init, TP sharding, all-reduce
│   └── bench/
│       ├── __init__.py
│       ├── latency.py     # Per-request latency benchmarks
│       ├── throughput.py   # Throughput under load
│       ├── swap_bench.py  # Multi-GPU swap timing
│       └── compare_vllm.py # Head-to-head vs dedicated vLLM
├── tests/
│   ├── test_memory_pool.py
│   ├── test_weight_manager.py
│   ├── test_distributed.py
│   ├── test_model_executor.py
│   ├── test_scheduler.py
│   └── test_e2e.py
└── PLAN.md
```

---

## Day 0: Gate Spike — COMPLETED 2026-03-30

### Result: Approach B confirmed. Approach A (vLLM models) failed.

**Approach A (vLLM model classes):** FAIL — `torch.compile`/`FakeTensorMode` in vLLM's forward path blocks manual inference. vLLM models are too coupled to their engine's execution pipeline.

**Approach B (transformers + meta device):** PASS — correct output, fast swaps.

### Confirmed technique (used in all subsequent work):

```python
# 1. Load safetensors into pinned host RAM (one-time, at startup)
pinned_weights = {}
for path in safetensors_paths:
    shard = load_file(str(path), device="cpu")
    for name, t in shard.items():
        buf = torch.empty_like(t).pin_memory()
        buf.copy_(t)
        pinned_weights[name] = buf

# 2. Create model skeleton on meta device (fast: 0.37s for 7B)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

# 3. Async transfer pinned → GPU (47 GB/s on H100 PCIe Gen5)
gpu_weights = {}
with torch.cuda.stream(copy_stream):
    for name, t in pinned_weights.items():
        gpu_weights[name] = t.to(device, non_blocking=True)
copy_stream.synchronize()

# 4. Assign weights (replaces meta tensors in-place, 0.006s)
model.load_state_dict(gpu_weights, strict=False, assign=True)
reinit_rotary_buffers(model, config)  # rotary inv_freq not in safetensors
model.to(device)
model.eval()
```

### Measured performance:

| Operation | 7B (15 GB) | 14B (30 GB) |
|---|---|---|
| Meta skeleton | 0.37s | ~0.4s |
| Pinned → GPU | 0.33s (47 GB/s) | 0.71s (42 GB/s) |
| Evict | 0.01-0.03s | 0.01-0.03s |
| **Total swap** | **0.41s** | **0.72s** |
| Inference | 49 tok/s | ~35 tok/s |

### Optimization for engine (Week 1-2):
Create model skeletons ONCE at startup, reuse across swaps. Don't recreate from_config each time — just `load_state_dict(assign=True)` with new weights.

---

## Week 1-2: Single-GPU Single-Model End-to-End

**Goal:** One model, one GPU, our weight manager, correct token generation. The smallest complete unit.

**Files:** `engine/config.py`, `engine/memory_pool.py`, `engine/weight_manager.py`, `engine/model_executor.py`, `engine/kv_cache.py`

### Task 1: Config + Memory Pool

**`engine/config.py`:**
```python
class ModelConfig:
    model_id: str          # HF path
    tp_size: int           # 1, 2, or 4
    dtype: str             # "fp16", "fp8"
    max_batch_size: int
    max_seq_len: int

class EngineConfig:
    models: list[ModelConfig]
    gpu_ids: list[int]         # [0, 1, 2, 3]
    t0_budget_gb: float        # HBM for weights per GPU (default: 40)
    t1_budget_gb: float        # Total pinned RAM budget (default: 400)
    kv_cache_budget_gb: float  # HBM for KV per GPU (default: 35)
```

**`engine/memory_pool.py`:**
- `CUDAPool(gpu_id, size_gb)`: Pre-allocate HBM on a specific GPU. Bump allocator.
- `PinnedPool(size_gb)`: Pre-allocate pinned host memory.
- `MultiGPUPool(gpu_ids, size_gb_per_gpu)`: One `CUDAPool` per GPU.
- `async_copy(src, dst, stream)`: Non-blocking H2D/D2H.
- Each GPU gets its own CUDA copy stream.

**Test:** Allocate 10GB GPU pool on GPU 0, 50GB pinned pool. Copy 1GB back and forth. Verify ~51 GB/s H2D bandwidth.

### Task 2: Weight Manager (single GPU, no sharding yet)

**`engine/weight_manager.py`:**
- `WeightManager(engine_config)`:
  - Load safetensors for each model into `PinnedPool` (T1).
  - Residency tracking: `{model_id: {gpu_id: "t0"|"t1"|"none"}}`.
- `load_to_gpu(model_id, gpu_ids, stream_per_gpu) -> dict[gpu_id, Event]`:
  - Week 1-2: only handles `gpu_ids=[single_gpu]`, no sharding.
  - Async copy from pinned → GPU. Return CUDA event.
- `evict(model_id)`: Free T0 space.
- `is_loaded(model_id) -> bool`
- `resident_on(gpu_id) -> list[model_id]`

**Test:** Load Qwen2.5-7B weights into pinned RAM. Transfer to GPU 0. Verify tensor shapes match. Evict. Transfer again. Print transfer times.

### Task 3: Model Executor (single GPU, single model)

**`engine/model_executor.py`:**
- Based on Day 0 spike result (Approach A or B).
- `ModelExecutor(engine_config)`:
  - Instantiate model skeleton(s) on meta device.
  - `activate(model_id, gpu_id)`: Load weights from weight manager into model.
- `prefill(model_id, token_ids_batch) -> logits`: First forward pass.
- `decode(model_id, last_token_ids) -> logits`: One decode step.
- `sample(logits, params) -> token_ids`: Greedy, top-k, top-p.
- Tokenization via `transformers.AutoTokenizer`.

**`engine/kv_cache.py`:**
- `KVCachePool(gpu_id, budget_gb, num_layers, num_kv_heads, head_dim, block_size=16)`:
  - Allocate KV blocks on GPU.
- `BlockAllocator`: Manage free/allocated blocks.
  - `allocate(seq_id, num_tokens) -> list[block_ids]`
  - `free(seq_id)`, `free_all()`

### Task 4: Integration test

```bash
python -m engine --model Qwen/Qwen2.5-7B-Instruct --tp 1 --gpu 0 --prompt "The capital of France is"
```

**Pass criteria:**
- Output is coherent.
- Greedy output matches `vllm.LLM` (or `transformers` generate) for same prompt.
- TTFT and TBT measured and printed.

**Deliverable:** Single-model inference works end-to-end through our weight manager and model executor on 1 GPU. This is the foundation everything else builds on.

---

## Week 3-4: Multi-GPU Tensor Parallelism

**Goal:** Same model executor, but now sharded across multiple GPUs. Prove TP=1, TP=2, and TP=4 all produce correct output.

**Files:** `engine/distributed.py`, updates to `engine/weight_manager.py`, `engine/model_executor.py`, `engine/kv_cache.py`

### Task 5: NCCL + TP Sharding

**`engine/distributed.py`:**
- `init_nccl(gpu_ids)`: Initialize NCCL process groups. Single-process multi-GPU.
- `TPShardingPlan(model_config)`: Sharding plan per model architecture.
  - Column-parallel: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`.
  - Row-parallel: `o_proj`, `down_proj`.
  - Replicated: embeddings, layer norms, LM head.
- `shard_weight(tensor, tp_rank, tp_size, mode) -> Tensor`

**Test:** Load Qwen2.5-7B weights, shard for TP=4. Verify each shard is 1/4 size for parallel layers, full size for replicated.

### Task 6: Weight Manager TP Support

Update `load_to_gpu()`:
- When `tp_size > 1`: for each GPU in the TP group, shard the weight for that rank, then async copy the shard.
- Launches `tp_size` parallel H2D transfers (one per GPU).
- `is_loaded(model_id)`: True only when ALL GPUs done (barrier).
- `evict(model_id)`: Free on ALL GPUs atomically.

**Test:** Load Qwen2.5-14B, shard to 4 GPUs. Print per-GPU transfer time and barrier skew.

### Task 7: Model Executor TP Support

Update forward pass:
- For TP>1: each GPU runs its shard. NCCL all-reduce after row-parallel layers.
- KV cache sharded: each GPU holds `num_kv_heads / tp_size` heads.
- Sampling on rank 0 only.

If using Approach A (vLLM models): set up `vllm.distributed.parallel_state` so `ColumnParallelLinear`/`RowParallelLinear` work correctly.
If using Approach B: hand-roll the TP loop (each GPU forward → all-reduce → next layer).

### Task 8: TP Integration Tests

```bash
# TP=1 on GPU 0
python -m engine --model Qwen/Qwen2.5-7B-Instruct --tp 1 --prompt "Hello"

# TP=2 on GPUs 0,1
python -m engine --model Qwen/Qwen2.5-14B-Instruct --tp 2 --prompt "Hello"

# TP=4 on GPUs 0,1,2,3
python -m engine --model Qwen/Qwen2.5-14B-Instruct --tp 4 --prompt "Hello"

# TP=4 large model
python -m engine --model Qwen/Qwen2.5-72B-Instruct --tp 4 --prompt "Hello"
```

**Pass criteria for each:**
- Coherent output.
- Greedy output matches vLLM with same TP size.
- TTFT/TBT printed.

**Deliverable:** TP works at all sizes. Per-GPU transfer times measured. We can now run any model at any TP on our cluster.

**Expected transfer times:**

| Model | TP | Weight/GPU | Transfer wall-clock |
|---|---|---|---|
| 7B FP16 | 1 | 14 GB | ~275ms |
| 14B FP16 | 2 | 14 GB/GPU | ~275ms (parallel) |
| 14B FP16 | 4 | 7 GB/GPU | ~137ms (parallel) |
| 72B FP16 | 4 | 36 GB/GPU | ~706ms (parallel) |

---

## Week 5-6: Multi-Model Swap + Async Prefetch

**Goal:** Two or more models, one set of GPUs. Swap between them. Overlap transfer with compute. This is the core innovation.

**Files:** `engine/prefetch.py`, updates to `engine/weight_manager.py`, `engine/model_executor.py`

### Task 9: Basic Swap (synchronous, no overlap)

The simplest multi-model test: two models, swap synchronously.

```
1. Load model A weights to GPU (via weight_manager)
2. Activate model A in executor
3. Run 1 request on model A (prefill + 50 decode steps)
4. Evict model A from GPU (free T0)
5. Load model B weights to GPU
6. Activate model B in executor
7. Run 1 request on model B
8. Print: total swap time (evict A + load B), request latencies
```

**Test this at multiple TP sizes:**
- 2× 7B swapping on GPU 0 (TP=1)
- 2× 14B swapping on GPUs 0-3 (TP=4) — coordinated 4-GPU swap

**Pass criteria:** Both models produce correct output after swap. Swap time matches expected bandwidth numbers.

### Task 10: Async Prefetch (overlapped transfer + compute)

**`engine/prefetch.py`:**
- `PrefetchController(weight_manager)`:
  - `start_prefetch(model_id, gpu_ids)`: Async H2D on copy streams (separate from compute stream).
  - `is_ready(model_id) -> bool`: All GPUs done.
  - `wait(model_id)`: Block until ready.

Now the swap becomes:
```
1. Model A is active, running decode steps
2. While model A decodes, start async prefetch of model B on copy streams
3. After N decode steps, model A's batch finishes
4. Check: is model B ready? If yes, switch instantly. If no, wait for remaining transfer.
5. Evict model A, activate model B
6. Run model B's request
```

**Overlap measurement script:**
- Instrument: decode step duration, transfer duration, overlap ratio.
- For 7B on 1 GPU: transfer ~275ms FP16, decode step ~20ms → ~14 decode steps of overlap.
- For 72B on 4 GPUs (TP=4): transfer ~706ms, decode step ~30ms → ~23 decode steps.

**Test:**
```bash
python -m engine.bench.swap_bench \
  --model-a Qwen/Qwen2.5-7B-Instruct --tp-a 1 \
  --model-b Qwen/Qwen2.5-14B-Instruct --tp-b 1 \
  --gpu 0 --overlap
```

**Pass criteria:**
- Prefetch overlap > 50% (most of transfer hidden behind compute).
- Model B's first request TTFT is within 2× of its standalone TTFT.

### Task 11: Multi-GPU Swap Benchmark

Full swap matrix:

| Swap | TP | GPUs | Expected wall-clock |
|---|---|---|---|
| 7B→7B | 1 | 1 | ~275ms (FP16) |
| 14B→14B | 2 | 2 | ~275ms (14 GB/GPU parallel) |
| 14B→14B | 4 | 4 | ~137ms (7 GB/GPU parallel) |
| 72B→72B | 4 | 4 | ~706ms (36 GB/GPU parallel) |
| 7B(TP=1)→72B(TP=4) | mixed | 4 | ~706ms + eviction |

Record: per-GPU transfer time, barrier skew, total idle time.

**Deliverable:** We can swap any two models at any TP size, with async prefetch overlapping transfer and compute. Swap timing data for all configurations.

---

## Week 7-8: Scheduler + API Server

**Goal:** Automatic multi-model scheduling (no manual swap commands) and HTTP API. Multiple models serve requests without human intervention.

**Files:** `engine/request_manager.py`, `engine/scheduler.py`, `engine/server.py`

### Task 12: Request Manager

**`engine/request_manager.py`:**
- `Request`: dataclass — `model_id`, `prompt_tokens`, `sampling_params`, `arrival_time`, `state`, `output_tokens`, `kv_block_ids`.
- `RequestManager`:
  - `add_request(model_id, prompt, params) -> request_id`
  - `get_pending(model_id) -> list[Request]`
  - `get_active(model_id) -> list[Request]`
  - `complete(request_id)`
  - `models_with_pending() -> list[model_id]`

### Task 13: Scheduler + GPU Allocator

**`engine/scheduler.py`:**

`GPUAllocator`:
- Track which models are claimed on which GPUs.
- `claim(model_id, gpu_ids) -> bool`
- `release(model_id)`
- `can_coexist(model_a, model_b) -> bool`: GPU sets don't overlap.
- `plan_evictions(model_id) -> list[model_id]`

`Scheduler` main loop:
```
while True:
  1. Check for completed requests, return results
  2. Determine schedulable set: models with pending requests
     whose weights are in T0 (or prefetch complete)
  3. Among schedulable models, find compatible GPU assignment:
     - TP=1 models on separate GPUs → concurrent (spatial multiplex)
     - TP=4 model → needs exclusive access
  4. For models that need loading:
     a. Compute eviction plan
     b. Sync on prefetch events (or pay sync transfer penalty)
     c. Barrier: wait for ALL GPUs in TP group
  5. For each active GPU group, form batch and execute
  6. Signal prefetch controller for next likely model
  7. Sample tokens, check stop conditions
  8. Loop
```

**Spatial multiplexing:** Model A (TP=1, GPU 0) and Model B (TP=1, GPU 1) run simultaneously. Two forward passes, two CUDA streams, one scheduler tick. (If GIL proves to be a bottleneck here, fall back to time-sharing — one model at a time. Still proves the core thesis, just with lower throughput.)

**Continuous batching:** After each decode step, check for new prefill requests for the current model. Add to batch without draining.

### Task 14: API Server

**`engine/server.py`:**
- `POST /v1/chat/completions`: Chat completions with `model` routing.
- `POST /v1/completions`: Text completions.
- `GET /v1/models`: List loaded models (TP size, dtype, GPU assignment).
- `GET /health`, `GET /metrics`.
- Streaming via SSE. `asyncio.Queue` per request.
- Backpressure: HTTP 429 when queues full.

### Task 15: Scheduler Integration Tests

**Scenario 1 — Spatial multiplexing (no swapping):**
4× Qwen2.5-7B-Instruct loaded as 4 aliases (model-a through model-d), each TP=1 on GPUs 0-3. Send concurrent requests to all 4. All serve simultaneously.
- Pass: all 4 respond correctly, latency ≈ dedicated.

**Scenario 2 — Oversubscribed single GPU:**
4× Qwen2.5-0.5B loaded on GPU 0 (all fit in T0 but only 1 active at a time for compute). Scheduler round-robins.
- Pass: all 4 respond, latency ≤ 2× dedicated.

**Scenario 3 — Mixed TP concurrent:**
1× Qwen2.5-14B (TP=2, GPUs 0-1) + 2× Qwen2.5-7B (TP=1, GPUs 2, 3). All concurrent.
- Pass: all 3 respond correctly. No GPU conflicts.

**Scenario 4 — TP=4 swap:**
2× Qwen2.5-14B loaded as two aliases, both TP=4 (forced). Only one fits in T0 at a time. Scheduler swaps between them.
- Pass: both respond. Swap is async-prefetched.

**Deliverable:** `python -m engine serve --config engine_config.yaml` starts the engine. `curl` requests to any loaded model get correct streaming responses. Scheduler handles all 4 scenarios automatically.

---

## Week 9-10: Optimization + Edge Cases

**Goal:** Close performance gaps. Handle scheduling edge cases.

### Task 16: Profiling and Bottleneck Identification

- Profile with `torch.cuda.nvtx` + Nsight Systems.
- Measure per-component time: weight transfer, prefill, decode, sampling, scheduling overhead, NCCL all-reduce.
- Identify: is the bottleneck compute, memory bandwidth, scheduling, or GIL?

### Task 17: Optimizations (prioritized by profile results)

Likely candidates:
- **CUDA graph capture** for decode steps (skip kernel launch overhead). Capture per model per batch size. Invalidate on model switch.
- **Weight tensor contiguous packing**: Single contiguous buffer per model in T1 for max DMA bandwidth. Profile scattered vs contiguous.
- **Prefetch threshold tuning**: Sweep `prefetch_queue_threshold` to maximize hit rate without wasting bandwidth.
- **Batch size tuning** per model size.

### Task 18: Scheduler Edge Cases

- **TP conflict resolution**: TP=4 model active, TP=1 request arrives. Wait for batch completion, then serve TP=1 on 1 GPU while TP=4 weights stay resident (if T0 budget allows).
- **Starvation prevention**: `max_consecutive_batches` forces yield. During yield, other models get GPU time.
- **Deadlock prevention**: GPU allocator uses global model-ID ordering for eviction.
- **Memory fragmentation**: Run 1000 load/evict cycles. Verify no OOM. Switch to slab allocator if needed.

### Task 19: Model Hot-Loading

`POST /v1/models/load` to add a model at runtime. Weight manager loads to T1 (pinned RAM). Model becomes schedulable on next request.

**Deliverable:** Measurable performance improvements with before/after numbers. All edge cases tested or documented as Phase 2.

---

## Week 11-12: End-to-End Benchmarking

**Goal:** Prove the MVP hypothesis with numbers.

### Task 20: Head-to-Head vs vLLM (`engine/bench/compare_vllm.py`)

**Scenario A — 4×7B, spatial multiplexing (best case):**
- Baseline: 4 dedicated vLLM instances, one 7B per GPU.
- Ours: 4×7B TP=1, one per GPU.
- Target: ≤1.1× latency (prove no overhead).

**Scenario B — 4×7B, single GPU (contention):**
- Baseline: 4 dedicated vLLM instances, one 7B per GPU.
- Ours: 4×7B all on GPU 0, scheduler swaps.
- Target: ≤2× latency (prove swap + prefetch works).

**Scenario C — Mixed TP (realistic):**
- Ours: 1×14B TP=2 (GPUs 0-1) + 2×7B TP=1 (GPUs 2,3), all concurrent.
- Target: ≤1.3× latency (prove spatial multiplexing with mixed TP).

**Scenario D — Oversubscribed TP=4 (swap-heavy):**
- Ours: 2×72B TP=4, scheduler swaps between them.
- Target: ≤2× latency (prove coordinated 4-GPU swap).

**Traffic patterns:**
- ShareGPT traces (multi-turn, bursty)
- Azure coding traces (single-turn, high throughput)
- Synthetic uniform (steady state)
- Synthetic bursty (10× spike to one model)

**Metrics per scenario:**
- TTFT (p50, p95, p99)
- TBT (p50, p95, p99)
- End-to-end latency, throughput (tok/s per model + aggregate)
- GPU utilization per device
- Switch count, prefetch hit rate
- Cost: aggregate tokens/sec per $GPU-hour

### Task 21: Latency Breakdown (`engine/bench/latency.py`)

- Weight transfer time (per GPU, barrier skew)
- Prefetch overlap ratio
- KV cache alloc, prefill/decode latency, scheduling overhead
- NCCL all-reduce time (TP>1)
- Model switch total idle time

### Task 22: Throughput Scaling (`engine/bench/throughput.py`)

- Ramp concurrent requests 1→128.
- Measure saturation per scenario.
- GPU utilization at each level.
- Identify per-scenario bottleneck.

### Task 23: Stress Test

- 1 hour continuous mixed traffic.
- Monitor: memory leaks, fragmentation, timeouts, NCCL errors.
- Inject: kill model mid-request, hot-load a model, traffic spike.

**Success criteria:**
- Scenario A: ≤1.1× latency vs dedicated.
- Scenario B: ≤2× latency vs dedicated.
- Scenario C: ≤1.3× latency.
- Scenario D: ≤2× latency.
- No crashes, leaks, or errors in 1-hour stress.

---

## Critical Path

```
Day 0: Gate Spike ─── PASS? ──→ Week 1-2: Single GPU, Single Model, E2E
  (vLLM model +                   (weight mgr + executor + KV cache
   external weights)                + correct token generation)
                                           │
                                           ▼
                                  Week 3-4: Multi-GPU TP
                                    (TP=1,2,4 all correct
                                     + transfer benchmarks)
                                           │
                                           ▼
                                  Week 5-6: Multi-Model Swap
                                    (sync swap → async prefetch
                                     + overlap measurement
                                     + multi-GPU swap matrix)
                                           │
                                           ▼
                                  Week 7-8: Scheduler + API
                                    (automatic scheduling
                                     + spatial multiplexing
                                     + HTTP serving
                                     + 4 integration scenarios)
                                           │
                                           ▼
                                  Week 9-10: Optimize + Harden
                                    (profile → optimize
                                     + edge cases + hot-load)
                                           │
                                           ▼
                                  Week 11-12: Benchmark
                                    (4 scenarios vs vLLM
                                     + stress test + report)
```

**Risk 1 — Day 0 gate:** RESOLVED. vLLM model extraction failed. Approach B (transformers) confirmed working with correct output and sub-second swaps.

**Risk 2 — Week 3-4:** Single-process multi-GPU TP. `torch.distributed` normally expects one-process-per-GPU. If single-process NCCL doesn't work cleanly, fall back to multi-process with shared memory coordination (more like current gpu_swap architecture but with our scheduler on top).

**Risk 3 — Week 7-8:** Spatial multiplexing GIL bottleneck. If concurrent forward passes on different GPUs serialize on GIL, fall back to time-sharing (one model active at a time). Still proves scheduling + prefetch, just at lower throughput.

---

## Dependencies & Prerequisites

- **Already installed:** `torch 2.10`, `vllm 0.18`, `transformers 4.57`, `safetensors 0.7`, `flashinfer 0.6`
- **Need to install:** `fastapi`, `uvicorn`, `sse-starlette`, `prometheus-client`
- **Models cached:** Qwen2.5-0.5B (spike), 1.5B, 7B-Instruct, 14B-Instruct, 72B-Instruct
- **Test traffic:** ShareGPT dataset, Azure LLM traces

---

## What This Plan Does NOT Cover (Phase 2+)

- Token-level scheduling (Aegaeon-style interleaving within a batch)
- KV cache preservation across model switches (Oneiros-style parameter remapping)
- GH200 optimization (900 GB/s C2C bandwidth — would cut all swap times by ~7×)
- NVMe cold tier (T2) for hundreds of models
- Cross-tenant security isolation (memory zeroing, timing side-channels)
- VLM support (vision encoders, cross-modal KV cache)
- SLO-aware scheduling with per-tenant latency budgets
- Traffic prediction / learned prefetch
- Multi-node distributed serving (TP > 4)
- Layer-by-layer pipelined prefetch (partial weight residency)
