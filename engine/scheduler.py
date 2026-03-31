"""Multi-model scheduler: decides which model gets GPU time, manages swaps."""

import time
import threading
import torch
from dataclasses import dataclass, field

from .config import EngineConfig, SchedulerConfig
from .memory_pool import PinnedPool, MultiGPUPool
from .weight_manager import WeightManager
from .kv_cache import PagedKVPool
from .fa_kv_cache import FlashAttnKVCache
from .model_executor import ModelExecutor
from .fa_executor_v2 import FlashAttnExecutorV2
from .tp_executor import TPModelExecutor
from .distributed import TPGroup
from .request_manager import RequestManager, Request, RequestState
from .prefetch import PrefetchController


class Scheduler:
    """Main scheduling loop.

    Round-robins across models with pending requests. For each model:
    1. Ensure weights are on GPU (via prefetch or sync load)
    2. Pop pending requests, run prefill
    3. Run decode steps for active requests
    4. Signal prefetch for the next likely model
    5. Yield after max_consecutive_batches
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        scheduler_config: SchedulerConfig,
        weight_manager: WeightManager,
        request_manager: RequestManager,
        prefetch_controller: PrefetchController,
        gpu_pool: MultiGPUPool,
    ):
        self.engine_config = engine_config
        self.scheduler_config = scheduler_config
        self.wm = weight_manager
        self.rm = request_manager
        self.prefetch = prefetch_controller
        self.gpu_pool = gpu_pool

        # Per-model KV caches (FlashAttnKVCache for TP=1, PagedKVPool list for TP>1)
        self._kv_pools: dict[str, PagedKVPool] = {}  # Legacy FlashInfer (TP>1)
        self._fa_kv: dict[str, FlashAttnKVCache] = {}  # flash_attn (TP=1)
        self._tp_kv_pools: dict[str, list[PagedKVPool]] = {}

        # Per-model executors
        self._executors: dict[str, FlashAttnExecutorV2 | TPModelExecutor] = {}

        # Architecture-keyed executor cache: models with same arch share graph
        # Key: (num_layers, hidden_size, num_heads, num_kv_heads, intermediate_size)
        self._arch_executors: dict[tuple, FlashAttnExecutorV2] = {}

        # Tracking
        self._current_model: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        # Stats
        self.stats = SchedulerStats()

    def _make_kv_pool(self, model_name: str, device: str, tp_kv_heads: int | None = None) -> PagedKVPool:
        """Create a KV pool for a model on a specific device."""
        state = self.wm.get_state(model_name)
        hf_config = state.hf_config
        dtype = getattr(torch, state.config.dtype)

        num_kv_heads = tp_kv_heads or hf_config.num_key_value_heads
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        page_size = self.engine_config.kv_page_size

        num_models = max(len(self.engine_config.models), 1)
        kv_budget = self.engine_config.kv_cache_budget_gb / num_models

        bytes_per_page = (
            2 * page_size * num_kv_heads * head_dim
            * dtype.itemsize * hf_config.num_hidden_layers
        )
        max_pages = max(int(kv_budget * 1e9 / bytes_per_page), 64)

        return PagedKVPool(
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_pages=max_pages,
            page_size=page_size,
            device=device,
            dtype=dtype,
        )

    def _get_kv_pool(self, model_name: str, gpu_id: int) -> PagedKVPool:
        """Get or create a KV pool for a TP=1 model."""
        if model_name in self._kv_pools:
            return self._kv_pools[model_name]
        pool = self._make_kv_pool(model_name, f"cuda:{gpu_id}")
        self._kv_pools[model_name] = pool
        return pool

    def _get_tp_kv_pools(self, model_name: str, gpu_ids: list[int]) -> list[PagedKVPool]:
        """Get or create KV pools for a TP>1 model (one per GPU)."""
        if model_name in self._tp_kv_pools:
            return self._tp_kv_pools[model_name]
        state = self.wm.get_state(model_name)
        tp_size = len(gpu_ids)
        kv_heads_per_gpu = state.hf_config.num_key_value_heads // tp_size
        pools = [self._make_kv_pool(model_name, f"cuda:{g}", kv_heads_per_gpu) for g in gpu_ids]
        self._tp_kv_pools[model_name] = pools
        return pools

    def _get_arch_key(self, hf_config) -> tuple:
        """Architecture fingerprint — models with same key share CUDA graphs."""
        return (
            hf_config.num_hidden_layers,
            hf_config.hidden_size,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.intermediate_size,
        )

    def _get_fa_kv(self, model_name: str, gpu_id: int) -> FlashAttnKVCache:
        """Get or create flash_attn KV cache.

        Models with the same architecture SHARE one KV cache (and one executor).
        KV sequences are cleared on model switch.
        """
        state = self.wm.get_state(model_name)
        arch_key = self._get_arch_key(state.hf_config)

        # Share KV cache across same-architecture models
        if model_name in self._fa_kv:
            return self._fa_kv[model_name]

        # Check if another model with same arch already has a cache
        for other_name, cache in self._fa_kv.items():
            other_state = self.wm.get_state(other_name)
            if other_state and self._get_arch_key(other_state.hf_config) == arch_key:
                self._fa_kv[model_name] = cache  # Share the same cache object
                return cache

        hf_config = state.hf_config
        dtype = getattr(torch, state.config.dtype)
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        page_size = 256

        kv_budget = self.engine_config.kv_cache_budget_gb
        bytes_per_page = (
            2 * page_size * num_kv_heads * head_dim * dtype.itemsize * hf_config.num_hidden_layers
        )
        max_pages = max(int(kv_budget * 1e9 / bytes_per_page), 16)

        cache = FlashAttnKVCache(
            hf_config.num_hidden_layers, num_kv_heads, head_dim,
            max_pages, page_size, f"cuda:{gpu_id}", dtype,
        )
        self._fa_kv[model_name] = cache
        return cache

    def _create_executor(self, model_name: str, gpu_id: int) -> FlashAttnExecutorV2 | TPModelExecutor:
        """Create the right executor for a model's TP size.

        For TP=1: uses FlashAttnExecutorV2 with CUDA graphs.
        If another model with the same architecture already has an executor,
        reuse it (swap weights into static buffers, no graph recapture).
        """
        state = self.wm.get_state(model_name)
        tp_size = state.config.tp_size

        if tp_size > 1:
            models = self.wm.get_models(model_name)
            gpu_ids = state.active_gpu_ids
            kv_pools = self._get_tp_kv_pools(model_name, gpu_ids)
            tp_group = TPGroup(gpu_ids)
            return TPModelExecutor(models, kv_pools, tp_group)

        # TP=1: use flash_attn with CUDA graphs
        model = self.wm.get_model(model_name)
        kv_cache = self._get_fa_kv(model_name, gpu_id)
        arch_key = self._get_arch_key(state.hf_config)

        # Check if we have an executor for this architecture (reuse graph + buffers)
        if arch_key in self._arch_executors:
            executor = self._arch_executors[arch_key]
            executor.kv = kv_cache
            executor.swap_weights(model)
            return executor

        # New architecture — free old executors to reclaim GPU memory
        for old_key, old_exec in list(self._arch_executors.items()):
            old_exec.invalidate_graph()
        self._arch_executors.clear()
        import gc; gc.collect()
        for gid in self.engine_config.gpu_ids:
            with torch.cuda.device(gid):
                torch.cuda.empty_cache()

        executor = FlashAttnExecutorV2(model, kv_cache, f"cuda:{gpu_id}")
        self._arch_executors[arch_key] = executor
        return executor

    def _ensure_model_loaded(self, model_name: str, gpu_id: int) -> ModelExecutor | TPModelExecutor:
        """Ensure model weights are on GPU and return an executor."""
        state = self.wm.get_state(model_name)
        tp_size = state.config.tp_size

        # Path 1: already loaded
        if self.wm.is_loaded(model_name):
            if model_name not in self._executors:
                self._executors[model_name] = self._create_executor(model_name, gpu_id)
            return self._executors[model_name]

        # Path 2: prefetch completed
        if self.prefetch.is_ready(model_name):
            self.prefetch.complete_prefetch(model_name)
            self.stats.prefetch_hits += 1
            self._executors[model_name] = self._create_executor(model_name, gpu_id)
            return self._executors[model_name]

        # Path 3: prefetch in-flight
        if self.prefetch.is_in_flight(model_name):
            self.prefetch.wait_ready(model_name)
            self.prefetch.complete_prefetch(model_name)
            self.stats.prefetch_hits += 1
            self._executors[model_name] = self._create_executor(model_name, gpu_id)
            return self._executors[model_name]

        # Path 4: cold load
        # For TP>1, need space on ALL GPUs in the TP group
        if tp_size > 1:
            gpu_ids = self.engine_config.gpu_ids[:tp_size]
            for gid in gpu_ids:
                pool = self.gpu_pool[gid]
                while not pool.can_fit_weights(state.gpu_bytes_per_device):
                    victim = self.wm.lru_candidate(gid)
                    if victim is None:
                        raise RuntimeError(f"Cannot fit {model_name} on GPU {gid}")
                    self._evict_model(victim)
                    self.stats.evictions += 1
            self.wm.load_to_gpu_tp(model_name, gpu_ids)
        else:
            pool = self.gpu_pool[gpu_id]
            while not pool.can_fit_weights(state.gpu_bytes_per_device):
                victim = self.wm.lru_candidate(gpu_id)
                if victim is None:
                    raise RuntimeError(f"Cannot fit {model_name} on GPU {gpu_id}")
                self._evict_model(victim)
                self.stats.evictions += 1
            self.wm.load_to_gpu(model_name, gpu_id)

        self.stats.sync_loads += 1
        self._executors[model_name] = self._create_executor(model_name, gpu_id)
        return self._executors[model_name]

    def _start_prefetch_for_next(self, current_model: str, gpu_id: int):
        """Start async prefetch for the next likely model.

        Only runs if current model is already loaded (we need it to be serving
        while the prefetch transfers on the copy stream). Does NOT reserve budget —
        that happens in complete_prefetch when the model is activated.
        """
        # Don't prefetch if current model isn't even loaded yet
        if not self.wm.is_loaded(current_model):
            return

        next_model = self.prefetch.suggest_prefetch(current_model, gpu_id)
        if next_model is None:
            return

        state = self.wm.get_state(next_model)
        if state is None:
            return

        # Check if there's PHYSICAL space (ignore budget — we'll evict on switch)
        # The async transfer just puts data on GPU; budget is tracked on activation
        self.prefetch.start_prefetch(next_model, gpu_id)
        self.stats.prefetch_triggers += 1

    def _evict_model(self, model_name: str):
        """Evict a model: free KV cache, requeue active requests, remove from GPU."""
        # Requeue any active requests — they'll need re-prefill
        active = self.rm.get_active(model_name)
        for req in active:
            if req.state == RequestState.DECODING:
                # Reset to queued state — scheduler will re-prefill when model reloads
                req.state = RequestState.QUEUED
                req.generated_tokens.clear()
                req.first_token_time = 0.0
                req.seq_id = -1
                self.stats.requeued += 1
        # Move active requests back to pending queue
        with self.rm._lock:
            requeued = self.rm._active.pop(model_name, [])
            for req in requeued:
                if req.state == RequestState.QUEUED:
                    self.rm._queues.setdefault(model_name, __import__('collections').deque()).append(req)

        # Free this model's KV sequences (not ALL sequences — other models' KV is preserved)
        if model_name in self._fa_kv:
            self._fa_kv[model_name].free_model_sequences(model_name)
            # Don't delete the cache object — it's shared with other same-arch models
        if model_name in self._kv_pools:
            self._kv_pools[model_name].free_all()
            del self._kv_pools[model_name]
        if model_name in self._tp_kv_pools:
            for pool in self._tp_kv_pools[model_name]:
                pool.free_all()
            del self._tp_kv_pools[model_name]
        # Don't delete arch_executors — they're reused across models
        if model_name in self._executors:
            del self._executors[model_name]
        self.wm.evict_from_gpu(model_name)
        print(f"[Scheduler] Evicted {model_name} (requeued {len(active)} requests)")

    def _pick_next_model(self) -> str | None:
        """Round-robin: pick next model with pending or active requests."""
        candidates = set()
        candidates.update(self.rm.models_with_pending())
        candidates.update(self.rm.models_with_active())

        if not candidates:
            return None

        # Simple round-robin: sort and pick next after current
        ordered = sorted(candidates)
        if self._current_model is None or self._current_model not in ordered:
            return ordered[0]

        idx = ordered.index(self._current_model)
        next_idx = (idx + 1) % len(ordered)
        return ordered[next_idx]

    def _kv_new_sequence(self, model_name: str, seq_id: int):
        """Create a new sequence in all KV pools for this model."""
        if model_name in self._fa_kv:
            self._fa_kv[model_name].new_sequence(seq_id, owner=model_name)
        if model_name in self._kv_pools:
            self._kv_pools[model_name].new_sequence(seq_id)
        if model_name in self._tp_kv_pools:
            for pool in self._tp_kv_pools[model_name]:
                pool.new_sequence(seq_id)

    def _kv_free_sequence(self, model_name: str, seq_id: int):
        """Free a sequence from all KV pools for this model."""
        if model_name in self._fa_kv:
            self._fa_kv[model_name].free_sequence(seq_id)
        if model_name in self._kv_pools:
            self._kv_pools[model_name].free_sequence(seq_id)
        if model_name in self._tp_kv_pools:
            for pool in self._tp_kv_pools[model_name]:
                pool.free_sequence(seq_id)

    def _serve_model(self, model_name: str, gpu_id: int):
        """Serve one batch cycle for a model: prefill new requests + decode active ones."""
        try:
            executor = self._ensure_model_loaded(model_name, gpu_id)
        except Exception as e:
            print(f"[Scheduler] Failed to load {model_name}: {e}")
            # Fail all pending requests for this model
            for req in self.rm.pop_pending(model_name, max_count=999):
                req.error = str(e)
                req.state = RequestState.DONE
                req.done_time = time.time()
                self.rm.complete_request(req.id)
            return

        state = self.wm.get_state(model_name)
        device = f"cuda:{gpu_id}"

        # Pop new requests and prefill them
        new_requests = self.rm.pop_pending(model_name, max_count=self.scheduler_config.max_batch_tokens)
        for req in new_requests:
            try:
                input_ids = torch.tensor([req.prompt_tokens], device=device)
                self._kv_new_sequence(model_name, req.seq_id)
                logits = executor.prefill(input_ids, req.seq_id)
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                req.generated_tokens.append(next_token)
                req.first_token_time = time.time()
                req.state = RequestState.DECODING
            except torch.cuda.OutOfMemoryError:
                print(f"[Scheduler] OOM during prefill for request {req.id}, freeing KV cache")
                self._kv_free_sequence(model_name, req.seq_id)
                req.error = "GPU out of memory during prefill"
                req.state = RequestState.DONE
                req.done_time = time.time()
                self.rm.complete_request(req.id)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Scheduler] Prefill error for request {req.id}: {e}")
                self._kv_free_sequence(model_name, req.seq_id)
                req.error = str(e)
                req.state = RequestState.DONE
                req.done_time = time.time()
                self.rm.complete_request(req.id)

        # Check stop conditions and collect active decoding requests
        active = self.rm.get_active(model_name)
        eos = state.tokenizer.eos_token_id
        to_decode = []
        for req in active:
            if req.state != RequestState.DECODING:
                continue
            last_token = req.generated_tokens[-1]
            if last_token == eos or len(req.generated_tokens) >= req.max_new_tokens:
                self._kv_free_sequence(model_name, req.seq_id)
                self.rm.complete_request(req.id)
                self.stats.completed += 1
                continue
            to_decode.append(req)

        # Batched decode: all active sequences in one forward pass
        if to_decode:
            try:
                token_ids = [req.generated_tokens[-1] for req in to_decode]
                seq_ids = [req.seq_id for req in to_decode]

                if len(to_decode) == 1:
                    token_input = torch.tensor([[token_ids[0]]], device=device)
                    logits = executor.decode_step(token_input, seq_ids[0])
                    logits = logits[:, -1, :]  # [1, vocab]
                else:
                    logits = executor.batched_decode_step(token_ids, seq_ids)

                for i, req in enumerate(to_decode):
                    if req.temperature <= 0:
                        next_token = logits[i].argmax(dim=-1).item()
                    else:
                        probs = torch.softmax(logits[i] / req.temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    req.generated_tokens.append(next_token)
                    self.stats.tokens_generated += 1
            except torch.cuda.OutOfMemoryError:
                print(f"[Scheduler] OOM during decode, freeing all KV for {model_name}")
                for req in to_decode:
                    self._kv_free_sequence(model_name, req.seq_id)
                    req.error = "GPU out of memory during decode"
                    req.state = RequestState.DONE
                    req.done_time = time.time()
                    self.rm.complete_request(req.id)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Scheduler] Decode error: {e}")
                for req in to_decode:
                    self._kv_free_sequence(model_name, req.seq_id)
                    req.error = str(e)
                    req.state = RequestState.DONE
                    req.done_time = time.time()
                    self.rm.complete_request(req.id)

    def step(self) -> bool:
        """Run one scheduler step. Returns True if work was done."""
        model_name = self._pick_next_model()
        if model_name is None:
            return False

        gpu_id = self.engine_config.gpu_ids[0]  # Single GPU for now
        prefetch_started = False

        # Continuous batching: serve this model, accepting new requests mid-generation
        batches = 0
        while batches < self.scheduler_config.max_consecutive_batches:
            # Check if there's any work for this model (pending OR active)
            has_pending = self.rm.pending_count(model_name) > 0
            has_active = self.rm.active_count(model_name) > 0
            if not has_pending and not has_active:
                break

            # _serve_model: prefills any new pending requests, then decodes all active
            # This means new requests arriving between decode steps get picked up
            # on the next _serve_model call — true continuous batching
            self._serve_model(model_name, gpu_id)
            batches += 1
            self.stats.batches += 1

            # After first batch: start async prefetch for next model
            if not prefetch_started:
                self._start_prefetch_for_next(model_name, gpu_id)
                prefetch_started = True

            # Yield to other models if they have pending requests
            other_pending = [
                n for n in self.rm.models_with_pending()
                if n != model_name
            ]
            if other_pending and batches >= 2:
                # Yield after at least 2 batches (prefill + some decode)
                break

        self._current_model = model_name
        return True

    def run(self, timeout: float | None = None):
        """Run the scheduler loop until stopped or timeout."""
        self._running = True
        self.prefetch.start()
        start = time.time()

        try:
            while self._running:
                if timeout and (time.time() - start) > timeout:
                    break
                try:
                    did_work = self.step()
                    if not did_work:
                        time.sleep(0.001)
                except Exception as e:
                    print(f"[Scheduler] Step error (recovering): {e}")
                    time.sleep(0.01)  # Brief pause before retry
        finally:
            self.prefetch.stop()
            self._running = False

    def run_background(self, timeout: float | None = None):
        """Run the scheduler in a background thread."""
        self._thread = threading.Thread(target=self.run, kwargs={"timeout": timeout}, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def cleanup(self):
        """Free all GPU resources — call before shutdown or benchmarks."""
        self.stop()

        # Free CUDA graphs
        for executor in self._arch_executors.values():
            executor.invalidate_graph()
        self._arch_executors.clear()
        self._executors.clear()

        # Free all KV caches
        freed_caches = set()
        for name, cache in self._fa_kv.items():
            if id(cache) not in freed_caches:
                cache.free_all()
                # Free the actual tensors
                for c in cache.k_caches + cache.v_caches:
                    del c
                cache.k_caches.clear()
                cache.v_caches.clear()
                freed_caches.add(id(cache))
        self._fa_kv.clear()

        for name, pool in self._kv_pools.items():
            pool.free_all()
        self._kv_pools.clear()

        for name, pools in self._tp_kv_pools.items():
            for p in pools:
                p.free_all()
        self._tp_kv_pools.clear()

        # Evict all models from GPU
        for name in list(self.wm.models.keys()):
            if self.wm.is_loaded(name):
                self.wm.evict_from_gpu(name)

        import gc
        gc.collect()
        for gpu_id in self.engine_config.gpu_ids:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()


@dataclass
class SchedulerStats:
    batches: int = 0
    switches: int = 0
    evictions: int = 0
    requeued: int = 0
    sync_loads: int = 0
    prefetch_triggers: int = 0
    prefetch_hits: int = 0
    completed: int = 0
    tokens_generated: int = 0
