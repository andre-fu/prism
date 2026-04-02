"""Scheduler v2: uses static weight pool for zero-allocation model swaps.

Simplified architecture:
  - One StaticWeightPool per architecture (pre-allocated at startup)
  - One FlashAttnExecutorV3 per architecture (CUDA graph captured once)
  - Model swap = copy_() from pinned RAM into static GPU buffers
  - Async prefetch = copy on GPU copy stream while current model generates
"""

import time
import threading
import torch
from dataclasses import dataclass, field
from collections import deque

from .config import EngineConfig, SchedulerConfig
from .memory_pool import PinnedPool, MultiGPUPool
from .weight_manager import WeightManager
from .kv_cache import FlashAttnKVCache
from .weight_pool import StaticWeightPool
from .executor import FlashAttnExecutorV3
from .request_manager import RequestManager, Request, RequestState


def _arch_key(hf_config):
    return (hf_config.num_hidden_layers, hf_config.hidden_size,
            hf_config.num_attention_heads, hf_config.num_key_value_heads,
            hf_config.intermediate_size)


class Scheduler:
    """Production scheduler with static weight pools."""

    def __init__(self, engine_config: EngineConfig, scheduler_config: SchedulerConfig,
                 weight_manager: WeightManager, request_manager: RequestManager,
                 gpu_pool: MultiGPUPool):
        self.engine_config = engine_config
        self.scheduler_config = scheduler_config
        self.wm = weight_manager
        self.rm = request_manager
        self.gpu_pool = gpu_pool

        self.default_gpu = engine_config.gpu_ids[0]

        # Per-architecture: weight pool + executor + KV cache + GPU assignment
        self._pools: dict[tuple, StaticWeightPool] = {}
        self._executors: dict[tuple, FlashAttnExecutorV3] = {}
        self._kv_caches: dict[tuple, FlashAttnKVCache] = {}
        self._arch_gpu: dict[tuple, int] = {}

        # Current state — per-architecture tracking for spatial multiplexing
        self._current_model: str | None = None  # Legacy, for single-arch compat
        self._current_arch: tuple | None = None
        self._current_model_for_arch: dict[tuple, str] = {}

        # Async prefetch state
        self._prefetch_model: str | None = None
        self._prefetch_event: torch.cuda.Event | None = None

        # Prefix cache per architecture
        from .prefix_cache import PrefixCache
        self._PrefixCache = PrefixCache
        self._prefix_caches: dict[tuple, PrefixCache] = {}

        # Control
        self._running = False
        self._thread: threading.Thread | None = None
        self.stats = SchedulerStats()

        # Initialize pools for all architectures
        self._init_pools()

    def _init_pools(self):
        """Allocate weight pools across GPUs. Each architecture gets its own GPU."""
        seen_archs = {}
        gpu_idx = 0
        for mc in self.engine_config.models:
            state = self.wm.get_state(mc.name)
            if state is None:
                continue
            key = _arch_key(state.hf_config)
            if key in seen_archs:
                continue
            seen_archs[key] = state.hf_config

            # Assign a GPU to this architecture (round-robin across available GPUs)
            gpu_id = self.engine_config.gpu_ids[gpu_idx % len(self.engine_config.gpu_ids)]
            device = f"cuda:{gpu_id}"
            gpu_idx += 1

            # Weight pool on this GPU
            pool = StaticWeightPool(state.hf_config, device, torch.bfloat16)
            self._pools[key] = pool

            # KV cache on same GPU
            nkv = state.hf_config.num_key_value_heads
            hd = state.hf_config.hidden_size // state.hf_config.num_attention_heads
            nl = state.hf_config.num_hidden_layers
            page_size = 256
            free_gb = torch.cuda.mem_get_info(gpu_id)[0] / 1e9
            kv_budget = min(self.engine_config.kv_cache_budget_gb, free_gb * 0.7)
            bytes_per_page = 2 * page_size * nkv * hd * 2 * nl
            max_pages = max(int(kv_budget * 1e9 / bytes_per_page), 16)
            kv = FlashAttnKVCache(nl, nkv, hd, max_pages, page_size, device, torch.bfloat16)
            self._kv_caches[key] = kv

            # Executor on same GPU
            executor = FlashAttnExecutorV3(pool, kv, device)
            self._executors[key] = executor

            # Prefix cache for this architecture
            self._prefix_caches[key] = self._PrefixCache(kv, page_size=page_size)

            # Track which GPU each arch is on
            self._arch_gpu[key] = gpu_id

            print(f"[Scheduler] Arch {key[:2]} on GPU {gpu_id}: {pool.total_gb:.1f}GB weights, "
                  f"{max_pages} KV pages ({max_pages * page_size} tokens), "
                  f"{free_gb - pool.total_gb:.1f}GB free")

    def _get_arch(self, model_name: str) -> tuple:
        state = self.wm.get_state(model_name)
        return _arch_key(state.hf_config)

    def _swap_to_model(self, model_name: str):
        """Swap weights to a model. If async prefetch completed, this is instant."""
        if self._current_model == model_name:
            return

        arch = self._get_arch(model_name)
        pool = self._pools[arch]
        state = self.wm.get_state(model_name)

        # Check if async prefetch already loaded this model
        if self._prefetch_model == model_name and self._prefetch_event is not None:
            self._prefetch_event.synchronize()
            self._prefetch_model = None
            self._prefetch_event = None
            self.stats.prefetch_hits += 1
        else:
            # Sync swap: copy from pinned RAM to GPU buffers
            t0 = time.perf_counter()
            pool.load_from_pinned(state.pinned_weights, state.hf_config)
            torch.cuda.synchronize()
            t_swap = time.perf_counter() - t0
            self.stats.sync_swaps += 1
            self.stats.swap_time_ms += t_swap * 1000

        # If switching architectures, invalidate graph
        if arch != self._current_arch:
            self._executors[arch].invalidate_graph()
            self._current_arch = arch

        self._current_model = model_name

    def _start_async_prefetch(self, model_name: str):
        """Start async copy of model weights on GPU copy stream."""
        if model_name == self._current_model:
            return
        if self._prefetch_model == model_name:
            return

        arch = self._get_arch(model_name)
        pool = self._pools[arch]
        state = self.wm.get_state(model_name)
        gpu_id = self._arch_gpu[arch]
        copy_stream = self.gpu_pool[gpu_id].copy_stream

        with torch.cuda.stream(copy_stream):
            pool.load_from_pinned(state.pinned_weights, state.hf_config)
            event = copy_stream.record_event()

        self._prefetch_model = model_name
        self._prefetch_event = event
        self.stats.prefetch_triggers += 1

    def _pick_next_model(self) -> str | None:
        candidates = set()
        candidates.update(self.rm.models_with_pending())
        candidates.update(self.rm.models_with_active())
        if not candidates:
            return None
        ordered = sorted(candidates)
        if self._current_model is None or self._current_model not in ordered:
            return ordered[0]
        idx = ordered.index(self._current_model)
        return ordered[(idx + 1) % len(ordered)]

    def _serve_model(self, model_name: str):
        """Serve one batch: prefill new requests + decode active ones."""
        arch = self._get_arch(model_name)
        executor = self._executors[arch]
        kv = self._kv_caches[arch]
        state = self.wm.get_state(model_name)
        device = executor.device

        # Prefill new requests (with prefix cache check)
        prefix_cache = self._prefix_caches.get(arch)
        new_reqs = self.rm.pop_pending(model_name, max_count=8)
        for req in new_reqs:
            try:
                input_ids = torch.tensor([req.prompt_tokens], device=device)

                # Check prefix cache
                prefix_pages, prefix_len = [], 0
                if prefix_cache:
                    prefix_pages, prefix_len = prefix_cache.lookup(req.prompt_tokens, model_name)
                    if prefix_pages:
                        prefix_cache.ref_pages(prefix_pages)

                kv.new_sequence(req.seq_id, owner=model_name,
                               prefix_pages=prefix_pages if prefix_pages else None,
                               prefix_len=prefix_len)
                logits = executor.prefill(input_ids, req.seq_id, prefix_len=prefix_len)

                # Store prefix for future requests
                if prefix_cache and not prefix_pages:
                    prefix_cache.store(req.prompt_tokens, kv.seq_pages.get(req.seq_id, []), model_name)
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                req.generated_tokens.append(next_token)
                req.first_token_time = time.time()
                req.state = RequestState.DECODING
            except torch.cuda.OutOfMemoryError:
                kv.free_sequence(req.seq_id)
                req.error = "OOM during prefill"
                req.state = RequestState.DONE
                req.done_time = time.time()
                self.rm.complete_request(req.id)
                torch.cuda.empty_cache()
            except Exception as e:
                kv.free_sequence(req.seq_id)
                req.error = str(e)
                req.state = RequestState.DONE
                req.done_time = time.time()
                self.rm.complete_request(req.id)

        # Collect active decode requests for THIS model
        active = self.rm.get_active(model_name)
        eos = state.tokenizer.eos_token_id
        to_decode = []
        for req in active:
            if req.state != RequestState.DECODING:
                continue
            last_token = req.generated_tokens[-1]
            if last_token == eos or len(req.generated_tokens) >= req.max_new_tokens:
                kv.free_sequence(req.seq_id)
                self.rm.complete_request(req.id)
                self.stats.completed += 1
                continue
            to_decode.append(req)

        if not to_decode:
            return

        try:
            token_ids = [r.generated_tokens[-1] for r in to_decode]
            seq_ids = [r.seq_id for r in to_decode]

            if len(to_decode) == 1:
                token_input = torch.tensor([[token_ids[0]]], device=device)
                logits = executor.decode_step(token_input, seq_ids[0])
                logits = logits[:, -1, :]
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
        except Exception as e:
            print(f"[Scheduler] Decode error: {e}")
            for req in to_decode:
                kv.free_sequence(req.seq_id)
                req.error = str(e)
                req.state = RequestState.DONE
                req.done_time = time.time()
                self.rm.complete_request(req.id)

    def _serve_gpu(self, arch_key: tuple):
        """Serve one batch cycle for all models on a specific GPU/architecture."""
        # Find the model with most urgent work for this architecture
        # Priority: highest priority first, then earliest deadline, then most work
        import time as _time
        now = _time.time()
        candidates = []
        for mc in self.engine_config.models:
            state = self.wm.get_state(mc.name)
            if state and _arch_key(state.hf_config) == arch_key:
                pending = self.rm.pending_count(mc.name)
                active = self.rm.active_count(mc.name)
                if pending > 0 or active > 0:
                    # Find highest priority among pending requests
                    max_priority = 0
                    earliest_deadline = float('inf')
                    with self.rm._lock:
                        for req in self.rm._queues.get(mc.name, []):
                            max_priority = max(max_priority, req.priority)
                            if req.slo_ttft_ms:
                                deadline = req.arrival_time + req.slo_ttft_ms / 1000
                                earliest_deadline = min(earliest_deadline, deadline)
                    candidates.append((max_priority, -earliest_deadline, pending + active, mc.name))

        if not candidates:
            return False

        # Sort: highest priority, earliest deadline, most work
        candidates.sort(reverse=True)
        model_name = candidates[0][3]

        # Swap weights if needed (same arch = just copy_(), different model)
        current = self._current_model_for_arch.get(arch_key)
        if current != model_name:
            self._swap_to_model(model_name)
            self._current_model_for_arch[arch_key] = model_name

        # Serve batches
        batches = 0
        while batches < self.scheduler_config.max_consecutive_batches:
            has_work = (self.rm.pending_count(model_name) > 0 or
                        self.rm.active_count(model_name) > 0)
            if not has_work:
                break
            self._serve_model(model_name)
            batches += 1
            self.stats.batches += 1

            # Yield if other models on same arch need attention
            other_same_arch = [
                mc.name for mc in self.engine_config.models
                if mc.name != model_name
                and self.wm.get_state(mc.name)
                and _arch_key(self.wm.get_state(mc.name).hf_config) == arch_key
                and self.rm.pending_count(mc.name) > 0
            ]
            if other_same_arch and batches >= 2:
                break

        return True

    def step(self) -> bool:
        """Serve all GPUs that have pending work — concurrent across architectures."""
        # Collect architectures with pending work
        active_archs = set()
        for mc in self.engine_config.models:
            state = self.wm.get_state(mc.name)
            if state is None:
                continue
            key = _arch_key(state.hf_config)
            if key in self._pools:
                if self.rm.pending_count(mc.name) > 0 or self.rm.active_count(mc.name) > 0:
                    active_archs.add(key)

        if not active_archs:
            return False

        if len(active_archs) == 1:
            # Single architecture — no threading overhead
            return self._serve_gpu(next(iter(active_archs)))

        # Multiple architectures on different GPUs — dispatch concurrently
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(active_archs)) as pool:
            futures = {pool.submit(self._serve_gpu, key): key for key in active_archs}
            did_work = False
            for future in as_completed(futures):
                try:
                    if future.result():
                        did_work = True
                except Exception as e:
                    print(f"[Scheduler] GPU dispatch error: {e}")
            return did_work

    def run(self, timeout: float | None = None):
        self._running = True
        start = time.time()
        try:
            while self._running:
                if timeout and (time.time() - start) > timeout:
                    break
                try:
                    if not self.step():
                        time.sleep(0.001)
                except Exception as e:
                    print(f"[Scheduler] Step error: {e}")
                    time.sleep(0.01)
        finally:
            self._running = False

    def run_background(self, timeout: float | None = None):
        self._thread = threading.Thread(target=self.run, kwargs={"timeout": timeout}, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def cleanup(self):
        self.stop()
        for executor in self._executors.values():
            executor.invalidate_graph()
        for kv in self._kv_caches.values():
            kv.free_all()
            for c in kv.k_caches + kv.v_caches:
                del c
            kv.k_caches.clear()
            kv.v_caches.clear()
        for pool in self._pools.values():
            pool.free()
        self._pools.clear()
        self._executors.clear()
        self._kv_caches.clear()

        import gc
        gc.collect()
        for gid in self.engine_config.gpu_ids:
            with torch.cuda.device(gid):
                torch.cuda.empty_cache()


@dataclass
class SchedulerStats:
    batches: int = 0
    completed: int = 0
    tokens_generated: int = 0
    sync_swaps: int = 0
    swap_time_ms: float = 0.0
    prefetch_triggers: int = 0
    prefetch_hits: int = 0
