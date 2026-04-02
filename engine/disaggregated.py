"""Disaggregated prefill/decode: separate GPUs for each phase.

Prefill GPUs: batch multiple prompts, compute-bound, high utilization
Decode GPUs: continuous batching with CUDA graphs, memory-bandwidth-bound

After prefill completes on a prefill GPU, KV cache is transferred D2D
(via NVLink ~300GB/s) to a decode GPU. The decode GPU continues generating.

Example layout on 4×H100:
  GPU 0-1: prefill (can batch 8+ prompts concurrently)
  GPU 2-3: decode (continuous batching with CUDA graphs)
"""

import time
import threading
import torch

from .config import EngineConfig, SchedulerConfig
from .memory_pool import MultiGPUPool
from .weight_manager import WeightManager
from .weight_pool import StaticWeightPool
from .kv_cache import FlashAttnKVCache
from .executor import FlashAttnExecutorV3
from .request_manager import RequestManager, Request, RequestState
from .fused_kernels import fused_rms_norm


class DisaggregatedScheduler:
    """Scheduler with separate prefill and decode GPU pools."""

    def __init__(
        self,
        engine_config: EngineConfig,
        scheduler_config: SchedulerConfig,
        weight_manager: WeightManager,
        request_manager: RequestManager,
        gpu_pool: MultiGPUPool,
        prefill_gpus: list[int],
        decode_gpus: list[int],
    ):
        self.engine_config = engine_config
        self.scheduler_config = scheduler_config
        self.wm = weight_manager
        self.rm = request_manager
        self.gpu_pool = gpu_pool
        self.prefill_gpus = prefill_gpus
        self.decode_gpus = decode_gpus

        # We need weight pools + executors on BOTH prefill and decode GPUs
        # (same architecture, different devices)
        self._prefill_pools: dict[int, StaticWeightPool] = {}
        self._prefill_kv: dict[int, FlashAttnKVCache] = {}
        self._prefill_executors: dict[int, FlashAttnExecutorV3] = {}

        self._decode_pools: dict[int, StaticWeightPool] = {}
        self._decode_kv: dict[int, FlashAttnKVCache] = {}
        self._decode_executors: dict[int, FlashAttnExecutorV3] = {}

        self._current_model: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        from dataclasses import dataclass, field

        @dataclass
        class Stats:
            prefills: int = 0
            decode_steps: int = 0
            kv_transfers: int = 0
            kv_transfer_ms: float = 0.0
            tokens_generated: int = 0
            completed: int = 0

        self.stats = Stats()

        self._init_pools()

    def _init_pools(self):
        """Allocate weight pools on prefill and decode GPUs for the first model's architecture."""
        # Use first model's architecture
        mc = self.engine_config.models[0]
        state = self.wm.get_state(mc.name)
        hf_config = state.hf_config
        nkv = hf_config.num_key_value_heads
        hd = hf_config.hidden_size // hf_config.num_attention_heads
        nl = hf_config.num_hidden_layers
        page_size = 256

        for gpu_id in self.prefill_gpus:
            device = f"cuda:{gpu_id}"
            pool = StaticWeightPool(hf_config, device, torch.bfloat16)
            pool.load_from_pinned(state.pinned_weights, hf_config)

            free_gb = torch.cuda.mem_get_info(gpu_id)[0] / 1e9
            kv_budget = min(self.engine_config.kv_cache_budget_gb, free_gb * 0.7)
            bytes_per_page = 2 * page_size * nkv * hd * 2 * nl
            max_pages = max(int(kv_budget * 1e9 / bytes_per_page), 16)

            kv = FlashAttnKVCache(nl, nkv, hd, max_pages, page_size, device, torch.bfloat16)
            executor = FlashAttnExecutorV3(pool, kv, device, use_cuda_graph=False)  # No graph for prefill

            self._prefill_pools[gpu_id] = pool
            self._prefill_kv[gpu_id] = kv
            self._prefill_executors[gpu_id] = executor
            print(f"[Disagg] Prefill GPU {gpu_id}: {pool.total_gb:.1f}GB, {max_pages} KV pages")

        for gpu_id in self.decode_gpus:
            device = f"cuda:{gpu_id}"
            pool = StaticWeightPool(hf_config, device, torch.bfloat16)
            pool.load_from_pinned(state.pinned_weights, hf_config)

            free_gb = torch.cuda.mem_get_info(gpu_id)[0] / 1e9
            kv_budget = min(self.engine_config.kv_cache_budget_gb, free_gb * 0.7)
            bytes_per_page = 2 * page_size * nkv * hd * 2 * nl
            max_pages = max(int(kv_budget * 1e9 / bytes_per_page), 16)

            kv = FlashAttnKVCache(nl, nkv, hd, max_pages, page_size, device, torch.bfloat16)
            executor = FlashAttnExecutorV3(pool, kv, device, use_cuda_graph=True)  # Graph for decode

            self._decode_pools[gpu_id] = pool
            self._decode_kv[gpu_id] = kv
            self._decode_executors[gpu_id] = executor
            print(f"[Disagg] Decode GPU {gpu_id}: {pool.total_gb:.1f}GB, {max_pages} KV pages")

    def _prefill_request(self, req: Request, prefill_gpu: int, decode_gpu: int):
        """Prefill a request on prefill_gpu, transfer KV to decode_gpu."""
        device = f"cuda:{prefill_gpu}"
        executor = self._prefill_executors[prefill_gpu]
        prefill_kv = self._prefill_kv[prefill_gpu]
        decode_kv = self._decode_kv[decode_gpu]
        state = self.wm.get_state(req.model_name)

        # Prefill on prefill GPU
        input_ids = torch.tensor([req.prompt_tokens], device=device)
        prefill_kv.new_sequence(req.seq_id)
        logits = executor.prefill(input_ids, req.seq_id)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        req.generated_tokens.append(next_token)
        req.first_token_time = time.time()
        self.stats.prefills += 1

        # Transfer KV from prefill GPU to decode GPU
        t_xfer = time.perf_counter()
        pages = prefill_kv.transfer_sequence(req.seq_id, decode_kv, req.seq_id)
        torch.cuda.synchronize()
        t_xfer = time.perf_counter() - t_xfer
        self.stats.kv_transfers += 1
        self.stats.kv_transfer_ms += t_xfer * 1000

        # Free prefill KV (no longer needed)
        prefill_kv.free_sequence(req.seq_id)

        req.state = RequestState.DECODING

    def _decode_batch(self, requests: list[Request], decode_gpu: int):
        """Run one decode step for a batch of requests on decode_gpu."""
        executor = self._decode_executors[decode_gpu]
        kv = self._decode_kv[decode_gpu]
        state = self.wm.get_state(requests[0].model_name)
        device = f"cuda:{decode_gpu}"
        eos = state.tokenizer.eos_token_id

        # Check completions
        to_decode = []
        for req in requests:
            last = req.generated_tokens[-1]
            if last == eos or len(req.generated_tokens) >= req.max_new_tokens:
                kv.free_sequence(req.seq_id)
                self.rm.complete_request(req.id)
                self.stats.completed += 1
                continue
            to_decode.append(req)

        if not to_decode:
            return

        token_ids = [r.generated_tokens[-1] for r in to_decode]
        seq_ids = [r.seq_id for r in to_decode]

        if len(to_decode) == 1:
            logits = executor.decode_step(
                torch.tensor([[token_ids[0]]], device=device), seq_ids[0])
            logits = logits[:, -1, :]
        else:
            logits = executor.batched_decode_step(token_ids, seq_ids)

        for i, req in enumerate(to_decode):
            next_token = logits[i].argmax(dim=-1).item()
            req.generated_tokens.append(next_token)
            self.stats.tokens_generated += 1

    def step(self) -> bool:
        """One scheduler step: prefill pending requests, then decode active ones."""
        did_work = False

        # Phase 1: Prefill any pending requests on prefill GPUs
        prefill_gpu_idx = 0
        decode_gpu_idx = 0
        pending_models = self.rm.models_with_pending()
        for model_name in pending_models:
            new_reqs = self.rm.pop_pending(model_name, max_count=4)
            for req in new_reqs:
                pgpu = self.prefill_gpus[prefill_gpu_idx % len(self.prefill_gpus)]
                dgpu = self.decode_gpus[decode_gpu_idx % len(self.decode_gpus)]
                try:
                    self._prefill_request(req, pgpu, dgpu)
                    did_work = True
                except Exception as e:
                    req.error = str(e)
                    req.state = RequestState.DONE
                    req.done_time = time.time()
                    self.rm.complete_request(req.id)
                prefill_gpu_idx += 1

        # Phase 2: Decode active requests on decode GPUs
        active_models = self.rm.models_with_active()
        for model_name in active_models:
            active = self.rm.get_active(model_name)
            if active:
                dgpu = self.decode_gpus[0]  # Route to first decode GPU for now
                try:
                    self._decode_batch(active, dgpu)
                    did_work = True
                except Exception as e:
                    print(f"[Disagg] Decode error: {e}")
                    for req in active:
                        req.error = str(e)
                        req.state = RequestState.DONE
                        req.done_time = time.time()
                        self.rm.complete_request(req.id)

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
                    print(f"[Disagg] Step error: {e}")
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
        for ex in list(self._prefill_executors.values()) + list(self._decode_executors.values()):
            ex.invalidate_graph()
        for kv in list(self._prefill_kv.values()) + list(self._decode_kv.values()):
            kv.free_all()
        for pool in list(self._prefill_pools.values()) + list(self._decode_pools.values()):
            pool.free()
        import gc
        gc.collect()
        for g in self.prefill_gpus + self.decode_gpus:
            with torch.cuda.device(g):
                torch.cuda.empty_cache()
