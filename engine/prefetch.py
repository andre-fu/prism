"""Async weight prefetch: load next model's weights while current model computes.

The flow:
1. Scheduler serves model A, signals prefetch: "I'll need model B next"
2. Scheduler evicts a victim model if needed (frees GPU budget)
3. Prefetch controller starts async H2D transfer for model B on the COPY stream
4. Scheduler continues serving model A on the COMPUTE stream (overlapped!)
5. When scheduler is ready to switch, it checks is_ready(B)
6. If ready: switch instantly (0ms). If not: wait for transfer to finish.

The copy stream and compute stream run concurrently on the GPU — this is how
we hide the 0.3-0.7s weight transfer latency behind active inference.
"""

import threading
import time
import torch
from pathlib import Path

from .weight_manager import WeightManager, ModelState
from .request_manager import RequestManager
from .memory_pool import MultiGPUPool


class PrefetchController:
    """Manages async weight prefetch on GPU copy streams."""

    def __init__(
        self,
        weight_manager: WeightManager,
        request_manager: RequestManager,
        gpu_pool: MultiGPUPool,
        queue_threshold: int = 1,
    ):
        self.wm = weight_manager
        self.rm = request_manager
        self.gpu_pool = gpu_pool
        self.queue_threshold = queue_threshold

        self._lock = threading.Lock()
        # model_name -> Event that fires when transfer is done
        self._events: dict[str, torch.cuda.Event] = {}
        # model_name -> True when transfer confirmed complete
        self._ready: dict[str, bool] = {}
        # model_name -> gpu_id where it's being prefetched
        self._target_gpu: dict[str, int] = {}

        # Stats
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def start(self):
        """No background thread needed — prefetch is triggered by scheduler."""
        pass

    def stop(self):
        pass

    def suggest_prefetch(self, exclude_model: str, gpu_id: int) -> str | None:
        """Which cold model should we prefetch next?"""
        candidates = []
        for name in self.rm.models_with_pending():
            if name == exclude_model:
                continue
            if self.wm.is_loaded(name):
                continue
            with self._lock:
                if name in self._events or name in self._ready:
                    continue
            pending = self.rm.pending_count(name)
            if pending >= self.queue_threshold:
                candidates.append((pending, name))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def start_prefetch(self, model_name: str, gpu_id: int):
        """Start async weight transfer for a model on the GPU's copy stream.

        IMPORTANT: Caller must ensure there's enough GPU budget BEFORE calling this.
        The scheduler handles eviction; prefetch just does the transfer.
        """
        with self._lock:
            if model_name in self._events or model_name in self._ready:
                return  # Already in flight or done
            self._target_gpu[model_name] = gpu_id

        state = self.wm.get_state(model_name)
        if state is None:
            return

        pool = self.gpu_pool[gpu_id]
        device = f"cuda:{gpu_id}"
        copy_stream = pool.copy_stream

        # Async transfer on the copy stream (doesn't block compute stream)
        with torch.cuda.stream(copy_stream):
            gpu_weights = {}
            for param_name, tensor in state.pinned_weights.items():
                gpu_weights[param_name] = tensor.to(device, non_blocking=True)
            event = copy_stream.record_event()

        with self._lock:
            self._events[model_name] = event
            # Store the GPU weights so the scheduler can use them when ready
            state._prefetch_gpu_weights = gpu_weights
            state._prefetch_gpu_id = gpu_id

    def is_ready(self, model_name: str) -> bool:
        """Check if prefetch transfer has completed (non-blocking)."""
        with self._lock:
            if model_name in self._ready:
                return True
            event = self._events.get(model_name)
            if event is None:
                return False
            if event.query():  # Non-blocking check
                self._ready[model_name] = True
                del self._events[model_name]
                return True
            return False

    def is_in_flight(self, model_name: str) -> bool:
        """Is a prefetch currently transferring?"""
        with self._lock:
            return model_name in self._events

    def wait_ready(self, model_name: str, timeout_s: float = 10.0) -> bool:
        """Block until prefetch completes. Returns True if completed."""
        with self._lock:
            event = self._events.get(model_name)
        if event is None:
            with self._lock:
                return model_name in self._ready
        event.synchronize()
        with self._lock:
            self._ready[model_name] = True
            self._events.pop(model_name, None)
        return True

    def complete_prefetch(self, model_name: str) -> bool:
        """Finalize a prefetched model: build the nn.Module from transferred weights.

        Called by the scheduler when it's ready to activate the prefetched model.
        Returns True if successful.
        """
        state = self.wm.get_state(model_name)
        if state is None:
            return False

        gpu_weights = getattr(state, '_prefetch_gpu_weights', None)
        gpu_id = getattr(state, '_prefetch_gpu_id', None)
        if gpu_weights is None or gpu_id is None:
            return False

        device = f"cuda:{gpu_id}"

        # Build model from prefetched weights (this is fast: ~0.1s)
        from transformers import AutoModelForCausalLM
        dtype = getattr(torch, state.config.dtype)

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(state.hf_config, dtype=dtype)

        model.load_state_dict(gpu_weights, strict=False, assign=True)

        # Reinit rotary buffers
        for buf_name, buf in list(model.named_buffers()):
            if buf.device == torch.device("meta") and "inv_freq" in buf_name:
                dim = buf.shape[0] * 2
                rope_theta = getattr(state.hf_config, "rope_theta", 10000.0)
                inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                parent = model
                for part in buf_name.rsplit(".", 1)[0].split("."):
                    parent = getattr(parent, part)
                parent.inv_freq = inv_freq.to(device)

        model.to(device)
        model.eval()

        # Update weight manager state
        pool = self.gpu_pool[gpu_id]
        pool.track_weights(state.gpu_bytes_per_device)
        state.gpu_weights[gpu_id] = gpu_weights
        state.active_model = model
        state.active_gpu_id = gpu_id
        state.last_served = time.time()

        # Cleanup prefetch state
        state._prefetch_gpu_weights = None
        state._prefetch_gpu_id = None
        with self._lock:
            self._ready.pop(model_name, None)
            self._target_gpu.pop(model_name, None)

        self.prefetch_hits += 1
        return True

    def cancel(self, model_name: str):
        """Cancel a prefetch (e.g., if the model's queue drained)."""
        with self._lock:
            self._events.pop(model_name, None)
            self._ready.pop(model_name, None)
            self._target_gpu.pop(model_name, None)
        state = self.wm.get_state(model_name)
        if state:
            gpu_weights = getattr(state, '_prefetch_gpu_weights', None)
            if gpu_weights:
                del gpu_weights
                state._prefetch_gpu_weights = None
                state._prefetch_gpu_id = None
                torch.cuda.empty_cache()
