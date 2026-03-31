"""GPU and host memory pool management."""

import torch
from dataclasses import dataclass


class PinnedPool:
    """Pre-allocated pinned host memory pool.

    Pinned (page-locked) memory enables fast async DMA transfers to GPU.
    This pool is shared across all GPUs — host RAM is not device-specific.
    """

    def __init__(self, budget_gb: float):
        self.budget_bytes = int(budget_gb * 1e9)
        self.allocated_bytes = 0

    def track(self, nbytes: int):
        """Track an allocation against the budget."""
        self.allocated_bytes += nbytes
        if self.allocated_bytes > self.budget_bytes:
            raise RuntimeError(
                f"Pinned pool exceeded budget: {self.allocated_bytes / 1e9:.1f} GB "
                f"> {self.budget_bytes / 1e9:.1f} GB"
            )

    def untrack(self, nbytes: int):
        self.allocated_bytes -= nbytes

    @property
    def used_gb(self) -> float:
        return self.allocated_bytes / 1e9

    @property
    def free_gb(self) -> float:
        return (self.budget_bytes - self.allocated_bytes) / 1e9


class GPUPool:
    """Tracks memory usage on a single GPU.

    We don't pre-allocate a contiguous block — PyTorch's CUDA allocator handles
    the actual GPU memory. This class tracks our logical budget so we know when
    a GPU is "full" for weight residency decisions.
    """

    def __init__(self, gpu_id: int, weight_budget_gb: float, kv_budget_gb: float):
        self.gpu_id = gpu_id
        self.weight_budget_bytes = int(weight_budget_gb * 1e9)
        self.kv_budget_bytes = int(kv_budget_gb * 1e9)
        self.weight_allocated_bytes = 0
        self.kv_allocated_bytes = 0

        # Dedicated CUDA stream for async H2D copies (separate from compute)
        self.copy_stream = torch.cuda.Stream(device=f"cuda:{gpu_id}")

    def can_fit_weights(self, nbytes: int) -> bool:
        return (self.weight_allocated_bytes + nbytes) <= self.weight_budget_bytes

    def track_weights(self, nbytes: int):
        self.weight_allocated_bytes += nbytes

    def untrack_weights(self, nbytes: int):
        self.weight_allocated_bytes = max(0, self.weight_allocated_bytes - nbytes)

    def track_kv(self, nbytes: int):
        self.kv_allocated_bytes += nbytes

    def untrack_kv(self, nbytes: int):
        self.kv_allocated_bytes = max(0, self.kv_allocated_bytes - nbytes)

    @property
    def weight_used_gb(self) -> float:
        return self.weight_allocated_bytes / 1e9

    @property
    def weight_free_gb(self) -> float:
        return (self.weight_budget_bytes - self.weight_allocated_bytes) / 1e9

    @property
    def device(self) -> str:
        return f"cuda:{self.gpu_id}"


class MultiGPUPool:
    """Manages memory pools across multiple GPUs."""

    def __init__(self, gpu_ids: list[int], weight_budget_gb: float, kv_budget_gb: float):
        self.gpu_ids = gpu_ids
        self.pools = {
            gpu_id: GPUPool(gpu_id, weight_budget_gb, kv_budget_gb)
            for gpu_id in gpu_ids
        }

    def __getitem__(self, gpu_id: int) -> GPUPool:
        return self.pools[gpu_id]

    def async_copy_h2d(self, src: torch.Tensor, gpu_id: int) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Async copy from pinned host tensor to GPU. Returns (gpu_tensor, completion_event)."""
        pool = self.pools[gpu_id]
        with torch.cuda.stream(pool.copy_stream):
            dst = src.to(pool.device, non_blocking=True)
            event = pool.copy_stream.record_event()
        return dst, event

    def sync_gpu(self, gpu_id: int):
        """Wait for all pending copies on a GPU's copy stream."""
        self.pools[gpu_id].copy_stream.synchronize()

    def sync_all(self):
        """Wait for all pending copies on all GPUs."""
        for pool in self.pools.values():
            pool.copy_stream.synchronize()
