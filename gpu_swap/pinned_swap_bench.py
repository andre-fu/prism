"""
Head-to-head: vLLM sync sleep/wake vs pinned async swap vs page table remap.

Tests three approaches on real model weights:
1. VLLM_SYNC: vLLM's built-in sleep/wake (sync cudaMemcpy + cuMemRelease/cuMemCreate)
2. PINNED_ASYNC: Pre-pinned host buffers + cudaMemcpyAsync (no reallocation)
3. PAGE_TABLE_REMAP: Both models resident in HBM, swap via cuMemMap only (no data copy)
"""

import gc
import os
import time

os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import torch.cuda

# Patch for PyTorch 2.10 compat
try:
    import torch._inductor.standalone_compile as _sc
    from torch._subclasses.fake_tensor import FakeTensorMode as _FTM
    if not hasattr(_sc, "FakeTensorMode"):
        _sc.FakeTensorMode = _FTM
except Exception:
    pass


def gpu_mem_mb(gpu_id=0):
    torch.cuda.synchronize(gpu_id)
    return torch.cuda.memory_allocated(gpu_id) / (1024**2)


def gpu_mem_reserved_mb(gpu_id=0):
    torch.cuda.synchronize(gpu_id)
    return torch.cuda.memory_reserved(gpu_id) / (1024**2)


# ─── Approach 1: Simulate vLLM sync (allocate pinned each time, sync copy) ──
def bench_vllm_sync(gpu_tensors: dict[str, torch.Tensor], gpu_id: int, n_iters: int = 5):
    """Simulate vLLM's sleep/wake: fresh pinned alloc + sync memcpy each time."""
    results = []

    for i in range(n_iters):
        # --- SLEEP: GPU -> fresh pinned host (sync) ---
        t0 = time.perf_counter()
        host_backups = {}
        with torch.cuda.device(gpu_id):
            for name, tensor in gpu_tensors.items():
                nbytes = tensor.nelement() * tensor.element_size()
                # vLLM allocates a fresh pinned buffer each sleep
                cpu_buf = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                # Synchronous copy
                cpu_buf.copy_(tensor.view(torch.uint8).contiguous())
                torch.cuda.synchronize(gpu_id)
                host_backups[name] = (cpu_buf, tensor.shape, tensor.dtype)
                # Simulate cuMemUnmap+cuMemRelease (just zero the tensor)
                tensor.zero_()
        t_sleep = time.perf_counter() - t0

        # --- WAKE: fresh GPU alloc + sync copy from pinned host ---
        t0 = time.perf_counter()
        with torch.cuda.device(gpu_id):
            for name, (cpu_buf, shape, dtype) in host_backups.items():
                # Simulate cuMemCreate+cuMemMap (just use existing tensor storage)
                gpu_tensors[name].copy_(cpu_buf.view(dtype).reshape(shape))
                torch.cuda.synchronize(gpu_id)
        t_wake = time.perf_counter() - t0

        results.append({"sleep": t_sleep, "wake": t_wake, "total": t_sleep + t_wake})
        # Free host backups
        del host_backups

    return results


# ─── Approach 2: Pre-pinned async (our fast path) ───────────────────────────
def bench_pinned_async(gpu_tensors: dict[str, torch.Tensor], gpu_id: int, n_iters: int = 5):
    """Pre-allocated pinned buffers + async memcpy."""
    results = []

    # Pre-allocate pinned host buffers ONCE (amortized cost)
    t_prealloc = time.perf_counter()
    host_buffers = {}
    total_bytes = 0
    for name, tensor in gpu_tensors.items():
        nbytes = tensor.nelement() * tensor.element_size()
        total_bytes += nbytes
        host_buffers[name] = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    t_prealloc = time.perf_counter() - t_prealloc
    print(f"    Pre-allocated {total_bytes/(1024**3):.2f} GB pinned buffers in {t_prealloc:.3f}s (one-time cost)")

    stream = torch.cuda.Stream(device=gpu_id)

    for i in range(n_iters):
        # --- SLEEP: GPU -> pre-pinned host (async) ---
        t0 = time.perf_counter()
        with torch.cuda.device(gpu_id), torch.cuda.stream(stream):
            for name, tensor in gpu_tensors.items():
                host_buffers[name].copy_(tensor.view(torch.uint8).contiguous(), non_blocking=True)
        stream.synchronize()
        t_sleep = time.perf_counter() - t0

        # Zero GPU tensors to simulate free
        for tensor in gpu_tensors.values():
            tensor.zero_()
        torch.cuda.synchronize(gpu_id)

        # --- WAKE: pre-pinned host -> GPU (async) ---
        t0 = time.perf_counter()
        with torch.cuda.device(gpu_id), torch.cuda.stream(stream):
            for name, tensor in gpu_tensors.items():
                nbytes = tensor.nelement() * tensor.element_size()
                tensor.view(torch.uint8).copy_(host_buffers[name][:nbytes], non_blocking=True)
        stream.synchronize()
        t_wake = time.perf_counter() - t0

        results.append({"sleep": t_sleep, "wake": t_wake, "total": t_sleep + t_wake})

    del host_buffers
    return results


# ─── Approach 3: Pre-pinned async with single contiguous buffer ──────────────
def bench_pinned_contiguous(gpu_tensors: dict[str, torch.Tensor], gpu_id: int, n_iters: int = 5):
    """Single contiguous pinned buffer + single async memcpy. Maximum bandwidth."""
    results = []

    # Pack all weights into a single contiguous GPU tensor
    total_bytes = sum(t.nelement() * t.element_size() for t in gpu_tensors.values())

    with torch.cuda.device(gpu_id):
        gpu_packed = torch.empty(total_bytes, dtype=torch.uint8, device=f"cuda:{gpu_id}")

    # Copy weights into packed buffer
    offset = 0
    param_layout = []  # (name, offset, nbytes, shape, dtype)
    with torch.cuda.device(gpu_id):
        for name, tensor in gpu_tensors.items():
            flat = tensor.contiguous().view(torch.uint8)
            nbytes = flat.numel()
            gpu_packed[offset:offset+nbytes].copy_(flat)
            param_layout.append((name, offset, nbytes, tensor.shape, tensor.dtype))
            offset += nbytes
    torch.cuda.synchronize(gpu_id)

    # Single contiguous pinned host buffer
    host_packed = torch.empty(total_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    stream = torch.cuda.Stream(device=gpu_id)

    print(f"    Contiguous buffer: {total_bytes/(1024**3):.2f} GB (single memcpy)")

    for i in range(n_iters):
        # --- SLEEP: single GPU->host async copy ---
        t0 = time.perf_counter()
        with torch.cuda.device(gpu_id), torch.cuda.stream(stream):
            host_packed.copy_(gpu_packed, non_blocking=True)
        stream.synchronize()
        t_sleep = time.perf_counter() - t0

        gpu_packed.zero_()
        torch.cuda.synchronize(gpu_id)

        # --- WAKE: single host->GPU async copy ---
        t0 = time.perf_counter()
        with torch.cuda.device(gpu_id), torch.cuda.stream(stream):
            gpu_packed.copy_(host_packed, non_blocking=True)
        stream.synchronize()
        t_wake = time.perf_counter() - t0

        results.append({"sleep": t_sleep, "wake": t_wake, "total": t_sleep + t_wake})

    del gpu_packed, host_packed
    return results


# ─── Approach 4: Dual-resident page table remap (simulated) ─────────────────
def bench_page_table_remap(size_bytes: int, gpu_id: int, n_iters: int = 5):
    """
    Simulate page table remap: both models in HBM, swap = pointer reassignment.
    No data movement at all — just measure the overhead of switching which
    tensor set the model uses.
    """
    results = []

    with torch.cuda.device(gpu_id):
        # Both "models" resident in HBM simultaneously
        model_a_weights = torch.randn(size_bytes // 2, dtype=torch.float16, device=f"cuda:{gpu_id}")
        model_b_weights = torch.randn(size_bytes // 2, dtype=torch.float16, device=f"cuda:{gpu_id}")
        active_ptr = model_a_weights

    print(f"    Both models in HBM: {size_bytes/(1024**3):.2f} GB x 2")

    for i in range(n_iters):
        torch.cuda.synchronize(gpu_id)
        t0 = time.perf_counter()
        # "Swap" = reassign pointer. Zero data movement.
        active_ptr = model_b_weights if active_ptr is model_a_weights else model_a_weights
        torch.cuda.synchronize(gpu_id)
        t_swap = time.perf_counter() - t0
        results.append({"sleep": 0, "wake": 0, "total": t_swap, "swap": t_swap})

    del model_a_weights, model_b_weights
    return results


def print_results(name, results):
    sleeps = [r["sleep"] for r in results]
    wakes = [r["wake"] for r in results]
    totals = [r["total"] for r in results]

    # Skip first iteration (warmup)
    if len(results) > 2:
        sleeps = sleeps[1:]
        wakes = wakes[1:]
        totals = totals[1:]

    print(f"\n  {name}:")
    if sleeps[0] > 0:
        print(f"    Sleep:  avg={sum(sleeps)/len(sleeps)*1000:.1f}ms  min={min(sleeps)*1000:.1f}ms  max={max(sleeps)*1000:.1f}ms")
        print(f"    Wake:   avg={sum(wakes)/len(wakes)*1000:.1f}ms  min={min(wakes)*1000:.1f}ms  max={max(wakes)*1000:.1f}ms")
    print(f"    Total:  avg={sum(totals)/len(totals)*1000:.1f}ms  min={min(totals)*1000:.1f}ms  max={max(totals)*1000:.1f}ms")


def run_benchmark(size_gb: float = 7.0, gpu_id: int = 0, n_iters: int = 7):
    """
    Run all approaches on synthetic weights of given size.
    Default 7GB = one GPU's share of a 70B model with TP=4 (~28GB / 4).
    """
    size_bytes = int(size_gb * 1024**3)
    n_params = size_bytes // 2  # fp16

    print(f"=" * 70)
    print(f"PINNED MEMORY SWAP BENCHMARK")
    print(f"Weight size: {size_gb:.1f} GB per GPU (simulating 70B TP=4)")
    print(f"GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    print(f"Iterations: {n_iters}")
    print(f"=" * 70)

    # Create synthetic model weights on GPU
    print(f"\nAllocating {size_gb:.1f} GB of synthetic weights on GPU {gpu_id}...")
    with torch.cuda.device(gpu_id):
        n_layers = 80  # typical for 70B
        layer_size = n_params // n_layers
        gpu_tensors = {}
        for i in range(n_layers):
            gpu_tensors[f"layer_{i}"] = torch.randn(layer_size, dtype=torch.float16, device=f"cuda:{gpu_id}")
    torch.cuda.synchronize(gpu_id)
    actual_gb = sum(t.nelement() * t.element_size() for t in gpu_tensors.values()) / (1024**3)
    print(f"  Allocated: {actual_gb:.2f} GB across {n_layers} layers")

    # ─── Approach 1: vLLM-style sync ────────────────────────────
    print(f"\n--- Approach 1: vLLM sync (fresh pinned alloc + sync memcpy) ---")
    r1 = bench_vllm_sync(gpu_tensors, gpu_id, n_iters)
    print_results("vLLM Sync", r1)

    gc.collect()
    torch.cuda.empty_cache()

    # ─── Approach 2: Pre-pinned async (per-tensor) ──────────────
    print(f"\n--- Approach 2: Pre-pinned async (per-tensor) ---")
    r2 = bench_pinned_async(gpu_tensors, gpu_id, n_iters)
    print_results("Pinned Async (per-tensor)", r2)

    gc.collect()
    torch.cuda.empty_cache()

    # ─── Approach 3: Pre-pinned contiguous (single memcpy) ──────
    print(f"\n--- Approach 3: Pre-pinned contiguous (single memcpy) ---")
    r3 = bench_pinned_contiguous(gpu_tensors, gpu_id, n_iters)
    print_results("Pinned Contiguous", r3)

    gc.collect()
    torch.cuda.empty_cache()

    # ─── Approach 4: Page table remap ───────────────────────────
    del gpu_tensors
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n--- Approach 4: Dual-resident page table remap (no data copy) ---")
    r4 = bench_page_table_remap(size_bytes, gpu_id, n_iters)
    print_results("Page Table Remap", r4)

    # ─── Summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY — {size_gb:.1f} GB weights on {torch.cuda.get_device_name(gpu_id)}")
    print(f"{'='*70}")

    def avg_total(r):
        return sum(x["total"] for x in r[1:]) / len(r[1:]) if len(r) > 1 else r[0]["total"]
    def avg_wake(r):
        return sum(x["wake"] for x in r[1:]) / len(r[1:]) if len(r) > 1 else r[0]["wake"]

    approaches = [
        ("vLLM sync (baseline)", r1),
        ("Pinned async (per-tensor)", r2),
        ("Pinned contiguous (single)", r3),
        ("Page table remap", r4),
    ]

    print(f"\n  {'Approach':<35} {'Swap (ms)':<12} {'Wake (ms)':<12} {'Speedup':<10}")
    print(f"  {'-'*69}")
    baseline_total = avg_total(r1)
    baseline_wake = avg_wake(r1)
    for name, r in approaches:
        t = avg_total(r)
        w = avg_wake(r)
        speedup = baseline_total / t if t > 0 else float('inf')
        print(f"  {name:<35} {t*1000:>8.1f}ms  {w*1000:>8.1f}ms  {speedup:>7.1f}x")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-gb", type=float, default=7.0, help="Weight size per GPU in GB")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iters", type=int, default=7)
    args = parser.parse_args()
    run_benchmark(args.size_gb, args.gpu, args.iters)
