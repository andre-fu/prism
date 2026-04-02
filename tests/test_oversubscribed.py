"""Test: oversubscribed GPU — models don't all fit, scheduler must swap.

Loads 7B (15GB) + 14B (30GB) = 45GB into 40GB weight budget.
Only one model fits at a time. Every model switch forces an eviction.
Sends interleaved requests to both models to stress the swap path.
"""

import time
import torch

from engine.config import ModelConfig, EngineConfig, SchedulerConfig
from engine.memory_pool import PinnedPool, MultiGPUPool
from engine.weight_manager import WeightManager
from engine.request_manager import RequestManager, RequestState
from engine.prefetch import PrefetchController
from engine.scheduler import Scheduler

GPU = 0


def main():
    print("=" * 60)
    print("OVERSUBSCRIBED TEST: 7B + 14B on 40GB budget")
    print("=" * 60)

    model_a = ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="qwen-7b")
    model_b = ModelConfig(model_id="Qwen/Qwen2.5-14B-Instruct", name="qwen-14b")

    engine_cfg = EngineConfig(
        models=[model_a, model_b],
        gpu_ids=[GPU],
        t0_budget_gb=30.0,     # Only 30GB for weights — 7B(15) fits OR 14B(30) fits, never both
        kv_cache_budget_gb=10.0,
    )
    sched_cfg = SchedulerConfig(
        max_consecutive_batches=32,  # Serve more before yielding (fewer swaps)
        prefetch_queue_threshold=1,
    )

    # Setup
    pinned_pool = PinnedPool(budget_gb=engine_cfg.t1_budget_gb)
    gpu_pool = MultiGPUPool(engine_cfg.gpu_ids, engine_cfg.t0_budget_gb, engine_cfg.kv_cache_budget_gb)
    wm = WeightManager(engine_cfg, pinned_pool, gpu_pool)
    rm = RequestManager()
    pc = PrefetchController(wm, rm, gpu_pool, queue_threshold=sched_cfg.prefetch_queue_threshold)
    scheduler = Scheduler(engine_cfg, sched_cfg, wm, rm, pc, gpu_pool)

    # Load to pinned RAM
    print("\n[1] Loading models to pinned RAM...")
    wm.load_model(model_a)
    wm.load_model(model_b)
    sa = wm.get_state("qwen-7b")
    sb = wm.get_state("qwen-14b")
    print(f"    7B: {sa.pinned_bytes/1e9:.1f} GB")
    print(f"    14B: {sb.pinned_bytes/1e9:.1f} GB")
    print(f"    Total: {(sa.pinned_bytes + sb.pinned_bytes)/1e9:.1f} GB (budget: {engine_cfg.t0_budget_gb} GB)")
    print(f"    => Only one fits at a time. Every switch = evict + load.")

    # Submit interleaved requests that force swaps
    prompts = [
        ("qwen-7b",  "What is the capital of France?"),
        ("qwen-14b", "What is the capital of Germany?"),
        ("qwen-7b",  "What is the capital of Japan?"),
        ("qwen-14b", "What is the capital of Brazil?"),
        ("qwen-7b",  "What is the capital of Australia?"),
        ("qwen-14b", "What is the capital of India?"),
    ]

    print(f"\n[2] Submitting {len(prompts)} interleaved requests (forces swaps)...")
    requests = []
    for model_name, prompt in prompts:
        state = wm.get_state(model_name)
        tokens = state.tokenizer.encode(prompt)
        req = rm.add_request(model_name, prompt, tokens, max_new_tokens=30, temperature=0.0)
        requests.append(req)
        print(f"    [{req.id}] {model_name}: '{prompt}'")

    # Run
    print(f"\n[3] Running scheduler...")
    t_start = time.time()
    scheduler.run_background()

    while True:
        done = all(r.state == RequestState.DONE for r in requests)
        if done:
            break
        if time.time() - t_start > 180:
            print("    TIMEOUT!")
            break
        time.sleep(0.1)

    scheduler.stop()
    t_total = time.time() - t_start

    # Results
    print(f"\n[4] Results ({t_total:.1f}s total)")
    print("-" * 60)
    for req in requests:
        state = wm.get_state(req.model_name)
        text = state.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        print(f"\n  [{req.id}] {req.model_name}")
        print(f"      Prompt:  '{req.prompt}'")
        print(f"      Output:  '{text[:100]}'")
        print(f"      Tokens:  {len(req.generated_tokens)}")
        print(f"      TTFT:    {req.ttft*1000:.0f}ms")
        print(f"      Total:   {req.total_time:.2f}s")

    # Stats
    stats = scheduler.stats
    total_tokens = sum(len(r.generated_tokens) for r in requests if r.state == RequestState.DONE)
    completed = sum(1 for r in requests if r.state == RequestState.DONE)

    print(f"\n--- Stats ---")
    print(f"Completed:       {completed}/{len(requests)}")
    print(f"Total tokens:    {total_tokens}")
    print(f"Aggregate:       {total_tokens/t_total:.1f} tok/s")
    print(f"Batches:         {stats.batches}")
    print(f"Evictions:       {stats.evictions}")
    print(f"Requeued:        {stats.requeued}")
    print(f"Sync loads:      {stats.sync_loads}")
    print(f"Prefetch starts: {stats.prefetch_triggers}")
    print(f"Prefetch hits:   {stats.prefetch_hits}")
    print(f"Wall clock:      {t_total:.1f}s")
    print(f"GPU memory:      {torch.cuda.memory_allocated(GPU)/1e9:.1f} GB")

    # Correctness check — all outputs should mention actual capitals
    print(f"\n--- Correctness ---")
    expected = {
        "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo",
        "Brazil": "Brasilia", "Australia": "Canberra", "India": "Delhi",
    }
    for req in requests:
        state = wm.get_state(req.model_name)
        text = state.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        for country, capital in expected.items():
            if country in req.prompt:
                found = capital.lower() in text.lower()
                print(f"  {country}: {'PASS' if found else 'FAIL'} — {'found' if found else 'missing'} '{capital}' in output")
                break


if __name__ == "__main__":
    main()
