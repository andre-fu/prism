"""Integration test: multi-model automatic scheduling.

Loads 2 models (7B and 0.5B to keep it fast), sends interleaved requests,
and lets the scheduler handle everything automatically — model loading,
swapping, prefetch, and generation.
"""

import time
import torch
import threading

from .config import ModelConfig, EngineConfig, SchedulerConfig
from .memory_pool import PinnedPool, MultiGPUPool
from .weight_manager import WeightManager
from .request_manager import RequestManager, RequestState
from .prefetch import PrefetchController
from .scheduler import Scheduler

GPU = 0
DEVICE = f"cuda:{GPU}"


def main():
    print("=" * 60)
    print("SCHEDULER TEST: Multi-model automatic scheduling")
    print("=" * 60)

    # Two models — use small ones for fast testing
    model_a = ModelConfig(model_id="Qwen/Qwen2.5-0.5B", name="qwen-0.5b", max_seq_len=2048)
    model_b = ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="qwen-7b", max_seq_len=2048)

    engine_cfg = EngineConfig(
        models=[model_a, model_b],
        gpu_ids=[GPU],
        t0_budget_gb=40.0,
        kv_cache_budget_gb=10.0,  # Smaller for testing
    )
    sched_cfg = SchedulerConfig(
        max_consecutive_batches=8,
        prefetch_queue_threshold=1,
    )

    # Setup
    pinned_pool = PinnedPool(budget_gb=engine_cfg.t1_budget_gb)
    gpu_pool = MultiGPUPool(engine_cfg.gpu_ids, engine_cfg.t0_budget_gb, engine_cfg.kv_cache_budget_gb)
    wm = WeightManager(engine_cfg, pinned_pool, gpu_pool)
    rm = RequestManager()
    pc = PrefetchController(wm, rm, gpu_pool, queue_threshold=sched_cfg.prefetch_queue_threshold)
    scheduler = Scheduler(engine_cfg, sched_cfg, wm, rm, pc, gpu_pool)

    # Load models into pinned RAM
    print("\n[1] Loading models to pinned RAM...")
    wm.load_model(model_a)
    wm.load_model(model_b)
    print(f"    Pinned RAM: {pinned_pool.used_gb:.1f} GB")

    # Tokenize prompts
    prompts = [
        ("qwen-0.5b", "The capital of France is"),
        ("qwen-7b", "Write a short poem about the ocean"),
        ("qwen-0.5b", "What is 2 + 2?"),
        ("qwen-7b", "Explain quantum computing in one sentence"),
        ("qwen-0.5b", "The meaning of life is"),
        ("qwen-7b", "What is the speed of light?"),
    ]

    # Submit all requests
    print(f"\n[2] Submitting {len(prompts)} requests...")
    requests = []
    for model_name, prompt in prompts:
        state = wm.get_state(model_name)
        tokens = state.tokenizer.encode(prompt)
        req = rm.add_request(
            model_name=model_name,
            prompt=prompt,
            prompt_tokens=tokens,
            max_new_tokens=50,
            temperature=0.0,
        )
        requests.append(req)
        print(f"    [{req.id}] {model_name}: '{prompt}'")

    # Run scheduler until all requests are done
    print(f"\n[3] Running scheduler...")
    t_start = time.time()

    scheduler.run_background()

    # Wait for all requests to complete
    while True:
        done = all(r.state == RequestState.DONE for r in requests)
        if done:
            break
        remaining = sum(1 for r in requests if r.state != RequestState.DONE)
        time.sleep(0.1)
        if time.time() - t_start > 120:
            print("    TIMEOUT after 120s!")
            break

    scheduler.stop()
    t_total = time.time() - t_start

    # Results
    print(f"\n[4] Results ({t_total:.1f}s total)")
    print("-" * 60)
    for req in requests:
        state = wm.get_state(req.model_name)
        text = state.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        status = "DONE" if req.state == RequestState.DONE else req.state.value
        print(f"\n  [{req.id}] {req.model_name} ({status})")
        print(f"      Prompt: '{req.prompt}'")
        print(f"      Output: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"      Tokens: {len(req.generated_tokens)}, TTFT: {req.ttft*1000:.0f}ms, "
              f"TBT: {req.tbt*1000:.1f}ms, Total: {req.total_time:.2f}s")

    # Stats
    stats = scheduler.stats
    print(f"\n--- Scheduler Stats ---")
    print(f"Batches:          {stats.batches}")
    print(f"Evictions:        {stats.evictions}")
    print(f"Sync loads:       {stats.sync_loads}")
    print(f"Prefetch triggers:{stats.prefetch_triggers}")
    print(f"Completed:        {stats.completed}/{len(requests)}")
    print(f"Tokens generated: {stats.tokens_generated}")
    total_tokens = sum(len(r.generated_tokens) for r in requests if r.state == RequestState.DONE)
    print(f"Aggregate tok/s:  {total_tokens/t_total:.1f}")
    print(f"GPU memory:       {torch.cuda.memory_allocated(GPU)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
