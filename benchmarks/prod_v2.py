"""Production scenarios using Scheduler (static weight pool)."""

import time
import torch
import threading

from engineconfig import ModelConfig, EngineConfig, SchedulerConfig
from enginememory_pool import PinnedPool, MultiGPUPool
from engineweight_manager import WeightManager
from enginerequest_manager import RequestManager, RequestState
from enginescheduler import Scheduler

GPU = 0


def setup(models_cfg, kv_budget=15):
    engine_cfg = EngineConfig(models=models_cfg, gpu_ids=[GPU], kv_cache_budget_gb=kv_budget)
    sched_cfg = SchedulerConfig(max_consecutive_batches=32)
    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], 60, kv_budget)
    wm = WeightManager(engine_cfg, pinned, gpu)
    rm = RequestManager()
    for mc in models_cfg:
        wm.load_model(mc)
    scheduler = Scheduler(engine_cfg, sched_cfg, wm, rm, gpu)
    return wm, rm, scheduler


def run_and_wait(scheduler, requests, timeout=120):
    t0 = time.time()
    scheduler.run_background()
    while not all(r.state == RequestState.DONE for r in requests):
        if time.time() - t0 > timeout:
            print("    TIMEOUT!")
            break
        time.sleep(0.05)
    scheduler.stop()
    return time.time() - t0


def report(requests, t_total, wm):
    completed = [r for r in requests if r.state == RequestState.DONE]
    errored = [r for r in requests if r.error]
    total_tokens = sum(len(r.generated_tokens) for r in completed)
    ttfts = sorted([r.ttft * 1000 for r in completed if r.ttft > 0])
    print(f"    Completed: {len(completed)}/{len(requests)}, Errors: {len(errored)}")
    print(f"    Tokens: {total_tokens}, Wall: {t_total:.2f}s, Tok/s: {total_tokens/t_total:.1f}")
    if ttfts:
        print(f"    TTFT: p50={ttfts[len(ttfts)//2]:.0f}ms, p95={ttfts[int(len(ttfts)*0.95)]:.0f}ms")
    if errored:
        for r in errored[:3]:
            print(f"    Error: {r.error[:80]}")


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def scenario_4_same_arch():
    section("1. FOUR SAME-ARCH MODELS (7B fine-tunes)")
    models = [ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name=f"customer-{i}") for i in range(4)]
    wm, rm, scheduler = setup(models, kv_budget=15)

    requests = []
    prompts = ["Contract law summary", "Parse CSV in Python", "Chest pain diagnosis",
               "Water damage claim", "HIPAA requirements", "Debug SQL query",
               "Legal clause translation", "Premium calculation", "Metformin side effects",
               "Auth module tests", "Fraud statute of limitations", "Medical report template"]
    for i, p in enumerate(prompts):
        state = wm.get_state(f"customer-{i%4}")
        requests.append(rm.add_request(f"customer-{i%4}", p, state.tokenizer.encode(p), max_new_tokens=30))

    t = run_and_wait(scheduler, requests)
    report(requests, t, wm)
    stats = scheduler.stats
    print(f"    Swaps: {stats.sync_swaps} sync ({stats.swap_time_ms:.0f}ms total), {stats.prefetch_hits} prefetch hits")
    scheduler.cleanup()


def scenario_mixed_arch():
    section("2. MIXED ARCHITECTURES (0.5B + 1.5B + 7B)")
    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-0.5B", name="tiny"),
        ModelConfig(model_id="Qwen/Qwen2.5-1.5B", name="small"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="medium"),
    ]
    wm, rm, scheduler = setup(models, kv_budget=10)

    requests = []
    for mn in ["tiny", "small", "medium"]:
        state = wm.get_state(mn)
        for p in ["What is 2+2?", "Explain gravity", "Write a poem"]:
            requests.append(rm.add_request(mn, p, state.tokenizer.encode(p), max_new_tokens=30))

    t = run_and_wait(scheduler, requests)
    report(requests, t, wm)
    for mn in ["tiny", "small", "medium"]:
        done = [r for r in requests if r.model_name == mn and r.state == RequestState.DONE]
        print(f"    {mn}: {len(done)}/3 done, {sum(len(r.generated_tokens) for r in done)} tok")
    scheduler.cleanup()


def scenario_bursty():
    section("3. BURSTY TRAFFIC (model-a gets 10x spike)")
    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-a"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-b"),
    ]
    wm, rm, scheduler = setup(models, kv_budget=15)

    requests = []
    state_b = wm.get_state("model-b")
    for p in ["Hello", "World"]:
        requests.append(rm.add_request("model-b", p, state_b.tokenizer.encode(p), max_new_tokens=30))
    state_a = wm.get_state("model-a")
    for i in range(10):
        requests.append(rm.add_request("model-a", f"Question {i}", state_a.tokenizer.encode(f"Question {i}"), max_new_tokens=30))

    t = run_and_wait(scheduler, requests)
    report(requests, t, wm)
    scheduler.cleanup()


def scenario_multiturn():
    section("4. MULTI-TURN (KV preservation)")
    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-a"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-b"),
    ]
    wm, rm, scheduler = setup(models, kv_budget=15)

    state_a, state_b = wm.get_state("model-a"), wm.get_state("model-b")
    reqs = [
        rm.add_request("model-a", "What is ML?", state_a.tokenizer.encode("What is ML?"), max_new_tokens=30),
        rm.add_request("model-b", "Capital of Japan?", state_b.tokenizer.encode("Capital of Japan?"), max_new_tokens=30),
        rm.add_request("model-a", "Give an example", state_a.tokenizer.encode("Give an example"), max_new_tokens=30),
        rm.add_request("model-b", "Speed of sound?", state_b.tokenizer.encode("Speed of sound?"), max_new_tokens=30),
        rm.add_request("model-a", "Compare to deep learning", state_a.tokenizer.encode("Compare to deep learning"), max_new_tokens=30),
    ]

    t = run_and_wait(scheduler, reqs)
    print(f"    Done in {t:.2f}s")
    for r in reqs:
        state = wm.get_state(r.model_name)
        text = state.tokenizer.decode(r.generated_tokens, skip_special_tokens=True)
        print(f"    [{r.model_name}] TTFT={r.ttft*1000:.0f}ms, {len(r.generated_tokens)} tok: {text[:50]}")
    scheduler.cleanup()


def scenario_cold_start():
    section("5. COLD START LATENCY")
    for model_id, name in [("Qwen/Qwen2.5-0.5B", "tiny"), ("Qwen/Qwen2.5-7B-Instruct", "medium")]:
        models = [ModelConfig(model_id=model_id, name=name)]
        wm, rm, scheduler = setup(models, kv_budget=10)
        state = wm.get_state(name)
        req = rm.add_request(name, "Hello world", state.tokenizer.encode("Hello world"), max_new_tokens=20)
        t = run_and_wait(scheduler, [req])
        text = state.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        err = f" ERR={req.error}" if req.error else ""
        print(f"    {name:>6s}: TTFT={req.ttft*1000:.0f}ms, total={t:.2f}s, {len(req.generated_tokens)} tok{err}: {text[:40]}")
        scheduler.cleanup()


def scenario_sustained():
    section("6. SUSTAINED LOAD (15s)")
    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name=f"model-{c}")
        for c in "abc"
    ]
    wm, rm, scheduler = setup(models, kv_budget=15)

    all_reqs = []
    done_event = threading.Event()

    def submit_loop():
        prompts = ["Theory of relativity", "Haiku about coding", "Benefits of exercise",
                    "Water cycle", "How encryption works"]
        i = 0
        t0 = time.time()
        while time.time() - t0 < 15 and not done_event.is_set():
            mn = f"model-{chr(ord('a') + i % 3)}"
            p = prompts[i % len(prompts)]
            state = wm.get_state(mn)
            all_reqs.append(rm.add_request(mn, p, state.tokenizer.encode(p), max_new_tokens=30))
            i += 1
            time.sleep(0.5)

    scheduler.run_background()
    t = threading.Thread(target=submit_loop)
    t.start()
    time.sleep(20)
    done_event.set()
    t.join()

    t_wait = time.time()
    while not all(r.state == RequestState.DONE for r in all_reqs):
        if time.time() - t_wait > 30: break
        time.sleep(0.1)
    scheduler.stop()

    completed = [r for r in all_reqs if r.state == RequestState.DONE]
    total_tokens = sum(len(r.generated_tokens) for r in completed)
    ttfts = sorted([r.ttft * 1000 for r in completed if r.ttft > 0])

    print(f"    Submitted: {len(all_reqs)}, Completed: {len(completed)}")
    print(f"    Tokens: {total_tokens}")
    if ttfts:
        print(f"    TTFT: p50={ttfts[len(ttfts)//2]:.0f}ms, p95={ttfts[int(len(ttfts)*0.95)]:.0f}ms")
    stats = scheduler.stats
    print(f"    Swaps: {stats.sync_swaps} sync, {stats.prefetch_hits} prefetch")

    scheduler.cleanup()
    gpu_after = torch.cuda.memory_allocated(GPU) / 1e9
    print(f"    GPU after cleanup: {gpu_after:.2f} GB {'OK' if gpu_after < 0.5 else 'LEAK'}")


def main():
    print("=" * 70)
    print(f"  PROD SCENARIOS V2 — Static Weight Pool + CUDA Graphs")
    print(f"  Hardware: H100 80GB, PyTorch {torch.__version__}")
    print("=" * 70)

    scenario_4_same_arch()
    scenario_mixed_arch()
    scenario_bursty()
    scenario_multiturn()
    scenario_cold_start()
    scenario_sustained()

    print(f"\n{'='*70}\n  ALL SCENARIOS COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()
