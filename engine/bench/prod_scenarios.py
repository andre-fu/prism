"""Production scenario benchmarks.

Tests realistic multi-model deployments:
  1. Four same-architecture models (simulating 4 customer fine-tunes)
  2. Mixed architectures (0.5B + 1.5B + 7B + 14B)
  3. Bursty traffic (one model gets 10x spike)
  4. Multi-turn conversations (KV preservation test)
  5. Cold start latency (new model, never seen before)
  6. Sustained load (continuous requests over 30s)
"""

import time
import torch
import threading
import random

from ..config import ModelConfig, EngineConfig, SchedulerConfig
from ..memory_pool import PinnedPool, MultiGPUPool
from ..weight_manager import WeightManager
from ..request_manager import RequestManager, RequestState
from ..prefetch import PrefetchController
from ..scheduler import Scheduler

GPU = 0
DEVICE = f"cuda:{GPU}"


def setup(models_cfg, t0_budget=40, kv_budget=15):
    engine_cfg = EngineConfig(
        models=models_cfg, gpu_ids=[GPU],
        t0_budget_gb=t0_budget, kv_cache_budget_gb=kv_budget,
    )
    sched_cfg = SchedulerConfig(max_consecutive_batches=32)
    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], t0_budget, kv_budget)
    wm = WeightManager(engine_cfg, pinned, gpu)
    rm = RequestManager()
    pc = PrefetchController(wm, rm, gpu)
    scheduler = Scheduler(engine_cfg, sched_cfg, wm, rm, pc, gpu)
    for mc in models_cfg:
        wm.load_model(mc)
    return wm, rm, scheduler


def run_and_wait(scheduler, rm, requests, timeout=120):
    t0 = time.time()
    scheduler.run_background()
    while not all(r.state == RequestState.DONE for r in requests):
        if time.time() - t0 > timeout:
            print("    TIMEOUT!")
            break
        time.sleep(0.05)
    scheduler.stop()
    return time.time() - t0


def teardown(scheduler):
    """Clean shutdown — free all GPU memory."""
    scheduler.cleanup()
    import gc
    gc.collect()
    for g in [0, 1, 2, 3]:
        try:
            with torch.cuda.device(g):
                torch.cuda.empty_cache()
        except Exception:
            pass


def report(requests, t_total, wm):
    completed = [r for r in requests if r.state == RequestState.DONE]
    errored = [r for r in requests if r.error]
    total_tokens = sum(len(r.generated_tokens) for r in completed)
    ttfts = sorted([r.ttft * 1000 for r in completed if r.ttft > 0])

    print(f"    Completed: {len(completed)}/{len(requests)}, Errors: {len(errored)}")
    print(f"    Tokens: {total_tokens}, Wall: {t_total:.2f}s, Agg tok/s: {total_tokens/t_total:.1f}")
    if ttfts:
        print(f"    TTFT: p50={ttfts[len(ttfts)//2]:.0f}ms, p95={ttfts[int(len(ttfts)*0.95)]:.0f}ms, min={ttfts[0]:.0f}ms, max={ttfts[-1]:.0f}ms")
    if errored:
        for r in errored[:3]:
            print(f"    Error: {r.error[:80]}")


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# =====================================================================
# Scenario 1: Four same-architecture models (customer fine-tunes)
# =====================================================================
def scenario_4_same_arch():
    section("1. FOUR SAME-ARCH MODELS (7B fine-tunes)")

    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name=f"customer-{i}")
        for i in range(4)
    ]
    wm, rm, scheduler = setup(models, t0_budget=40, kv_budget=15)

    prompts = [
        "Summarize the key points of contract law",
        "Write a Python function to parse CSV files",
        "Explain the diagnosis for chest pain symptoms",
        "Draft an insurance claim for water damage",
        "What are the HIPAA compliance requirements",
        "Debug this SQL query: SELECT * FROM",
        "Translate this legal clause into plain English:",
        "Calculate the premium for a 30-year-old male",
        "List the side effects of metformin",
        "Write unit tests for the authentication module",
        "Explain the statute of limitations for fraud",
        "Generate a medical report template",
    ]

    requests = []
    for i, prompt in enumerate(prompts):
        model_name = f"customer-{i % 4}"
        state = wm.get_state(model_name)
        req = rm.add_request(model_name, prompt, state.tokenizer.encode(prompt), max_new_tokens=50)
        requests.append(req)

    print(f"    12 requests across 4 models (3 each)")
    t = run_and_wait(scheduler, rm, requests)
    report(requests, t, wm)

    for i in range(4):
        model_reqs = [r for r in requests if r.model_name == f"customer-{i}"]
        tokens = sum(len(r.generated_tokens) for r in model_reqs if r.state == RequestState.DONE)
        ttfts = [r.ttft*1000 for r in model_reqs if r.ttft > 0]
        avg_ttft = sum(ttfts)/len(ttfts) if ttfts else 0
        print(f"    customer-{i}: {tokens} tok, avg TTFT={avg_ttft:.0f}ms")

    teardown(scheduler)


# =====================================================================
# Scenario 2: Mixed architectures (0.5B + 1.5B + 7B + 14B)
# =====================================================================
def scenario_mixed_arch():
    section("2. MIXED ARCHITECTURES (0.5B + 1.5B + 7B + 14B)")

    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-0.5B", name="tiny"),
        ModelConfig(model_id="Qwen/Qwen2.5-1.5B", name="small"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="medium"),
        ModelConfig(model_id="Qwen/Qwen2.5-14B-Instruct", name="large"),
    ]
    # Budget: 14B=30GB, 7B=15GB, 1.5B=3GB, 0.5B=1GB. Only one big model at a time.
    wm, rm, scheduler = setup(models, t0_budget=45, kv_budget=10)

    prompts_by_model = {
        "tiny": ["What is 2+2?", "Hello world", "Count to 5"],
        "small": ["Explain gravity", "What is DNA?", "Define entropy"],
        "medium": ["Write a poem about the stars", "Explain quantum physics", "History of Rome"],
        "large": ["Analyze this legal document:", "Comprehensive market analysis:", "Detailed medical review:"],
    }

    requests = []
    for model_name, prompts in prompts_by_model.items():
        state = wm.get_state(model_name)
        for p in prompts:
            req = rm.add_request(model_name, p, state.tokenizer.encode(p), max_new_tokens=40)
            requests.append(req)

    print(f"    12 requests across 4 architectures")
    t = run_and_wait(scheduler, rm, requests)
    report(requests, t, wm)

    for mn in ["tiny", "small", "medium", "large"]:
        model_reqs = [r for r in requests if r.model_name == mn]
        done = [r for r in model_reqs if r.state == RequestState.DONE]
        tokens = sum(len(r.generated_tokens) for r in done)
        print(f"    {mn:>6s}: {len(done)}/{len(model_reqs)} done, {tokens} tok")

    stats = scheduler.stats
    print(f"    Evictions: {stats.evictions}, Sync loads: {stats.sync_loads}, Prefetch hits: {stats.prefetch_hits}")

    teardown(scheduler)


# =====================================================================
# Scenario 3: Bursty traffic (one model gets 10x spike)
# =====================================================================
def scenario_bursty():
    section("3. BURSTY TRAFFIC (model-a gets 10x spike)")

    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-a"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-b"),
    ]
    wm, rm, scheduler = setup(models, t0_budget=40, kv_budget=15)

    requests = []
    # 2 baseline requests for model-b
    state_b = wm.get_state("model-b")
    for p in ["Hello", "World"]:
        requests.append(rm.add_request("model-b", p, state_b.tokenizer.encode(p), max_new_tokens=30))

    # 10 burst requests for model-a
    state_a = wm.get_state("model-a")
    burst_prompts = [f"Question {i}: explain topic {i}" for i in range(10)]
    for p in burst_prompts:
        requests.append(rm.add_request("model-a", p, state_a.tokenizer.encode(p), max_new_tokens=30))

    print(f"    2 model-b + 10 model-a (burst)")
    t = run_and_wait(scheduler, rm, requests)
    report(requests, t, wm)

    a_reqs = [r for r in requests if r.model_name == "model-a"]
    b_reqs = [r for r in requests if r.model_name == "model-b"]
    a_done = sum(1 for r in a_reqs if r.state == RequestState.DONE)
    b_done = sum(1 for r in b_reqs if r.state == RequestState.DONE)
    print(f"    model-a: {a_done}/{len(a_reqs)} done")
    print(f"    model-b: {b_done}/{len(b_reqs)} done")

    teardown(scheduler)


# =====================================================================
# Scenario 4: Multi-turn conversations (KV preservation)
# =====================================================================
def scenario_multiturn():
    section("4. MULTI-TURN CONVERSATIONS (KV preservation)")

    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-a"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-b"),
    ]
    wm, rm, scheduler = setup(models, t0_budget=40, kv_budget=15)

    state_a = wm.get_state("model-a")
    state_b = wm.get_state("model-b")

    # Simulate: user-a sends 3 messages, with model-b requests in between
    # The key metric: does user-a's 2nd and 3rd message have low TTFT?
    # (If KV is preserved, no re-prefill needed)

    all_reqs = []

    # Turn 1: user-a
    r1 = rm.add_request("model-a", "What is machine learning?",
                        state_a.tokenizer.encode("What is machine learning?"), max_new_tokens=30)
    all_reqs.append(r1)

    # Interleave: model-b request
    rb1 = rm.add_request("model-b", "What is the capital of Japan?",
                         state_b.tokenizer.encode("What is the capital of Japan?"), max_new_tokens=30)
    all_reqs.append(rb1)

    # Turn 2: user-a follow-up
    r2 = rm.add_request("model-a", "Can you give me an example?",
                        state_a.tokenizer.encode("Can you give me an example?"), max_new_tokens=30)
    all_reqs.append(r2)

    # Interleave: another model-b request
    rb2 = rm.add_request("model-b", "What is the speed of sound?",
                         state_b.tokenizer.encode("What is the speed of sound?"), max_new_tokens=30)
    all_reqs.append(rb2)

    # Turn 3: user-a follow-up
    r3 = rm.add_request("model-a", "How does that compare to deep learning?",
                        state_a.tokenizer.encode("How does that compare to deep learning?"), max_new_tokens=30)
    all_reqs.append(r3)

    print(f"    3 model-a turns + 2 model-b interleaved")
    t = run_and_wait(scheduler, rm, all_reqs)

    print(f"    Done in {t:.2f}s")
    print(f"    Model-a TTFTs:")
    for i, r in enumerate([r1, r2, r3]):
        state = wm.get_state(r.model_name)
        text = state.tokenizer.decode(r.generated_tokens, skip_special_tokens=True)
        print(f"      Turn {i+1}: TTFT={r.ttft*1000:.0f}ms, {len(r.generated_tokens)} tok: {text[:50]}")
    print(f"    Model-b TTFTs:")
    for r in [rb1, rb2]:
        state = wm.get_state(r.model_name)
        text = state.tokenizer.decode(r.generated_tokens, skip_special_tokens=True)
        print(f"      TTFT={r.ttft*1000:.0f}ms, {len(r.generated_tokens)} tok: {text[:50]}")

    teardown(scheduler)


# =====================================================================
# Scenario 5: Cold start latency
# =====================================================================
def scenario_cold_start():
    section("5. COLD START LATENCY")

    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-0.5B", name="tiny"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="medium"),
        ModelConfig(model_id="Qwen/Qwen2.5-14B-Instruct", name="large"),
    ]
    wm, rm, scheduler = setup(models, t0_budget=45, kv_budget=10)

    # Each model gets one request — measures cold start (first load)
    for mn in ["tiny", "medium", "large"]:
        state = wm.get_state(mn)
        req = rm.add_request(mn, "Hello world", state.tokenizer.encode("Hello world"), max_new_tokens=20)
        scheduler.run_background()
        t0 = time.time()
        while req.state != RequestState.DONE:
            if time.time() - t0 > 60: break
            time.sleep(0.01)
        scheduler.stop()
        t = time.time() - t0
        text = state.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        print(f"    {mn:>6s}: TTFT={req.ttft*1000:.0f}ms, total={t:.2f}s, {len(req.generated_tokens)} tok: {text[:40]}")

    teardown(scheduler)


# =====================================================================
# Scenario 6: Sustained load (30s continuous)
# =====================================================================
def scenario_sustained():
    section("6. SUSTAINED LOAD (15s continuous)")

    models = [
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-a"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-b"),
        ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="model-c"),
    ]
    wm, rm, scheduler = setup(models, t0_budget=40, kv_budget=15)

    prompts = [
        "Explain the theory of relativity",
        "Write a haiku about coding",
        "What are the benefits of exercise",
        "Describe the water cycle",
        "How does encryption work",
    ]

    all_reqs = []
    done_event = threading.Event()

    def submit_loop():
        """Submit requests continuously for 15 seconds."""
        t0 = time.time()
        i = 0
        while time.time() - t0 < 15 and not done_event.is_set():
            model_name = f"model-{chr(ord('a') + (i % 3))}"
            prompt = prompts[i % len(prompts)]
            state = wm.get_state(model_name)
            req = rm.add_request(model_name, prompt, state.tokenizer.encode(prompt), max_new_tokens=30)
            all_reqs.append(req)
            i += 1
            time.sleep(0.5)  # 2 req/sec

    scheduler.run_background()
    submit_thread = threading.Thread(target=submit_loop)
    submit_thread.start()

    # Let it run for 20s total (15s submitting + 5s drain)
    time.sleep(20)
    done_event.set()
    submit_thread.join()

    # Wait for remaining requests
    t_wait = time.time()
    while not all(r.state == RequestState.DONE for r in all_reqs):
        if time.time() - t_wait > 30: break
        time.sleep(0.1)
    scheduler.stop()

    completed = [r for r in all_reqs if r.state == RequestState.DONE]
    total_tokens = sum(len(r.generated_tokens) for r in completed)
    ttfts = sorted([r.ttft * 1000 for r in completed if r.ttft > 0])

    print(f"    Submitted: {len(all_reqs)} requests over 15s")
    print(f"    Completed: {len(completed)}/{len(all_reqs)}")
    print(f"    Tokens: {total_tokens}")
    if ttfts:
        print(f"    TTFT: p50={ttfts[len(ttfts)//2]:.0f}ms, p95={ttfts[int(len(ttfts)*0.95)]:.0f}ms")

    stats = scheduler.stats
    print(f"    Evictions: {stats.evictions}, Prefetch hits: {stats.prefetch_hits}")

    # Check memory
    gpu_mem = torch.cuda.memory_allocated(GPU) / 1e9
    print(f"    GPU memory (during): {gpu_mem:.1f} GB")

    teardown(scheduler)
    gpu_mem_after = torch.cuda.memory_allocated(GPU) / 1e9
    print(f"    GPU after cleanup: {gpu_mem_after:.2f} GB {'OK' if gpu_mem_after < 0.5 else 'LEAK'}")


def main():
    print("=" * 70)
    print("  PRODUCTION SCENARIO BENCHMARKS")
    print(f"  Hardware: H100 80GB, PyTorch {torch.__version__}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    scenario_4_same_arch()
    scenario_mixed_arch()
    scenario_bursty()
    scenario_multiturn()
    scenario_cold_start()
    scenario_sustained()

    print("\n" + "=" * 70)
    print("  ALL SCENARIOS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
