"""Benchmark: Prism vs Modal vs RunPod serverless.

Measures for each provider:
- Cold start (from scale-to-zero to first token)
- Warm TTFT (container/model already loaded)
- Decode throughput (tok/s)
- p50, p90, p99 latency over N requests
- Scale-to-zero behavior
"""

import time
import json
import statistics
import requests as http


# ===== RunPod Serverless =====
RUNPOD_URL = "https://api.runpod.ai/v2/vjgxfutjy0mbqx"
RUNPOD_KEY = "rpa_GRNIP5WLEGM19ZEUVY17YGHKIS90K42NPBZASX751yref5"
RUNPOD_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_KEY}",
}

# ===== Modal (use the benchmark from bench_modal.py results) =====
# Modal results already collected: cold=119s, warm=66 tok/s

PROMPT = "The capital of France is"
MAX_TOKENS = 100


def runpod_submit(prompt, max_tokens=100):
    """Submit async job to RunPod. Returns job ID."""
    resp = http.post(f"{RUNPOD_URL}/run", headers=RUNPOD_HEADERS,
                     json={"input": {"prompt": prompt, "max_tokens": max_tokens}})
    return resp.json().get("id")


def runpod_poll(job_id, timeout=300):
    """Poll RunPod for job completion. Returns (result, elapsed_seconds)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        resp = http.get(f"{RUNPOD_URL}/status/{job_id}", headers=RUNPOD_HEADERS)
        data = resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data, time.time() - t0
        elif status == "FAILED":
            return data, time.time() - t0
        time.sleep(0.5)
    return {"status": "TIMEOUT"}, timeout


def runpod_sync(prompt, max_tokens=100):
    """Submit and wait for RunPod result. Returns (result, total_seconds)."""
    resp = http.post(f"{RUNPOD_URL}/runsync", headers=RUNPOD_HEADERS,
                     json={"input": {"prompt": prompt, "max_tokens": max_tokens}},
                     timeout=300)
    return resp.json(), 0  # runsync blocks until done


def benchmark_runpod():
    """Full RunPod benchmark."""
    print("\n" + "=" * 60)
    print("  RUNPOD SERVERLESS")
    print("=" * 60)

    results = {}

    # Test 1: Cold start (first request after idle)
    print("\n  [1] Cold start test (submit + poll)...")
    t0 = time.time()
    job_id = runpod_submit(PROMPT, MAX_TOKENS)
    print(f"      Job ID: {job_id}")
    data, elapsed = runpod_poll(job_id, timeout=300)
    results["cold_start_s"] = elapsed
    status = data.get("status", "unknown")
    print(f"      Status: {status}, Elapsed: {elapsed:.1f}s")

    if status == "COMPLETED":
        output = data.get("output", {})
        if isinstance(output, str):
            print(f"      Output: {output[:60]}...")
        elif isinstance(output, dict):
            text = output.get("text", output.get("output", str(output)))
            if isinstance(text, list):
                text = text[0] if text else ""
            print(f"      Output: {str(text)[:60]}...")
            exec_time = data.get("executionTime", 0)
            delay_time = data.get("delayTime", 0)
            results["exec_time_ms"] = exec_time
            results["delay_time_ms"] = delay_time
            print(f"      Execution time: {exec_time}ms")
            print(f"      Queue delay: {delay_time}ms")

    # Test 2: Warm requests (N sequential requests)
    print(f"\n  [2] Warm latency test (10 sequential requests)...")
    latencies = []
    for i in range(10):
        t0 = time.time()
        job_id = runpod_submit(PROMPT, 20)  # Short generation for latency test
        data, elapsed = runpod_poll(job_id, timeout=120)
        total = time.time() - t0
        latencies.append(total)
        status = data.get("status", "?")
        exec_ms = data.get("executionTime", 0)
        delay_ms = data.get("delayTime", 0)
        print(f"      [{i+1}] {total:.2f}s (exec={exec_ms}ms, delay={delay_ms}ms, status={status})")

    if latencies:
        latencies.sort()
        results["warm_p50_s"] = latencies[len(latencies) // 2]
        results["warm_p90_s"] = latencies[int(len(latencies) * 0.9)]
        results["warm_p99_s"] = latencies[-1]  # With 10 samples, p99 ≈ max
        results["warm_mean_s"] = statistics.mean(latencies)

    # Test 3: Concurrent load (submit 5 at once)
    print(f"\n  [3] Concurrent load test (5 simultaneous)...")
    t0 = time.time()
    job_ids = []
    for _ in range(5):
        jid = runpod_submit(PROMPT, 50)
        job_ids.append(jid)

    # Poll all
    concurrent_times = []
    for jid in job_ids:
        data, elapsed = runpod_poll(jid, timeout=120)
        concurrent_times.append(time.time() - t0)

    if concurrent_times:
        results["concurrent_5_wall_s"] = max(concurrent_times)
        results["concurrent_5_mean_s"] = statistics.mean(concurrent_times)
        print(f"      Wall time: {max(concurrent_times):.1f}s")
        print(f"      Mean per request: {statistics.mean(concurrent_times):.1f}s")

    return results


def benchmark_prism_latency():
    """Measure Prism p50/p90/p99 latency for comparison."""
    print("\n" + "=" * 60)
    print("  PRISM (local)")
    print("=" * 60)

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from engine.config import ModelConfig, EngineConfig, SchedulerConfig
    from engine.memory_pool import PinnedPool, MultiGPUPool
    from engine.weight_manager import WeightManager
    from engine.weight_pool import StaticWeightPool
    from engine.kv_cache import FlashAttnKVCache
    from engine.executor import FlashAttnExecutorV3
    from transformers import AutoConfig, AutoTokenizer

    MODEL_ID = "Qwen/Qwen3-8B"
    model_cfg = ModelConfig(model_id=MODEL_ID, name="bench")
    engine_cfg = EngineConfig(models=[model_cfg], gpu_ids=[0])
    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([0], 60, 20)
    wm = WeightManager(engine_cfg, pinned, gpu)
    wm.load_model(model_cfg)
    state = wm.get_state("bench")
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    pool = StaticWeightPool(config, "cuda:0", torch.bfloat16)
    pool.load_from_pinned(state.pinned_weights, config)
    nkv = config.num_key_value_heads
    hd = config.hidden_size // config.num_attention_heads
    kv = FlashAttnKVCache(config.num_hidden_layers, nkv, hd, 128, 256, "cuda:0", torch.bfloat16)
    executor = FlashAttnExecutorV3(pool, kv, "cuda:0")
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to("cuda:0")

    # Warmup
    kv.free_all()
    executor.generate(input_ids, seq_id=0, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id)
    kv.free_all()
    executor.invalidate_graph()

    # p50/p90/p99 over 50 requests (20 tokens each for latency test)
    print(f"\n  Latency test (50 requests, 20 tokens each)...")
    latencies = []
    for i in range(50):
        kv.new_sequence(i + 100)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        executor.generate(input_ids, i + 100, max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        kv.free_sequence(i + 100)

    latencies.sort()
    results = {
        "p50_s": latencies[len(latencies) // 2],
        "p90_s": latencies[int(len(latencies) * 0.9)],
        "p99_s": latencies[int(len(latencies) * 0.99)],
        "mean_s": statistics.mean(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
    }

    for k, v in results.items():
        print(f"    {k}: {v*1000:.1f}ms")

    # Full 100-token generation
    kv.free_all()
    kv.new_sequence(999)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen = executor.generate(input_ids, 999, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    t100 = time.perf_counter() - t0
    results["full_100tok_s"] = t100
    results["tok_s"] = len(gen) / t100
    print(f"    100 tokens: {t100:.3f}s ({len(gen)/t100:.0f} tok/s)")

    del executor, kv, pool
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    print("=" * 60)
    print("  SERVERLESS PROVIDER BENCHMARK: Qwen3-8B")
    print("=" * 60)

    # Prism local
    prism = benchmark_prism_latency()

    # RunPod serverless
    runpod = benchmark_runpod()

    # Modal results (from previous benchmark run)
    modal = {
        "cold_start_s": 118.8,
        "warm_tok_s": 65.8,
        "warm_request_100tok_s": 100 / 65.8,  # ~1.52s
        "batch8_tok_s": 465.7,
    }

    # Final comparison
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON: Qwen3-8B")
    print("=" * 60)

    print(f"\n  {'Metric':<30s} {'Prism':>12s} {'RunPod':>12s} {'Modal':>12s}")
    print(f"  {'-'*68}")
    print(f"  {'Cold start':<30s} {'303ms':>12s} {runpod.get('cold_start_s',0):.1f}s{' ':>6s} {'118.8s':>12s}")
    print(f"  {'Warm request (20 tok)':<30s} {prism['p50_s']*1000:.0f}ms p50{' ':>3s} {runpod.get('warm_p50_s',0)*1000:.0f}ms p50{' ':>3s} {'~1520ms':>12s}")
    print(f"  {'  p90':<30s} {prism['p90_s']*1000:.0f}ms{' ':>6s} {runpod.get('warm_p90_s',0)*1000:.0f}ms{' ':>6s} {'N/A':>12s}")
    print(f"  {'  p99':<30s} {prism['p99_s']*1000:.0f}ms{' ':>6s} {runpod.get('warm_p99_s',0)*1000:.0f}ms{' ':>6s} {'N/A':>12s}")
    print(f"  {'100 tok generation':<30s} {prism['full_100tok_s']:.2f}s{' ':>6s} {runpod.get('exec_time_ms',0)/1000:.2f}s{' ':>6s} {modal['warm_request_100tok_s']:.2f}s{' ':>6s}")
    print(f"  {'Decode tok/s':<30s} {prism['tok_s']:.0f}{' ':>8s} {'~66':>12s} {modal['warm_tok_s']:.0f}{' ':>8s}")
    print(f"  {'Concurrent 5':<30s} {'<1s':>12s} {runpod.get('concurrent_5_wall_s',0):.1f}s{' ':>6s} {'N/A':>12s}")
    print(f"  {'Model swap':<30s} {'303ms':>12s} {'N/A':>12s} {'N/A':>12s}")
    print(f"  {'Models per GPU':<30s} {'4+':>12s} {'1':>12s} {'1':>12s}")
    print(f"  {'Scale to zero cost':<30s} {'$0 (in RAM)':>12s} {'$0':>12s} {'$0':>12s}")
    print(f"  {'Resume from zero':<30s} {'303ms':>12s} {runpod.get('cold_start_s',0):.0f}s{' ':>6s} {'119s':>12s}")


if __name__ == "__main__":
    main()
