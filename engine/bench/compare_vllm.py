"""Head-to-head benchmark: our engine vs dedicated vLLM.

Scenarios:
  A: Single model, single request — raw per-token speed comparison
  B: Single model, 8 concurrent requests — batched throughput
  C: Two models, interleaved requests — multi-model scheduling overhead

For each scenario, runs against both our engine and vLLM, reports latency and throughput.
"""

import time
import torch
import subprocess
import requests as http_requests
import json
import sys

from ..config import ModelConfig, EngineConfig, SchedulerConfig
from ..memory_pool import PinnedPool, MultiGPUPool
from ..weight_manager import WeightManager
from ..kv_cache import PagedKVPool
from ..model_executor import ModelExecutor
from ..request_manager import RequestManager, RequestState
from ..prefetch import PrefetchController
from ..scheduler import Scheduler

GPU = 0
DEVICE = f"cuda:{GPU}"
MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a short poem about the ocean:",
    "The speed of light is approximately",
    "Python is a programming language that",
    "The meaning of life according to philosophy is",
    "Water consists of two hydrogen atoms and",
    "Machine learning algorithms can be categorized as",
]


def benchmark_engine_single(model_id, prompt, max_tokens=100, n_runs=3):
    """Scenario A: single request latency."""
    model_cfg = ModelConfig(model_id=model_id, name="bench-model")
    engine_cfg = EngineConfig(models=[model_cfg], gpu_ids=[GPU], kv_cache_budget_gb=20)
    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], 40, 20)
    wm = WeightManager(engine_cfg, pinned, gpu)
    wm.load_model(model_cfg)
    model = wm.load_to_gpu("bench-model", GPU)
    state = wm.get_state("bench-model")
    config = state.hf_config
    tokenizer = state.tokenizer

    kv = PagedKVPool(
        config.num_hidden_layers, config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
        1024, 16, DEVICE, torch.bfloat16,
    )
    executor = ModelExecutor(model, kv, DEVICE)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    # Warmup
    kv.new_sequence(0)
    executor.generate(input_ids, 0, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id)
    kv.free_all()

    results = []
    for run in range(n_runs):
        kv.new_sequence(run + 1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Prefill
        logits = executor.prefill(input_ids, run + 1)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0

        # Decode
        next_tok = logits[:, -1, :].argmax(dim=-1).item()
        gen = [next_tok]
        t_decode_start = time.perf_counter()
        for _ in range(max_tokens - 1):
            if gen[-1] == tokenizer.eos_token_id:
                break
            tok_in = torch.tensor([[gen[-1]]], device=DEVICE)
            logits = executor.decode_step(tok_in, run + 1)
            gen.append(logits[:, -1, :].argmax(dim=-1).item())
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t_decode_start
        t_total = time.perf_counter() - t0

        n = len(gen)
        results.append({
            "tokens": n,
            "ttft_ms": t_prefill * 1000,
            "decode_s": t_decode,
            "tbt_ms": t_decode / max(n - 1, 1) * 1000,
            "tok_s": n / t_total,
            "total_s": t_total,
        })
        kv.free_all()

    # Cleanup
    wm.evict_from_gpu("bench-model")
    del executor, kv
    torch.cuda.empty_cache()
    return results


def benchmark_engine_batched(model_id, prompts, max_tokens=50, n_runs=2):
    """Scenario B: multiple concurrent requests, batched decode."""
    model_cfg = ModelConfig(model_id=model_id, name="bench-model")
    engine_cfg = EngineConfig(models=[model_cfg], gpu_ids=[GPU], kv_cache_budget_gb=20)
    sched_cfg = SchedulerConfig(max_consecutive_batches=128)

    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], 40, 20)
    wm = WeightManager(engine_cfg, pinned, gpu)
    rm = RequestManager()
    pc = PrefetchController(wm, rm)
    scheduler = Scheduler(engine_cfg, sched_cfg, wm, rm, pc, gpu)

    wm.load_model(model_cfg)
    state = wm.get_state("bench-model")
    tokenizer = state.tokenizer

    results = []
    for run in range(n_runs):
        reqs = []
        for p in prompts:
            tokens = tokenizer.encode(p)
            req = rm.add_request("bench-model", p, tokens, max_new_tokens=max_tokens)
            reqs.append(req)

        t0 = time.time()
        scheduler.run_background()
        while not all(r.state == RequestState.DONE for r in reqs):
            if time.time() - t0 > 60:
                break
            time.sleep(0.01)
        scheduler.stop()
        t_total = time.time() - t0

        total_tokens = sum(len(r.generated_tokens) for r in reqs)
        ttfts = [r.ttft * 1000 for r in reqs if r.ttft > 0]
        tbts = [r.tbt * 1000 for r in reqs if r.tbt > 0]

        results.append({
            "requests": len(reqs),
            "total_tokens": total_tokens,
            "wall_s": t_total,
            "agg_tok_s": total_tokens / t_total,
            "ttft_p50_ms": sorted(ttfts)[len(ttfts) // 2] if ttfts else 0,
            "tbt_p50_ms": sorted(tbts)[len(tbts) // 2] if tbts else 0,
        })

    wm.evict_from_gpu("bench-model")
    del scheduler
    torch.cuda.empty_cache()
    return results


def benchmark_vllm_single(model_id, prompt, max_tokens=100, n_runs=3):
    """Benchmark vLLM via its Python API for single request."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.5,
              enforce_eager=True, max_model_len=2048)
    params = SamplingParams(max_tokens=max_tokens, temperature=0)

    # Warmup
    llm.generate([prompt], params)

    results = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], params)
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        n = len(outputs[0].outputs[0].token_ids)
        results.append({
            "tokens": n,
            "total_s": t_total,
            "tok_s": n / t_total,
        })

    del llm
    torch.cuda.empty_cache()
    return results


def benchmark_vllm_batched(model_id, prompts, max_tokens=50, n_runs=2):
    """Benchmark vLLM with batched requests."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.5,
              enforce_eager=True, max_model_len=2048)
    params = SamplingParams(max_tokens=max_tokens, temperature=0)

    llm.generate(prompts[:2], params)  # warmup

    results = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        results.append({
            "requests": len(prompts),
            "total_tokens": total_tokens,
            "wall_s": t_total,
            "agg_tok_s": total_tokens / t_total,
        })

    del llm
    torch.cuda.empty_cache()
    return results


def print_comparison(label, engine_results, vllm_results, key):
    eng = sum(r[key] for r in engine_results) / len(engine_results)
    vlm = sum(r[key] for r in vllm_results) / len(vllm_results)
    ratio = eng / vlm if vlm > 0 else float("inf")
    print(f"  {label:25s}  Engine: {eng:8.1f}  vLLM: {vlm:8.1f}  Ratio: {ratio:.2f}x")


def main():
    print("=" * 70)
    print("BENCHMARK: Engine vs vLLM")
    print(f"Model: {MODEL_7B}")
    print("=" * 70)

    # Scenario A: Single request
    print("\n--- Scenario A: Single request, 100 tokens ---")
    prompt = PROMPTS[0]

    print("  Running engine...")
    eng_a = benchmark_engine_single(MODEL_7B, prompt, max_tokens=100, n_runs=3)

    print("  Running vLLM...")
    vlm_a = benchmark_vllm_single(MODEL_7B, prompt, max_tokens=100, n_runs=3)

    print_comparison("Tok/s", eng_a, vlm_a, "tok_s")
    print_comparison("Total time (s)", eng_a, vlm_a, "total_s")
    avg_ttft = sum(r["ttft_ms"] for r in eng_a) / len(eng_a)
    avg_tbt = sum(r["tbt_ms"] for r in eng_a) / len(eng_a)
    print(f"  Engine TTFT: {avg_ttft:.1f} ms, TBT: {avg_tbt:.1f} ms")

    # Scenario B: 8 concurrent requests
    print("\n--- Scenario B: 8 concurrent requests, 50 tokens each ---")

    print("  Running engine...")
    eng_b = benchmark_engine_batched(MODEL_7B, PROMPTS, max_tokens=50, n_runs=2)

    print("  Running vLLM...")
    vlm_b = benchmark_vllm_batched(MODEL_7B, PROMPTS, max_tokens=50, n_runs=2)

    print_comparison("Aggregate tok/s", eng_b, vlm_b, "agg_tok_s")
    print_comparison("Wall time (s)", eng_b, vlm_b, "wall_s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    eng_single = sum(r["tok_s"] for r in eng_a) / len(eng_a)
    vlm_single = sum(r["tok_s"] for r in vlm_a) / len(vlm_a)
    eng_batch = sum(r["agg_tok_s"] for r in eng_b) / len(eng_b)
    vlm_batch = sum(r["agg_tok_s"] for r in vlm_b) / len(vlm_b)

    print(f"  Single request:  Engine {eng_single:.1f} tok/s  vs  vLLM {vlm_single:.1f} tok/s  ({eng_single/vlm_single:.2f}x)")
    print(f"  8 concurrent:    Engine {eng_batch:.1f} tok/s  vs  vLLM {vlm_batch:.1f} tok/s  ({eng_batch/vlm_batch:.2f}x)")

    target = 0.5  # ≤2× means we need ≥0.5× vLLM speed
    single_pass = eng_single / vlm_single >= target
    batch_pass = eng_batch / vlm_batch >= target
    print(f"\n  Target: ≤2× vLLM latency (≥0.5× throughput)")
    print(f"  Single: {'PASS' if single_pass else 'FAIL'} ({eng_single/vlm_single:.2f}x)")
    print(f"  Batched: {'PASS' if batch_pass else 'FAIL'} ({eng_batch/vlm_batch:.2f}x)")


if __name__ == "__main__":
    main()
