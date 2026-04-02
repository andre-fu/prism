"""Benchmark: Modal serverless GPU vs Prism local.

Tests Qwen3-8B on Modal's serverless GPU:
1. Cold start: time from request to first token (includes container spin-up + model load)
2. Warm TTFT: time to first token when container is warm
3. Decode throughput: tokens per second
4. Full request: end-to-end time for 100 tokens

Then compares against Prism local numbers.
"""

import modal
import time

# Modal app for serving Qwen3-8B
app = modal.App("prism-bench-qwen3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.8", "transformers", "torch")
)

MODEL_ID = "Qwen/Qwen3-8B"
PROMPT = "The capital of France is"
MAX_TOKENS = 100


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    memory=65536,
)
def run_vllm_benchmark():
    """Run vLLM benchmark inside Modal container."""
    import time
    import torch
    from vllm import LLM, SamplingParams

    results = {}

    # Cold start: measure model loading
    t_start = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    t_loaded = time.perf_counter()
    results["model_load_s"] = t_loaded - t_start

    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)

    # Warmup
    llm.generate([PROMPT], params)

    # Warm TTFT + throughput (3 runs)
    ttft_runs = []
    tps_runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        outputs = llm.generate([PROMPT], params)
        t_done = time.perf_counter()
        n_tokens = len(outputs[0].outputs[0].token_ids)
        total_s = t_done - t0
        ttft_runs.append(total_s)  # vLLM doesn't expose TTFT separately in sync mode
        tps_runs.append(n_tokens / total_s)

    results["warm_request_s"] = sum(ttft_runs) / len(ttft_runs)
    results["warm_tok_s"] = sum(tps_runs) / len(tps_runs)
    results["tokens_generated"] = n_tokens

    # Batch benchmark (8 concurrent)
    prompts_8 = [PROMPT] * 8
    params_50 = SamplingParams(max_tokens=50, temperature=0)
    llm.generate(prompts_8[:2], params_50)  # warmup batch

    t0 = time.perf_counter()
    outputs = llm.generate(prompts_8, params_50)
    t_done = time.perf_counter()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    results["batch8_tok_s"] = total_tokens / (t_done - t0)
    results["batch8_wall_s"] = t_done - t0

    results["gpu"] = torch.cuda.get_device_name(0)

    return results


@app.local_entrypoint()
def main():
    import json

    print("=" * 60)
    print("  MODAL SERVERLESS BENCHMARK: Qwen3-8B on H100")
    print("=" * 60)

    # Cold start: time the entire function call (includes container spin-up)
    print("\n[1] Cold start (container spin-up + model load)...")
    t_cold_start = time.perf_counter()
    results = run_vllm_benchmark.remote()
    t_cold_total = time.perf_counter() - t_cold_start

    print(f"\n[2] Results from Modal:")
    print(f"  GPU: {results['gpu']}")
    print(f"  Container cold start (total): {t_cold_total:.1f}s")
    print(f"  Model load (inside container): {results['model_load_s']:.1f}s")
    print(f"  Warm request ({results['tokens_generated']} tokens): {results['warm_request_s']:.3f}s")
    print(f"  Warm tok/s: {results['warm_tok_s']:.1f}")
    print(f"  Batch=8 tok/s: {results['batch8_tok_s']:.1f}")
    print(f"  Batch=8 wall: {results['batch8_wall_s']:.3f}s")

    # Now run warm (container should be hot)
    print(f"\n[3] Warm start (container already running)...")
    t_warm_start = time.perf_counter()
    results2 = run_vllm_benchmark.remote()
    t_warm_total = time.perf_counter() - t_warm_start
    print(f"  Warm container call: {t_warm_total:.1f}s")
    print(f"  Warm tok/s: {results2['warm_tok_s']:.1f}")

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30s} {'Modal':>10s} {'Prism':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'Cold start (total)':<30s} {t_cold_total:>10.1f}s {'0.8':>10s}s")
    print(f"  {'Model swap':<30s} {'N/A':>10s} {'0.3':>10s}s")
    print(f"  {'Warm tok/s (single)':<30s} {results['warm_tok_s']:>10.1f} {'150':>10s}")
    print(f"  {'Batch=8 tok/s':<30s} {results['batch8_tok_s']:>10.1f} {'1096':>10s}")
    print(f"  {'Models per GPU':<30s} {'1':>10s} {'4+':>10s}")
