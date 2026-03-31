"""Full benchmark suite: single model, multi-model, swap overhead, TP, and vs vLLM.

Runs all scenarios and produces a summary report.
"""

import time
import torch
import gc
import sys

from ..config import ModelConfig, EngineConfig, SchedulerConfig
from ..memory_pool import PinnedPool, MultiGPUPool
from ..weight_manager import WeightManager
from ..kv_cache import PagedKVPool
from ..model_executor import ModelExecutor
from ..request_manager import RequestManager, RequestState
from ..prefetch import PrefetchController
from ..scheduler import Scheduler
from ..distributed import TPGroup, shard_all_weights
from ..tp_executor import TPModelExecutor
from ..test_tp import create_model_on_device

GPU = 0
ALL_GPUS = [0, 1, 2, 3]
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


def cleanup():
    gc.collect()
    for g in ALL_GPUS:
        with torch.cuda.device(g):
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(g)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =====================================================================
# Benchmark 1: Single model raw decode speed
# =====================================================================
def bench_single_model_decode():
    section("1. SINGLE MODEL DECODE SPEED (7B)")
    cleanup()

    model_cfg = ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="bench")
    engine_cfg = EngineConfig(models=[model_cfg], gpu_ids=[GPU])
    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], 40, 20)
    wm = WeightManager(engine_cfg, pinned, gpu)
    wm.load_model(model_cfg)
    model = wm.load_to_gpu("bench", GPU)
    state = wm.get_state("bench")
    config = state.hf_config
    tokenizer = state.tokenizer

    kv = PagedKVPool(config.num_hidden_layers, config.num_key_value_heads,
                     config.hidden_size // config.num_attention_heads,
                     2048, 16, f"cuda:{GPU}", torch.bfloat16)
    executor = ModelExecutor(model, kv, f"cuda:{GPU}")
    input_ids = tokenizer.encode(PROMPTS[0], return_tensors="pt").to(f"cuda:{GPU}")

    # Warmup
    kv.new_sequence(0)
    executor.generate(input_ids, 0, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id)
    kv.free_all()

    results = {}

    for batch_size in [1, 4, 8, 16, 32]:
        # Setup sequences
        for i in range(batch_size):
            kv.new_sequence(i + 100)
            executor.prefill(input_ids, i + 100)

        # Warmup batched
        if batch_size > 1:
            for _ in range(3):
                executor.batched_decode_step([100]*batch_size, list(range(100, 100+batch_size)))

        N = 200 if batch_size <= 8 else 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            if batch_size == 1:
                executor.decode_step(torch.tensor([[100]], device=f"cuda:{GPU}"), 100)
            else:
                executor.batched_decode_step([100]*batch_size, list(range(100, 100+batch_size)))
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        ms_per_step = t / N * 1000
        tok_s = batch_size * N / t
        results[batch_size] = {"ms": ms_per_step, "tok_s": tok_s}
        print(f"  Batch={batch_size:>3d}: {ms_per_step:6.2f} ms/step  {tok_s:8.1f} tok/s")
        kv.free_all()

    wm.evict_from_gpu("bench")
    cleanup()
    return results


# =====================================================================
# Benchmark 2: Model swap overhead
# =====================================================================
def bench_swap_overhead():
    section("2. MODEL SWAP OVERHEAD")
    cleanup()

    model_7b = ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="qwen-7b")
    model_14b = ModelConfig(model_id="Qwen/Qwen2.5-14B-Instruct", name="qwen-14b")
    engine_cfg = EngineConfig(models=[model_7b, model_14b], gpu_ids=[GPU], t0_budget_gb=30)
    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], 30, 10)
    wm = WeightManager(engine_cfg, pinned, gpu)
    wm.load_model(model_7b)
    wm.load_model(model_14b)

    results = {}

    # 7B load
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wm.load_to_gpu("qwen-7b", GPU)
    torch.cuda.synchronize()
    t_load_7b = time.perf_counter() - t0
    results["load_7b_s"] = t_load_7b
    print(f"  Load 7B:     {t_load_7b:.3f}s")

    # Evict 7B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wm.evict_from_gpu("qwen-7b")
    torch.cuda.synchronize()
    t_evict_7b = time.perf_counter() - t0
    results["evict_7b_s"] = t_evict_7b
    print(f"  Evict 7B:    {t_evict_7b:.3f}s")

    # 14B load
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wm.load_to_gpu("qwen-14b", GPU)
    torch.cuda.synchronize()
    t_load_14b = time.perf_counter() - t0
    results["load_14b_s"] = t_load_14b
    print(f"  Load 14B:    {t_load_14b:.3f}s")

    # Full swap: evict 14B + load 7B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wm.evict_from_gpu("qwen-14b")
    wm.load_to_gpu("qwen-7b", GPU)
    torch.cuda.synchronize()
    t_swap = time.perf_counter() - t0
    results["swap_14b_to_7b_s"] = t_swap
    print(f"  Swap 14B→7B: {t_swap:.3f}s")

    wm.evict_from_gpu("qwen-7b")
    cleanup()
    return results


# =====================================================================
# Benchmark 3: Multi-model scheduler throughput (oversubscribed)
# =====================================================================
def bench_multimodel_scheduler():
    section("3. MULTI-MODEL SCHEDULER (OVERSUBSCRIBED)")
    cleanup()

    model_7b = ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="qwen-7b")
    model_14b = ModelConfig(model_id="Qwen/Qwen2.5-14B-Instruct", name="qwen-14b")
    engine_cfg = EngineConfig(models=[model_7b, model_14b], gpu_ids=[GPU],
                              t0_budget_gb=30, kv_cache_budget_gb=10)
    sched_cfg = SchedulerConfig(max_consecutive_batches=32)

    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], 30, 10)
    wm = WeightManager(engine_cfg, pinned, gpu)
    rm = RequestManager()
    pc = PrefetchController(wm, rm, gpu)
    scheduler = Scheduler(engine_cfg, sched_cfg, wm, rm, pc, gpu)

    wm.load_model(model_7b)
    wm.load_model(model_14b)

    # Submit 8 interleaved requests
    requests = []
    for i, prompt in enumerate(PROMPTS):
        model_name = "qwen-7b" if i % 2 == 0 else "qwen-14b"
        state = wm.get_state(model_name)
        tokens = state.tokenizer.encode(prompt)
        req = rm.add_request(model_name, prompt, tokens, max_new_tokens=30)
        requests.append(req)

    t0 = time.time()
    scheduler.run_background()
    while not all(r.state == RequestState.DONE for r in requests):
        if time.time() - t0 > 120:
            print("  TIMEOUT!")
            break
        time.sleep(0.05)
    scheduler.stop()
    t_total = time.time() - t0

    total_tokens = sum(len(r.generated_tokens) for r in requests if r.state == RequestState.DONE)
    completed = sum(1 for r in requests if r.state == RequestState.DONE)
    ttfts = [r.ttft * 1000 for r in requests if r.ttft > 0]
    tbts = [r.tbt * 1000 for r in requests if r.tbt > 0]

    stats = scheduler.stats
    print(f"  Requests:        {completed}/{len(requests)} completed")
    print(f"  Total tokens:    {total_tokens}")
    print(f"  Wall time:       {t_total:.2f}s")
    print(f"  Aggregate tok/s: {total_tokens/t_total:.1f}")
    print(f"  TTFT p50:        {sorted(ttfts)[len(ttfts)//2]:.0f} ms" if ttfts else "  TTFT: N/A")
    print(f"  TBT p50:         {sorted(tbts)[len(tbts)//2]:.1f} ms" if tbts else "  TBT: N/A")
    print(f"  Evictions:       {stats.evictions}")
    print(f"  Prefetch hits:   {stats.prefetch_hits}")
    print(f"  Sync loads:      {stats.sync_loads}")

    cleanup()
    return {
        "total_tokens": total_tokens, "wall_s": t_total,
        "agg_tok_s": total_tokens / t_total,
        "evictions": stats.evictions, "prefetch_hits": stats.prefetch_hits,
    }


# =====================================================================
# Benchmark 4: TP=4 performance
# =====================================================================
def bench_tp4():
    section("4. TENSOR PARALLELISM (TP=4)")
    cleanup()

    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    for model_id, label in [
        ("Qwen/Qwen2.5-7B-Instruct", "7B"),
        ("Qwen/Qwen2.5-14B-Instruct", "14B"),
    ]:
        print(f"\n  --- {label} TP=4 ---")
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        cache_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.gguf"])
        paths = sorted(Path(cache_dir).glob("*.safetensors"))
        weights = {}
        for p in paths:
            weights.update(load_file(str(p), device="cpu"))
        if getattr(config, "tie_word_embeddings", False):
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        tp = TPGroup(ALL_GPUS)
        dtype = torch.bfloat16
        kv_heads_per_gpu = config.num_key_value_heads // 4
        head_dim = config.hidden_size // config.num_attention_heads

        models = []
        for rank in range(4):
            dev = f"cuda:{ALL_GPUS[rank]}"
            sharded = shard_all_weights(weights, rank, 4)
            gpu_w = {n: t.to(dtype).to(dev) for n, t in sharded.items()}
            model = create_model_on_device(config, gpu_w, dev, dtype)
            models.append(model)
            del gpu_w

        kv_pools = [PagedKVPool(config.num_hidden_layers, kv_heads_per_gpu, head_dim,
                                256, 16, f"cuda:{g}", dtype) for g in ALL_GPUS]
        executor = TPModelExecutor(models, kv_pools, tp)

        input_ids = tokenizer.encode(PROMPTS[0], return_tensors="pt").to("cuda:0")

        # Warmup
        executor.generate(input_ids, seq_id=0, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)
        for pool in kv_pools:
            pool.free_all()

        # Benchmark
        for pool in kv_pools:
            pool.new_sequence(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = executor.prefill(input_ids, 1)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0

        gen = [logits[:, -1, :].argmax(dim=-1).item()]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            token = torch.tensor([[gen[-1]]], device="cuda:0")
            logits = executor.decode_step(token, 1)
            gen.append(logits[:, -1, :].argmax(dim=-1).item())
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t0

        text = tokenizer.decode(gen, skip_special_tokens=True)
        mem = [torch.cuda.memory_allocated(g) / 1e9 for g in ALL_GPUS]

        print(f"  Output:     {text[:60]}")
        print(f"  Prefill:    {t_prefill*1000:.1f} ms")
        print(f"  Decode:     {t_decode/N*1000:.1f} ms/step, {N/t_decode:.1f} tok/s")
        print(f"  Memory:     {' | '.join(f'GPU{g}: {m:.1f}GB' for g, m in zip(ALL_GPUS, mem))}")

        del models, executor
        for pool in kv_pools:
            del pool
        cleanup()


# =====================================================================
# Benchmark 5: vs vLLM
# =====================================================================
def bench_vs_vllm():
    section("5. VS vLLM (7B, enforce_eager)")
    cleanup()

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    # Our engine — single request
    model_cfg = ModelConfig(model_id=model_id, name="bench")
    engine_cfg = EngineConfig(models=[model_cfg], gpu_ids=[GPU])
    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([GPU], 40, 20)
    wm = WeightManager(engine_cfg, pinned, gpu)
    wm.load_model(model_cfg)
    model = wm.load_to_gpu("bench", GPU)
    state = wm.get_state("bench")
    config = state.hf_config
    tokenizer = state.tokenizer

    kv = PagedKVPool(config.num_hidden_layers, config.num_key_value_heads,
                     config.hidden_size // config.num_attention_heads,
                     2048, 16, f"cuda:{GPU}", torch.bfloat16)
    executor = ModelExecutor(model, kv, f"cuda:{GPU}")
    input_ids = tokenizer.encode(PROMPTS[0], return_tensors="pt").to(f"cuda:{GPU}")

    # Warmup
    kv.new_sequence(0)
    executor.generate(input_ids, 0, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id)
    kv.free_all()

    # Benchmark: single request, 100 tokens
    kv.new_sequence(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen = executor.generate(input_ids, 1, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    t_engine_single = time.perf_counter() - t0
    engine_single_toks = len(gen) / t_engine_single

    # Benchmark: 8 concurrent
    kv.free_all()
    rm = RequestManager()
    pc = PrefetchController(wm, rm, gpu)
    sched_cfg = SchedulerConfig(max_consecutive_batches=128)
    sched = Scheduler(engine_cfg, sched_cfg, wm, rm, pc, gpu)
    wm.load_model(model_cfg)  # no-op, already loaded

    reqs = []
    for p in PROMPTS:
        tokens = tokenizer.encode(p)
        reqs.append(rm.add_request("bench", p, tokens, max_new_tokens=50))

    t0 = time.time()
    sched.run_background()
    while not all(r.state == RequestState.DONE for r in reqs):
        if time.time() - t0 > 60: break
        time.sleep(0.01)
    sched.stop()
    t_engine_batch = time.time() - t0
    engine_batch_tokens = sum(len(r.generated_tokens) for r in reqs)
    engine_batch_toks = engine_batch_tokens / t_engine_batch

    wm.evict_from_gpu("bench")
    cleanup()

    # vLLM
    print("  Loading vLLM...")
    # Force full GPU cleanup before vLLM
    gc.collect()
    torch.cuda.empty_cache()
    for g in ALL_GPUS:
        torch.cuda.synchronize(g)
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85,
              enforce_eager=True, max_model_len=2048, disable_log_stats=True)
    params = SamplingParams(max_tokens=100, temperature=0)

    llm.generate([PROMPTS[0]], params)  # warmup

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([PROMPTS[0]], params)
    torch.cuda.synchronize()
    t_vllm_single = time.perf_counter() - t0
    vllm_single_toks = len(out[0].outputs[0].token_ids) / t_vllm_single

    params50 = SamplingParams(max_tokens=50, temperature=0)
    llm.generate(PROMPTS[:2], params50)  # warmup

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, params50)
    torch.cuda.synchronize()
    t_vllm_batch = time.perf_counter() - t0
    vllm_batch_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    vllm_batch_toks = vllm_batch_tokens / t_vllm_batch

    del llm
    cleanup()

    print(f"\n  {'Metric':<25s} {'Engine':>10s} {'vLLM':>10s} {'Ratio':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'Single (tok/s)':<25s} {engine_single_toks:>10.1f} {vllm_single_toks:>10.1f} {engine_single_toks/vllm_single_toks:>8.2f}x")
    print(f"  {'Batch 8 (tok/s)':<25s} {engine_batch_toks:>10.1f} {vllm_batch_toks:>10.1f} {engine_batch_toks/vllm_batch_toks:>8.2f}x")
    print(f"  {'Single latency':<25s} {t_engine_single:>10.3f}s {t_vllm_single:>10.3f}s {t_engine_single/t_vllm_single:>8.2f}x")

    return {
        "engine_single": engine_single_toks,
        "vllm_single": vllm_single_toks,
        "engine_batch": engine_batch_toks,
        "vllm_batch": vllm_batch_toks,
    }


def main():
    print("=" * 70)
    print("  FULL BENCHMARK SUITE — Multi-Tenant GPU Inference Engine")
    print(f"  Hardware: 4×H100 80GB, PyTorch {torch.__version__}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    r1 = bench_single_model_decode()
    r2 = bench_swap_overhead()
    r3 = bench_multimodel_scheduler()
    bench_tp4()
    r5 = bench_vs_vllm()

    # Final summary
    section("SUMMARY")
    print(f"\n  Single model decode (7B):")
    for bs, r in r1.items():
        print(f"    Batch={bs:>3d}: {r['tok_s']:>8.1f} tok/s ({r['ms']:.1f} ms/step)")

    print(f"\n  Swap overhead:")
    print(f"    Load 7B:       {r2['load_7b_s']:.3f}s")
    print(f"    Load 14B:      {r2['load_14b_s']:.3f}s")
    print(f"    Evict:         {r2['evict_7b_s']:.3f}s")
    print(f"    Full swap:     {r2['swap_14b_to_7b_s']:.3f}s")

    print(f"\n  Multi-model oversubscribed (7B+14B, 8 requests):")
    print(f"    Aggregate:     {r3['agg_tok_s']:.1f} tok/s")
    print(f"    Evictions:     {r3['evictions']}")
    print(f"    Prefetch hits: {r3['prefetch_hits']}")

    print(f"\n  vs vLLM (7B, enforce_eager):")
    print(f"    Single:  {r5['engine_single']:.1f} vs {r5['vllm_single']:.1f} tok/s ({r5['engine_single']/r5['vllm_single']:.2f}x)")
    print(f"    Batch 8: {r5['engine_batch']:.1f} vs {r5['vllm_batch']:.1f} tok/s ({r5['engine_batch']/r5['vllm_batch']:.2f}x)")

    target = 0.5
    single_pass = r5['engine_single'] / r5['vllm_single'] >= target
    batch_pass = r5['engine_batch'] / r5['vllm_batch'] >= target
    print(f"\n  ≤2× vLLM target (≥0.5× throughput):")
    print(f"    Single:  {'PASS' if single_pass else 'FAIL'} ({r5['engine_single']/r5['vllm_single']:.2f}x)")
    print(f"    Batch 8: {'PASS' if batch_pass else 'FAIL'} ({r5['engine_batch']/r5['vllm_batch']:.2f}x)")


if __name__ == "__main__":
    main()
