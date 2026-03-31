"""Realistic swap benchmark: 7B ↔ 14B with real workloads.

Simulates actual production usage:
  - Long prompts (500+ tokens) — realistic prefill cost
  - Multi-turn conversations — KV cache accumulates
  - Interleaved requests between two different-sized models
  - Measures every phase: swap, prefill, decode, total
"""

import time
import torch
from transformers import AutoConfig, AutoTokenizer

from ..config import ModelConfig, EngineConfig, SchedulerConfig
from ..memory_pool import PinnedPool, MultiGPUPool
from ..weight_manager import WeightManager
from ..request_manager import RequestManager, RequestState
from ..scheduler_v2 import SchedulerV2

GPU = 0  # for memory reporting

# Realistic long prompts (will be ~200-500 tokens after tokenization)
LONG_PROMPTS = {
    "legal": """Review the following contract clause and identify potential issues:

INDEMNIFICATION. The Service Provider shall indemnify, defend, and hold harmless the Client,
its officers, directors, employees, agents, and affiliates from and against any and all claims,
damages, losses, costs, and expenses (including reasonable attorneys' fees) arising out of or
relating to: (a) any breach of this Agreement by the Service Provider; (b) any negligent or
wrongful act or omission of the Service Provider in connection with the performance of the
Services; (c) any violation of applicable laws or regulations by the Service Provider; or
(d) any infringement or misappropriation of any intellectual property right by the Service
Provider or the Deliverables. The Service Provider's obligation to indemnify shall survive
the termination or expiration of this Agreement for a period of three (3) years. The Client
shall provide prompt written notice of any claim and shall cooperate with the Service Provider
in the defense thereof. The Service Provider shall have the right to control the defense of
any such claim, provided that the Client shall have the right to participate in the defense
at its own expense. This indemnification shall not apply to claims arising solely from the
Client's negligence or willful misconduct. Analyze this clause for completeness and suggest improvements.""",

    "medical": """Patient presents with the following clinical history and examination findings.
Please provide a differential diagnosis and recommended workup:

Chief Complaint: Progressive shortness of breath over 3 weeks
History of Present Illness: 58-year-old male with progressive dyspnea on exertion, initially
noticed when climbing stairs, now occurring with minimal activity. Associated with dry cough,
especially at night. Denies chest pain, hemoptysis, fever, or weight loss. Reports bilateral
ankle swelling for the past week. Orthopnea present - sleeps with 3 pillows. Occasional
paroxysmal nocturnal dyspnea.
Past Medical History: Hypertension (15 years), Type 2 Diabetes (8 years), Hyperlipidemia,
Former smoker (30 pack-years, quit 5 years ago)
Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily
Physical Examination: BP 148/92, HR 96, RR 22, O2 Sat 94% on room air, Temp 37.1C
Cardiovascular: Irregular rhythm, S3 gallop present, JVD to 10cm
Pulmonary: Bilateral basal crackles, dullness to percussion at bases
Extremities: 2+ pitting edema bilateral lower extremities
Provide differential diagnosis ranked by probability and recommended diagnostic workup.""",

    "code": """Review this Python codebase architecture and suggest improvements for scalability:

The system is a real-time data pipeline that processes financial market data. It consists of:
1. A WebSocket ingestion layer that receives tick data from 50+ exchanges
2. A normalization service that converts exchange-specific formats to a unified schema
3. A time-series database (TimescaleDB) for storage with 1-second aggregation
4. A real-time analytics engine that computes VWAP, TWAP, and volatility metrics
5. A REST API serving historical and real-time data to trading algorithms
6. A monitoring dashboard with Grafana

Current issues:
- Ingestion layer drops messages during market open (peak: 500K messages/second)
- Database write latency spikes to 200ms during high volume
- Analytics engine falls behind real-time during volatile markets
- API response time exceeds 100ms for complex queries

The codebase uses asyncio for I/O, numpy for computation, and SQLAlchemy for database access.
Total team size is 4 engineers. What architectural changes would you recommend to handle
10x current load while maintaining sub-10ms latency for real-time data? Be specific about
technology choices and migration path.""",
}


def main():
    print("=" * 70)
    print("  REALISTIC SWAP BENCHMARK: 7B ↔ 14B")
    print(f"  Hardware: H100 80GB, PyTorch {torch.__version__}")
    print("=" * 70)

    # Setup
    model_7b = ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="7b-legal")
    model_14b = ModelConfig(model_id="Qwen/Qwen2.5-14B-Instruct", name="14b-medical")
    engine_cfg = EngineConfig(
        models=[model_7b, model_14b], gpu_ids=[0, 1, 2, 3], kv_cache_budget_gb=30,
    )
    sched_cfg = SchedulerConfig(max_consecutive_batches=256)

    pinned = PinnedPool(budget_gb=400)
    gpu = MultiGPUPool([0, 1, 2, 3], 60, 30)
    wm = WeightManager(engine_cfg, pinned, gpu)
    rm = RequestManager()

    print("\nLoading models to pinned RAM...")
    wm.load_model(model_7b)
    wm.load_model(model_14b)

    state_7b = wm.get_state("7b-legal")
    state_14b = wm.get_state("14b-medical")
    tok_7b = state_7b.tokenizer
    tok_14b = state_14b.tokenizer

    # Tokenize long prompts
    legal_tokens = tok_7b.encode(LONG_PROMPTS["legal"])
    medical_tokens = tok_14b.encode(LONG_PROMPTS["medical"])
    code_tokens = tok_7b.encode(LONG_PROMPTS["code"])

    print(f"  Legal prompt: {len(legal_tokens)} tokens")
    print(f"  Medical prompt: {len(medical_tokens)} tokens")
    print(f"  Code prompt: {len(code_tokens)} tokens")

    scheduler = SchedulerV2(engine_cfg, sched_cfg, wm, rm, gpu)

    # ================================================================
    # Test 1: Cold start with long prompt (7B)
    # ================================================================
    print(f"\n{'='*70}")
    print("  TEST 1: Cold start, 7B, long legal prompt ({} tokens)".format(len(legal_tokens)))
    print(f"{'='*70}")

    req1 = rm.add_request("7b-legal", LONG_PROMPTS["legal"], legal_tokens, max_new_tokens=200)
    t0 = time.time()
    scheduler.run_background()
    while req1.state != RequestState.DONE:
        if time.time() - t0 > 60: break
        time.sleep(0.01)
    scheduler.stop()
    t_total = time.time() - t0

    text = tok_7b.decode(req1.generated_tokens, skip_special_tokens=True)
    n_gen = len(req1.generated_tokens)
    decode_time = req1.done_time - req1.first_token_time if req1.first_token_time > 0 else 0
    tbt = decode_time / max(n_gen - 1, 1) * 1000

    print(f"  TTFT:          {req1.ttft*1000:.0f} ms")
    print(f"  Generated:     {n_gen} tokens")
    print(f"  Decode time:   {decode_time:.2f}s")
    print(f"  TBT:           {tbt:.1f} ms")
    print(f"  Decode tok/s:  {(n_gen-1)/decode_time:.1f}" if decode_time > 0 else "  Decode: N/A")
    print(f"  Total time:    {t_total:.2f}s")
    print(f"  Output: {text[:100]}...")

    # ================================================================
    # Test 2: Swap 7B → 14B with long medical prompt
    # ================================================================
    print(f"\n{'='*70}")
    print("  TEST 2: Swap 7B→14B, long medical prompt ({} tokens)".format(len(medical_tokens)))
    print(f"{'='*70}")

    req2 = rm.add_request("14b-medical", LONG_PROMPTS["medical"], medical_tokens, max_new_tokens=200)
    t0 = time.time()
    scheduler.run_background()
    while req2.state != RequestState.DONE:
        if time.time() - t0 > 120: break
        time.sleep(0.01)
    scheduler.stop()
    t_total = time.time() - t0

    text = tok_14b.decode(req2.generated_tokens, skip_special_tokens=True)
    n_gen = len(req2.generated_tokens)
    decode_time = req2.done_time - req2.first_token_time if req2.first_token_time > 0 else 0
    tbt = decode_time / max(n_gen - 1, 1) * 1000
    swap_time = scheduler.stats.swap_time_ms

    print(f"  Swap time:     {swap_time:.0f} ms (7B→14B)")
    print(f"  TTFT:          {req2.ttft*1000:.0f} ms (includes swap + prefill)")
    print(f"  Generated:     {n_gen} tokens")
    print(f"  Decode time:   {decode_time:.2f}s")
    print(f"  TBT:           {tbt:.1f} ms")
    print(f"  Decode tok/s:  {(n_gen-1)/decode_time:.1f}" if decode_time > 0 else "  Decode: N/A")
    print(f"  Total time:    {t_total:.2f}s")
    print(f"  Output: {text[:100]}...")

    # ================================================================
    # Test 3: Swap back 14B → 7B with code review prompt
    # ================================================================
    print(f"\n{'='*70}")
    print("  TEST 3: Swap 14B→7B, code review prompt ({} tokens)".format(len(code_tokens)))
    print(f"{'='*70}")

    old_swap_ms = scheduler.stats.swap_time_ms
    req3 = rm.add_request("7b-legal", LONG_PROMPTS["code"], code_tokens, max_new_tokens=200)
    t0 = time.time()
    scheduler.run_background()
    while req3.state != RequestState.DONE:
        if time.time() - t0 > 60: break
        time.sleep(0.01)
    scheduler.stop()
    t_total = time.time() - t0

    text = tok_7b.decode(req3.generated_tokens, skip_special_tokens=True)
    n_gen = len(req3.generated_tokens)
    decode_time = req3.done_time - req3.first_token_time if req3.first_token_time > 0 else 0
    tbt = decode_time / max(n_gen - 1, 1) * 1000
    swap_time = scheduler.stats.swap_time_ms - old_swap_ms

    print(f"  Swap time:     {swap_time:.0f} ms (14B→7B)")
    print(f"  TTFT:          {req3.ttft*1000:.0f} ms")
    print(f"  Generated:     {n_gen} tokens")
    print(f"  Decode time:   {decode_time:.2f}s")
    print(f"  TBT:           {tbt:.1f} ms")
    print(f"  Decode tok/s:  {(n_gen-1)/decode_time:.1f}" if decode_time > 0 else "  Decode: N/A")
    print(f"  Total time:    {t_total:.2f}s")
    print(f"  Output: {text[:100]}...")

    # ================================================================
    # Test 4: Interleaved concurrent — 3 requests to each model
    # ================================================================
    print(f"\n{'='*70}")
    print("  TEST 4: Interleaved — 3 requests each, long prompts")
    print(f"{'='*70}")

    old_stats_prefetch = scheduler.stats.prefetch_hits
    reqs = []
    for i, (mn, prompt, tokens, tok) in enumerate([
        ("7b-legal", LONG_PROMPTS["legal"], legal_tokens, tok_7b),
        ("14b-medical", LONG_PROMPTS["medical"], medical_tokens, tok_14b),
        ("7b-legal", LONG_PROMPTS["code"], code_tokens, tok_7b),
        ("14b-medical", LONG_PROMPTS["legal"], tok_14b.encode(LONG_PROMPTS["legal"]), tok_14b),
        ("7b-legal", LONG_PROMPTS["medical"], tok_7b.encode(LONG_PROMPTS["medical"]), tok_7b),
        ("14b-medical", LONG_PROMPTS["code"], tok_14b.encode(LONG_PROMPTS["code"]), tok_14b),
    ]):
        reqs.append(rm.add_request(mn, prompt[:50], tokens, max_new_tokens=100))

    t0 = time.time()
    scheduler.run_background()
    while not all(r.state == RequestState.DONE for r in reqs):
        if time.time() - t0 > 120: break
        time.sleep(0.05)
    scheduler.stop()
    t_total = time.time() - t0

    completed = [r for r in reqs if r.state == RequestState.DONE]
    errored = [r for r in reqs if r.error]
    total_tokens = sum(len(r.generated_tokens) for r in completed)
    ttfts = sorted([r.ttft * 1000 for r in completed if r.ttft > 0])
    prompt_sizes = [len(r.prompt_tokens) for r in reqs]
    prefetch_new = scheduler.stats.prefetch_hits - old_stats_prefetch

    print(f"  Requests:      {len(completed)}/{len(reqs)} completed, {len(errored)} errors")
    print(f"  Prompt sizes:  {prompt_sizes}")
    print(f"  Total tokens:  {total_tokens} generated")
    print(f"  Wall time:     {t_total:.2f}s")
    print(f"  Aggregate:     {total_tokens/t_total:.1f} tok/s")
    if ttfts:
        print(f"  TTFT p50:      {ttfts[len(ttfts)//2]:.0f} ms")
        print(f"  TTFT p95:      {ttfts[int(len(ttfts)*0.95)]:.0f} ms")
        print(f"  TTFT min/max:  {ttfts[0]:.0f} / {ttfts[-1]:.0f} ms")
    print(f"  Prefetch hits: {prefetch_new}")

    # Per-model breakdown
    for mn in ["7b-legal", "14b-medical"]:
        model_reqs = [r for r in completed if r.model_name == mn]
        tok = sum(len(r.generated_tokens) for r in model_reqs)
        ttft_list = sorted([r.ttft*1000 for r in model_reqs if r.ttft > 0])
        avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
        print(f"    {mn}: {len(model_reqs)} reqs, {tok} tok, avg TTFT={avg_ttft:.0f}ms")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    stats = scheduler.stats
    print(f"  Total swaps:      {stats.sync_swaps} sync, {stats.prefetch_hits} prefetch")
    print(f"  Avg swap time:    {stats.swap_time_ms/max(stats.sync_swaps,1):.0f} ms")
    print(f"  Total tokens:     {stats.tokens_generated}")
    print(f"  GPU memory:       {torch.cuda.memory_allocated(GPU)/1e9:.1f} GB")

    scheduler.cleanup()
    print(f"  GPU after cleanup: {torch.cuda.memory_allocated(GPU)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
