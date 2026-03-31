"""Test: load two models, swap between them, verify correct output."""

import time
import torch

from .config import ModelConfig, EngineConfig
from .memory_pool import PinnedPool, MultiGPUPool
from .weight_manager import WeightManager
from .kv_cache import PagedKVPool
from .model_executor import ModelExecutor

GPU = 0
DEVICE = f"cuda:{GPU}"
PROMPT = "The capital of France is"
MAX_TOKENS = 50


def make_kv_pool(hf_config, dtype, device, budget_gb=35.0, page_size=16):
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    bytes_per_page = 2 * page_size * num_kv_heads * head_dim * dtype.itemsize * hf_config.num_hidden_layers
    max_pages = int(budget_gb * 1e9 / bytes_per_page)
    return PagedKVPool(
        num_layers=hf_config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_pages=max_pages,
        page_size=page_size,
        device=device,
        dtype=dtype,
    )


def run_one(wm, model_name, kv_pool, prompt, max_tokens):
    """Load model, generate, return text and timing."""
    state = wm.get_state(model_name)
    tokenizer = state.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model = wm.load_to_gpu(model_name, GPU)
    executor = ModelExecutor(model, kv_pool, DEVICE)

    kv_pool.free_all()
    gen_ids = executor.generate(
        input_ids, seq_id=0, max_new_tokens=max_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    all_ids = torch.cat([input_ids[0], torch.tensor(gen_ids, device=DEVICE)])
    text = tokenizer.decode(all_ids, skip_special_tokens=True)
    return text, t_total, len(gen_ids)


def main():
    print("=" * 60)
    print("SWAP TEST: Two models, one GPU")
    print("=" * 60)

    # Config
    model_a = ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct", name="qwen-7b")
    model_b = ModelConfig(model_id="Qwen/Qwen2.5-14B-Instruct", name="qwen-14b")
    engine_cfg = EngineConfig(models=[model_a, model_b], gpu_ids=[GPU])

    # Pools
    pinned_pool = PinnedPool(budget_gb=engine_cfg.t1_budget_gb)
    gpu_pool = MultiGPUPool(engine_cfg.gpu_ids, engine_cfg.t0_budget_gb, engine_cfg.kv_cache_budget_gb)

    # Weight manager
    wm = WeightManager(engine_cfg, pinned_pool, gpu_pool)

    # Load both models into pinned RAM
    print("\n[1] Loading models to pinned RAM...")
    state_a = wm.load_model(model_a)
    state_b = wm.load_model(model_b)

    dtype = torch.bfloat16

    # --- Round 1: Model A ---
    print("\n[2] Serving Model A (7B)...")
    kv_a = make_kv_pool(state_a.hf_config, dtype, DEVICE)
    text_a, t_a, n_a = run_one(wm, "qwen-7b", kv_a, PROMPT, MAX_TOKENS)
    print(f"    Output: {text_a[:80]}...")
    print(f"    {n_a} tokens, {t_a:.3f}s, {n_a/t_a:.1f} tok/s")
    del kv_a

    # --- SWAP: Evict A, Load B ---
    print("\n[3] SWAP: Evict 7B, Load 14B...")
    torch.cuda.synchronize()
    t_swap_start = time.perf_counter()
    wm.evict_from_gpu("qwen-7b")
    kv_b = make_kv_pool(state_b.hf_config, dtype, DEVICE)
    text_b, t_b, n_b = run_one(wm, "qwen-14b", kv_b, PROMPT, MAX_TOKENS)
    torch.cuda.synchronize()
    t_swap_total = time.perf_counter() - t_swap_start
    print(f"    Output: {text_b[:80]}...")
    print(f"    {n_b} tokens, {t_b:.3f}s, {n_b/t_b:.1f} tok/s")
    print(f"    Total swap+generate: {t_swap_total:.3f}s")
    del kv_b

    # --- SWAP BACK: Evict B, Load A ---
    print("\n[4] SWAP BACK: Evict 14B, Load 7B...")
    torch.cuda.synchronize()
    t_swap_start = time.perf_counter()
    wm.evict_from_gpu("qwen-14b")
    kv_a2 = make_kv_pool(state_a.hf_config, dtype, DEVICE)
    text_a2, t_a2, n_a2 = run_one(wm, "qwen-7b", kv_a2, PROMPT, MAX_TOKENS)
    torch.cuda.synchronize()
    t_swap_total2 = time.perf_counter() - t_swap_start
    print(f"    Output: {text_a2[:80]}...")
    print(f"    {n_a2} tokens, {t_a2:.3f}s, {n_a2/t_a2:.1f} tok/s")
    print(f"    Match first run: {text_a == text_a2}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model A (7B):  {n_a/t_a:.1f} tok/s")
    print(f"Model B (14B): {n_b/t_b:.1f} tok/s")
    print(f"Swap A->B+gen: {t_swap_total:.3f}s")
    print(f"Swap B->A+gen: {t_swap_total2:.3f}s")
    print(f"Deterministic: {text_a == text_a2}")
    print(f"GPU memory:    {torch.cuda.memory_allocated(GPU)/1e9:.1f} GB")
    print(f"Pinned RAM:    {pinned_pool.used_gb:.1f} GB")


if __name__ == "__main__":
    main()
