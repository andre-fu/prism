"""Test: oversubscribed TP=4 swap.

Loads 7B (TP=4) and 14B (TP=4) with a tight per-GPU budget.
Only one model fits at a time. Scheduler must evict across all 4 GPUs to swap.
"""

import time
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download

from .distributed import TPGroup, shard_all_weights
from .kv_cache import PagedKVPool
from .tp_executor import TPModelExecutor
from .test_tp import create_model_on_device

TP_SIZE = 4
GPU_IDS = [0, 1, 2, 3]
PROMPT = "The capital of France is"
MAX_TOKENS = 20


def load_tp_model(model_id, tp_group, dtype=torch.bfloat16):
    """Load, shard, and place a model across TP GPUs. Returns (models, config, tokenizer)."""
    cache_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.gguf"])
    paths = sorted(Path(cache_dir).glob("*.safetensors"))
    weights = {}
    for p in paths:
        weights.update(load_file(str(p), device="cpu"))
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "tie_word_embeddings", False):
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    models = []
    for rank in range(tp_group.tp_size):
        device = tp_group.devices[rank]
        sharded = shard_all_weights(weights, rank, tp_group.tp_size)
        gpu_w = {n: t.to(dtype).to(device) for n, t in sharded.items()}
        model = create_model_on_device(config, gpu_w, device, dtype)
        models.append(model)
        del gpu_w

    return models, config, tokenizer


def make_kv_pools(config, tp_group, dtype=torch.bfloat16):
    kv_heads = config.num_key_value_heads // tp_group.tp_size
    head_dim = config.hidden_size // config.num_attention_heads
    pools = []
    for gpu_id in tp_group.gpu_ids:
        pools.append(PagedKVPool(
            config.num_hidden_layers, kv_heads, head_dim, 128, 16,
            f"cuda:{gpu_id}", dtype,
        ))
    return pools


def free_tp_model(models):
    """Delete model shards and free GPU memory."""
    for m in models:
        del m
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("TP=4 SWAP TEST: 7B <-> 14B across 4 GPUs")
    print("=" * 60)

    tp = TPGroup(GPU_IDS)
    dtype = torch.bfloat16

    # Load model A: 7B TP=4
    print("\n[1] Loading 7B (TP=4)...")
    t0 = time.perf_counter()
    models_a, config_a, tok_a = load_tp_model("Qwen/Qwen2.5-7B-Instruct", tp, dtype)
    t_load_a = time.perf_counter() - t0
    mem_a = torch.cuda.memory_allocated(0) / 1e9
    print(f"    {mem_a:.1f} GB/GPU, loaded in {t_load_a:.1f}s")

    # Generate with model A
    kv_a = make_kv_pools(config_a, tp, dtype)
    exec_a = TPModelExecutor(models_a, kv_a, tp)
    input_ids = tok_a.encode(PROMPT, return_tensors="pt").to("cuda:0")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_a = exec_a.generate(input_ids, seq_id=0, max_new_tokens=MAX_TOKENS, eos_token_id=tok_a.eos_token_id)
    torch.cuda.synchronize()
    t_gen_a = time.perf_counter() - t0
    text_a = tok_a.decode(gen_a, skip_special_tokens=True)
    print(f"    Output: '{text_a}'")
    print(f"    {len(gen_a)} tokens, {t_gen_a:.3f}s, {len(gen_a)/t_gen_a:.1f} tok/s")

    # SWAP: Evict A, Load B
    print(f"\n[2] SWAP: Evict 7B -> Load 14B (TP=4)...")
    torch.cuda.synchronize()
    t_swap_start = time.perf_counter()

    # Evict A
    del exec_a
    for p in kv_a:
        del p
    free_tp_model(models_a)
    t_evict = time.perf_counter() - t_swap_start

    # Load B
    models_b, config_b, tok_b = load_tp_model("Qwen/Qwen2.5-14B-Instruct", tp, dtype)
    torch.cuda.synchronize()
    t_load_b = time.perf_counter() - t_swap_start - t_evict
    t_swap = time.perf_counter() - t_swap_start

    mem_b = torch.cuda.memory_allocated(0) / 1e9
    print(f"    Evict: {t_evict:.3f}s")
    print(f"    Load 14B: {t_load_b:.1f}s")
    print(f"    Total swap: {t_swap:.1f}s")
    print(f"    {mem_b:.1f} GB/GPU")

    # Generate with model B
    kv_b = make_kv_pools(config_b, tp, dtype)
    exec_b = TPModelExecutor(models_b, kv_b, tp)
    input_ids_b = tok_b.encode(PROMPT, return_tensors="pt").to("cuda:0")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_b = exec_b.generate(input_ids_b, seq_id=0, max_new_tokens=MAX_TOKENS, eos_token_id=tok_b.eos_token_id)
    torch.cuda.synchronize()
    t_gen_b = time.perf_counter() - t0
    text_b = tok_b.decode(gen_b, skip_special_tokens=True)
    print(f"    Output: '{text_b}'")
    print(f"    {len(gen_b)} tokens, {t_gen_b:.3f}s, {len(gen_b)/t_gen_b:.1f} tok/s")

    # SWAP BACK: Evict B, Load A
    print(f"\n[3] SWAP BACK: Evict 14B -> Load 7B (TP=4)...")
    torch.cuda.synchronize()
    t_swap_start = time.perf_counter()

    del exec_b
    for p in kv_b:
        del p
    free_tp_model(models_b)
    t_evict2 = time.perf_counter() - t_swap_start

    models_a2, _, _ = load_tp_model("Qwen/Qwen2.5-7B-Instruct", tp, dtype)
    torch.cuda.synchronize()
    t_load_a2 = time.perf_counter() - t_swap_start - t_evict2
    t_swap2 = time.perf_counter() - t_swap_start

    print(f"    Evict: {t_evict2:.3f}s")
    print(f"    Load 7B: {t_load_a2:.1f}s")
    print(f"    Total swap: {t_swap2:.1f}s")

    kv_a2 = make_kv_pools(config_a, tp, dtype)
    exec_a2 = TPModelExecutor(models_a2, kv_a2, tp)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_a2 = exec_a2.generate(input_ids, seq_id=0, max_new_tokens=MAX_TOKENS, eos_token_id=tok_a.eos_token_id)
    torch.cuda.synchronize()
    t_gen_a2 = time.perf_counter() - t0
    text_a2 = tok_a.decode(gen_a2, skip_special_tokens=True)
    print(f"    Output: '{text_a2}'")
    print(f"    Match first run: {text_a == text_a2}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"7B TP=4:  {len(gen_a)/t_gen_a:.1f} tok/s, {mem_a:.1f} GB/GPU")
    print(f"14B TP=4: {len(gen_b)/t_gen_b:.1f} tok/s, {mem_b:.1f} GB/GPU")
    print(f"Swap 7B->14B: evict {t_evict:.3f}s + load {t_load_b:.1f}s = {t_swap:.1f}s")
    print(f"Swap 14B->7B: evict {t_evict2:.3f}s + load {t_load_a2:.1f}s = {t_swap2:.1f}s")
    print(f"Deterministic: {text_a == text_a2}")


if __name__ == "__main__":
    main()
