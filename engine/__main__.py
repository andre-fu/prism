"""Engine CLI: load models and run inference."""

import argparse
import time
import torch

from .config import ModelConfig, EngineConfig
from .memory_pool import PinnedPool, MultiGPUPool
from .weight_manager import WeightManager
from .kv_cache import PagedKVPool
from .model_executor import ModelExecutor


def main():
    parser = argparse.ArgumentParser(description="Multi-model inference engine")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--name", default="", help="Model display name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--prompt", default="The capital of France is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Config
    model_cfg = ModelConfig(model_id=args.model, name=args.name or args.model.split("/")[-1], dtype=args.dtype)
    engine_cfg = EngineConfig(models=[model_cfg], gpu_ids=[args.gpu])

    # Memory pools
    pinned_pool = PinnedPool(budget_gb=engine_cfg.t1_budget_gb)
    gpu_pool = MultiGPUPool(engine_cfg.gpu_ids, engine_cfg.t0_budget_gb, engine_cfg.kv_cache_budget_gb)

    # Weight manager
    wm = WeightManager(engine_cfg, pinned_pool, gpu_pool)

    # Load model to pinned RAM
    state = wm.load_model(model_cfg)

    # Transfer to GPU
    t0 = time.perf_counter()
    model = wm.load_to_gpu(model_cfg.name, args.gpu)
    t_load = time.perf_counter() - t0

    # Create KV cache pool
    hf_config = state.hf_config
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    dtype = getattr(torch, args.dtype)

    # Calculate max pages from KV budget
    bytes_per_page = 2 * engine_cfg.kv_page_size * num_kv_heads * head_dim * dtype.itemsize * hf_config.num_hidden_layers
    max_pages = int(engine_cfg.kv_cache_budget_gb * 1e9 / bytes_per_page)
    print(f"[KV Cache] {max_pages} pages, {engine_cfg.kv_page_size} tokens/page, "
          f"{max_pages * engine_cfg.kv_page_size} max tokens")

    kv_pool = PagedKVPool(
        num_layers=hf_config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_pages=max_pages,
        page_size=engine_cfg.kv_page_size,
        device=device,
        dtype=dtype,
    )

    # Executor
    executor = ModelExecutor(model, kv_pool, device)

    # Tokenize
    tokenizer = state.tokenizer
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: '{args.prompt}' ({input_ids.shape[1]} tokens)")

    # Warmup
    print("Warming up...")
    gen_ids = executor.generate(input_ids, seq_id=0, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)
    kv_pool.free_all()

    # Benchmark
    print("Generating...\n")
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # Prefill
    kv_pool.new_sequence(seq_id=1)
    t_prefill_start = time.perf_counter()
    logits = executor.prefill(input_ids, seq_id=1)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t_prefill_start

    next_token = logits[:, -1, :].argmax(dim=-1)
    generated = [next_token.item()]

    # Decode
    t_decode_start = time.perf_counter()
    for _ in range(args.max_tokens - 1):
        if generated[-1] == tokenizer.eos_token_id:
            break
        token_input = torch.tensor([[generated[-1]]], device=device)
        logits = executor.decode_step(token_input, seq_id=1)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t_decode_start
    t_total = time.perf_counter() - t_start

    # Output
    all_ids = torch.cat([input_ids[0], torch.tensor(generated, device=device)])
    text = tokenizer.decode(all_ids, skip_special_tokens=True)

    n_gen = len(generated)
    n_decode = max(n_gen - 1, 1)

    print(f"Output: {text}")
    print()
    print(f"--- Performance ---")
    print(f"TTFT (prefill):    {t_prefill*1000:.1f} ms ({input_ids.shape[1]} tokens)")
    print(f"Decode:            {t_decode:.3f}s ({n_decode} steps)")
    print(f"TBT:               {t_decode/n_decode*1000:.1f} ms")
    print(f"Decode tok/s:      {n_decode/t_decode:.1f}")
    print(f"Total tok/s:       {n_gen/t_total:.1f}")
    print(f"KV pages used:     {kv_pool.pages_used}")
    print(f"GPU memory:        {torch.cuda.memory_allocated(args.gpu)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
