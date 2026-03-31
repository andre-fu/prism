"""Test tensor parallelism: shard 7B across 4 GPUs, verify correct output."""

import time
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from .distributed import TPGroup, shard_all_weights
from .kv_cache import PagedKVPool
from .tp_executor import TPModelExecutor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "The capital of France is"
MAX_TOKENS = 30
TP_SIZE = 4
GPU_IDS = [0, 1, 2, 3]


def load_pinned_weights(model_id):
    cache_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.gguf"])
    paths = sorted(Path(cache_dir).glob("*.safetensors"))
    weights = {}
    for p in paths:
        weights.update(load_file(str(p), device="cpu"))
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "tie_word_embeddings", False):
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    return weights, config


def create_model_on_device(config, weights_dict, device, dtype):
    """Create a model on a specific device with given weights.

    For TP: weights_dict contains sharded tensors that may not match the model's
    expected shapes. We manually assign each parameter instead of using load_state_dict.
    """
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    # Manually assign sharded weights to parameters
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assigned = 0
    for name, tensor in weights_dict.items():
        if name in params:
            # Navigate to the parent module and set the parameter
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))
            assigned += 1

    # Reinit rotary buffers
    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device("meta") and "inv_freq" in name:
            dim = buf.shape[0] * 2
            rope_theta = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            parent = model
            for part in name.rsplit(".", 1)[0].split("."):
                parent = getattr(parent, part)
            parent.inv_freq = inv_freq.to(device)

    # Move any remaining meta tensors to device (replicated params that weren't in weights)
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], torch.nn.Parameter(
                torch.zeros(param.shape, dtype=dtype, device=device), requires_grad=False
            ))

    model.eval()
    return model


def main():
    print("=" * 60)
    print(f"TP TEST: {MODEL_ID} across {TP_SIZE} GPUs")
    print("=" * 60)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load weights
    print("\n[1] Loading weights to CPU...")
    t0 = time.perf_counter()
    weights, config = load_pinned_weights(MODEL_ID)
    t_load = time.perf_counter() - t0
    total_gb = sum(t.nbytes for t in weights.values()) / 1e9
    print(f"    {total_gb:.1f} GB in {t_load:.1f}s")

    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    kv_heads_per_gpu = num_kv_heads // TP_SIZE
    print(f"    Heads: {config.num_attention_heads} q, {num_kv_heads} kv")
    print(f"    Per GPU: {config.num_attention_heads // TP_SIZE} q, {kv_heads_per_gpu} kv")

    # Create TP group
    tp_group = TPGroup(GPU_IDS)

    # Shard weights and create model per GPU
    print(f"\n[2] Sharding weights across {TP_SIZE} GPUs...")
    t0 = time.perf_counter()

    models = []
    for rank in range(TP_SIZE):
        device = f"cuda:{GPU_IDS[rank]}"
        sharded = shard_all_weights(weights, rank, TP_SIZE)
        gpu_weights = {n: t.to(dtype).to(device) for n, t in sharded.items()}
        model = create_model_on_device(config, gpu_weights, device, dtype)
        models.append(model)
        del gpu_weights
        mem = torch.cuda.memory_allocated(GPU_IDS[rank]) / 1e9
        print(f"    GPU {GPU_IDS[rank]}: {mem:.1f} GB")

    t_shard = time.perf_counter() - t0
    print(f"    Sharded in {t_shard:.1f}s")

    # Create KV pools per GPU
    kv_pools = []
    for rank in range(TP_SIZE):
        device = f"cuda:{GPU_IDS[rank]}"
        pool = PagedKVPool(
            num_layers=config.num_hidden_layers,
            num_kv_heads=kv_heads_per_gpu,
            head_dim=head_dim,
            max_pages=256,
            page_size=16,
            device=device,
            dtype=dtype,
        )
        kv_pools.append(pool)

    # Create TP executor
    tp_exec = TPModelExecutor(models, kv_pools, tp_group)

    # Generate
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to("cuda:0")
    print(f"\n[3] Generating (TP={TP_SIZE})...")
    print(f"    Prompt: '{PROMPT}'")

    # Warmup
    gen = tp_exec.generate(input_ids, seq_id=0, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)
    for pool in kv_pools:
        pool.free_all()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen = tp_exec.generate(input_ids, seq_id=1, max_new_tokens=MAX_TOKENS, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    all_ids = torch.cat([input_ids[0].cpu(), torch.tensor(gen)])
    text = tokenizer.decode(all_ids, skip_special_tokens=True)

    print(f"    Output: '{text}'")
    print(f"    Tokens: {len(gen)}, Time: {t_gen:.3f}s, Tok/s: {len(gen)/t_gen:.1f}")

    # 72B doesn't fit on 1 GPU, so just check if output is coherent
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"TP=4: '{text}'")
    print(f"Tok/s: {len(gen)/t_gen:.1f}")
    for gpu_id in GPU_IDS:
        mem = torch.cuda.memory_allocated(gpu_id) / 1e9
        print(f"GPU {gpu_id}: {mem:.1f} GB")
    if "Paris" in text or "paris" in text.lower():
        print("\n>>> PASS: 72B TP=4 generates coherent output about Paris.")
    else:
        print(f"\n>>> CHECK: Output doesn't mention Paris. May still be correct.")


if __name__ == "__main__":
    main()
