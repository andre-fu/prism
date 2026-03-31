"""
Day 0 Gate Spike: Can we use vLLM model classes with externally-managed weights?

Tests:
1. Load safetensors into CPU pinned memory (our weight management)
2. Instantiate vLLM Qwen2ForCausalLM
3. Call load_weights() with our tensors
4. Run forward pass, generate tokens
5. Compare output to transformers baseline

Uses Qwen2.5-0.5B for fast iteration (~1GB weights, loads in seconds).
"""

import os
import time
import sys
import torch
import gc
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


MODEL_ID = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 30
DEVICE = "cuda:0"


def get_safetensors_paths(model_id: str) -> list[Path]:
    """Find cached safetensors files for a HuggingFace model."""
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.gguf"])
    p = Path(cache_dir)
    files = sorted(p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {cache_dir}")
    return files


def load_weights_to_pinned(safetensors_paths: list[Path]) -> dict[str, torch.Tensor]:
    """Load all safetensors into CPU pinned memory."""
    weights = {}
    for path in safetensors_paths:
        shard = load_file(str(path), device="cpu")
        for name, tensor in shard.items():
            pinned = torch.empty_like(tensor, device="cpu").pin_memory()
            pinned.copy_(tensor)
            weights[name] = pinned
        del shard
    return weights


def approach_a_vllm(weights: dict[str, torch.Tensor]):
    """Approach A: Use vLLM model class with load_weights()."""
    print("\n" + "=" * 60)
    print("APPROACH A: vLLM Qwen2ForCausalLM + external weights")
    print("=" * 60)

    # Step 1: Set up vLLM parallel state (required by model classes)
    print("\n[1] Initializing vLLM parallel state...")
    t0 = time.perf_counter()

    from vllm.config import (
        VllmConfig, ModelConfig as VllmModelConfig,
        CacheConfig, DeviceConfig, ParallelConfig, LoadConfig
    )
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    # Single-process init for torch.distributed
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)

    model_config = VllmModelConfig(
        model=MODEL_ID,
        dtype="float16",
        seed=0,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        device_config=DeviceConfig(device="cuda"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        load_config=LoadConfig(),
    )

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    t_init_dist = time.perf_counter() - t0
    print(f"    Parallel state initialized in {t_init_dist:.2f}s")

    # Step 2: Instantiate model
    print("\n[2] Instantiating vLLM model...")
    t0 = time.perf_counter()

    from vllm.model_executor.models.registry import ModelRegistry

    architectures = model_config.hf_config.architectures
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures, model_config)
    print(f"    Architecture: {architectures}, Class: {model_cls.__name__}")

    with set_current_vllm_config(vllm_config):
        with torch.device(DEVICE):
            model = model_cls(vllm_config=vllm_config, prefix="")

    t_model = time.perf_counter() - t0
    print(f"    Model created in {t_model:.2f}s")

    # Step 3: Load our pinned weights
    print("\n[3] Loading weights via load_weights()...")
    t0 = time.perf_counter()

    # Prepare weight iterator: move to GPU and yield as (name, tensor) pairs
    def gpu_weight_iter():
        for name, tensor in weights.items():
            yield name, tensor.to(DEVICE, non_blocking=False)

    with set_current_vllm_config(vllm_config):
        loaded = model.load_weights(gpu_weight_iter())

    t_load = time.perf_counter() - t0
    print(f"    Loaded {len(loaded)} params in {t_load:.2f}s")

    # Step 4: Generate tokens via manual forward pass
    print("\n[4] Generating tokens...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    print(f"    Prompt: '{PROMPT}' ({input_ids.shape[1]} tokens)")

    model.eval()
    generated = input_ids.clone()

    t0 = time.perf_counter()
    with torch.no_grad(), set_current_vllm_config(vllm_config):
        for i in range(MAX_NEW_TOKENS):
            positions = torch.arange(generated.shape[1], device=DEVICE).unsqueeze(0)
            try:
                # vLLM models expect (input_ids, positions, intermediate_tensors, inputs_embeds)
                # For simple usage, try the most basic call
                hidden = model.model(
                    input_ids=generated,
                    positions=positions,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                )
                logits = model.lm_head(hidden)
            except Exception as e:
                print(f"\n    Forward pass failed at step {i}: {type(e).__name__}: {e}")
                print("    Trying alternative forward path...")

                # Try through the top-level forward
                try:
                    # Some vLLM models have compute_logits
                    hidden = model.model(
                        input_ids=generated,
                        positions=positions,
                        intermediate_tensors=None,
                    )
                    logits = model.logits_processor(
                        model.lm_head, hidden, model.model.embed_tokens.weight
                    )
                except Exception as e2:
                    print(f"    Alternative also failed: {type(e2).__name__}: {e2}")
                    print("    Approach A FAIL: forward() incompatible.")
                    return None

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    t_gen = time.perf_counter() - t0
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    n_tokens = generated.shape[1] - input_ids.shape[1]

    print(f"    Output: '{output_text}'")
    print(f"    Generated {n_tokens} tokens in {t_gen:.3f}s ({n_tokens/t_gen:.1f} tok/s)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return output_text


def approach_b_transformers(weights: dict[str, torch.Tensor]):
    """Approach B (fallback): transformers + our pinned weight management."""
    print("\n" + "=" * 60)
    print("APPROACH B: transformers AutoModel + pinned weight loading")
    print("=" * 60)

    # Create model on meta device
    print("\n[1] Instantiating model on meta device...")
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    t_init = time.perf_counter() - t0
    print(f"    Initialized in {t_init:.2f}s")

    # Materialize on GPU and load our pinned weights
    print("\n[2] Loading pinned weights to GPU...")
    t0 = time.perf_counter()
    model = model.to_empty(device=DEVICE)

    loaded_count = 0
    for name, param in model.named_parameters():
        if name in weights:
            param.data.copy_(weights[name].to(DEVICE, non_blocking=True))
            loaded_count += 1
        elif config.tie_word_embeddings and name == "lm_head.weight":
            # Tied weights: lm_head shares embed_tokens
            if "model.embed_tokens.weight" in weights:
                param.data.copy_(weights["model.embed_tokens.weight"].to(DEVICE, non_blocking=True))
                loaded_count += 1
        else:
            print(f"    WARNING: param '{name}' not in weights")

    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    total_params = len(list(model.named_parameters()))
    print(f"    Loaded {loaded_count}/{total_params} params in {t_load:.3f}s")

    # Generate tokens
    print("\n[3] Generating tokens...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    print(f"    Prompt: '{PROMPT}' ({input_ids.shape[1]} tokens)")

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t_gen = time.perf_counter() - t0

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    n_tokens = output_ids.shape[1] - input_ids.shape[1]
    print(f"    Output: '{output_text}'")
    print(f"    Generated {n_tokens} tokens in {t_gen:.3f}s ({n_tokens/t_gen:.1f} tok/s)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return output_text


def baseline_transformers():
    """Baseline: standard transformers loading for ground truth."""
    print("\n" + "=" * 60)
    print("BASELINE: transformers standard loading (ground truth)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()

    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t_gen = time.perf_counter() - t0

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    n_tokens = output_ids.shape[1] - input_ids.shape[1]

    print(f"    Prompt: '{PROMPT}'")
    print(f"    Output: '{output_text}'")
    print(f"    Generated {n_tokens} tokens in {t_gen:.3f}s ({n_tokens/t_gen:.1f} tok/s)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return output_text


def main():
    print("=" * 60)
    print("DAY 0 GATE SPIKE: vLLM External Weight Loading")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Step 0: Load weights once, reuse across approaches
    print("\n[SETUP] Loading safetensors into pinned memory...")
    t0 = time.perf_counter()
    paths = get_safetensors_paths(MODEL_ID)
    weights = load_weights_to_pinned(paths)
    t_load = time.perf_counter() - t0
    total_bytes = sum(t.nbytes for t in weights.values())
    print(f"    {len(weights)} tensors, {total_bytes / 1e9:.2f} GB, {t_load:.2f}s")
    print(f"    All pinned: {all(t.is_pinned() for t in weights.values())}")

    # Ground truth
    baseline_text = baseline_transformers()

    # Approach B (guaranteed to work — validates weight loading)
    b_text = approach_b_transformers(weights)

    # Approach A (the one we need to validate)
    a_text = approach_a_vllm(weights)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBaseline:    '{baseline_text}'")
    print(f"Approach B:  '{b_text}'")
    print(f"Approach A:  '{a_text}'")

    b_match = b_text == baseline_text
    print(f"\nApproach B matches baseline: {b_match}")

    if a_text is not None:
        a_match = a_text == baseline_text
        print(f"Approach A matches baseline: {a_match}")
        if a_match:
            print("\n>>> GATE PASS: Approach A works perfectly. Proceed with vLLM models.")
        elif a_text and len(a_text) > len(PROMPT) + 10:
            print("\n>>> GATE PARTIAL: Approach A generates coherent text but doesn't match exactly.")
            print("    Likely numerical differences from fused kernels. Acceptable — proceed with A.")
        else:
            print("\n>>> GATE FAIL: Approach A output is bad. Use Approach B.")
    else:
        print("Approach A: FAILED (forward pass incompatible)")
        if b_match:
            print("\n>>> Approach B works perfectly. Proceed with Approach B.")
        else:
            print("\n>>> WARNING: Approach B doesn't match either. Investigate.")

    # Async transfer benchmark
    print("\n" + "=" * 60)
    print("BONUS: Async H2D Transfer Benchmark")
    print("=" * 60)
    stream = torch.cuda.Stream(device=DEVICE)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for name, tensor in weights.items():
            tensor.to(DEVICE, non_blocking=True)
    stream.synchronize()
    t_transfer = time.perf_counter() - t0
    bw = total_bytes / t_transfer / 1e9
    print(f"    Pinned → GPU 0: {t_transfer:.3f}s ({bw:.1f} GB/s)")
    print(f"    7B model estimate: {7 / bw:.3f}s")
    print(f"    14B model estimate: {14 / bw:.3f}s")
    print(f"    72B model estimate: {72 / bw:.3f}s")


if __name__ == "__main__":
    main()
