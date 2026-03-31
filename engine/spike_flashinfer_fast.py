"""
Optimized FlashInfer paged attention + transformers model.
Fixes from naive spike:
  1. Use flashinfer.append_paged_kv_cache kernel (not Python per-element writes)
  2. Cache plan() args across layers (same page table for all layers in a step)
  3. Minimize Python overhead in decode loop
"""

import torch
import flashinfer
import time
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 100
PAGE_SIZE = 16


class PagedKVPool:
    """Paged KV cache pool with FlashInfer integration."""

    def __init__(self, num_layers, num_kv_heads, head_dim, max_pages, page_size, device, dtype):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_pages = max_pages
        self.page_size = page_size
        self.device = device
        self.dtype = dtype

        # Per-layer page pool: [max_pages, 2, page_size, num_kv_heads, head_dim]
        self.pools = [
            torch.zeros(max_pages, 2, page_size, num_kv_heads, head_dim,
                        dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        self.free_pages = list(range(max_pages))
        self.seq_pages: dict[int, list[int]] = {}
        self.seq_len: dict[int, int] = {}

    def new_sequence(self, seq_id: int):
        self.seq_pages[seq_id] = [self.free_pages.pop()]
        self.seq_len[seq_id] = 0

    def _ensure_capacity(self, seq_id: int, new_tokens: int):
        needed_after = self.seq_len[seq_id] + new_tokens
        pages_needed = (needed_after + self.page_size - 1) // self.page_size
        while len(self.seq_pages[seq_id]) < pages_needed:
            self.seq_pages[seq_id].append(self.free_pages.pop())

    def prepare_append(self, seq_id: int, num_tokens: int):
        """Ensure capacity, update seq_len, return (old_len, flashinfer_args)."""
        old_len = self.seq_len[seq_id]
        self._ensure_capacity(seq_id, num_tokens)
        self.seq_len[seq_id] += num_tokens
        return old_len

    def get_flashinfer_args(self, seq_ids: list[int]):
        indptr = [0]
        indices = []
        last_page_lens = []
        for sid in seq_ids:
            pages = self.seq_pages[sid]
            slen = self.seq_len[sid]
            if slen == 0:
                indptr.append(indptr[-1])
                last_page_lens.append(0)
                continue
            num_pages = (slen + self.page_size - 1) // self.page_size
            indices.extend(pages[:num_pages])
            indptr.append(len(indices))
            last = slen % self.page_size
            last_page_lens.append(last if last > 0 else self.page_size)
        return (
            torch.tensor(indptr, dtype=torch.int32, device=self.device),
            torch.tensor(indices, dtype=torch.int32, device=self.device),
            torch.tensor(last_page_lens, dtype=torch.int32, device=self.device),
        )

    def free_all(self):
        self.free_pages = list(range(self.max_pages))
        self.seq_pages.clear()
        self.seq_len.clear()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class FlashInferExecutor:
    """Runs transformers model with FlashInfer paged attention."""

    def __init__(self, model, kv_pool: PagedKVPool):
        self.model = model
        self.kv_pool = kv_pool
        self.config = model.config
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.num_layers = self.config.num_hidden_layers

        # Pre-allocate FlashInfer workspaces (reuse across all calls)
        self.prefill_ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
        self.decode_ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_ws, kv_layout="NHD"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.decode_ws, kv_layout="NHD", use_tensor_cores=True
        )

    def prefill(self, input_ids: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run prefill. Returns logits for last token. input_ids: [1, seq_len]."""
        model = self.model
        seq_len = input_ids.shape[1]

        # Prepare KV cache pages
        old_len = self.kv_pool.prepare_append(seq_id, seq_len)
        kv_indptr, kv_indices, kv_last_page_len = self.kv_pool.get_flashinfer_args([seq_id])

        # Plan once — reuse across all layers
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
        self.prefill_wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
            self.num_heads, self.num_kv_heads, self.head_dim, PAGE_SIZE,
            causal=True, pos_encoding_mode="NONE",
            q_data_type="bfloat16", kv_data_type="bfloat16",
        )

        # For append_paged_kv_cache — seq_lens must be AFTER append (total length)
        append_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
        seq_lens_after = torch.tensor([old_len + seq_len], dtype=torch.int32, device=DEVICE)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens_after, seq_len
        )

        hidden = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(old_len, old_len + seq_len, device=DEVICE).unsqueeze(0)

        for layer_idx, layer in enumerate(model.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            q = layer.self_attn.q_proj(hidden).view(1, seq_len, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden).view(1, seq_len, self.num_kv_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden).view(1, seq_len, self.num_kv_heads, self.head_dim)

            cos, sin = model.model.rotary_emb(v, position_ids)
            q, k = apply_rotary_emb(q, k, cos, sin)

            # Append KV using FlashInfer kernel (fast!)
            flashinfer.append_paged_kv_cache(
                k.reshape(-1, self.num_kv_heads, self.head_dim),
                v.reshape(-1, self.num_kv_heads, self.head_dim),
                batch_indices, positions,
                self.kv_pool.pools[layer_idx],
                kv_indices, kv_indptr, kv_last_page_len,
                kv_layout="NHD",
            )

            # Attention via FlashInfer (plan already done, reused across layers)
            attn_out = self.prefill_wrapper.run(
                q.reshape(-1, self.num_heads, self.head_dim),
                self.kv_pool.pools[layer_idx],
            )
            attn_out = layer.self_attn.o_proj(attn_out.reshape(1, seq_len, -1))

            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden[:, -1:, :])
        return logits

    def decode_step(self, token_id: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run one decode step. token_id: [1, 1]. Returns logits."""
        model = self.model
        cur_pos = self.kv_pool.seq_len[seq_id]

        # Prepare KV cache
        old_len = self.kv_pool.prepare_append(seq_id, 1)
        kv_indptr, kv_indices, kv_last_page_len = self.kv_pool.get_flashinfer_args([seq_id])

        # Plan decode once — reuse across layers
        self.decode_wrapper.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            self.num_heads, self.num_kv_heads, self.head_dim, PAGE_SIZE,
            pos_encoding_mode="NONE",
            q_data_type="bfloat16", kv_data_type="bfloat16",
        )

        # For append — seq_lens must be AFTER append
        append_indptr = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        seq_lens_after = torch.tensor([old_len + 1], dtype=torch.int32, device=DEVICE)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens_after, 1
        )

        position_ids = torch.tensor([[old_len]], device=DEVICE)
        hidden = model.model.embed_tokens(token_id)

        for layer_idx, layer in enumerate(model.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            q = layer.self_attn.q_proj(hidden).view(1, 1, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden).view(1, 1, self.num_kv_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden).view(1, 1, self.num_kv_heads, self.head_dim)

            cos, sin = model.model.rotary_emb(v, position_ids)
            q, k = apply_rotary_emb(q, k, cos, sin)

            # Append KV
            flashinfer.append_paged_kv_cache(
                k.reshape(1, self.num_kv_heads, self.head_dim),
                v.reshape(1, self.num_kv_heads, self.head_dim),
                batch_indices, positions,
                self.kv_pool.pools[layer_idx],
                kv_indices, kv_indptr, kv_last_page_len,
                kv_layout="NHD",
            )

            # Attention
            attn_out = self.decode_wrapper.run(
                q.reshape(1, self.num_heads, self.head_dim),
                self.kv_pool.pools[layer_idx],
            )
            attn_out = layer.self_attn.o_proj(attn_out.reshape(1, 1, -1))

            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = model.model.norm(hidden)
        return model.lm_head(hidden)

    def generate(self, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, dict]:
        """Full generation with timing breakdown."""
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        seq_id = 0
        self.kv_pool.new_sequence(seq_id)

        # Prefill
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        logits = self.prefill(input_ids, seq_id)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t_prefill_start

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token.item()]

        # Decode
        torch.cuda.synchronize()
        t_decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break
            logits = self.decode_step(next_token, seq_id)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t_decode_start

        n_prompt = input_ids.shape[1]
        n_generated = len(generated)
        all_ids = torch.cat([input_ids[0], torch.tensor(generated, device=DEVICE)])
        text = tokenizer.decode(all_ids, skip_special_tokens=True)

        stats = {
            "prompt_tokens": n_prompt,
            "generated_tokens": n_generated,
            "ttft": t_prefill,
            "decode_time": t_decode,
            "tbt": t_decode / max(n_generated - 1, 1),
            "decode_tok_s": (n_generated - 1) / t_decode if t_decode > 0 else 0,
            "total_time": t_prefill + t_decode,
            "total_tok_s": n_generated / (t_prefill + t_decode),
        }
        return text, stats


def load_model(model_id):
    config = AutoConfig.from_pretrained(model_id)
    cache_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.gguf"])
    paths = sorted(Path(cache_dir).glob("*.safetensors"))
    weights = {}
    for p in paths:
        weights.update(load_file(str(p), device="cpu"))
    if config.tie_word_embeddings:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    gpu_w = {n: t.to(DEVICE) for n, t in weights.items()}
    model.load_state_dict(gpu_w, strict=False, assign=True)
    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device("meta") and "inv_freq" in name:
            dim = buf.shape[0] * 2
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            parent = model
            for part in name.rsplit(".", 1)[0].split("."):
                parent = getattr(parent, part)
            parent.inv_freq = inv_freq.to(DEVICE)
    model.to(DEVICE)
    model.eval()
    del weights, gpu_w
    torch.cuda.empty_cache()
    return model, config


def main():
    print(f"Model: {MODEL_ID}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.perf_counter()
    model, config = load_model(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    t_load = time.perf_counter() - t0
    print(f"Loaded in {t_load:.1f}s")

    mem_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"GPU memory: {mem_gb:.1f} GB")
    print()

    # Baseline: transformers generate
    print("=== Baseline: transformers generate() ===")
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    mask = torch.ones_like(input_ids)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, attention_mask=mask, max_new_tokens=5,
                           do_sample=False, pad_token_id=tokenizer.eos_token_id)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    t_baseline = time.perf_counter() - t0
    n_gen_baseline = out.shape[1] - input_ids.shape[1]
    baseline_text = tokenizer.decode(out[0], skip_special_tokens=True)
    baseline_toks = n_gen_baseline / t_baseline

    print(f"Output: {baseline_text[:100]}...")
    print(f"Tokens: {n_gen_baseline}, Time: {t_baseline:.3f}s, Tok/s: {baseline_toks:.1f}")
    print()

    # FlashInfer
    print("=== FlashInfer paged attention ===")
    kv_pool = PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_pages=512,
        page_size=PAGE_SIZE,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    executor = FlashInferExecutor(model, kv_pool)

    # Warmup
    with torch.no_grad():
        executor.generate(tokenizer, PROMPT, max_new_tokens=5)
    kv_pool.free_all()

    # Benchmark
    with torch.no_grad():
        fi_text, stats = executor.generate(tokenizer, PROMPT, MAX_NEW_TOKENS)

    print(f"Output: {fi_text[:100]}...")
    print(f"Tokens: {stats['generated_tokens']}")
    print(f"TTFT (prefill):  {stats['ttft']*1000:.1f} ms")
    print(f"Decode time:     {stats['decode_time']:.3f}s")
    print(f"TBT (per token): {stats['tbt']*1000:.1f} ms")
    print(f"Decode tok/s:    {stats['decode_tok_s']:.1f}")
    print(f"Total tok/s:     {stats['total_tok_s']:.1f}")
    print()

    # Summary
    print("=== Summary ===")
    print(f"{'Metric':<20} {'Baseline':>12} {'FlashInfer':>12} {'Ratio':>8}")
    print(f"{'Total tok/s':<20} {baseline_toks:>12.1f} {stats['total_tok_s']:>12.1f} {stats['total_tok_s']/baseline_toks:>8.2f}x")
    print(f"{'Total time':<20} {t_baseline:>12.3f}s {stats['total_time']:>12.3f}s")

    pages_used = kv_pool.max_pages - len(kv_pool.free_pages)
    print(f"\nKV pages used: {pages_used}/{kv_pool.max_pages}")
    print(f"GPU memory: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
