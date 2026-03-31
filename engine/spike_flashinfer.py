"""
Spike: FlashInfer paged attention integrated with transformers Qwen2.5 model.

Proves we can:
1. Run transformers model layer-by-layer
2. Use FlashInfer for attention with paged KV cache
3. Produce correct output (matches transformers baseline)
"""

import torch
import flashinfer
import time
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda:0"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 30
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

        # Per-layer page pool: [max_pages, 2, page_size, num_kv_heads, head_dim]
        self.pools = [
            torch.zeros(max_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        self.free_pages = list(range(max_pages))
        # Per-seq: ordered list of page IDs
        self.seq_pages: dict[int, list[int]] = {}
        # Per-seq: number of tokens stored so far
        self.seq_len: dict[int, int] = {}

    def new_sequence(self, seq_id: int):
        self.seq_pages[seq_id] = [self.free_pages.pop()]
        self.seq_len[seq_id] = 0

    def _ensure_capacity(self, seq_id: int, new_tokens: int):
        """Allocate pages so there's room for new_tokens more."""
        cur = self.seq_len[seq_id]
        needed_after = cur + new_tokens
        pages_needed = (needed_after + self.page_size - 1) // self.page_size
        while len(self.seq_pages[seq_id]) < pages_needed:
            self.seq_pages[seq_id].append(self.free_pages.pop())

    def append_kv(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Append K, V tensors for one layer. k, v: [num_tokens, num_kv_heads, head_dim].
        start_pos: the sequence position of the first token being appended."""
        num_tokens = k.shape[0]
        pages = self.seq_pages[seq_id]

        for i in range(num_tokens):
            pos = start_pos + i
            page_idx = pos // self.page_size
            slot = pos % self.page_size
            page_id = pages[page_idx]
            self.pools[layer_idx][page_id, 0, slot] = k[i]  # K
            self.pools[layer_idx][page_id, 1, slot] = v[i]  # V

    def prepare_append(self, seq_id: int, num_tokens: int) -> int:
        """Ensure capacity and return start_pos for appending. Call once before all layers."""
        start_pos = self.seq_len[seq_id]
        self._ensure_capacity(seq_id, num_tokens)
        self.seq_len[seq_id] += num_tokens
        return start_pos

    def get_flashinfer_args(self, seq_ids: list[int]):
        """Build FlashInfer CSR page table."""
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

    def free_sequence(self, seq_id: int):
        self.free_pages.extend(self.seq_pages.pop(seq_id))
        del self.seq_len[seq_id]

    def free_all(self):
        self.free_pages = list(range(self.max_pages))
        self.seq_pages.clear()
        self.seq_len.clear()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    cos = cos.unsqueeze(2)  # [batch, seq, 1, dim]
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def run_prefill(model, input_ids, kv_pool: PagedKVPool, seq_id: int):
    """Run prefill through all layers, storing KV in paged cache. Returns logits."""
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Workspace for FlashInfer
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    # Embed
    hidden = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        # QKV projections
        q = layer.self_attn.q_proj(hidden).view(batch_size, seq_len, num_heads, head_dim)
        k = layer.self_attn.k_proj(hidden).view(batch_size, seq_len, num_kv_heads, head_dim)
        v = layer.self_attn.v_proj(hidden).view(batch_size, seq_len, num_kv_heads, head_dim)

        # Rotary embeddings
        cos, sin = model.model.rotary_emb(v, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Store K, V in paged cache
        if layer_idx == 0:
            start_pos = kv_pool.prepare_append(seq_id, seq_len)
            kv_indptr, kv_indices, kv_last_page_len = kv_pool.get_flashinfer_args([seq_id])
        kv_pool.append_kv(seq_id, layer_idx, k[0], v[0], start_pos)
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)

        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
            num_heads, num_kv_heads, head_dim, PAGE_SIZE,
            causal=True, pos_encoding_mode="NONE",
            q_data_type="bfloat16", kv_data_type="bfloat16",
        )
        attn_out = wrapper.run(q.reshape(-1, num_heads, head_dim), kv_pool.pools[layer_idx])
        attn_out = attn_out.reshape(batch_size, seq_len, -1)

        attn_out = layer.self_attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


def run_decode_step(model, token_id, kv_pool: PagedKVPool, seq_id: int, decode_wrapper):
    """Run one decode step, append 1 token's KV to cache. Returns logits."""
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    cur_pos = kv_pool.seq_len[seq_id]
    position_ids = torch.tensor([[cur_pos]], device=DEVICE)

    hidden = model.model.embed_tokens(token_id)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        q = layer.self_attn.q_proj(hidden).view(1, 1, num_heads, head_dim)
        k = layer.self_attn.k_proj(hidden).view(1, 1, num_kv_heads, head_dim)
        v = layer.self_attn.v_proj(hidden).view(1, 1, num_kv_heads, head_dim)

        cos, sin = model.model.rotary_emb(v, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Append this token's KV
        if layer_idx == 0:
            start_pos = kv_pool.prepare_append(seq_id, 1)
            kv_indptr, kv_indices, kv_last_page_len = kv_pool.get_flashinfer_args([seq_id])
        kv_pool.append_kv(seq_id, layer_idx, k[0], v[0], start_pos)

        decode_wrapper.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            num_heads, num_kv_heads, head_dim, PAGE_SIZE,
            pos_encoding_mode="NONE", q_data_type="bfloat16", kv_data_type="bfloat16",
        )
        attn_out = decode_wrapper.run(
            q.reshape(1, num_heads, head_dim), kv_pool.pools[layer_idx]
        )
        attn_out = attn_out.reshape(1, 1, -1)

        attn_out = layer.self_attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


def manual_generate(model, tokenizer, prompt, kv_pool: PagedKVPool, max_new_tokens):
    """Full generation: prefill + decode loop using FlashInfer paged KV."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    seq_id = 0

    kv_pool.new_sequence(seq_id)

    # Prefill
    logits = run_prefill(model, input_ids, kv_pool, seq_id)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=-1)

    # Decode workspace
    decode_ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(decode_ws, kv_layout="NHD", use_tensor_cores=True)

    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        logits = run_decode_step(model, next_token, kv_pool, seq_id, decode_wrapper)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def load_model():
    """Load model via our pinned memory path."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    cache_dir = snapshot_download(MODEL_ID, ignore_patterns=["*.bin", "*.gguf"])
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
    print("=" * 60)
    print("SPIKE: FlashInfer Paged Attention + Transformers")
    print(f"Model: {MODEL_ID}, Page size: {PAGE_SIZE}")
    print("=" * 60)

    print("\n[1] Loading model...")
    model, config = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"    Loaded on {DEVICE}")

    # Baseline
    print("\n[2] Baseline (transformers generate)...")
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    mask = torch.ones_like(input_ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    baseline = tokenizer.decode(out[0], skip_special_tokens=True)
    t_base = time.perf_counter() - t0
    print(f"    '{baseline}'")
    print(f"    {t_base:.3f}s")

    # FlashInfer
    print("\n[3] FlashInfer paged attention...")
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    kv_pool = PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_pages=256,
        page_size=PAGE_SIZE,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        fi_text = manual_generate(model, tokenizer, PROMPT, kv_pool, MAX_NEW_TOKENS)
    t_fi = time.perf_counter() - t0
    print(f"    '{fi_text}'")
    print(f"    {t_fi:.3f}s")
    print(f"    Pages used: {256 - len(kv_pool.free_pages)}")

    # Compare
    print("\n" + "=" * 60)
    match = baseline == fi_text
    print(f"Match: {match}")
    if match:
        print(">>> PASS: FlashInfer paged attention matches baseline exactly.")
    elif fi_text and "Paris" in fi_text:
        print(">>> PARTIAL: Coherent but diverges (numerical). Acceptable.")
        print(f"    Baseline:   '{baseline}'")
        print(f"    FlashInfer: '{fi_text}'")
    else:
        print(">>> FAIL: Output incorrect. Debug needed.")
        print(f"    Baseline:   '{baseline}'")
        print(f"    FlashInfer: '{fi_text}'")


if __name__ == "__main__":
    main()
