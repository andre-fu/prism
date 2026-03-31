"""Model execution: layer-by-layer forward pass with FlashInfer paged attention."""

import torch
import flashinfer
from .kv_cache import PagedKVPool
from .fused_kernels import fused_rms_norm


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(q, k, cos, sin):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class ModelExecutor:
    """Runs inference through a transformers model using FlashInfer paged attention.

    The model is executed layer-by-layer so we can intercept attention and route
    it through FlashInfer's paged KV cache instead of the model's default attention.
    """

    def __init__(self, model: torch.nn.Module, kv_pool: PagedKVPool, device: str):
        self.model = model
        self.kv_pool = kv_pool
        self.device = device

        config = model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        # Pre-extract layer references to avoid repeated attribute lookups
        self._embed = model.model.embed_tokens
        self._norm = model.model.norm
        self._lm_head = model.lm_head
        self._rotary = model.model.rotary_emb
        # Pre-extract RMSNorm weights for fused kernel
        self._input_ln_weights = [l.input_layernorm.weight.data for l in model.model.layers]
        self._post_ln_weights = [l.post_attention_layernorm.weight.data for l in model.model.layers]
        self._final_norm_weight = model.model.norm.weight.data
        self._rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        self._layers_input_ln = [l.input_layernorm for l in model.model.layers]
        self._layers_post_ln = [l.post_attention_layernorm for l in model.model.layers]
        self._layers_q = [l.self_attn.q_proj for l in model.model.layers]
        self._layers_k = [l.self_attn.k_proj for l in model.model.layers]
        self._layers_v = [l.self_attn.v_proj for l in model.model.layers]
        self._layers_o = [l.self_attn.o_proj for l in model.model.layers]
        self._layers_mlp = [l.mlp for l in model.model.layers]

        # Pre-stack QKV weights for fused matmul (3 launches → 1)
        self._qkv_weights = []
        self._qkv_biases = []
        self._q_size = self.num_heads * self.head_dim
        self._kv_size = self.num_kv_heads * self.head_dim
        for l in model.model.layers:
            qkv_w = torch.cat([l.self_attn.q_proj.weight.data,
                               l.self_attn.k_proj.weight.data,
                               l.self_attn.v_proj.weight.data], dim=0)
            self._qkv_weights.append(qkv_w)
            # Handle biases (Qwen has them on QKV)
            q_bias = getattr(l.self_attn.q_proj, 'bias', None)
            k_bias = getattr(l.self_attn.k_proj, 'bias', None)
            v_bias = getattr(l.self_attn.v_proj, 'bias', None)
            if q_bias is not None:
                self._qkv_biases.append(torch.cat([q_bias.data, k_bias.data, v_bias.data]))
            else:
                self._qkv_biases.append(None)

        # Pre-stack gate+up weights for fused MLP matmul (2 launches → 1)
        self._gate_up_weights = []
        self._mlp_size = config.intermediate_size
        for l in model.model.layers:
            gu_w = torch.cat([l.mlp.gate_proj.weight.data,
                              l.mlp.up_proj.weight.data], dim=0)
            self._gate_up_weights.append(gu_w)
        self._layers_down = [l.mlp.down_proj for l in model.model.layers]
        self._layers_act = [l.mlp.act_fn for l in model.model.layers]

        # Raw weight tensors for F.linear (avoids nn.Module dispatch)
        self._o_weights = [l.self_attn.o_proj.weight.data for l in model.model.layers]
        self._o_biases = [getattr(l.self_attn.o_proj, 'bias', None) for l in model.model.layers]
        self._o_biases = [b.data if b is not None else None for b in self._o_biases]
        self._down_weights = [l.mlp.down_proj.weight.data for l in model.model.layers]
        self._down_biases = [getattr(l.mlp.down_proj, 'bias', None) for l in model.model.layers]
        self._down_biases = [b.data if b is not None else None for b in self._down_biases]

        # Pre-compute rotary embeddings for positions 0..max_seq_len
        max_pos = getattr(config, 'max_position_embeddings', 4096)
        max_pos = min(max_pos, 4096)  # Cap to save memory
        with torch.no_grad():
            dummy = torch.zeros(1, max_pos, self.hidden_size, device=device, dtype=torch.bfloat16)
            pos_range = torch.arange(max_pos, device=device).unsqueeze(0)
            cos, sin = self._rotary(dummy, pos_range)
        self._cos_cache = cos.unsqueeze(2)  # [1, max_pos, 1, head_dim]
        self._sin_cache = sin.unsqueeze(2)

        # Pre-allocate reusable tensors for single-token decode
        self._decode_append_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self._decode_seq_lens_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self._decode_pos_buf = torch.zeros(1, 1, dtype=torch.long, device=device)

    def prefill(self, input_ids: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run prefill phase. input_ids: [1, seq_len]. Returns logits: [1, 1, vocab]."""
        model = self.model
        kv = self.kv_pool
        seq_len = input_ids.shape[1]

        # Allocate KV pages and update page table
        old_len = kv.prepare_append(seq_id, seq_len)
        kv_indptr, kv_indices, kv_last_page_len = kv.build_page_table([seq_id])

        # Plan prefill attention once (reused across all layers)
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        kv.plan_prefill(qo_indptr, kv_indptr, kv_indices, kv_last_page_len, self.num_heads)

        # Compute append positions for KV write
        append_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        seq_lens_after = torch.tensor([old_len + seq_len], dtype=torch.int32, device=self.device)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens_after, seq_len
        )

        # Embedding
        hidden = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(old_len, old_len + seq_len, device=self.device).unsqueeze(0)

        # Layer-by-layer
        for layer_idx, layer in enumerate(model.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            # QKV projections
            q = layer.self_attn.q_proj(hidden).view(1, seq_len, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden).view(1, seq_len, self.num_kv_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden).view(1, seq_len, self.num_kv_heads, self.head_dim)

            # Rotary embeddings
            cos, sin = model.model.rotary_emb(v, position_ids)
            q, k = _apply_rotary_emb(q, k, cos, sin)

            # Write KV to paged cache
            kv.append_kv(
                layer_idx,
                k.reshape(-1, self.num_kv_heads, self.head_dim),
                v.reshape(-1, self.num_kv_heads, self.head_dim),
                kv_indptr, kv_indices, kv_last_page_len,
                batch_indices, positions,
            )

            # FlashInfer attention (plan already done, reused)
            attn_out = kv.run_prefill(layer_idx, q.reshape(-1, self.num_heads, self.head_dim))
            attn_out = layer.self_attn.o_proj(attn_out.reshape(1, seq_len, -1))

            # Residual + MLP
            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = model.model.norm(hidden)
        return model.lm_head(hidden[:, -1:, :])

    def decode_step(self, token_id: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run one decode step. token_id: [1, 1]. Returns logits: [1, 1, vocab].

        Maximally optimized: pre-computed rotary, raw weight tensors, fused norms.
        """
        kv = self.kv_pool
        nh = self.num_heads
        nkv = self.num_kv_heads
        hd = self.head_dim
        F = torch.nn.functional

        old_len = kv.prepare_append(seq_id, 1)
        kv_indptr, kv_indices, kv_last_page_len = kv.build_page_table([seq_id])
        kv.plan_decode(kv_indptr, kv_indices, kv_last_page_len, nh)

        self._decode_seq_lens_buf[0] = old_len + 1
        bi, pos = flashinfer.get_batch_indices_positions(
            self._decode_append_indptr, self._decode_seq_lens_buf, 1
        )

        hidden = self._embed(token_id)
        cu = self._cos_cache[:, old_len:old_len+1]
        su = self._sin_cache[:, old_len:old_len+1]

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, self._input_ln_weights[i], self._rms_eps)

            qkv = F.linear(hidden, self._qkv_weights[i], self._qkv_biases[i])
            q = qkv[:, :, :self._q_size].view(1, 1, nh, hd)
            k = qkv[:, :, self._q_size:self._q_size + self._kv_size].view(1, 1, nkv, hd)
            v = qkv[:, :, self._q_size + self._kv_size:].view(1, 1, nkv, hd)

            q = (q * cu) + (_rotate_half(q) * su)
            k = (k * cu) + (_rotate_half(k) * su)

            kv.append_kv(i, k.reshape(1, nkv, hd), v.reshape(1, nkv, hd),
                         kv_indptr, kv_indices, kv_last_page_len, bi, pos)

            attn = kv.run_decode(i, q.reshape(1, nh, hd))
            hidden = residual + F.linear(attn.reshape(1, 1, -1), self._o_weights[i], self._o_biases[i])

            residual = hidden
            hidden = fused_rms_norm(hidden, self._post_ln_weights[i], self._rms_eps)
            gu = F.linear(hidden, self._gate_up_weights[i])
            hidden = residual + F.linear(
                F.silu(gu[:, :, :self._mlp_size]) * gu[:, :, self._mlp_size:],
                self._down_weights[i], self._down_biases[i]
            )

        return self._lm_head(fused_rms_norm(hidden, self._final_norm_weight, self._rms_eps))

    def batched_decode_step(self, token_ids: list[int], seq_ids: list[int]) -> torch.Tensor:
        """Run one decode step for multiple sequences at once.

        token_ids: list of last token per sequence
        seq_ids: list of KV cache sequence IDs
        Returns logits: [batch_size, vocab_size]
        """
        model = self.model
        kv = self.kv_pool
        batch_size = len(seq_ids)

        # Prepare all sequences: allocate KV pages, build combined page table
        old_lens = []
        for sid in seq_ids:
            old_lens.append(kv.prepare_append(sid, 1))

        kv_indptr, kv_indices, kv_last_page_len = kv.build_page_table(seq_ids)

        # Plan decode for whole batch
        kv.plan_decode(kv_indptr, kv_indices, kv_last_page_len, self.num_heads)

        # Append positions for each sequence (each appending 1 token)
        # Build ragged append: each seq appends 1 token
        append_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
        seq_lens_after = torch.tensor(
            [old + 1 for old in old_lens], dtype=torch.int32, device=self.device
        )
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens_after, batch_size
        )

        # Embed all tokens: [batch_size, 1, hidden]
        input_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(old_lens, device=self.device).unsqueeze(1)
        hidden = model.model.embed_tokens(input_tensor)

        nh = self.num_heads
        nkv = self.num_kv_heads
        hd = self.head_dim
        F = torch.nn.functional

        # Pre-computed rotary (gather positions for each sequence in batch)
        cos_u = self._cos_cache[:, old_lens, :, :]  # [1, batch, 1, hd] — wrong shape, need [batch, 1, 1, hd]
        # Actually: _cos_cache is [1, max_pos, 1, hd], we need [batch, 1, 1, hd]
        cos_u = torch.stack([self._cos_cache[0, p] for p in old_lens]).unsqueeze(1)  # [batch, 1, 1, hd]
        sin_u = torch.stack([self._sin_cache[0, p] for p in old_lens]).unsqueeze(1)

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, self._input_ln_weights[i], self._rms_eps)

            qkv = F.linear(hidden, self._qkv_weights[i], self._qkv_biases[i])
            q = qkv[:, :, :self._q_size].view(batch_size, 1, nh, hd)
            k = qkv[:, :, self._q_size:self._q_size + self._kv_size].view(batch_size, 1, nkv, hd)
            v = qkv[:, :, self._q_size + self._kv_size:].view(batch_size, 1, nkv, hd)

            q = (q * cos_u) + (_rotate_half(q) * sin_u)
            k = (k * cos_u) + (_rotate_half(k) * sin_u)

            kv.append_kv(
                i, k.reshape(batch_size, nkv, hd), v.reshape(batch_size, nkv, hd),
                kv_indptr, kv_indices, kv_last_page_len, batch_indices, positions,
            )

            attn_out = kv.run_decode(i, q.reshape(batch_size, nh, hd))
            hidden = residual + F.linear(attn_out.reshape(batch_size, 1, -1), self._o_weights[i], self._o_biases[i])

            residual = hidden
            hidden = fused_rms_norm(hidden, self._post_ln_weights[i], self._rms_eps)
            gu = F.linear(hidden, self._gate_up_weights[i])
            hidden = residual + F.linear(
                F.silu(gu[:, :, :self._mlp_size]) * gu[:, :, self._mlp_size:],
                self._down_weights[i], self._down_biases[i]
            )

        hidden = fused_rms_norm(hidden, self._final_norm_weight, self._rms_eps)
        return self._lm_head(hidden[:, -1, :])  # [batch_size, vocab]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        seq_id: int,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Generate tokens. Returns list of generated token IDs."""
        self.kv_pool.new_sequence(seq_id)

        # Prefill
        logits = self.prefill(input_ids, seq_id)
        next_token = self._sample(logits, temperature)
        generated = [next_token.item()]

        # Decode loop
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and generated[-1] == eos_token_id:
                break
            token_input = torch.tensor([[generated[-1]]], device=self.device)
            logits = self.decode_step(token_input, seq_id)
            next_token = self._sample(logits, temperature)
            generated.append(next_token.item())

        return generated

    def _sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample next token from logits. temperature=0 for greedy."""
        logits = logits[:, -1, :]
        if temperature <= 0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
