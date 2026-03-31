"""Static GPU weight pool: pre-allocated buffers for zero-allocation model swaps.

At engine startup, we allocate ONE set of weight buffers on GPU, sized for
the largest model architecture. Every model swap is just copy_() into these
fixed buffers — no GPU memory allocation, no fragmentation, no OOM.

The CUDA graph captures the addresses of these buffers. Since copy_() changes
values but not addresses, the graph replays correctly with new weights.

Memory layout:
  - Per-layer stacked QKV weight: [q_size + 2*kv_size, hidden_size]
  - Per-layer stacked QKV bias: [q_size + 2*kv_size]
  - Per-layer stacked gate+up weight: [2*intermediate_size, hidden_size]
  - Per-layer O projection weight: [hidden_size, hidden_size]
  - Per-layer down projection weight: [hidden_size, intermediate_size]
  - Per-layer input/post layernorm weights: [hidden_size]
  - Embedding weight: [vocab_size, hidden_size]
  - LM head weight: [vocab_size, hidden_size]
  - Final norm weight: [hidden_size]
"""

import torch
from transformers import AutoConfig


class StaticWeightPool:
    """Pre-allocated weight buffers on GPU. Zero allocation during serving."""

    def __init__(self, hf_config, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.config = hf_config

        nl = hf_config.num_hidden_layers
        hs = hf_config.hidden_size
        nh = hf_config.num_attention_heads
        nkv = hf_config.num_key_value_heads
        hd = hs // nh
        mlps = hf_config.intermediate_size
        vocab = hf_config.vocab_size

        q_size = nh * hd
        kv_size = nkv * hd
        qkv_size = q_size + 2 * kv_size

        # Allocate all buffers
        self.embed_w = torch.zeros(vocab, hs, dtype=dtype, device=device)
        self.lm_head_w = torch.zeros(vocab, hs, dtype=dtype, device=device)
        self.final_norm_w = torch.zeros(hs, dtype=dtype, device=device)

        self.input_ln_w = [torch.zeros(hs, dtype=dtype, device=device) for _ in range(nl)]
        self.post_ln_w = [torch.zeros(hs, dtype=dtype, device=device) for _ in range(nl)]

        self.qkv_w = [torch.zeros(qkv_size, hs, dtype=dtype, device=device) for _ in range(nl)]
        self.qkv_b = [torch.zeros(qkv_size, dtype=dtype, device=device) for _ in range(nl)]

        self.o_w = [torch.zeros(hs, q_size, dtype=dtype, device=device) for _ in range(nl)]
        self.o_b = [torch.zeros(hs, dtype=dtype, device=device) for _ in range(nl)]
        self.has_o_bias = False

        # QK Norm (Qwen3+): RMSNorm applied to Q and K before rotary + attention
        self.q_norm_w = [torch.zeros(hd, dtype=dtype, device=device) for _ in range(nl)]
        self.k_norm_w = [torch.zeros(hd, dtype=dtype, device=device) for _ in range(nl)]
        self.has_qk_norm = False

        self.gu_w = [torch.zeros(2 * mlps, hs, dtype=dtype, device=device) for _ in range(nl)]

        self.down_w = [torch.zeros(hs, mlps, dtype=dtype, device=device) for _ in range(nl)]
        self.down_b = [torch.zeros(hs, dtype=dtype, device=device) for _ in range(nl)]
        self.has_down_bias = False

        # Track sizes for budget reporting
        self._total_bytes = sum(t.nbytes for t in self._all_tensors())

    def _all_tensors(self):
        yield self.embed_w
        yield self.lm_head_w
        yield self.final_norm_w
        for lst in [self.input_ln_w, self.post_ln_w, self.qkv_w, self.qkv_b,
                    self.o_w, self.o_b, self.gu_w, self.down_w, self.down_b]:
            yield from lst

    @property
    def total_gb(self) -> float:
        return self._total_bytes / 1e9

    def load_from_model(self, model: torch.nn.Module):
        """Copy a model's weights into the static buffers. ~15ms for 7B."""
        layers = model.model.layers
        nl = len(layers)

        self.embed_w.copy_(model.model.embed_tokens.weight.data)
        self.lm_head_w.copy_(model.lm_head.weight.data)
        self.final_norm_w.copy_(model.model.norm.weight.data)

        for i in range(nl):
            l = layers[i]
            self.input_ln_w[i].copy_(l.input_layernorm.weight.data)
            self.post_ln_w[i].copy_(l.post_attention_layernorm.weight.data)

            # Stack QKV
            q_w = l.self_attn.q_proj.weight.data
            k_w = l.self_attn.k_proj.weight.data
            v_w = l.self_attn.v_proj.weight.data
            qs = q_w.shape[0]
            ks = k_w.shape[0]
            self.qkv_w[i][:qs].copy_(q_w)
            self.qkv_w[i][qs:qs+ks].copy_(k_w)
            self.qkv_w[i][qs+ks:].copy_(v_w)

            q_b = getattr(l.self_attn.q_proj, 'bias', None)
            if q_b is not None:
                self.qkv_b[i][:qs].copy_(q_b.data)
                self.qkv_b[i][qs:qs+ks].copy_(l.self_attn.k_proj.bias.data)
                self.qkv_b[i][qs+ks:].copy_(l.self_attn.v_proj.bias.data)

            self.o_w[i].copy_(l.self_attn.o_proj.weight.data)
            o_bias = getattr(l.self_attn.o_proj, 'bias', None)
            if o_bias is not None:
                self.o_b[i].copy_(o_bias.data)
                self.has_o_bias = True

            # QK Norm (Qwen3+)
            q_norm = getattr(l.self_attn, 'q_norm', None)
            if q_norm is not None and hasattr(q_norm, 'weight'):
                self.q_norm_w[i].copy_(q_norm.weight.data)
                self.k_norm_w[i].copy_(l.self_attn.k_norm.weight.data)
                self.has_qk_norm = True

            # Stack gate+up
            g_w = l.mlp.gate_proj.weight.data
            gs = g_w.shape[0]
            self.gu_w[i][:gs].copy_(g_w)
            self.gu_w[i][gs:].copy_(l.mlp.up_proj.weight.data)

            self.down_w[i].copy_(l.mlp.down_proj.weight.data)
            d_bias = getattr(l.mlp.down_proj, 'bias', None)
            if d_bias is not None:
                self.down_b[i].copy_(d_bias.data)
                self.has_down_bias = True

    def load_from_pinned(self, pinned_weights: dict[str, torch.Tensor], hf_config):
        """Copy directly from pinned RAM weights into static GPU buffers.

        This is the fastest path: pinned RAM → GPU buffer via copy_().
        Skips creating an nn.Module entirely.
        """
        nh = hf_config.num_attention_heads
        nkv = hf_config.num_key_value_heads
        hd = hf_config.hidden_size // nh
        q_size = nh * hd
        kv_size = nkv * hd
        nl = hf_config.num_hidden_layers
        mlps = hf_config.intermediate_size

        self.embed_w.copy_(pinned_weights["model.embed_tokens.weight"].to(self.dtype))
        if "lm_head.weight" in pinned_weights:
            self.lm_head_w.copy_(pinned_weights["lm_head.weight"].to(self.dtype))
        else:
            self.lm_head_w.copy_(self.embed_w)  # Tied weights
        self.final_norm_w.copy_(pinned_weights["model.norm.weight"].to(self.dtype))

        for i in range(nl):
            prefix = f"model.layers.{i}"
            self.input_ln_w[i].copy_(pinned_weights[f"{prefix}.input_layernorm.weight"].to(self.dtype))
            self.post_ln_w[i].copy_(pinned_weights[f"{prefix}.post_attention_layernorm.weight"].to(self.dtype))

            # QKV stacked
            q_w = pinned_weights[f"{prefix}.self_attn.q_proj.weight"].to(self.dtype)
            k_w = pinned_weights[f"{prefix}.self_attn.k_proj.weight"].to(self.dtype)
            v_w = pinned_weights[f"{prefix}.self_attn.v_proj.weight"].to(self.dtype)
            self.qkv_w[i][:q_size].copy_(q_w)
            self.qkv_w[i][q_size:q_size+kv_size].copy_(k_w)
            self.qkv_w[i][q_size+kv_size:].copy_(v_w)

            q_b_key = f"{prefix}.self_attn.q_proj.bias"
            if q_b_key in pinned_weights:
                self.qkv_b[i][:q_size].copy_(pinned_weights[q_b_key].to(self.dtype))
                self.qkv_b[i][q_size:q_size+kv_size].copy_(
                    pinned_weights[f"{prefix}.self_attn.k_proj.bias"].to(self.dtype))
                self.qkv_b[i][q_size+kv_size:].copy_(
                    pinned_weights[f"{prefix}.self_attn.v_proj.bias"].to(self.dtype))

            self.o_w[i].copy_(pinned_weights[f"{prefix}.self_attn.o_proj.weight"].to(self.dtype))

            # QK Norm (Qwen3+)
            q_norm_key = f"{prefix}.self_attn.q_norm.weight"
            if q_norm_key in pinned_weights:
                self.q_norm_w[i].copy_(pinned_weights[q_norm_key].to(self.dtype))
                self.k_norm_w[i].copy_(pinned_weights[f"{prefix}.self_attn.k_norm.weight"].to(self.dtype))
                self.has_qk_norm = True

            # Gate+up stacked
            g_w = pinned_weights[f"{prefix}.mlp.gate_proj.weight"].to(self.dtype)
            self.gu_w[i][:mlps].copy_(g_w)
            self.gu_w[i][mlps:].copy_(
                pinned_weights[f"{prefix}.mlp.up_proj.weight"].to(self.dtype))

            self.down_w[i].copy_(pinned_weights[f"{prefix}.mlp.down_proj.weight"].to(self.dtype))

    def free(self):
        """Release all GPU memory."""
        for t in self._all_tensors():
            t.data = torch.empty(0, dtype=self.dtype, device=self.device)
