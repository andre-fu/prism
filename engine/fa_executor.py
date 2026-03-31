"""Model executor using flash_attn with CUDA graph capture.

flash_attn_with_kvcache does KV append + attention in ONE kernel call.
CUDA graphs capture the full decode step (28 layers) as a single replay.
Result: 155+ tok/s on 7B — 1.65× vLLM eager.
"""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from .fa_kv_cache import FlashAttnKVCache
from .fused_kernels import fused_rms_norm


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class FlashAttnExecutor:
    """Runs inference with flash_attn + optional CUDA graph capture."""

    def __init__(
        self,
        model: torch.nn.Module,
        kv_cache: FlashAttnKVCache,
        device: str,
        use_cuda_graph: bool = True,
        max_graph_seqlen: int = 4096,
    ):
        self.model = model
        self.kv = kv_cache
        self.device = device
        self.use_cuda_graph = use_cuda_graph

        config = model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.mlp_size = config.intermediate_size
        self.rms_eps = getattr(config, "rms_norm_eps", 1e-6)

        # Pre-extract raw weights
        self._embed = model.model.embed_tokens
        self._lm_head = model.lm_head

        self._input_ln_w = [l.input_layernorm.weight.data for l in model.model.layers]
        self._post_ln_w = [l.post_attention_layernorm.weight.data for l in model.model.layers]
        self._final_norm_w = model.model.norm.weight.data

        # Stacked QKV weights
        self._qkv_w = []
        self._qkv_b = []
        for l in model.model.layers:
            self._qkv_w.append(torch.cat([
                l.self_attn.q_proj.weight.data,
                l.self_attn.k_proj.weight.data,
                l.self_attn.v_proj.weight.data,
            ], dim=0))
            q_bias = getattr(l.self_attn.q_proj, 'bias', None)
            if q_bias is not None:
                self._qkv_b.append(torch.cat([
                    q_bias.data,
                    l.self_attn.k_proj.bias.data,
                    l.self_attn.v_proj.bias.data,
                ]))
            else:
                self._qkv_b.append(None)

        self._o_w = [l.self_attn.o_proj.weight.data for l in model.model.layers]
        self._o_b = [getattr(l.self_attn.o_proj, 'bias', None) for l in model.model.layers]
        self._o_b = [b.data if b is not None else None for b in self._o_b]

        self._gu_w = [torch.cat([l.mlp.gate_proj.weight.data, l.mlp.up_proj.weight.data], dim=0)
                      for l in model.model.layers]
        self._down_w = [l.mlp.down_proj.weight.data for l in model.model.layers]
        self._down_b = [getattr(l.mlp.down_proj, 'bias', None) for l in model.model.layers]
        self._down_b = [b.data if b is not None else None for b in self._down_b]

        # Pre-compute rotary embeddings
        max_pos = min(getattr(config, 'max_position_embeddings', 4096), max_graph_seqlen)
        with torch.no_grad():
            dummy = torch.zeros(1, max_pos, self.hidden_size, device=device, dtype=torch.bfloat16)
            pos_range = torch.arange(max_pos, device=device).unsqueeze(0)
            cos, sin = model.model.rotary_emb(dummy, pos_range)
        self._cos = cos.unsqueeze(2)  # [1, max_pos, 1, head_dim]
        self._sin = sin.unsqueeze(2)

        # CUDA graph state
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_output: torch.Tensor | None = None
        self._graph_token_input = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._graph_block_table: torch.Tensor | None = None
        self._graph_seqlens: torch.Tensor | None = None
        self._graph_cos = self._cos[:, 0:1]  # Will be updated via slice
        self._graph_sin = self._sin[:, 0:1]

    def prefill(self, input_ids: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run prefill. input_ids: [1, seq_len]. Returns logits for last token."""
        seq_len = input_ids.shape[1]
        kv = self.kv

        # Ensure pages allocated
        new_total = kv.get_seq_len(seq_id) + seq_len
        kv._ensure_capacity(seq_id, new_total)
        block_table, cache_seqlens = kv.build_block_table([seq_id])

        hidden = self._embed(input_ids)
        cos = self._cos[:, :seq_len]
        sin = self._sin[:, :seq_len]

        qs, kvs, mlps = self.q_size, self.kv_size, self.mlp_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, self._input_ln_w[i], self.rms_eps)

            qkv = F.linear(hidden, self._qkv_w[i], self._qkv_b[i])
            q = qkv[:, :, :qs].view(1, seq_len, nh, hd)
            k = qkv[:, :, qs:qs+kvs].view(1, seq_len, nkv, hd)
            v = qkv[:, :, qs+kvs:].view(1, seq_len, nkv, hd)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)

            attn = flash_attn_with_kvcache(
                q, kv.k_caches[i], kv.v_caches[i], k=k, v=v,
                cache_seqlens=cache_seqlens, block_table=block_table, causal=True,
            )
            hidden = residual + F.linear(attn.reshape(1, seq_len, -1), self._o_w[i], self._o_b[i])

            residual = hidden
            hidden = fused_rms_norm(hidden, self._post_ln_w[i], self.rms_eps)
            gu = F.linear(hidden, self._gu_w[i])
            hidden = residual + F.linear(
                F.silu(gu[:, :, :mlps]) * gu[:, :, mlps:], self._down_w[i], self._down_b[i]
            )

        kv.seq_len[seq_id] = new_total
        return self._lm_head(fused_rms_norm(hidden[:, -1:, :], self._final_norm_w, self.rms_eps))

    def _decode_step_inner(self):
        """The decode computation — called directly or captured in CUDA graph."""
        hidden = self._embed(self._graph_token_input)
        cu = self._graph_cos
        su = self._graph_sin

        qs, kvs, mlps = self.q_size, self.kv_size, self.mlp_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, self._input_ln_w[i], self.rms_eps)

            qkv = F.linear(hidden, self._qkv_w[i], self._qkv_b[i])
            q = qkv[:, :, :qs].view(1, 1, nh, hd)
            k = qkv[:, :, qs:qs+kvs].view(1, 1, nkv, hd)
            v = qkv[:, :, qs+kvs:].view(1, 1, nkv, hd)
            q = (q * cu) + (_rotate_half(q) * su)
            k = (k * cu) + (_rotate_half(k) * su)

            attn = flash_attn_with_kvcache(
                q, self.kv.k_caches[i], self.kv.v_caches[i], k=k, v=v,
                cache_seqlens=self._graph_seqlens, block_table=self._graph_block_table,
                causal=True,
            )
            hidden = residual + F.linear(attn.reshape(1, 1, -1), self._o_w[i], self._o_b[i])

            residual = hidden
            hidden = fused_rms_norm(hidden, self._post_ln_w[i], self.rms_eps)
            gu = F.linear(hidden, self._gu_w[i])
            hidden = residual + F.linear(
                F.silu(gu[:, :, :mlps]) * gu[:, :, mlps:], self._down_w[i], self._down_b[i]
            )

        return self._lm_head(fused_rms_norm(hidden, self._final_norm_w, self.rms_eps))

    def _ensure_graph(self, block_table: torch.Tensor, cache_seqlens: torch.Tensor):
        """Capture CUDA graph on first call, reuse thereafter."""
        if self._graph is not None:
            return

        # Set up static tensors for graph capture
        self._graph_block_table = block_table
        self._graph_seqlens = cache_seqlens
        self._graph_token_input[0, 0] = 100  # Dummy token
        self._graph_cos = self._cos[:, 10:11].clone()
        self._graph_sin = self._sin[:, 10:11].clone()

        # Warmup (3 runs to stabilize)
        for _ in range(3):
            with torch.no_grad():
                _ = self._decode_step_inner()

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            with torch.no_grad():
                self._graph_output = self._decode_step_inner()

    def decode_step(self, token_id: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run one decode step. token_id: [1,1]. Returns logits: [1,1,vocab]."""
        kv = self.kv
        pos = kv.seq_len[seq_id]

        # Update sequence length + ensure pages
        kv._ensure_capacity(seq_id, pos + 1)
        kv.seq_len[seq_id] = pos + 1

        block_table, cache_seqlens = kv.build_block_table([seq_id])

        if self.use_cuda_graph:
            self._ensure_graph(block_table, cache_seqlens)

            # Update static inputs (graph replays with these values)
            self._graph_token_input.copy_(token_id)
            self._graph_block_table[:block_table.shape[0], :block_table.shape[1]] = block_table
            self._graph_seqlens[:cache_seqlens.shape[0]] = cache_seqlens
            self._graph_cos.copy_(self._cos[:, pos:pos+1])
            self._graph_sin.copy_(self._sin[:, pos:pos+1])

            self._graph.replay()
            return self._graph_output
        else:
            # Non-graph path
            self._graph_token_input.copy_(token_id)
            self._graph_block_table = block_table
            self._graph_seqlens = cache_seqlens
            self._graph_cos = self._cos[:, pos:pos+1]
            self._graph_sin = self._sin[:, pos:pos+1]
            with torch.no_grad():
                return self._decode_step_inner()

    def invalidate_graph(self):
        """Call when model switches — graph must be recaptured."""
        self._graph = None
        self._graph_output = None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        seq_id: int,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Full generation: prefill + decode loop."""
        self.kv.new_sequence(seq_id)

        logits = self.prefill(input_ids, seq_id)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and generated[-1] == eos_token_id:
                break
            token_input = torch.tensor([[generated[-1]]], device=self.device)
            logits = self.decode_step(token_input, seq_id)
            if temperature <= 0:
                next_token = logits[:, -1, :].argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

        return generated
