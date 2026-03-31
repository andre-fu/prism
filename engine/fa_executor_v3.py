"""Model executor v3: flash_attn + CUDA graphs + static weight pool.

Uses StaticWeightPool for zero-allocation model swaps.
All weight buffers are pre-allocated at startup.
Model swap = copy_() from pinned RAM into fixed GPU buffers.
CUDA graph replays with new weight values at same addresses.
"""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from .fa_kv_cache import FlashAttnKVCache
from .weight_pool import StaticWeightPool
from .fused_kernels import fused_rms_norm


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class FlashAttnExecutorV3:
    """Model executor with pre-allocated static weight pool."""

    def __init__(self, weight_pool: StaticWeightPool, kv_cache: FlashAttnKVCache, device: str,
                 use_cuda_graph: bool = True):
        self.w = weight_pool
        self.kv = kv_cache
        self.device = device
        self.use_cuda_graph = use_cuda_graph

        config = weight_pool.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.mlp_size = config.intermediate_size
        self.rms_eps = getattr(config, "rms_norm_eps", 1e-6)

        # Pre-compute rotary
        max_pos = min(getattr(config, 'max_position_embeddings', 4096), 4096)
        # Compute rotary from scratch (no model needed)
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        dim = self.head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(torch.bfloat16).unsqueeze(0).unsqueeze(2)  # [1, max_pos, 1, dim/2]
        sin = freqs.sin().to(torch.bfloat16).unsqueeze(0).unsqueeze(2)
        # Expand to full head_dim (repeat for both halves)
        self._cos = torch.cat([cos, cos], dim=-1)  # [1, max_pos, 1, head_dim]
        self._sin = torch.cat([sin, sin], dim=-1)

        # CUDA graph state — single-sequence (batch=1)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_output: torch.Tensor | None = None
        self._graph_stream: torch.cuda.Stream | None = None
        self._graph_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._graph_bt: torch.Tensor | None = None
        self._graph_sl: torch.Tensor | None = None
        self._graph_cos = self._cos[:, 0:1].clone()
        self._graph_sin = self._sin[:, 0:1].clone()

        # Batched CUDA graph state — one graph per captured batch size
        self._batch_graph_sizes = [2, 4, 8, 16, 32]
        self._batch_graphs: dict[int, dict] = {}  # bs -> {graph, output, stream, token, bt, sl, cos, sin}
        self._max_blocks_per_seq = 32  # Max KV pages per sequence for graph capture

    def _embed(self, token_ids):
        return F.embedding(token_ids, self.w.embed_w)

    def prefill(self, input_ids: torch.Tensor, seq_id: int, prefix_len: int = 0) -> torch.Tensor:
        """Run prefill. If prefix_len > 0, skip those tokens (already in KV cache)."""
        kv = self.kv
        w = self.w

        if prefix_len > 0:
            # Skip prefix tokens — they're already in KV cache from prefix cache
            input_ids = input_ids[:, prefix_len:]

        seq_len = input_ids.shape[1]
        if seq_len == 0:
            # All tokens cached — just need to get logits for last token
            # Do a single decode step at position prefix_len-1
            # Actually we need at least 1 token for the forward pass
            # Use the last prefix token
            return self.decode_step(input_ids[:, -1:] if input_ids.shape[1] > 0
                                    else torch.tensor([[0]], device=self.device), seq_id)

        new_total = kv.get_seq_len(seq_id) + seq_len
        kv._ensure_capacity(seq_id, new_total)
        bt, sl = kv.build_block_table([seq_id])

        hidden = self._embed(input_ids)
        cos = self._cos[:, prefix_len:prefix_len + seq_len]
        sin = self._sin[:, prefix_len:prefix_len + seq_len]

        qs, kvs, mlps = self.q_size, self.kv_size, self.mlp_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, w.input_ln_w[i], self.rms_eps)
            qkv = F.linear(hidden, w.qkv_w[i], w.qkv_b[i])
            q = qkv[:, :, :qs].view(1, seq_len, nh, hd)
            k = qkv[:, :, qs:qs+kvs].view(1, seq_len, nkv, hd)
            v = qkv[:, :, qs+kvs:].view(1, seq_len, nkv, hd)
            # QK Norm (Qwen3+): RMSNorm per-head before rotary
            if w.has_qk_norm:
                q = fused_rms_norm(q, w.q_norm_w[i], self.rms_eps)
                k = fused_rms_norm(k, w.k_norm_w[i], self.rms_eps)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            attn = flash_attn_with_kvcache(q, kv.k_caches[i], kv.v_caches[i], k=k, v=v,
                                           cache_seqlens=sl, block_table=bt, causal=True)
            o_b = w.o_b[i] if w.has_o_bias else None
            hidden = residual + F.linear(attn.reshape(1, seq_len, -1), w.o_w[i], o_b)
            residual = hidden
            hidden = fused_rms_norm(hidden, w.post_ln_w[i], self.rms_eps)
            gu = F.linear(hidden, w.gu_w[i])
            d_b = w.down_b[i] if w.has_down_bias else None
            hidden = residual + F.linear(F.silu(gu[:, :, :mlps]) * gu[:, :, mlps:], w.down_w[i], d_b)

        kv.seq_len[seq_id] = new_total
        return F.linear(fused_rms_norm(hidden[:, -1:, :], w.final_norm_w, self.rms_eps), w.lm_head_w)

    def _decode_inner(self):
        hidden = self._embed(self._graph_token)
        cu, su = self._graph_cos, self._graph_sin
        w = self.w
        qs, kvs, mlps = self.q_size, self.kv_size, self.mlp_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, w.input_ln_w[i], self.rms_eps)
            qkv = F.linear(hidden, w.qkv_w[i], w.qkv_b[i])
            q = qkv[:, :, :qs].view(1, 1, nh, hd)
            k = qkv[:, :, qs:qs+kvs].view(1, 1, nkv, hd)
            v = qkv[:, :, qs+kvs:].view(1, 1, nkv, hd)
            if w.has_qk_norm:
                q = fused_rms_norm(q, w.q_norm_w[i], self.rms_eps)
                k = fused_rms_norm(k, w.k_norm_w[i], self.rms_eps)
            q = (q * cu) + (_rotate_half(q) * su)
            k = (k * cu) + (_rotate_half(k) * su)
            attn = flash_attn_with_kvcache(q, self.kv.k_caches[i], self.kv.v_caches[i], k=k, v=v,
                                           cache_seqlens=self._graph_sl, block_table=self._graph_bt,
                                           causal=True)
            o_b = w.o_b[i] if w.has_o_bias else None
            hidden = residual + F.linear(attn.reshape(1, 1, -1), w.o_w[i], o_b)
            residual = hidden
            hidden = fused_rms_norm(hidden, w.post_ln_w[i], self.rms_eps)
            gu = F.linear(hidden, w.gu_w[i])
            d_b = w.down_b[i] if w.has_down_bias else None
            hidden = residual + F.linear(F.silu(gu[:, :, :mlps]) * gu[:, :, mlps:], w.down_w[i], d_b)

        return F.linear(fused_rms_norm(hidden, w.final_norm_w, self.rms_eps), w.lm_head_w)

    def _ensure_graph(self, bt, sl):
        if self._graph is not None:
            return
        if not self.use_cuda_graph:
            self._graph = False
            return

        # Check memory
        gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
        free = torch.cuda.mem_get_info(gpu_id)[0]
        if free < 2 * 1024 * 1024 * 1024:
            print(f"[Executor] Insufficient memory for CUDA graph ({free/1e9:.1f}GB free), using eager")
            self._graph = False
            return

        gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
        self._graph_bt = bt.clone()
        self._graph_sl = sl.clone()
        self._graph_token[0, 0] = 100
        self._graph_cos.copy_(self._cos[:, 10:11])
        self._graph_sin.copy_(self._sin[:, 10:11])
        with torch.cuda.device(gpu_id):
            for _ in range(3):
                with torch.no_grad():
                    self._decode_inner()

        try:
            gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            self._graph_stream = torch.cuda.Stream(device=gpu_id)
            self._graph_stream.wait_stream(torch.cuda.current_stream(gpu_id))
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self._graph_stream):
                with torch.cuda.graph(self._graph, stream=self._graph_stream):
                    with torch.no_grad():
                        self._graph_output = self._decode_inner()
        except Exception as e:
            print(f"[Executor] Graph capture failed: {e}")
            self._graph = False

    def decode_step(self, token_id: torch.Tensor, seq_id: int) -> torch.Tensor:
        kv = self.kv
        pos = kv.seq_len[seq_id]
        kv._ensure_capacity(seq_id, pos + 1)
        kv.seq_len[seq_id] = pos + 1
        bt, sl = kv.build_block_table([seq_id])

        if self.use_cuda_graph:
            self._ensure_graph(bt, sl)

        if isinstance(self._graph, torch.cuda.CUDAGraph):
            self._graph_token.copy_(token_id)
            nb = bt.shape[1]
            if nb <= self._graph_bt.shape[1]:
                self._graph_bt[0, :nb] = bt[0, :nb]
            else:
                # Block table grew beyond graph capture — extend
                self.invalidate_graph()
                return self.decode_step(token_id, seq_id)  # Recapture
            self._graph_sl[0] = sl[0]
            self._graph_cos.copy_(self._cos[:, pos:pos+1])
            self._graph_sin.copy_(self._sin[:, pos:pos+1])
            self._graph.replay()
            if self._graph_stream:
                self._graph_stream.synchronize()
            return self._graph_output

        # Eager fallback
        self._graph_token.copy_(token_id)
        self._graph_bt = bt
        self._graph_sl = sl
        self._graph_cos = self._cos[:, pos:pos+1]
        self._graph_sin = self._sin[:, pos:pos+1]
        with torch.no_grad():
            return self._decode_inner()

    def _batched_decode_inner(self, bs: int):
        """Batched decode computation — graph-capturable for fixed batch size."""
        g = self._batch_graphs[bs]
        hidden = self._embed(g["token"])
        cu, su = g["cos"], g["sin"]
        w = self.w
        qs, kvs, mlps = self.q_size, self.kv_size, self.mlp_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, w.input_ln_w[i], self.rms_eps)
            qkv = F.linear(hidden, w.qkv_w[i], w.qkv_b[i])
            q = qkv[:, :, :qs].view(bs, 1, nh, hd)
            k = qkv[:, :, qs:qs+kvs].view(bs, 1, nkv, hd)
            v = qkv[:, :, qs+kvs:].view(bs, 1, nkv, hd)
            if w.has_qk_norm:
                q = fused_rms_norm(q, w.q_norm_w[i], self.rms_eps)
                k = fused_rms_norm(k, w.k_norm_w[i], self.rms_eps)
            q = (q * cu) + (_rotate_half(q) * su)
            k = (k * cu) + (_rotate_half(k) * su)
            attn = flash_attn_with_kvcache(q, self.kv.k_caches[i], self.kv.v_caches[i], k=k, v=v,
                                           cache_seqlens=g["sl"], block_table=g["bt"], causal=True)
            o_b = w.o_b[i] if w.has_o_bias else None
            hidden = residual + F.linear(attn.reshape(bs, 1, -1), w.o_w[i], o_b)
            residual = hidden
            hidden = fused_rms_norm(hidden, w.post_ln_w[i], self.rms_eps)
            gu = F.linear(hidden, w.gu_w[i])
            d_b = w.down_b[i] if w.has_down_bias else None
            hidden = residual + F.linear(F.silu(gu[:, :, :mlps]) * gu[:, :, mlps:], w.down_w[i], d_b)

        return F.linear(fused_rms_norm(hidden[:, -1:, :], w.final_norm_w, self.rms_eps), w.lm_head_w).squeeze(1)

    def _capture_batch_graph(self, bs: int):
        """Capture a CUDA graph for a specific batch size."""
        gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
        free = torch.cuda.mem_get_info(gpu_id)[0]
        if free < 3 * 1024 * 1024 * 1024:
            return  # Not enough memory

        # Allocate static buffers for this batch size
        self._batch_graphs[bs] = {
            "token": torch.zeros(bs, 1, dtype=torch.long, device=self.device),
            "bt": torch.zeros(bs, self._max_blocks_per_seq, dtype=torch.int32, device=self.device),
            "sl": torch.zeros(bs, dtype=torch.int32, device=self.device),
            "cos": self._cos[:, 10:11].expand(bs, -1, -1, -1).clone(),  # [bs, 1, 1, hd]
            "sin": self._sin[:, 10:11].expand(bs, -1, -1, -1).clone(),
        }
        g = self._batch_graphs[bs]

        # Warmup
        with torch.cuda.device(gpu_id):
            for _ in range(3):
                with torch.no_grad():
                    self._batched_decode_inner(bs)

        # Capture
        try:
            stream = torch.cuda.Stream(device=gpu_id)
            stream.wait_stream(torch.cuda.current_stream(gpu_id))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    with torch.no_grad():
                        output = self._batched_decode_inner(bs)
            g["graph"] = graph
            g["output"] = output
            g["stream"] = stream
            print(f"[Executor] Captured batch={bs} CUDA graph on {self.device}")
        except Exception as e:
            print(f"[Executor] Batch={bs} graph capture failed: {e}")
            del self._batch_graphs[bs]

    def warmup_batch_graphs(self, sizes: list[int] | None = None):
        """Pre-capture CUDA graphs for multiple batch sizes."""
        if not self.use_cuda_graph:
            return
        for bs in (sizes or self._batch_graph_sizes):
            if bs not in self._batch_graphs:
                self._capture_batch_graph(bs)

    def batched_decode_step(self, token_ids: list[int], seq_ids: list[int]) -> torch.Tensor:
        """Batched decode — uses CUDA graph if available for this batch size."""
        kv = self.kv
        actual_bs = len(seq_ids)

        # Update KV cache state
        positions = []
        for sid in seq_ids:
            pos = kv.seq_len[sid]
            kv._ensure_capacity(sid, pos + 1)
            kv.seq_len[sid] = pos + 1
            positions.append(pos)

        bt, sl = kv.build_block_table(seq_ids)

        # Find smallest captured batch size >= actual
        graph_bs = None
        if self.use_cuda_graph:
            for bs in sorted(self._batch_graphs.keys()):
                if bs >= actual_bs:
                    graph_bs = bs
                    break

        if graph_bs is not None and bt.shape[1] <= self._max_blocks_per_seq:
            g = self._batch_graphs[graph_bs]

            # Copy inputs into static buffers, pad remainder with zeros
            g["token"][:actual_bs, 0] = torch.tensor(token_ids, device=self.device)
            if actual_bs < graph_bs:
                g["token"][actual_bs:] = 0
            g["bt"][:] = 0
            g["bt"][:actual_bs, :bt.shape[1]] = bt
            g["sl"][:] = 0
            g["sl"][:actual_bs] = sl
            for i in range(actual_bs):
                g["cos"][i, 0].copy_(self._cos[0, positions[i]])
                g["sin"][i, 0].copy_(self._sin[0, positions[i]])

            g["graph"].replay()
            if g.get("stream"):
                g["stream"].synchronize()
            return g["output"][:actual_bs]

        # Eager fallback
        w = self.w
        qs, kvs, mlps = self.q_size, self.kv_size, self.mlp_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        cos = torch.stack([self._cos[0, p] for p in positions]).unsqueeze(1)
        sin = torch.stack([self._sin[0, p] for p in positions]).unsqueeze(1)
        token_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(1)
        hidden = self._embed(token_tensor)

        for i in range(self.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, w.input_ln_w[i], self.rms_eps)
            qkv = F.linear(hidden, w.qkv_w[i], w.qkv_b[i])
            q = qkv[:, :, :qs].view(actual_bs, 1, nh, hd)
            k = qkv[:, :, qs:qs+kvs].view(actual_bs, 1, nkv, hd)
            v = qkv[:, :, qs+kvs:].view(actual_bs, 1, nkv, hd)
            if w.has_qk_norm:
                q = fused_rms_norm(q, w.q_norm_w[i], self.rms_eps)
                k = fused_rms_norm(k, w.k_norm_w[i], self.rms_eps)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            attn = flash_attn_with_kvcache(q, kv.k_caches[i], kv.v_caches[i], k=k, v=v,
                                           cache_seqlens=sl, block_table=bt, causal=True)
            o_b = w.o_b[i] if w.has_o_bias else None
            hidden = residual + F.linear(attn.reshape(actual_bs, 1, -1), w.o_w[i], o_b)
            residual = hidden
            hidden = fused_rms_norm(hidden, w.post_ln_w[i], self.rms_eps)
            gu = F.linear(hidden, w.gu_w[i])
            d_b = w.down_b[i] if w.has_down_bias else None
            hidden = residual + F.linear(F.silu(gu[:, :, :mlps]) * gu[:, :, mlps:], w.down_w[i], d_b)

        logits = F.linear(fused_rms_norm(hidden[:, -1:, :], w.final_norm_w, self.rms_eps), w.lm_head_w)
        return logits.squeeze(1)

    def invalidate_graph(self):
        if isinstance(self._graph, torch.cuda.CUDAGraph):
            self._graph.reset()
        self._graph = None
        self._graph_output = None
        self._graph_bt = None
        # Also clear batched graphs
        for bs, g in self._batch_graphs.items():
            if "graph" in g:
                g["graph"].reset()
        self._batch_graphs.clear()
        self._graph_sl = None

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, seq_id: int, max_new_tokens: int = 100,
                 temperature: float = 0.0, eos_token_id: int | None = None) -> list[int]:
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
