"""Tensor-parallel model executor: runs a model sharded across multiple GPUs."""

import torch
import flashinfer
from .kv_cache import PagedKVPool
from .distributed import TPGroup, get_shard_plan, ShardPlan
from .model_executor import _rotate_half, _apply_rotary_emb


class TPModelExecutor:
    """Runs inference with a model sharded across multiple GPUs.

    Each GPU holds:
    - Its shard of the model weights (column/row split)
    - Its shard of the KV cache (split by KV heads)

    The forward pass runs on all GPUs in parallel (from Python's perspective,
    CUDA kernels on different devices run concurrently). After row-parallel
    layers (o_proj, down_proj), we all-reduce across GPUs.
    """

    def __init__(
        self,
        models: list[torch.nn.Module],  # One model per GPU, each with sharded weights
        kv_pools: list[PagedKVPool],     # One KV pool per GPU
        tp_group: TPGroup,
    ):
        self.models = models  # models[rank] is on gpu_ids[rank]
        self.kv_pools = kv_pools
        self.tp = tp_group
        self.tp_size = tp_group.tp_size

        config = models[0].config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        # Per-GPU head counts
        self.heads_per_gpu = self.num_heads // self.tp_size
        self.kv_heads_per_gpu = self.num_kv_heads // self.tp_size

    def prefill(self, input_ids: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run TP prefill. input_ids on GPU 0. Returns logits on GPU 0."""
        seq_len = input_ids.shape[1]
        device0 = self.tp.devices[0]

        # Allocate KV pages on all GPUs (same page IDs across GPUs)
        old_len = self.kv_pools[0].prepare_append(seq_id, seq_len)
        for rank in range(1, self.tp_size):
            self.kv_pools[rank].prepare_append(seq_id, seq_len)

        # Build page table (same across all GPUs since we use same seq tracking)
        kv_indptr, kv_indices, kv_last_page_len = self.kv_pools[0].build_page_table([seq_id])

        # Plan FlashInfer prefill on all GPUs
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device0)
        for rank in range(self.tp_size):
            dev = self.tp.devices[rank]
            self.kv_pools[rank].plan_prefill(
                qo_indptr.to(dev), kv_indptr.to(dev), kv_indices.to(dev),
                kv_last_page_len.to(dev), self.heads_per_gpu,
            )

        # Compute append positions
        append_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device0)
        seq_lens_after = torch.tensor([old_len + seq_len], dtype=torch.int32, device=device0)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens_after, seq_len
        )

        # Embedding (replicated — same on all GPUs)
        hidden_per_gpu = []
        for rank in range(self.tp_size):
            dev = self.tp.devices[rank]
            ids = input_ids.to(dev)
            h = self.models[rank].model.embed_tokens(ids)
            hidden_per_gpu.append(h)

        position_ids = torch.arange(old_len, old_len + seq_len, device=device0).unsqueeze(0)

        for layer_idx in range(self.num_layers):
            residuals = [h.clone() for h in hidden_per_gpu]

            # Layernorm (replicated weights, same input → same output per GPU)
            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                hidden_per_gpu[rank] = layer.input_layernorm(hidden_per_gpu[rank])

            # QKV projections (column-parallel — each GPU computes its head shard)
            q_per_gpu = []
            k_per_gpu = []
            v_per_gpu = []
            for rank in range(self.tp_size):
                dev = self.tp.devices[rank]
                layer = self.models[rank].model.layers[layer_idx]
                h = hidden_per_gpu[rank]

                q = layer.self_attn.q_proj(h).view(1, seq_len, self.heads_per_gpu, self.head_dim)
                k = layer.self_attn.k_proj(h).view(1, seq_len, self.kv_heads_per_gpu, self.head_dim)
                v = layer.self_attn.v_proj(h).view(1, seq_len, self.kv_heads_per_gpu, self.head_dim)

                cos, sin = self.models[rank].model.rotary_emb(v, position_ids.to(dev))
                q, k = _apply_rotary_emb(q, k, cos, sin)

                q_per_gpu.append(q)
                k_per_gpu.append(k)
                v_per_gpu.append(v)

            # Append KV to paged cache (each GPU writes its head shard)
            for rank in range(self.tp_size):
                dev = self.tp.devices[rank]
                self.kv_pools[rank].append_kv(
                    layer_idx,
                    k_per_gpu[rank].reshape(-1, self.kv_heads_per_gpu, self.head_dim),
                    v_per_gpu[rank].reshape(-1, self.kv_heads_per_gpu, self.head_dim),
                    kv_indptr.to(dev), kv_indices.to(dev), kv_last_page_len.to(dev),
                    batch_indices.to(dev), positions.to(dev),
                )

            # FlashInfer attention (each GPU on its head shard)
            attn_per_gpu = []
            for rank in range(self.tp_size):
                q_flat = q_per_gpu[rank].reshape(-1, self.heads_per_gpu, self.head_dim)
                attn_out = self.kv_pools[rank].run_prefill(layer_idx, q_flat)
                attn_out = attn_out.reshape(1, seq_len, self.heads_per_gpu * self.head_dim)
                attn_per_gpu.append(attn_out)

            # o_proj (row-parallel — each GPU has a column shard of o_proj)
            # Output needs all-reduce
            o_per_gpu = []
            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                o = layer.self_attn.o_proj(attn_per_gpu[rank])
                o_per_gpu.append(o)

            # All-reduce o_proj outputs (NCCL in-place, result on all GPUs)
            o_reduced = self.tp.all_reduce(o_per_gpu)

            # Residual (already on correct GPU)
            for rank in range(self.tp_size):
                hidden_per_gpu[rank] = residuals[rank] + o_reduced[rank]

            # MLP: layernorm → gate_proj/up_proj (column-parallel) → down_proj (row-parallel) → all-reduce
            residuals = [h.clone() for h in hidden_per_gpu]
            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                hidden_per_gpu[rank] = layer.post_attention_layernorm(hidden_per_gpu[rank])

            mlp_per_gpu = []
            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                m = layer.mlp(hidden_per_gpu[rank])
                mlp_per_gpu.append(m)

            mlp_reduced = self.tp.all_reduce(mlp_per_gpu)
            for rank in range(self.tp_size):
                hidden_per_gpu[rank] = residuals[rank] + mlp_reduced[rank]

        # Final norm + lm_head on GPU 0
        hidden = self.models[0].model.norm(hidden_per_gpu[0])
        logits = self.models[0].lm_head(hidden[:, -1:, :])
        return logits

    def decode_step(self, token_id: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run one TP decode step. Returns logits on GPU 0."""
        device0 = self.tp.devices[0]

        old_len = self.kv_pools[0].prepare_append(seq_id, 1)
        for rank in range(1, self.tp_size):
            self.kv_pools[rank].prepare_append(seq_id, 1)

        kv_indptr, kv_indices, kv_last_page_len = self.kv_pools[0].build_page_table([seq_id])

        for rank in range(self.tp_size):
            dev = self.tp.devices[rank]
            self.kv_pools[rank].plan_decode(
                kv_indptr.to(dev), kv_indices.to(dev), kv_last_page_len.to(dev),
                self.heads_per_gpu,
            )

        append_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device0)
        seq_lens_after = torch.tensor([old_len + 1], dtype=torch.int32, device=device0)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens_after, 1
        )

        position_ids = torch.tensor([[old_len]], device=device0)

        # Embedding (replicated)
        hidden_per_gpu = []
        for rank in range(self.tp_size):
            dev = self.tp.devices[rank]
            h = self.models[rank].model.embed_tokens(token_id.to(dev))
            hidden_per_gpu.append(h)

        for layer_idx in range(self.num_layers):
            residuals = [h.clone() for h in hidden_per_gpu]

            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                hidden_per_gpu[rank] = layer.input_layernorm(hidden_per_gpu[rank])

            q_per_gpu, k_per_gpu, v_per_gpu = [], [], []
            for rank in range(self.tp_size):
                dev = self.tp.devices[rank]
                layer = self.models[rank].model.layers[layer_idx]
                h = hidden_per_gpu[rank]

                q = layer.self_attn.q_proj(h).view(1, 1, self.heads_per_gpu, self.head_dim)
                k = layer.self_attn.k_proj(h).view(1, 1, self.kv_heads_per_gpu, self.head_dim)
                v = layer.self_attn.v_proj(h).view(1, 1, self.kv_heads_per_gpu, self.head_dim)

                cos, sin = self.models[rank].model.rotary_emb(v, position_ids.to(dev))
                q, k = _apply_rotary_emb(q, k, cos, sin)
                q_per_gpu.append(q)
                k_per_gpu.append(k)
                v_per_gpu.append(v)

            for rank in range(self.tp_size):
                dev = self.tp.devices[rank]
                self.kv_pools[rank].append_kv(
                    layer_idx,
                    k_per_gpu[rank].reshape(1, self.kv_heads_per_gpu, self.head_dim),
                    v_per_gpu[rank].reshape(1, self.kv_heads_per_gpu, self.head_dim),
                    kv_indptr.to(dev), kv_indices.to(dev), kv_last_page_len.to(dev),
                    batch_indices.to(dev), positions.to(dev),
                )

            attn_per_gpu = []
            for rank in range(self.tp_size):
                q_flat = q_per_gpu[rank].reshape(1, self.heads_per_gpu, self.head_dim)
                attn_out = self.kv_pools[rank].run_decode(layer_idx, q_flat)
                attn_out = attn_out.reshape(1, 1, self.heads_per_gpu * self.head_dim)
                attn_per_gpu.append(attn_out)

            o_per_gpu = []
            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                o_per_gpu.append(layer.self_attn.o_proj(attn_per_gpu[rank]))

            o_reduced = self.tp.all_reduce(o_per_gpu)
            for rank in range(self.tp_size):
                hidden_per_gpu[rank] = residuals[rank] + o_reduced[rank]

            residuals = [h.clone() for h in hidden_per_gpu]
            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                hidden_per_gpu[rank] = layer.post_attention_layernorm(hidden_per_gpu[rank])

            mlp_per_gpu = []
            for rank in range(self.tp_size):
                layer = self.models[rank].model.layers[layer_idx]
                mlp_per_gpu.append(layer.mlp(hidden_per_gpu[rank]))

            mlp_reduced = self.tp.all_reduce(mlp_per_gpu)
            for rank in range(self.tp_size):
                hidden_per_gpu[rank] = residuals[rank] + mlp_reduced[rank]

        hidden = self.models[0].model.norm(hidden_per_gpu[0])
        return self.models[0].lm_head(hidden)

    def batched_decode_step(self, token_ids: list[int], seq_ids: list[int]) -> torch.Tensor:
        """Batched decode for TP: run each sequence through decode_step.

        For TP models, true batching across sequences requires inter-GPU
        coordination for variable page tables. For now, loop over sequences.
        Returns logits: [batch_size, vocab_size].
        """
        logits_list = []
        for token_id, seq_id in zip(token_ids, seq_ids):
            token_input = torch.tensor([[token_id]], device=self.tp.devices[0])
            logits = self.decode_step(token_input, seq_id)
            logits_list.append(logits[:, -1, :])
        return torch.cat(logits_list, dim=0)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, seq_id: int, max_new_tokens: int = 50,
                 eos_token_id: int | None = None) -> list[int]:
        """Generate tokens with TP. input_ids on GPU 0."""
        for pool in self.kv_pools:
            pool.new_sequence(seq_id)

        logits = self.prefill(input_ids, seq_id)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated = [next_token.item()]

        for _ in range(max_new_tokens - 1):
            if eos_token_id and generated[-1] == eos_token_id:
                break
            token_input = torch.tensor([[generated[-1]]], device=self.tp.devices[0])
            logits = self.decode_step(token_input, seq_id)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token.item())

        return generated
