"""Speculative decoding: small draft model proposes tokens, large target verifies.

Uses a small model (e.g., 0.5B) to draft k tokens autoregressively,
then the large model (e.g., 7B) verifies all k tokens in a single forward pass.
Accepted tokens are kept; rejected tokens trigger a rollback.

At temperature=0 (greedy): accept if draft_token == target_token at each position.
Expected: 2-3× throughput improvement depending on draft/target agreement.
"""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from .weight_pool import StaticWeightPool
from .kv_cache import FlashAttnKVCache
from .executor import FlashAttnExecutorV3
from .fused_kernels import fused_rms_norm


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SpeculativeExecutor:
    """Speculative decoding with draft + target models."""

    def __init__(
        self,
        draft_pool: StaticWeightPool,
        draft_kv: FlashAttnKVCache,
        target_pool: StaticWeightPool,
        target_kv: FlashAttnKVCache,
        device: str,
        speculative_k: int = 4,
    ):
        self.draft = FlashAttnExecutorV3(draft_pool, draft_kv, device, use_cuda_graph=True)
        self.target = FlashAttnExecutorV3(target_pool, target_kv, device, use_cuda_graph=False)
        self.draft_kv = draft_kv
        self.target_kv = target_kv
        self.device = device
        self.k = speculative_k

    def prefill(self, input_ids: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Prefill both draft and target models."""
        # Use same seq_id for both (they track independently)
        self.draft_kv.new_sequence(seq_id)
        self.target_kv.new_sequence(seq_id)
        self.draft.prefill(input_ids, seq_id)
        logits = self.target.prefill(input_ids, seq_id)
        return logits

    def speculative_decode_step(self, last_token: int, seq_id: int) -> list[int]:
        """Generate tokens using speculative decoding.

        Returns list of accepted tokens (between 1 and k+1).
        """
        draft_tokens = []
        draft_logits = []

        # Step 1: Draft model generates k tokens
        current_token = last_token
        for _ in range(self.k):
            token_input = torch.tensor([[current_token]], device=self.device)
            logits = self.draft.decode_step(token_input, seq_id)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            draft_tokens.append(next_token)
            current_token = next_token

        # Step 2: Target model verifies all k tokens in one pass
        # Build the verification input: [last_token, draft_0, draft_1, ..., draft_{k-1}]
        verify_tokens = [last_token] + draft_tokens
        verify_input = torch.tensor([verify_tokens], device=self.device)

        # Target prefill with the verification sequence
        # This runs k+1 tokens through the target and produces logits at each position
        target_logits = self.target.prefill(verify_input, seq_id)
        # Actually we need logits at ALL positions, not just the last one
        # Let me use a manual forward pass instead

        # Manual target verification: run all tokens and get per-position logits
        target_logits_all = self._target_verify(verify_input, seq_id)

        # Step 3: Compare draft vs target at each position (greedy)
        accepted = []
        for i in range(self.k):
            target_token = target_logits_all[i].argmax(dim=-1).item()
            if target_token == draft_tokens[i]:
                accepted.append(draft_tokens[i])
            else:
                # Reject: use target's token instead
                accepted.append(target_token)
                break  # Stop at first rejection

        # If all k tokens accepted, also take the target's next prediction
        if len(accepted) == self.k:
            bonus_token = target_logits_all[self.k].argmax(dim=-1).item()
            accepted.append(bonus_token)

        # Rollback draft KV cache to match accepted length
        accepted_len = len(accepted)
        draft_current = self.draft_kv.seq_len.get(seq_id, 0)
        if accepted_len < self.k:
            # Draft generated k tokens but we only accepted fewer
            # Roll back draft KV to: original_len + accepted_len
            rollback_to = draft_current - self.k + accepted_len
            self.draft_kv.seq_len[seq_id] = rollback_to

        return accepted

    def _target_verify(self, tokens: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Run target model on multiple tokens and return logits at each position.

        tokens: [1, seq_len] — the verification sequence
        Returns: [seq_len, vocab_size] logits
        """
        kv = self.target_kv
        w = self.target.w
        seq_len = tokens.shape[1]

        # Prepare KV cache for these tokens
        new_total = kv.get_seq_len(seq_id) + seq_len
        kv._ensure_capacity(seq_id, new_total)
        bt, sl = kv.build_block_table([seq_id])

        start_pos = kv.seq_len[seq_id]
        hidden = F.embedding(tokens, w.embed_w)
        cos = self.target._cos[:, start_pos:start_pos + seq_len]
        sin = self.target._sin[:, start_pos:start_pos + seq_len]

        qs = self.target.q_size
        kvs = self.target.kv_size
        mlps = self.target.mlp_size
        nh = self.target.num_heads
        nkv = self.target.num_kv_heads
        hd = self.target.head_dim
        eps = self.target.rms_eps

        for i in range(self.target.num_layers):
            residual = hidden
            hidden = fused_rms_norm(hidden, w.input_ln_w[i], eps)
            qkv = F.linear(hidden, w.qkv_w[i], w.qkv_b[i])
            q = qkv[:, :, :qs].view(1, seq_len, nh, hd)
            k = qkv[:, :, qs:qs+kvs].view(1, seq_len, nkv, hd)
            v = qkv[:, :, qs+kvs:].view(1, seq_len, nkv, hd)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            attn = flash_attn_with_kvcache(
                q, kv.k_caches[i], kv.v_caches[i], k=k, v=v,
                cache_seqlens=sl, block_table=bt, causal=True,
            )
            o_b = w.o_b[i] if w.has_o_bias else None
            hidden = residual + F.linear(attn.reshape(1, seq_len, -1), w.o_w[i], o_b)
            residual = hidden
            hidden = fused_rms_norm(hidden, w.post_ln_w[i], eps)
            gu = F.linear(hidden, w.gu_w[i])
            d_b = w.down_b[i] if w.has_down_bias else None
            hidden = residual + F.linear(F.silu(gu[:, :, :mlps]) * gu[:, :, mlps:], w.down_w[i], d_b)

        kv.seq_len[seq_id] = new_total
        logits = F.linear(fused_rms_norm(hidden, w.final_norm_w, eps), w.lm_head_w)
        return logits.squeeze(0)  # [seq_len, vocab_size]

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, seq_id: int, max_new_tokens: int = 100,
                 eos_token_id: int | None = None) -> list[int]:
        """Generate with speculative decoding."""
        logits = self.prefill(input_ids, seq_id)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        generated = [next_token]

        while len(generated) < max_new_tokens:
            accepted = self.speculative_decode_step(generated[-1], seq_id)
            generated.extend(accepted)
            if eos_token_id and generated[-1] == eos_token_id:
                break

        return generated[:max_new_tokens]
