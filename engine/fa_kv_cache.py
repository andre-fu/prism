"""Paged KV cache for flash_attn.

flash_attn_with_kvcache expects:
  k_cache: [num_blocks, page_size, num_kv_heads, head_dim]
  v_cache: [num_blocks, page_size, num_kv_heads, head_dim]
  block_table: [batch_size, max_blocks_per_seq]  (int32)
  cache_seqlens: [batch_size]  (int32)

The function handles both KV append AND attention in one kernel call.
"""

import torch


class FlashAttnKVCache:
    """Paged KV cache backed by flash_attn's block table format."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_pages: int,
        page_size: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_pages = max_pages
        self.page_size = page_size
        self.device = device
        self.dtype = dtype

        # Per-layer K and V caches: [max_pages, page_size, num_kv_heads, head_dim]
        self.k_caches = [
            torch.zeros(max_pages, page_size, num_kv_heads, head_dim,
                        dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros(max_pages, page_size, num_kv_heads, head_dim,
                        dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # Free page list
        self.free_pages = list(range(max_pages - 1, -1, -1))

        # Per-sequence tracking
        self.seq_pages: dict[int, list[int]] = {}
        self.seq_len: dict[int, int] = {}
        self.seq_owner: dict[int, str] = {}  # seq_id -> model_name

        # Pre-allocated block table buffer (resized as needed)
        self._max_blocks_per_seq = 32
        self._block_table_buf = torch.zeros(
            1, self._max_blocks_per_seq, dtype=torch.int32, device=device
        )
        self._seqlens_buf = torch.zeros(1, dtype=torch.int32, device=device)

    @property
    def pages_used(self) -> int:
        return self.max_pages - len(self.free_pages)

    @property
    def pages_free(self) -> int:
        return len(self.free_pages)

    def new_sequence(self, seq_id: int, owner: str = "", prefix_pages: list[int] | None = None,
                     prefix_len: int = 0):
        """Start a new sequence, optionally with pre-cached prefix pages.

        prefix_pages: page IDs from prefix cache (already contain KV data)
        prefix_len: number of tokens already cached in those pages
        """
        if prefix_pages:
            self.seq_pages[seq_id] = list(prefix_pages) + [self.free_pages.pop()]
            self.seq_len[seq_id] = prefix_len
        else:
            page = self.free_pages.pop()
            self.seq_pages[seq_id] = [page]
            self.seq_len[seq_id] = 0
        self.seq_owner[seq_id] = owner

    def free_sequence(self, seq_id: int):
        if seq_id in self.seq_pages:
            self.free_pages.extend(self.seq_pages.pop(seq_id))
            del self.seq_len[seq_id]
            self.seq_owner.pop(seq_id, None)

    def free_model_sequences(self, model_name: str):
        """Free all sequences belonging to a specific model."""
        to_free = [sid for sid, owner in self.seq_owner.items() if owner == model_name]
        for sid in to_free:
            self.free_sequence(sid)

    def free_all(self):
        self.free_pages = list(range(self.max_pages - 1, -1, -1))
        self.seq_pages.clear()
        self.seq_len.clear()
        self.seq_owner.clear()

    def sequences_for_model(self, model_name: str) -> list[int]:
        """Get all sequence IDs owned by a model."""
        return [sid for sid, owner in self.seq_owner.items() if owner == model_name]

    def memory_used_by_model(self, model_name: str) -> int:
        """Pages used by a model's sequences."""
        return sum(len(self.seq_pages.get(sid, [])) for sid in self.sequences_for_model(model_name))

    def get_seq_len(self, seq_id: int) -> int:
        return self.seq_len.get(seq_id, 0)

    def _ensure_capacity(self, seq_id: int, new_total: int):
        pages_needed = (new_total + self.page_size - 1) // self.page_size
        while len(self.seq_pages[seq_id]) < pages_needed:
            if not self.free_pages:
                raise RuntimeError(f"Out of KV cache pages ({self.max_pages} total)")
            self.seq_pages[seq_id].append(self.free_pages.pop())

    def build_block_table(self, seq_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build flash_attn block_table and cache_seqlens for a batch.

        Returns:
            block_table: [batch_size, max_blocks] int32
            cache_seqlens: [batch_size] int32
        """
        batch_size = len(seq_ids)
        max_blocks = max(len(self.seq_pages.get(sid, [1])) for sid in seq_ids)
        max_blocks = max(max_blocks, 1)

        # Resize buffers if needed
        if batch_size > self._block_table_buf.shape[0] or max_blocks > self._block_table_buf.shape[1]:
            self._block_table_buf = torch.zeros(
                max(batch_size, self._block_table_buf.shape[0]),
                max(max_blocks, self._block_table_buf.shape[1]),
                dtype=torch.int32, device=self.device,
            )
        if batch_size > self._seqlens_buf.shape[0]:
            self._seqlens_buf = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

        bt = self._block_table_buf[:batch_size, :max_blocks]
        sl = self._seqlens_buf[:batch_size]

        for i, sid in enumerate(seq_ids):
            pages = self.seq_pages[sid]
            for j, page_id in enumerate(pages):
                bt[i, j] = page_id
            sl[i] = self.seq_len[sid]

        return bt, sl

    def update_seqlens(self, seq_ids: list[int], num_new_tokens: int | list[int]):
        """Update sequence lengths after flash_attn_with_kvcache writes KV.

        flash_attn handles the actual cache write internally — we just track lengths.
        """
        if isinstance(num_new_tokens, int):
            for sid in seq_ids:
                new_total = self.seq_len[sid] + num_new_tokens
                self._ensure_capacity(sid, new_total)
                self.seq_len[sid] = new_total
        else:
            for sid, n in zip(seq_ids, num_new_tokens):
                new_total = self.seq_len[sid] + n
                self._ensure_capacity(sid, new_total)
                self.seq_len[sid] = new_total

    def has_capacity(self, num_pages: int = 1) -> bool:
        return len(self.free_pages) >= num_pages

    def transfer_sequence(self, seq_id: int, target_kv: "FlashAttnKVCache",
                          target_seq_id: int | None = None) -> int:
        """Copy a sequence's KV data to another KV cache (potentially on a different GPU).

        Used for disaggregated prefill/decode: prefill GPU → decode GPU.
        Returns the number of pages transferred.
        """
        if seq_id not in self.seq_pages:
            return 0

        dst_seq_id = target_seq_id if target_seq_id is not None else seq_id
        src_pages = self.seq_pages[seq_id]
        src_len = self.seq_len[seq_id]
        num_pages_used = (src_len + self.page_size - 1) // self.page_size

        # Allocate pages on target
        target_kv.new_sequence(dst_seq_id)
        target_kv._ensure_capacity(dst_seq_id, src_len)
        dst_pages = target_kv.seq_pages[dst_seq_id]
        target_kv.seq_len[dst_seq_id] = src_len

        # Copy KV data page-by-page, layer-by-layer
        for layer_idx in range(self.num_layers):
            for i in range(num_pages_used):
                src_pid = src_pages[i]
                dst_pid = dst_pages[i]
                # K cache
                target_kv.k_caches[layer_idx][dst_pid].copy_(
                    self.k_caches[layer_idx][src_pid], non_blocking=True
                )
                # V cache
                target_kv.v_caches[layer_idx][dst_pid].copy_(
                    self.v_caches[layer_idx][src_pid], non_blocking=True
                )

        return num_pages_used
