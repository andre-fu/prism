"""Paged KV cache backed by FlashInfer."""

import torch
import flashinfer


class PagedKVPool:
    """Per-model paged KV cache pool.

    Each model gets its own pool because different models have different
    num_kv_heads and head_dim. The pool is a stack of per-layer page tensors.

    Page layout (NHD): [max_pages, 2, page_size, num_kv_heads, head_dim]
      - dim 0: page index
      - dim 1: 0=K, 1=V
      - dim 2: token slot within page
      - dim 3: KV head index
      - dim 4: head dimension
    """

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

        # Per-layer page pool
        self.pools = [
            torch.zeros(max_pages, 2, page_size, num_kv_heads, head_dim,
                        dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # Free page list (LIFO for cache locality)
        self.free_pages = list(range(max_pages - 1, -1, -1))

        # Per-sequence state
        self.seq_pages: dict[int, list[int]] = {}
        self.seq_len: dict[int, int] = {}

        # FlashInfer workspaces (allocated once, reused)
        self.prefill_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.decode_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, kv_layout="NHD"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace, kv_layout="NHD", use_tensor_cores=True
        )

    @property
    def pages_used(self) -> int:
        return self.max_pages - len(self.free_pages)

    @property
    def pages_free(self) -> int:
        return len(self.free_pages)

    def new_sequence(self, seq_id: int):
        """Start tracking a new sequence."""
        page = self.free_pages.pop()
        self.seq_pages[seq_id] = [page]
        self.seq_len[seq_id] = 0

    def free_sequence(self, seq_id: int):
        """Free all pages for a sequence."""
        if seq_id in self.seq_pages:
            self.free_pages.extend(self.seq_pages.pop(seq_id))
            del self.seq_len[seq_id]

    def free_all(self):
        """Free all sequences and reset the pool."""
        self.free_pages = list(range(self.max_pages - 1, -1, -1))
        self.seq_pages.clear()
        self.seq_len.clear()

    def has_capacity(self, num_pages: int = 1) -> bool:
        """Check if there are enough free pages."""
        return len(self.free_pages) >= num_pages

    def prepare_append(self, seq_id: int, num_tokens: int) -> int:
        """Ensure capacity for new tokens and update seq_len.

        Returns the old sequence length (start position for the new tokens).
        Must be called once before appending KV for all layers.
        Raises RuntimeError if out of pages.
        """
        old_len = self.seq_len[seq_id]
        needed_after = old_len + num_tokens
        pages_needed = (needed_after + self.page_size - 1) // self.page_size
        while len(self.seq_pages[seq_id]) < pages_needed:
            if not self.free_pages:
                raise RuntimeError(
                    f"Out of KV cache pages: {self.max_pages} total, "
                    f"{len(self.seq_pages)} active sequences. "
                    f"Free some sequences or increase kv_cache_budget_gb."
                )
            self.seq_pages[seq_id].append(self.free_pages.pop())
        self.seq_len[seq_id] = needed_after
        return old_len

    def build_page_table(self, seq_ids: list[int]):
        """Build FlashInfer CSR-format page table for a batch of sequences.

        Returns (kv_indptr, kv_indices, kv_last_page_len) tensors.
        """
        indptr = [0]
        indices = []
        last_page_lens = []
        for sid in seq_ids:
            slen = self.seq_len[sid]
            pages = self.seq_pages[sid]
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

    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Append K, V to the paged cache for one layer using FlashInfer kernel.

        k, v: [nnz, num_kv_heads, head_dim]
        """
        flashinfer.append_paged_kv_cache(
            k, v, batch_indices, positions,
            self.pools[layer_idx],
            kv_indices, kv_indptr, kv_last_page_len,
            kv_layout="NHD",
        )

    def plan_prefill(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
    ):
        """Plan a prefill attention operation (call once, reuse across layers)."""
        dtype_str = "bfloat16" if self.dtype == torch.bfloat16 else "float16"
        self.prefill_wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
            num_qo_heads, self.num_kv_heads, self.head_dim, self.page_size,
            causal=True, pos_encoding_mode="NONE",
            q_data_type=dtype_str, kv_data_type=dtype_str,
        )

    def run_prefill(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        """Run prefill attention for one layer. q: [nnz, num_qo_heads, head_dim]."""
        return self.prefill_wrapper.run(q, self.pools[layer_idx])

    def plan_decode(
        self,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
    ):
        """Plan a decode attention operation (call once, reuse across layers)."""
        dtype_str = "bfloat16" if self.dtype == torch.bfloat16 else "float16"
        self.decode_wrapper.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            num_qo_heads, self.num_kv_heads, self.head_dim, self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype_str, kv_data_type=dtype_str,
        )

    def run_decode(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        """Run decode attention for one layer. q: [batch_size, num_qo_heads, head_dim]."""
        return self.decode_wrapper.run(q, self.pools[layer_idx])
