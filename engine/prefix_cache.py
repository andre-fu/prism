"""Prefix caching: share KV cache pages across requests with identical prefixes.

When 100 requests share the same system prompt, only the first one runs prefill.
Subsequent requests reuse the cached KV pages and only prefill their unique suffix.

Design:
  - Prefix tokens are hashed per page-sized block
  - Cached pages are immutable and ref-counted
  - New tokens always go to fresh pages
  - LRU eviction when free pages are low
"""

import torch
import time
from collections import OrderedDict


class PrefixCache:
    """Caches KV pages for shared token prefixes."""

    def __init__(self, kv_cache, page_size: int = 256):
        self.kv = kv_cache
        self.page_size = page_size

        # prefix_hash -> CachedPrefix
        self._cache: OrderedDict[int, "CachedPrefix"] = OrderedDict()
        # page_id -> refcount (how many sequences use this page)
        self._page_refs: dict[int, int] = {}

    def lookup(self, token_ids: list[int], model_name: str) -> tuple[list[int], int]:
        """Look up cached KV pages for a prefix.

        Returns (cached_page_ids, num_cached_tokens).
        The caller should only prefill tokens AFTER num_cached_tokens.
        """
        if not token_ids:
            return [], 0

        # Hash prefix in page-sized blocks
        blocks = []
        for start in range(0, len(token_ids), self.page_size):
            end = min(start + self.page_size, len(token_ids))
            block_tokens = tuple(token_ids[start:end])
            if len(block_tokens) < self.page_size:
                break  # Partial block — can't cache (mutable)
            blocks.append(block_tokens)

        if not blocks:
            return [], 0

        # Build prefix hash (cumulative — each block depends on all previous)
        prefix_key = hash((model_name, blocks[0]))
        cached_pages = []
        cached_tokens = 0

        for i, block in enumerate(blocks):
            if i > 0:
                prefix_key = hash((prefix_key, block))

            if prefix_key in self._cache:
                entry = self._cache[prefix_key]
                cached_pages.append(entry.page_id)
                cached_tokens += self.page_size
                # Move to end (LRU)
                self._cache.move_to_end(prefix_key)
            else:
                break  # Cache miss — remaining blocks not cached

        return cached_pages, cached_tokens

    def store(self, token_ids: list[int], page_ids: list[int], model_name: str):
        """Store KV pages for a prefix after prefill.

        page_ids: the pages allocated during prefill for this sequence.
        Only full pages (page_size tokens) are cached.
        """
        num_full_pages = len(token_ids) // self.page_size

        prefix_key = None
        for i in range(num_full_pages):
            start = i * self.page_size
            block = tuple(token_ids[start:start + self.page_size])

            if prefix_key is None:
                prefix_key = hash((model_name, block))
            else:
                prefix_key = hash((prefix_key, block))

            if prefix_key not in self._cache and i < len(page_ids):
                self._cache[prefix_key] = CachedPrefix(
                    page_id=page_ids[i],
                    token_hash=prefix_key,
                    created=time.time(),
                )
                self._page_refs[page_ids[i]] = self._page_refs.get(page_ids[i], 0) + 1

    def ref_pages(self, page_ids: list[int]):
        """Increment ref counts for cached pages being used by a new sequence."""
        for pid in page_ids:
            self._page_refs[pid] = self._page_refs.get(pid, 0) + 1

    def unref_pages(self, page_ids: list[int]):
        """Decrement ref counts. Returns pages that can be freed (refcount=0)."""
        freeable = []
        for pid in page_ids:
            if pid in self._page_refs:
                self._page_refs[pid] -= 1
                if self._page_refs[pid] <= 0:
                    del self._page_refs[pid]
                    freeable.append(pid)
        return freeable

    def is_cached_page(self, page_id: int) -> bool:
        """Is this page in the prefix cache (shared, don't free on sequence end)."""
        return self._page_refs.get(page_id, 0) > 0

    def evict_lru(self, num_pages: int = 1) -> list[int]:
        """Evict oldest cached prefixes to free pages. Returns freed page IDs."""
        freed = []
        while len(freed) < num_pages and self._cache:
            _, entry = self._cache.popitem(last=False)  # Oldest first
            if entry.page_id in self._page_refs:
                self._page_refs[entry.page_id] -= 1
                if self._page_refs[entry.page_id] <= 0:
                    del self._page_refs[entry.page_id]
                    freed.append(entry.page_id)
        return freed

    @property
    def num_cached_pages(self) -> int:
        return len(self._cache)


class CachedPrefix:
    __slots__ = ['page_id', 'token_hash', 'created']

    def __init__(self, page_id: int, token_hash: int, created: float):
        self.page_id = page_id
        self.token_hash = token_hash
        self.created = created
