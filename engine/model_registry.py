"""Model registry: lazy loading from disk, LRU eviction from pinned RAM.

Supports hundreds of registered models without loading all at startup.
Models are loaded to pinned RAM on first request and LRU-evicted when pool is full.
"""

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download

from .config import ModelConfig


@dataclass
class RegistryEntry:
    config: ModelConfig
    hf_config: object | None = None
    tokenizer: object | None = None
    safetensors_paths: list[Path] = field(default_factory=list)
    pinned_weights: dict[str, torch.Tensor] | None = None
    pinned_bytes: int = 0
    state: str = "cold"  # cold | loading | pinned
    last_accessed: float = 0.0
    _load_event: threading.Event | None = None


class ModelRegistry:
    """Lazy model registry with LRU eviction from pinned RAM."""

    def __init__(self, pinned_budget_gb: float = 400.0):
        self._entries: OrderedDict[str, RegistryEntry] = OrderedDict()
        self._pinned_budget = int(pinned_budget_gb * 1e9)
        self._pinned_used = 0
        self._lock = threading.Lock()
        self._loader = threading.Thread(target=self._loader_loop, daemon=True)
        self._load_queue: list[str] = []
        self._loader_running = False

    def register(self, config: ModelConfig) -> RegistryEntry:
        """Register a model. Only fetches HF config (fast), not weights."""
        name = config.name
        if name in self._entries:
            return self._entries[name]

        # Fetch HF config (small JSON, fast)
        hf_config = AutoConfig.from_pretrained(config.model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)

        # Locate safetensors (may trigger download)
        cache_dir = snapshot_download(config.model_id, ignore_patterns=["*.bin", "*.gguf"])
        paths = sorted(Path(cache_dir).glob("*.safetensors"))

        entry = RegistryEntry(
            config=config,
            hf_config=hf_config,
            tokenizer=tokenizer,
            safetensors_paths=paths,
        )
        self._entries[name] = entry
        print(f"[Registry] Registered {name} ({config.model_id}, {len(paths)} shards)")
        return entry

    def ensure_pinned(self, name: str, blocking: bool = True) -> RegistryEntry:
        """Ensure model weights are in pinned RAM. Load from disk if needed."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"Model '{name}' not registered")

        if entry.state == "pinned":
            entry.last_accessed = time.time()
            self._entries.move_to_end(name)  # LRU refresh
            return entry

        if entry.state == "loading":
            if blocking and entry._load_event:
                entry._load_event.wait(timeout=60)
            return entry

        # Need to load from disk
        entry.state = "loading"
        entry._load_event = threading.Event()

        self._load_from_disk(name)

        if blocking:
            entry._load_event.wait(timeout=60)

        return entry

    def _load_from_disk(self, name: str):
        """Load safetensors into pinned RAM."""
        entry = self._entries[name]
        t0 = time.perf_counter()

        # Evict from pinned RAM if needed
        self._evict_until_free(entry._estimate_bytes())

        # Load weights
        weights = {}
        total_bytes = 0
        dtype = getattr(torch, entry.config.dtype)
        for path in entry.safetensors_paths:
            shard = load_file(str(path), device="cpu")
            for pname, tensor in shard.items():
                pinned = torch.empty_like(tensor, dtype=dtype).pin_memory()
                pinned.copy_(tensor.to(dtype))
                weights[pname] = pinned
                total_bytes += pinned.nbytes
            del shard

        # Handle tied weights
        if getattr(entry.hf_config, "tie_word_embeddings", False):
            if "model.embed_tokens.weight" in weights:
                weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        entry.pinned_weights = weights
        entry.pinned_bytes = total_bytes
        entry.state = "pinned"
        entry.last_accessed = time.time()

        with self._lock:
            self._pinned_used += total_bytes

        t_load = time.perf_counter() - t0
        print(f"[Registry] Loaded {name} to pinned RAM: {total_bytes/1e9:.1f}GB in {t_load:.1f}s")

        if entry._load_event:
            entry._load_event.set()

    def _evict_until_free(self, needed_bytes: int):
        """LRU evict models from pinned RAM until there's enough space."""
        with self._lock:
            while self._pinned_used + needed_bytes > self._pinned_budget:
                # Find oldest pinned model
                victim = None
                for name, entry in self._entries.items():
                    if entry.state == "pinned":
                        victim = name
                        break  # OrderedDict: first = oldest

                if victim is None:
                    break  # Nothing to evict

                self._evict_from_pinned(victim)

    def _evict_from_pinned(self, name: str):
        """Remove a model's weights from pinned RAM."""
        entry = self._entries.get(name)
        if entry is None or entry.state != "pinned":
            return

        freed = entry.pinned_bytes
        del entry.pinned_weights
        entry.pinned_weights = None
        entry.pinned_bytes = 0
        entry.state = "cold"
        self._pinned_used -= freed
        print(f"[Registry] Evicted {name} from pinned RAM ({freed/1e9:.1f}GB freed)")

    def get(self, name: str) -> RegistryEntry | None:
        return self._entries.get(name)

    def list_models(self) -> list[dict]:
        return [
            {"name": name, "model_id": e.config.model_id, "state": e.state,
             "pinned_gb": e.pinned_bytes / 1e9}
            for name, e in self._entries.items()
        ]

    @property
    def pinned_used_gb(self) -> float:
        return self._pinned_used / 1e9

    def _loader_loop(self):
        """Background thread for async loading."""
        while self._loader_running:
            if self._load_queue:
                name = self._load_queue.pop(0)
                try:
                    self._load_from_disk(name)
                except Exception as e:
                    print(f"[Registry] Background load failed for {name}: {e}")
            else:
                time.sleep(0.1)

    def start_background_loader(self):
        self._loader_running = True
        self._loader.start()

    def stop(self):
        self._loader_running = False

    def _estimate_bytes_for_entry(self, entry: RegistryEntry) -> int:
        """Estimate pinned RAM needed for a model (without loading)."""
        return entry._estimate_bytes()


# Add _estimate_bytes to RegistryEntry
def _estimate_bytes(self):
    if self.pinned_bytes > 0:
        return self.pinned_bytes
    # Rough estimate from HF config
    if self.hf_config:
        params = getattr(self.hf_config, 'num_parameters', None)
        if params:
            return params * 2  # bf16
    # Fallback: estimate from file sizes
    total = 0
    for p in self.safetensors_paths:
        total += p.stat().st_size
    return total

RegistryEntry._estimate_bytes = _estimate_bytes
