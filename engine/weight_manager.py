"""Model weight management: pinned RAM storage, async GPU transfer, residency tracking."""

import time
import torch
from pathlib import Path
from dataclasses import dataclass, field
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from .config import ModelConfig, EngineConfig
from .memory_pool import PinnedPool, MultiGPUPool


@dataclass
class ModelState:
    """Tracks a model's weight residency and metadata."""
    config: ModelConfig
    hf_config: object                # HuggingFace model config
    tokenizer: object                # HuggingFace tokenizer
    pinned_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    pinned_bytes: int = 0
    # Per-GPU: gpu_id -> dict of GPU tensors (after load_to_gpu)
    gpu_weights: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    gpu_bytes_per_device: int = 0
    # Model skeleton (meta device, reused across swaps)
    skeleton: object = None
    # Active model(s) on GPU — single model for TP=1, list for TP>1
    active_model: object = None
    active_models: list = field(default_factory=list)  # For TP: one model per rank
    active_gpu_id: int | None = None     # Primary GPU (TP=1) or first GPU (TP>1)
    active_gpu_ids: list[int] = field(default_factory=list)  # All GPUs used
    last_served: float = 0.0


class WeightManager:
    """Manages model weights across pinned RAM and GPU memory."""

    def __init__(self, engine_config: EngineConfig, pinned_pool: PinnedPool, gpu_pool: MultiGPUPool):
        self.engine_config = engine_config
        self.pinned_pool = pinned_pool
        self.gpu_pool = gpu_pool
        self.models: dict[str, ModelState] = {}

    def load_model(self, model_config: ModelConfig) -> ModelState:
        """Load a model's weights into pinned RAM and create its skeleton.

        This is a one-time startup cost per model.
        """
        name = model_config.name
        if name in self.models:
            return self.models[name]

        print(f"[WeightManager] Loading {name} ({model_config.model_id})...")

        # Download / locate safetensors
        t0 = time.perf_counter()
        cache_dir = snapshot_download(
            model_config.model_id,
            ignore_patterns=["*.bin", "*.gguf"],
        )
        paths = sorted(Path(cache_dir).glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"No safetensors found for {model_config.model_id}")

        # Load HF config and tokenizer
        hf_config = AutoConfig.from_pretrained(model_config.model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

        # Load weights into pinned memory
        dtype = getattr(torch, model_config.dtype)
        pinned_weights = {}
        total_bytes = 0
        for path in paths:
            shard = load_file(str(path), device="cpu")
            for param_name, tensor in shard.items():
                pinned = torch.empty_like(tensor, dtype=dtype).pin_memory()
                pinned.copy_(tensor.to(dtype))
                pinned_weights[param_name] = pinned
                total_bytes += pinned.nbytes
            del shard

        # Handle tied weights
        if getattr(hf_config, "tie_word_embeddings", False):
            if "model.embed_tokens.weight" in pinned_weights:
                pinned_weights["lm_head.weight"] = pinned_weights["model.embed_tokens.weight"]

        self.pinned_pool.track(total_bytes)

        # Create model skeleton on meta device (fast: ~0.4s)
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(hf_config, dtype=dtype)

        t_load = time.perf_counter() - t0
        gb = total_bytes / 1e9
        print(f"[WeightManager] {name}: {len(pinned_weights)} tensors, {gb:.2f} GB, {t_load:.1f}s")

        # Estimate per-device bytes after TP sharding
        # Rough: replicated params (embed, norms, lm_head) + sharded params / tp_size
        tp_size = model_config.tp_size
        if tp_size > 1:
            from .distributed import get_shard_plan, ShardPlan
            replicated_bytes = 0
            sharded_bytes = 0
            for pname, t in pinned_weights.items():
                plan = get_shard_plan(pname)
                if plan == ShardPlan.REPLICATE:
                    replicated_bytes += t.nbytes
                else:
                    sharded_bytes += t.nbytes
            per_device = replicated_bytes + sharded_bytes // tp_size
        else:
            # Include ~60% overhead for executor's stacked weight buffers (QKV, gate+up)
            # These are created by torch.cat in FlashAttnExecutorV2
            per_device = int(total_bytes * 1.6)

        state = ModelState(
            config=model_config,
            hf_config=hf_config,
            tokenizer=tokenizer,
            pinned_weights=pinned_weights,
            pinned_bytes=total_bytes,
            skeleton=skeleton,
            gpu_bytes_per_device=per_device,
        )
        self.models[name] = state
        return state

    def load_to_gpu(self, name: str, gpu_id: int) -> torch.nn.Module:
        """Transfer a model's weights from pinned RAM to GPU and return an active model.

        Returns the model ready for inference on the specified GPU.
        """
        state = self.models[name]
        device = f"cuda:{gpu_id}"

        # Check if already loaded on this GPU
        if state.active_gpu_id == gpu_id and state.active_model is not None:
            state.last_served = time.time()
            return state.active_model

        # Check GPU budget
        pool = self.gpu_pool[gpu_id]
        if not pool.can_fit_weights(state.gpu_bytes_per_device):
            raise RuntimeError(
                f"GPU {gpu_id} cannot fit {name}: needs {state.gpu_bytes_per_device / 1e9:.1f} GB, "
                f"only {pool.weight_free_gb:.1f} GB free"
            )

        t0 = time.perf_counter()

        # Async copy pinned -> GPU
        copy_stream = pool.copy_stream
        gpu_weights = {}
        with torch.cuda.stream(copy_stream):
            for param_name, tensor in state.pinned_weights.items():
                gpu_weights[param_name] = tensor.to(device, non_blocking=True)
        copy_stream.synchronize()

        t_transfer = time.perf_counter() - t0
        bw = state.pinned_bytes / 1e9 / t_transfer

        # Create fresh model from skeleton pattern and load weights
        dtype = getattr(torch, state.config.dtype)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(state.hf_config, dtype=dtype)

        model.load_state_dict(gpu_weights, strict=False, assign=True)

        # Reinit rotary embedding buffers (not in safetensors)
        for buf_name, buf in list(model.named_buffers()):
            if buf.device == torch.device("meta") and "inv_freq" in buf_name:
                dim = buf.shape[0] * 2
                rope_theta = getattr(state.hf_config, "rope_theta", 10000.0)
                inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                parent = model
                for part in buf_name.rsplit(".", 1)[0].split("."):
                    parent = getattr(parent, part)
                parent.inv_freq = inv_freq.to(device)

        model.to(device)
        model.eval()

        t_total = time.perf_counter() - t0

        # Track in memory pool
        pool.track_weights(state.gpu_bytes_per_device)

        # Update state
        state.gpu_weights[gpu_id] = gpu_weights
        state.active_model = model
        state.active_gpu_id = gpu_id
        state.last_served = time.time()

        del gpu_weights
        print(f"[WeightManager] {name} -> GPU {gpu_id}: {t_transfer:.3f}s transfer ({bw:.1f} GB/s), {t_total:.3f}s total")

        return model

    def load_to_gpu_tp(self, name: str, gpu_ids: list[int]) -> list[torch.nn.Module]:
        """Load a model sharded across multiple GPUs (tensor parallelism).

        Returns a list of models, one per GPU rank.
        """
        from .distributed import shard_all_weights
        from .test_tp import create_model_on_device

        state = self.models[name]
        tp_size = len(gpu_ids)
        dtype = getattr(torch, state.config.dtype)

        t0 = time.perf_counter()

        models = []
        for rank, gpu_id in enumerate(gpu_ids):
            device = f"cuda:{gpu_id}"
            sharded = shard_all_weights(state.pinned_weights, rank, tp_size)
            gpu_weights = {n: t.to(dtype).to(device) for n, t in sharded.items()}
            model = create_model_on_device(state.hf_config, gpu_weights, device, dtype)
            models.append(model)
            state.gpu_weights[gpu_id] = gpu_weights
            self.gpu_pool[gpu_id].track_weights(state.gpu_bytes_per_device)

        t_total = time.perf_counter() - t0

        state.active_models = models
        state.active_model = models[0]  # Primary for compatibility
        state.active_gpu_id = gpu_ids[0]
        state.active_gpu_ids = list(gpu_ids)
        state.last_served = time.time()

        print(f"[WeightManager] {name} -> GPUs {gpu_ids} (TP={tp_size}): {t_total:.1f}s")
        return models

    def evict_from_gpu(self, name: str):
        """Remove a model's weights from GPU(s), freeing HBM."""
        state = self.models[name]
        if state.active_gpu_id is None:
            return

        t0 = time.perf_counter()
        gpu_ids = state.active_gpu_ids if state.active_gpu_ids else [state.active_gpu_id]

        # Delete GPU models and tensors
        if state.active_models:
            for m in state.active_models:
                del m
            state.active_models.clear()
        if state.active_model is not None:
            del state.active_model
            state.active_model = None

        for gpu_id in gpu_ids:
            if gpu_id in state.gpu_weights:
                del state.gpu_weights[gpu_id]
            self.gpu_pool[gpu_id].untrack_weights(state.gpu_bytes_per_device)

        # Force garbage collection to break reference cycles (tied weights, etc.)
        import gc
        gc.collect()
        # Empty cache on each device (torch.cuda.empty_cache only affects current device)
        for gid in gpu_ids:
            with torch.cuda.device(gid):
                torch.cuda.empty_cache()

        state.active_gpu_id = None
        state.active_gpu_ids.clear()

        t_evict = time.perf_counter() - t0
        print(f"[WeightManager] Evicted {name} from GPUs {gpu_ids}: {t_evict:.3f}s")

    def resident_on(self, gpu_id: int) -> list[str]:
        """Which models have weights loaded on this GPU."""
        result = []
        for name, state in self.models.items():
            if state.active_gpu_id == gpu_id:
                result.append(name)
            elif gpu_id in state.active_gpu_ids:
                result.append(name)
        return result

    def is_loaded(self, name: str) -> bool:
        """Is this model currently on a GPU?"""
        state = self.models.get(name)
        return state is not None and state.active_gpu_id is not None

    def get_model(self, name: str) -> torch.nn.Module | None:
        """Get the active GPU model (TP=1), or None if not loaded."""
        state = self.models.get(name)
        if state and state.active_model is not None:
            return state.active_model
        return None

    def get_models(self, name: str) -> list[torch.nn.Module]:
        """Get all active GPU models (TP>1). Returns empty list if not loaded."""
        state = self.models.get(name)
        if state and state.active_models:
            return state.active_models
        if state and state.active_model is not None:
            return [state.active_model]
        return []

    def get_state(self, name: str) -> ModelState | None:
        return self.models.get(name)

    def lru_candidate(self, gpu_id: int) -> str | None:
        """Return the least-recently-served model on this GPU, for eviction."""
        candidates = []
        for name, state in self.models.items():
            if state.active_gpu_id == gpu_id or gpu_id in (state.active_gpu_ids or []):
                candidates.append((state.last_served, name))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]
