"""Engine configuration."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str                   # HuggingFace model ID or local path
    name: str = ""                  # Display name (defaults to model_id)
    tp_size: int = 1                # Tensor parallel size (1, 2, or 4)
    dtype: str = "bfloat16"         # "bfloat16", "float16"
    max_batch_size: int = 32
    max_seq_len: int = 4096

    def __post_init__(self):
        if not self.name:
            self.name = self.model_id.split("/")[-1]


@dataclass
class EngineConfig:
    """Top-level engine configuration."""
    models: list[ModelConfig] = field(default_factory=list)
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    t0_budget_gb: float = 40.0      # HBM budget for weights per GPU
    t1_budget_gb: float = 400.0     # Total pinned RAM budget
    kv_cache_budget_gb: float = 35.0 # HBM for KV cache per GPU
    kv_page_size: int = 16          # Tokens per KV cache page


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    policy: str = "round_robin"
    prefetch_queue_threshold: int = 1   # Queue depth to trigger prefetch
    max_batch_tokens: int = 4096
    max_consecutive_batches: int = 16   # Before yielding to next model


@dataclass
class ServerConfig:
    """HTTP server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_queue_depth: int = 64           # Per-model max pending requests
    request_timeout_s: float = 120.0


def load_config(path: str) -> tuple[EngineConfig, SchedulerConfig, ServerConfig]:
    """Load configuration from a YAML file.

    Example YAML:
        models:
          - model_id: Qwen/Qwen2.5-7B-Instruct
            name: qwen-7b
            tp_size: 1
          - model_id: Qwen/Qwen2.5-14B-Instruct
            name: qwen-14b
            tp_size: 1
        gpu_ids: [0, 1, 2, 3]
        t0_budget_gb: 40.0
        kv_cache_budget_gb: 35.0
        scheduler:
          max_consecutive_batches: 32
        server:
          port: 8000
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = [ModelConfig(**m) for m in raw.get("models", [])]
    engine = EngineConfig(
        models=models,
        gpu_ids=raw.get("gpu_ids", [0]),
        t0_budget_gb=raw.get("t0_budget_gb", 40.0),
        t1_budget_gb=raw.get("t1_budget_gb", 400.0),
        kv_cache_budget_gb=raw.get("kv_cache_budget_gb", 35.0),
        kv_page_size=raw.get("kv_page_size", 16),
    )

    sched_raw = raw.get("scheduler", {})
    scheduler = SchedulerConfig(
        policy=sched_raw.get("policy", "round_robin"),
        prefetch_queue_threshold=sched_raw.get("prefetch_queue_threshold", 1),
        max_batch_tokens=sched_raw.get("max_batch_tokens", 4096),
        max_consecutive_batches=sched_raw.get("max_consecutive_batches", 16),
    )

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "0.0.0.0"),
        port=server_raw.get("port", 8000),
        max_queue_depth=server_raw.get("max_queue_depth", 64),
        request_timeout_s=server_raw.get("request_timeout_s", 120.0),
    )

    return engine, scheduler, server
