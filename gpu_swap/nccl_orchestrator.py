"""
NCCL teardown/rebuild orchestrator for multi-GPU vLLM sleep/wake.

vLLM's sleep mode releases GPU memory but does NOT tear down NCCL communicators.
On multi-GPU (TP>1), NCCL state prevents two vLLM processes from sharing GPUs.
This module adds NCCL teardown before sleep and rebuild after wake.

The key insight: vLLM workers communicate via shared-memory MessageQueues (not NCCL),
so we can tear down NCCL and still send RPCs to workers.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

import torch

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Captures distributed init params so we can rebuild after teardown."""
    world_size: int
    rank: int
    local_rank: int
    distributed_init_method: str
    backend: str
    tensor_parallel_size: int
    pipeline_parallel_size: int


def capture_distributed_config() -> DistributedConfig:
    """Capture current distributed configuration for later reinit."""
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
        get_pipeline_model_parallel_world_size,
    )

    return DistributedConfig(
        world_size=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        local_rank=torch.cuda.current_device(),
        distributed_init_method="env://",
        backend=torch.distributed.get_backend(),
        tensor_parallel_size=get_tensor_model_parallel_world_size(),
        pipeline_parallel_size=get_pipeline_model_parallel_world_size(),
    )


def teardown_nccl() -> dict:
    """
    Tear down all NCCL communicators and torch.distributed process groups.
    Returns a dict of state needed for rebuild.

    Must be called on each worker process (via collective_rpc).
    """
    import vllm.distributed.parallel_state as ps

    # Capture config before teardown
    config = capture_distributed_config()
    config_dict = {
        "world_size": config.world_size,
        "rank": config.rank,
        "local_rank": config.local_rank,
        "distributed_init_method": config.distributed_init_method,
        "backend": config.backend,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
    }

    logger.info(
        "Tearing down NCCL on rank %d (TP=%d, PP=%d)",
        config.rank,
        config.tensor_parallel_size,
        config.pipeline_parallel_size,
    )

    # Destroy all parallel groups (TP, PP, DP, EP, etc.)
    ps.destroy_model_parallel()

    # Destroy the world group and torch.distributed
    ps.destroy_distributed_environment()

    # Reset the group name counter so reinit creates groups with the same names
    ps._group_name_counter.clear()

    logger.info("NCCL teardown complete on rank %d", config.rank)
    return config_dict


def rebuild_nccl(config_dict: dict) -> None:
    """
    Rebuild NCCL communicators and torch.distributed process groups.
    Takes the config dict returned by teardown_nccl().

    Must be called on each worker process (via collective_rpc).
    """
    import vllm.distributed.parallel_state as ps

    logger.info(
        "Rebuilding NCCL on rank %d (TP=%d, PP=%d)",
        config_dict["rank"],
        config_dict["tensor_parallel_size"],
        config_dict["pipeline_parallel_size"],
    )

    # Reinitialize torch.distributed
    ps.init_distributed_environment(
        world_size=config_dict["world_size"],
        rank=config_dict["rank"],
        local_rank=config_dict["local_rank"],
        distributed_init_method=config_dict["distributed_init_method"],
        backend=config_dict["backend"],
    )

    # Rebuild all parallel groups
    ps.ensure_model_parallel_initialized(
        tensor_model_parallel_size=config_dict["tensor_parallel_size"],
        pipeline_model_parallel_size=config_dict["pipeline_parallel_size"],
    )

    logger.info("NCCL rebuild complete on rank %d", config_dict["rank"])
