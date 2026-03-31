"""Tensor parallelism: weight sharding and cross-GPU communication."""

import torch
from dataclasses import dataclass


@dataclass
class ShardPlan:
    """How to shard a single weight tensor."""
    COLUMN = "column"   # Split along dim 0 (output dim) — QKV, gate, up projections
    ROW = "row"         # Split along dim 1 (input dim) — o_proj, down_proj (needs all-reduce)
    REPLICATE = "replicate"  # Same on all GPUs — embeddings, layernorms


def get_shard_plan(param_name: str) -> str:
    """Determine sharding strategy for a parameter based on its name."""
    # Column-parallel: split output dimension
    if any(k in param_name for k in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
        return ShardPlan.COLUMN

    # Row-parallel: split input dimension (needs all-reduce after)
    if any(k in param_name for k in ["o_proj", "down_proj"]):
        return ShardPlan.ROW

    # Everything else is replicated
    return ShardPlan.REPLICATE


def shard_weight(tensor: torch.Tensor, tp_rank: int, tp_size: int, plan: str) -> torch.Tensor:
    """Extract the shard for a given TP rank."""
    if plan == ShardPlan.REPLICATE or tp_size == 1:
        return tensor

    if plan == ShardPlan.COLUMN:
        # Split along dim 0 (output features)
        chunk_size = tensor.shape[0] // tp_size
        return tensor[tp_rank * chunk_size : (tp_rank + 1) * chunk_size].contiguous()

    if plan == ShardPlan.ROW:
        # Split along dim 1 (input features) for weights
        if tensor.dim() == 1:
            # Bias for row-parallel: only rank 0 keeps it, others zero
            if tp_rank == 0:
                return tensor
            return torch.zeros_like(tensor)
        chunk_size = tensor.shape[1] // tp_size
        return tensor[:, tp_rank * chunk_size : (tp_rank + 1) * chunk_size].contiguous()

    raise ValueError(f"Unknown shard plan: {plan}")


def shard_all_weights(
    weights: dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
) -> dict[str, torch.Tensor]:
    """Shard all model weights for a given TP rank."""
    sharded = {}
    for name, tensor in weights.items():
        plan = get_shard_plan(name)
        sharded[name] = shard_weight(tensor, tp_rank, tp_size, plan)
    return sharded


def all_reduce_sum(tensors_per_gpu: list[torch.Tensor], target_device: str) -> torch.Tensor:
    """NCCL all-reduce: sum tensors across GPUs in-place.

    Uses torch.cuda.nccl for single-process multi-GPU. Each tensor must be
    on a different GPU. After the call, all tensors contain the sum.
    Returns the tensor on the target device.
    """
    if len(tensors_per_gpu) == 1:
        return tensors_per_gpu[0]
    torch.cuda.nccl.all_reduce(tensors_per_gpu)
    # Return the copy on the target device
    for t in tensors_per_gpu:
        if str(t.device) == target_device:
            return t
    return tensors_per_gpu[0].to(target_device)


class TPGroup:
    """Manages a tensor-parallel group of GPUs for one model."""

    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self.tp_size = len(gpu_ids)
        self.devices = [f"cuda:{g}" for g in gpu_ids]

    def shard_and_place_weights(
        self, weights: dict[str, torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Shard weights and place each shard on its GPU. Returns per-GPU weight dicts."""
        per_gpu = []
        for rank, gpu_id in enumerate(self.gpu_ids):
            device = f"cuda:{gpu_id}"
            sharded = shard_all_weights(weights, rank, self.tp_size)
            gpu_weights = {name: t.to(device) for name, t in sharded.items()}
            per_gpu.append(gpu_weights)
        return per_gpu

    def all_reduce(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Sum tensors from all GPUs. In-place NCCL — result on ALL GPUs."""
        if self.tp_size == 1:
            return tensors
        torch.cuda.nccl.all_reduce(tensors)
        return tensors
