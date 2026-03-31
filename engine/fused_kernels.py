"""Triton fused kernels for model execution."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused SiLU(gate) * up — replaces separate silu + elementwise multiply."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    out = silu * up
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute SiLU(gate) * up with a single fused kernel."""
    out = torch.empty_like(gate)
    n = gate.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _fused_silu_mul_kernel[grid](gate, up, out, n, BLOCK_SIZE=1024)
    return out


@triton.jit
def _fused_rms_norm_kernel(
    x_ptr, weight_ptr, out_ptr,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm: out = x * weight / rms(x)."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    x = tl.load(x_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # RMS
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / hidden_size
    rms = tl.rsqrt(mean_sq + eps)

    out = x * rms * w
    tl.store(out_ptr + row * hidden_size + offsets, out.to(tl.bfloat16), mask=mask)


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm. x: [..., hidden_size], weight: [hidden_size]."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    out = torch.empty_like(x_flat)
    hidden_size = orig_shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    # Ensure Triton launches on the correct GPU
    with torch.cuda.device(x.device):
        _fused_rms_norm_kernel[(x_flat.shape[0],)](
            x_flat, weight, out, hidden_size, eps, BLOCK_SIZE=BLOCK_SIZE,
        )
    return out.reshape(orig_shape)
