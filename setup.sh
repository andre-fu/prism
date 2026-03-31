#!/bin/bash
set -euo pipefail

echo "=== GPU Swap Setup for 4xH100 Cluster ==="

# Check GPUs
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check driver version (need 580+ for sleep mode GPU migration)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver: $DRIVER_VERSION"

# Check CUDA
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'nvcc not found')"

# Install vLLM
echo ""
echo "--- Installing vLLM ---"
pip install --upgrade pip
pip install vllm

# Verify vLLM install
echo ""
echo "--- Verifying vLLM ---"
python -c "
import vllm
print(f'vLLM version: {vllm.__version__}')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Check sleep mode support
echo ""
echo "--- Checking sleep mode support ---"
python -c "
try:
    from vllm.device_allocator.cumem import CuMemAllocator
    print('CuMemAllocator found - sleep mode supported')
except ImportError:
    print('WARNING: CuMemAllocator not found - sleep mode may not be supported')
    print('Try: pip install vllm>=0.7.0')

try:
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    print('NCCL teardown functions found')
except ImportError as e:
    print(f'WARNING: NCCL teardown functions not found: {e}')
"

echo ""
echo "--- Setup complete ---"
echo "Next: python -m gpu_swap.orchestrator launch --model <model> --tp 4 --name <name>"
