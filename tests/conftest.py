"""Shared fixtures for pytest test suite."""

import pytest
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@pytest.fixture(scope="session")
def gpu_available():
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def num_gpus():
    return torch.cuda.device_count()


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean GPU memory after each test."""
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        for g in range(torch.cuda.device_count()):
            with torch.cuda.device(g):
                torch.cuda.empty_cache()
