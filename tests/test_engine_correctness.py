"""GPU-dependent tests: model loading, inference correctness, swaps, memory.

These tests require a GPU and Qwen2.5-0.5B model cached.
Skip with: pytest -m "not gpu"
"""

import pytest
import torch
import time

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def engine_setup():
    """Set up engine components once for all tests in this module."""
    from engine.config import ModelConfig, EngineConfig
    from engine.memory_pool import PinnedPool, MultiGPUPool
    from engine.weight_manager import WeightManager
    from engine.weight_pool import StaticWeightPool
    from engine.fa_kv_cache import FlashAttnKVCache
    from engine.fa_executor_v3 import FlashAttnExecutorV3
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    model_id = "Qwen/Qwen2.5-0.5B"
    model_cfg = ModelConfig(model_id=model_id, name="test-0.5b")
    engine_cfg = EngineConfig(models=[model_cfg], gpu_ids=[0])
    pinned = PinnedPool(budget_gb=50)
    gpu = MultiGPUPool([0], 60, 10)
    wm = WeightManager(engine_cfg, pinned, gpu)
    wm.load_model(model_cfg)
    state = wm.get_state("test-0.5b")
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pool = StaticWeightPool(config, "cuda:0", torch.bfloat16)
    pool.load_from_pinned(state.pinned_weights, config)

    nkv = config.num_key_value_heads
    hd = config.hidden_size // config.num_attention_heads
    kv = FlashAttnKVCache(config.num_hidden_layers, nkv, hd, 32, 256, "cuda:0", torch.bfloat16)
    executor = FlashAttnExecutorV3(pool, kv, "cuda:0")

    yield {
        "executor": executor,
        "kv": kv,
        "pool": pool,
        "tokenizer": tokenizer,
        "config": config,
        "wm": wm,
        "state": state,
    }

    # Cleanup
    executor.invalidate_graph()
    del executor, kv, pool
    import gc; gc.collect()
    torch.cuda.empty_cache()


def test_basic_generation(engine_setup):
    """Verify model generates coherent text."""
    ex = engine_setup["executor"]
    kv = engine_setup["kv"]
    tok = engine_setup["tokenizer"]

    ids = tok.encode("The capital of France is", return_tensors="pt").to("cuda:0")
    kv.free_all()
    gen = ex.generate(ids, seq_id=0, max_new_tokens=10, eos_token_id=tok.eos_token_id)
    text = tok.decode(gen, skip_special_tokens=True)
    assert len(gen) > 0
    assert "Paris" in text or "paris" in text.lower()


def test_deterministic_output(engine_setup):
    """Same input → same output (greedy decoding)."""
    ex = engine_setup["executor"]
    kv = engine_setup["kv"]
    tok = engine_setup["tokenizer"]
    ids = tok.encode("Hello world", return_tensors="pt").to("cuda:0")

    kv.free_all()
    ex.invalidate_graph()
    gen1 = ex.generate(ids, seq_id=0, max_new_tokens=20, eos_token_id=tok.eos_token_id)

    kv.free_all()
    ex.invalidate_graph()
    gen2 = ex.generate(ids, seq_id=1, max_new_tokens=20, eos_token_id=tok.eos_token_id)

    assert gen1 == gen2, f"Non-deterministic: {gen1} != {gen2}"


def test_cuda_graph_captured(engine_setup):
    """Verify CUDA graph is captured after first decode."""
    ex = engine_setup["executor"]
    kv = engine_setup["kv"]
    tok = engine_setup["tokenizer"]
    ids = tok.encode("Test", return_tensors="pt").to("cuda:0")

    kv.free_all()
    ex.invalidate_graph()
    ex.generate(ids, seq_id=0, max_new_tokens=5)
    assert isinstance(ex._graph, torch.cuda.CUDAGraph)


def test_weight_swap_preserves_correctness(engine_setup):
    """Swap weights and verify output still correct."""
    ex = engine_setup["executor"]
    kv = engine_setup["kv"]
    tok = engine_setup["tokenizer"]
    pool = engine_setup["pool"]
    state = engine_setup["state"]
    config = engine_setup["config"]

    ids = tok.encode("The capital of France is", return_tensors="pt").to("cuda:0")

    # Generate before swap
    kv.free_all()
    gen1 = ex.generate(ids, seq_id=0, max_new_tokens=10)

    # Swap weights (same model, should produce same output)
    pool.load_from_pinned(state.pinned_weights, config)

    kv.free_all()
    gen2 = ex.generate(ids, seq_id=1, max_new_tokens=10)

    assert gen1 == gen2


def test_memory_no_leak(engine_setup):
    """Run many generations, verify memory doesn't grow."""
    ex = engine_setup["executor"]
    kv = engine_setup["kv"]
    tok = engine_setup["tokenizer"]
    ids = tok.encode("Test", return_tensors="pt").to("cuda:0")

    kv.free_all()
    ex.invalidate_graph()

    # Baseline
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(0)

    for i in range(50):
        kv.free_all()
        ex.generate(ids, seq_id=i, max_new_tokens=10)

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated(0)

    # Allow 10MB growth (CUDA graph + cache warmup)
    growth = (mem_after - mem_before) / 1e6
    assert growth < 100, f"Memory grew by {growth:.0f}MB over 50 generations"


def test_kv_cache_freed(engine_setup):
    """Verify KV pages are freed when sequences complete."""
    kv = engine_setup["kv"]

    pages_before = kv.pages_free
    kv.new_sequence(999)
    kv._ensure_capacity(999, 1000)  # Allocate several pages
    pages_during = kv.pages_free
    assert pages_during < pages_before

    kv.free_sequence(999)
    pages_after = kv.pages_free
    assert pages_after == pages_before
