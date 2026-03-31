"""Test model upload: validation, storage, integrity checking."""

import json
import pytest
import torch
from safetensors.torch import save_file
from engine.model_upload import ModelUploadManager


@pytest.fixture
def upload_mgr(tmp_path):
    return ModelUploadManager(upload_dir=str(tmp_path / "uploads"))


def _make_fake_model(hidden=256, layers=2, heads=4, kv_heads=2, vocab=1000, mlp=512):
    """Create minimal fake safetensors for testing."""
    hd = hidden // heads
    config = {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "intermediate_size": mlp,
        "vocab_size": vocab,
        "architectures": ["TestForCausalLM"],
    }

    weights = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.randn(hidden),
    }
    for i in range(layers):
        p = f"model.layers.{i}"
        weights[f"{p}.self_attn.q_proj.weight"] = torch.randn(heads * hd, hidden)
        weights[f"{p}.self_attn.k_proj.weight"] = torch.randn(kv_heads * hd, hidden)
        weights[f"{p}.self_attn.v_proj.weight"] = torch.randn(kv_heads * hd, hidden)
        weights[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, heads * hd)
        weights[f"{p}.mlp.gate_proj.weight"] = torch.randn(mlp, hidden)
        weights[f"{p}.mlp.up_proj.weight"] = torch.randn(mlp, hidden)
        weights[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, mlp)
        weights[f"{p}.input_layernorm.weight"] = torch.randn(hidden)
        weights[f"{p}.post_attention_layernorm.weight"] = torch.randn(hidden)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(weights, f.name)
        f.seek(0)
    with open(f.name, "rb") as f:
        data = f.read()
    import os
    os.unlink(f.name)
    return config, [("model.safetensors", data)]


def test_upload_valid_model(upload_mgr):
    config, files = _make_fake_model()
    result = upload_mgr.validate_and_store("test-model", config, files)
    assert result["status"] == "ok"
    assert result["name"] == "test-model"
    assert result["num_params"] > 0
    assert result["architecture"] == "TestForCausalLM"


def test_upload_missing_config_field(upload_mgr):
    config = {"hidden_size": 256}  # Missing num_hidden_layers, num_attention_heads
    with pytest.raises(ValueError, match="missing required field"):
        upload_mgr.validate_and_store("bad", config, [])


def test_upload_mismatched_shapes(upload_mgr):
    config, files = _make_fake_model(hidden=256, heads=4)
    # Corrupt: change config to claim different hidden_size
    config["hidden_size"] = 512
    config["num_attention_heads"] = 8
    with pytest.raises(ValueError, match="validation failed"):
        upload_mgr.validate_and_store("bad-shapes", config, files)


def test_upload_duplicate_name(upload_mgr):
    config, files = _make_fake_model()
    upload_mgr.validate_and_store("dup", config, files)
    with pytest.raises(ValueError, match="already exists"):
        upload_mgr.validate_and_store("dup", config, files)


def test_upload_non_safetensors_file(upload_mgr):
    config, _ = _make_fake_model()
    with pytest.raises(ValueError, match="must be .safetensors"):
        upload_mgr.validate_and_store("bad-ext", config, [("model.bin", b"data")])


def test_delete_model(upload_mgr):
    config, files = _make_fake_model()
    upload_mgr.validate_and_store("to-delete", config, files)
    assert len(upload_mgr.list_uploaded()) == 1
    upload_mgr.delete_model("to-delete")
    assert len(upload_mgr.list_uploaded()) == 0


def test_list_uploaded(upload_mgr):
    config1, files1 = _make_fake_model(hidden=256)
    config2, files2 = _make_fake_model(hidden=512, heads=8)
    upload_mgr.validate_and_store("model-a", config1, files1)
    upload_mgr.validate_and_store("model-b", config2, files2)
    listed = upload_mgr.list_uploaded()
    assert len(listed) == 2
    names = {m["name"] for m in listed}
    assert names == {"model-a", "model-b"}
