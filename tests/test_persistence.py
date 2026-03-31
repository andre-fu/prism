"""Test persistence layer: SQLite storage survives process restarts."""

import os
import pytest
from engine.persistence import PersistenceStore


@pytest.fixture
def db(tmp_path):
    return PersistenceStore(str(tmp_path / "test.db"))


def test_tenant_roundtrip(db):
    db.save_tenant("acme", name="Acme Corp", rate_limit_rps=5, allowed_models=["model-a"])
    loaded = db.load_tenant("acme")
    assert loaded is not None
    assert loaded["name"] == "Acme Corp"
    assert loaded["rate_limit_rps"] == 5
    assert loaded["allowed_models"] == ["model-a"]


def test_tenant_survives_reconnect(tmp_path):
    db_path = str(tmp_path / "persist.db")
    db1 = PersistenceStore(db_path)
    db1.save_tenant("test", name="Test Tenant", api_key_hash="abc123")
    del db1  # Close connection

    db2 = PersistenceStore(db_path)
    loaded = db2.load_tenant("test")
    assert loaded is not None
    assert loaded["name"] == "Test Tenant"
    assert loaded["api_key_hash"] == "abc123"


def test_usage_tracking(db):
    db.save_tenant("t1")
    db.update_usage("t1", prompt_tokens=100, completion_tokens=50)
    db.update_usage("t1", prompt_tokens=200, completion_tokens=75, error=True)
    usage = db.load_usage("t1")
    assert usage["total_requests"] == 2
    assert usage["total_prompt_tokens"] == 300
    assert usage["total_completion_tokens"] == 125
    assert usage["total_errors"] == 1


def test_request_log(db):
    db.log_request("req-1", "t1", "model-a", 100, 50, 150.0, 2000.0, "ok")
    db.log_request("req-2", "t1", "model-b", 200, 75, 300.0, 5000.0, "ok")
    db.log_request("req-3", "t2", "model-a", 50, 25, 100.0, 1000.0, "error", "OOM")

    logs = db.query_request_log(tenant_id="t1")
    assert len(logs) == 2
    assert logs[0]["request_id"] == "req-2"  # Most recent first

    logs_model = db.query_request_log(model_name="model-a")
    assert len(logs_model) == 2


def test_model_storage(db):
    db.save_model("my-model", "org/model-7b", tp_size=1, upload_path="/data/models/my-model")
    loaded = db.load_model("my-model")
    assert loaded is not None
    assert loaded["model_id"] == "org/model-7b"
    assert loaded["upload_path"] == "/data/models/my-model"

    all_models = db.load_all_models()
    assert len(all_models) == 1


def test_api_key_map(db):
    db.save_tenant("t1", api_key_hash="hash1")
    db.save_tenant("t2", api_key_hash="hash2")
    db.save_tenant("default")

    key_map = db.load_api_key_map()
    assert key_map == {"hash1": "t1", "hash2": "t2"}
