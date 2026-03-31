"""Test multi-tenant management: auth, rate limits, usage tracking."""

import pytest
from engine.tenant_manager import TenantManager, TenantConfig


@pytest.fixture
def tm(tmp_path):
    return TenantManager(db_path=str(tmp_path / "test.db"))


def test_register_and_authenticate(tm):
    key = tm.register_tenant(TenantConfig(tenant_id="acme", name="Acme"))
    assert key.startswith("sk-acme-")

    tenant_id = tm.authenticate(key)
    assert tenant_id == "acme"


def test_bad_key_returns_default(tm):
    assert tm.authenticate("sk-bad-key") == "default"
    assert tm.authenticate(None) == "default"
    assert tm.authenticate("") == "default"


def test_model_access_control(tm):
    tm.register_tenant(TenantConfig(
        tenant_id="restricted",
        allowed_models=["model-a", "model-b"]
    ))
    assert tm.check_model_access("restricted", "model-a") is True
    assert tm.check_model_access("restricted", "model-c") is False


def test_model_access_empty_allows_all(tm):
    tm.register_tenant(TenantConfig(tenant_id="open", allowed_models=[]))
    assert tm.check_model_access("open", "anything") is True


def test_rate_limiting(tm):
    tm.register_tenant(TenantConfig(tenant_id="limited", rate_limit_rps=3, max_concurrent=2))

    # Should allow first 2
    ok1, _ = tm.check_rate_limit("limited")
    assert ok1
    tm.on_request_start("limited", 10)

    ok2, _ = tm.check_rate_limit("limited")
    assert ok2
    tm.on_request_start("limited", 10)

    # Third should fail (max_concurrent=2)
    ok3, reason = tm.check_rate_limit("limited")
    assert not ok3
    assert "concurrent" in reason.lower()

    # Complete one, should allow again
    tm.on_request_complete("limited", 20)
    ok4, _ = tm.check_rate_limit("limited")
    assert ok4


def test_usage_tracking(tm):
    tm.register_tenant(TenantConfig(tenant_id="tracked"))
    tm.on_request_start("tracked", 100)
    tm.on_request_complete("tracked", 50)
    tm.on_request_start("tracked", 200)
    tm.on_request_complete("tracked", 75, error=True)

    usage = tm.get_usage("tracked")
    assert usage["total_requests"] == 2
    assert usage["total_prompt_tokens"] == 300
    assert usage["total_completion_tokens"] == 125
    assert usage["total_errors"] == 1


def test_persistence_across_instances(tmp_path):
    db_path = str(tmp_path / "persist.db")

    tm1 = TenantManager(db_path=db_path)
    key = tm1.register_tenant(TenantConfig(tenant_id="persist-test", name="Test"))
    tm1.on_request_start("persist-test", 100)
    tm1.on_request_complete("persist-test", 50)
    del tm1

    # New instance should restore state
    tm2 = TenantManager(db_path=db_path)
    assert "persist-test" in tm2._tenants
    assert tm2.authenticate(key) == "persist-test"
    usage = tm2.get_usage("persist-test")
    assert usage["total_completion_tokens"] == 50
