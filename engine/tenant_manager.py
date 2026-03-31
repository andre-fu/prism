"""Multi-tenant management: rate limiting, usage metering, API keys, quotas.

Each tenant gets:
  - API key for authentication
  - Rate limit (requests/second)
  - Max concurrent requests
  - Model access list (which models they can use)
  - Usage tracking (tokens consumed, requests served)
  - Priority level (affects scheduling order)
"""

import time
import threading
import secrets
import hashlib
from dataclasses import dataclass, field


@dataclass
class TenantConfig:
    tenant_id: str
    name: str = ""
    api_key_hash: str = ""          # SHA-256 of the API key
    rate_limit_rps: float = 10.0    # Requests per second
    max_concurrent: int = 16        # Max in-flight requests
    allowed_models: list[str] = field(default_factory=list)  # Empty = all models
    priority: int = 0               # Higher = served first
    slo_ttft_ms: float = 2000.0     # Default TTFT SLO
    max_tokens_per_request: int = 4096
    monthly_token_limit: int = 0    # 0 = unlimited


@dataclass
class TenantUsage:
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_errors: int = 0
    current_concurrent: int = 0
    last_request_time: float = 0.0
    # Sliding window for rate limiting
    _request_times: list[float] = field(default_factory=list)


class TenantManager:
    """Manages tenants, authentication, rate limiting, and usage tracking."""

    def __init__(self):
        self._tenants: dict[str, TenantConfig] = {}
        self._usage: dict[str, TenantUsage] = {}
        self._api_keys: dict[str, str] = {}  # key_hash -> tenant_id
        self._lock = threading.Lock()

        # Create default tenant (no auth required)
        self.register_tenant(TenantConfig(
            tenant_id="default",
            name="Default Tenant",
            rate_limit_rps=100,
            max_concurrent=64,
        ))

    def register_tenant(self, config: TenantConfig) -> str:
        """Register a tenant. Returns the API key (only shown once)."""
        with self._lock:
            self._tenants[config.tenant_id] = config
            self._usage[config.tenant_id] = TenantUsage()

            # Generate API key if not default tenant
            if config.tenant_id != "default":
                api_key = f"sk-{config.tenant_id}-{secrets.token_hex(16)}"
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                config.api_key_hash = key_hash
                self._api_keys[key_hash] = config.tenant_id
                return api_key
            return ""

    def authenticate(self, api_key: str | None) -> str:
        """Authenticate an API key. Returns tenant_id or 'default'."""
        if not api_key:
            return "default"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            return self._api_keys.get(key_hash, "default")

    def check_rate_limit(self, tenant_id: str) -> tuple[bool, str]:
        """Check if request is allowed. Returns (allowed, reason)."""
        with self._lock:
            config = self._tenants.get(tenant_id)
            usage = self._usage.get(tenant_id)
            if not config or not usage:
                return False, "Unknown tenant"

            now = time.time()

            # Check concurrent limit
            if usage.current_concurrent >= config.max_concurrent:
                return False, f"Max concurrent requests exceeded ({config.max_concurrent})"

            # Check rate limit (sliding window)
            window = 1.0  # 1 second window
            usage._request_times = [t for t in usage._request_times if now - t < window]
            if len(usage._request_times) >= config.rate_limit_rps:
                return False, f"Rate limit exceeded ({config.rate_limit_rps} req/s)"

            # Check monthly token limit
            if config.monthly_token_limit > 0:
                total_tokens = usage.total_prompt_tokens + usage.total_completion_tokens
                if total_tokens >= config.monthly_token_limit:
                    return False, "Monthly token limit exceeded"

            return True, ""

    def check_model_access(self, tenant_id: str, model_name: str) -> bool:
        """Check if tenant can access a model."""
        config = self._tenants.get(tenant_id)
        if not config:
            return False
        if not config.allowed_models:  # Empty = all models
            return True
        return model_name in config.allowed_models

    def on_request_start(self, tenant_id: str, prompt_tokens: int):
        """Record request start. Call when request is accepted."""
        with self._lock:
            usage = self._usage.get(tenant_id)
            if usage:
                usage.current_concurrent += 1
                usage.total_requests += 1
                usage.total_prompt_tokens += prompt_tokens
                usage.last_request_time = time.time()
                usage._request_times.append(time.time())

    def on_request_complete(self, tenant_id: str, completion_tokens: int, error: bool = False):
        """Record request completion."""
        with self._lock:
            usage = self._usage.get(tenant_id)
            if usage:
                usage.current_concurrent = max(0, usage.current_concurrent - 1)
                usage.total_completion_tokens += completion_tokens
                if error:
                    usage.total_errors += 1

    def get_tenant_config(self, tenant_id: str) -> TenantConfig | None:
        return self._tenants.get(tenant_id)

    def get_usage(self, tenant_id: str) -> dict:
        """Get usage stats for a tenant."""
        usage = self._usage.get(tenant_id)
        config = self._tenants.get(tenant_id)
        if not usage or not config:
            return {}
        return {
            "tenant_id": tenant_id,
            "name": config.name,
            "total_requests": usage.total_requests,
            "total_prompt_tokens": usage.total_prompt_tokens,
            "total_completion_tokens": usage.total_completion_tokens,
            "total_tokens": usage.total_prompt_tokens + usage.total_completion_tokens,
            "total_errors": usage.total_errors,
            "current_concurrent": usage.current_concurrent,
            "rate_limit_rps": config.rate_limit_rps,
            "max_concurrent": config.max_concurrent,
            "monthly_token_limit": config.monthly_token_limit,
        }

    def get_all_usage(self) -> list[dict]:
        """Get usage for all tenants."""
        return [self.get_usage(tid) for tid in self._tenants]
