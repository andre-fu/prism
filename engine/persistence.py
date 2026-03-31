"""Persistent storage: SQLite backend for tenants, API keys, usage, and request logs.

Survives process restarts. All tenant state, API keys, and usage metrics
are written to disk immediately on change.
"""

import sqlite3
import json
import time
import threading
from pathlib import Path
from contextlib import contextmanager


class PersistenceStore:
    """SQLite-backed persistent storage."""

    def __init__(self, db_path: str = "prism.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    @contextmanager
    def _conn(self):
        """Thread-safe connection (one per thread)."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        yield self._local.conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    api_key_hash TEXT DEFAULT '',
                    rate_limit_rps REAL DEFAULT 10.0,
                    max_concurrent INTEGER DEFAULT 16,
                    allowed_models TEXT DEFAULT '[]',
                    priority INTEGER DEFAULT 0,
                    slo_ttft_ms REAL DEFAULT 2000.0,
                    max_tokens_per_request INTEGER DEFAULT 4096,
                    monthly_token_limit INTEGER DEFAULT 0,
                    created_at REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS usage (
                    tenant_id TEXT PRIMARY KEY,
                    total_requests INTEGER DEFAULT 0,
                    total_prompt_tokens INTEGER DEFAULT 0,
                    total_completion_tokens INTEGER DEFAULT 0,
                    total_errors INTEGER DEFAULT 0,
                    last_request_time REAL DEFAULT 0.0,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
                );

                CREATE TABLE IF NOT EXISTS request_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    tenant_id TEXT,
                    model_name TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    ttft_ms REAL,
                    total_ms REAL,
                    status TEXT,
                    error TEXT,
                    created_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_request_log_tenant ON request_log(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_request_log_model ON request_log(model_name);
                CREATE INDEX IF NOT EXISTS idx_request_log_created ON request_log(created_at);

                CREATE TABLE IF NOT EXISTS models (
                    name TEXT PRIMARY KEY,
                    model_id TEXT,
                    tp_size INTEGER DEFAULT 1,
                    dtype TEXT DEFAULT 'bfloat16',
                    upload_path TEXT DEFAULT '',
                    status TEXT DEFAULT 'registered',
                    created_at REAL DEFAULT 0.0,
                    config_json TEXT DEFAULT '{}'
                );
            """)
            conn.commit()

    # --- Tenant operations ---

    def save_tenant(self, tenant_id: str, **kwargs):
        with self._conn() as conn:
            allowed = json.dumps(kwargs.get('allowed_models', []))
            conn.execute("""
                INSERT OR REPLACE INTO tenants
                (tenant_id, name, api_key_hash, rate_limit_rps, max_concurrent,
                 allowed_models, priority, slo_ttft_ms, max_tokens_per_request,
                 monthly_token_limit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tenant_id,
                kwargs.get('name', ''),
                kwargs.get('api_key_hash', ''),
                kwargs.get('rate_limit_rps', 10.0),
                kwargs.get('max_concurrent', 16),
                allowed,
                kwargs.get('priority', 0),
                kwargs.get('slo_ttft_ms', 2000.0),
                kwargs.get('max_tokens_per_request', 4096),
                kwargs.get('monthly_token_limit', 0),
                time.time(),
            ))
            # Initialize usage row
            conn.execute(
                "INSERT OR IGNORE INTO usage (tenant_id) VALUES (?)", (tenant_id,))
            conn.commit()

    def load_tenant(self, tenant_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d['allowed_models'] = json.loads(d['allowed_models'])
            return d

    def load_all_tenants(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM tenants").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['allowed_models'] = json.loads(d['allowed_models'])
                result.append(d)
            return result

    def load_api_key_map(self) -> dict[str, str]:
        """Returns {api_key_hash: tenant_id} for all tenants."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT api_key_hash, tenant_id FROM tenants WHERE api_key_hash != ''").fetchall()
            return {row['api_key_hash']: row['tenant_id'] for row in rows}

    # --- Usage operations ---

    def update_usage(self, tenant_id: str, prompt_tokens: int = 0,
                     completion_tokens: int = 0, error: bool = False):
        with self._conn() as conn:
            conn.execute("""
                UPDATE usage SET
                    total_requests = total_requests + 1,
                    total_prompt_tokens = total_prompt_tokens + ?,
                    total_completion_tokens = total_completion_tokens + ?,
                    total_errors = total_errors + ?,
                    last_request_time = ?
                WHERE tenant_id = ?
            """, (prompt_tokens, completion_tokens, 1 if error else 0, time.time(), tenant_id))
            conn.commit()

    def load_usage(self, tenant_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM usage WHERE tenant_id = ?", (tenant_id,)).fetchone()
            return dict(row) if row else None

    # --- Request log operations ---

    def log_request(self, request_id: str, tenant_id: str, model_name: str,
                    prompt_tokens: int, completion_tokens: int,
                    ttft_ms: float, total_ms: float, status: str, error: str = ""):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO request_log
                (request_id, tenant_id, model_name, prompt_tokens, completion_tokens,
                 ttft_ms, total_ms, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (request_id, tenant_id, model_name, prompt_tokens, completion_tokens,
                  ttft_ms, total_ms, status, error, time.time()))
            conn.commit()

    def query_request_log(self, tenant_id: str | None = None, model_name: str | None = None,
                          limit: int = 100, since: float = 0) -> list[dict]:
        with self._conn() as conn:
            query = "SELECT * FROM request_log WHERE created_at > ?"
            params = [since]
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # --- Model operations ---

    def save_model(self, name: str, model_id: str, tp_size: int = 1,
                   dtype: str = "bfloat16", upload_path: str = "", config_json: str = "{}"):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models
                (name, model_id, tp_size, dtype, upload_path, status, created_at, config_json)
                VALUES (?, ?, ?, ?, ?, 'registered', ?, ?)
            """, (name, model_id, tp_size, dtype, upload_path, time.time(), config_json))
            conn.commit()

    def load_model(self, name: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def load_all_models(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM models").fetchall()]

    def update_model_status(self, name: str, status: str):
        with self._conn() as conn:
            conn.execute("UPDATE models SET status = ? WHERE name = ?", (status, name))
            conn.commit()
