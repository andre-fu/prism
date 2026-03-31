"""Structured JSON logging for production.

All logs are JSON-formatted for easy parsing by log aggregators
(Datadog, ELK, CloudWatch, etc.).
"""

import logging
import json
import sys
import time
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger("prism")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        if json_format:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
        logger.addHandler(handler)

    return logger


def get_logger(name: str = "prism") -> logging.Logger:
    return logging.getLogger(name)


class RequestLogger:
    """Log every request for audit trail. Writes to persistence + structured log."""

    def __init__(self, persistence_store=None):
        self._db = persistence_store
        self._log = get_logger("prism.requests")

    def log_request(self, request_id: str, tenant_id: str, model_name: str,
                    prompt_tokens: int, completion_tokens: int,
                    ttft_ms: float, total_ms: float, status: str, error: str = ""):
        """Log a completed request to both structured log and database."""
        record = {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": round(ttft_ms, 1),
            "total_ms": round(total_ms, 1),
            "status": status,
        }
        if error:
            record["error"] = error

        self._log.info("request_completed", extra={"extra_fields": record})

        if self._db:
            self._db.log_request(request_id, tenant_id, model_name,
                                 prompt_tokens, completion_tokens,
                                 ttft_ms, total_ms, status, error)
