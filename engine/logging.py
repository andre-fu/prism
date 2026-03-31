"""Structured logging for the engine."""

import logging
import sys
import time


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the engine."""
    logger = logging.getLogger("engine")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


log = setup_logging()
