"""Process lifecycle: graceful shutdown with request draining.

On SIGTERM:
1. Stop accepting new requests (return 503)
2. Wait for in-flight requests to complete (drain timeout)
3. Save state (usage metrics, etc.)
4. Cleanup GPU memory
5. Exit
"""

import signal
import time
import threading
from .logging import get_logger

log = get_logger("prism.lifecycle")


class LifecycleManager:
    """Manages graceful startup and shutdown."""

    def __init__(self, drain_timeout_s: float = 30.0):
        self.drain_timeout = drain_timeout_s
        self.is_draining = False
        self.is_ready = False
        self._shutdown_callbacks: list[callable] = []
        self._drain_complete = threading.Event()

    def register_shutdown(self, callback: callable):
        """Register a callback to run during shutdown."""
        self._shutdown_callbacks.append(callback)

    def start(self):
        """Register signal handlers and mark ready."""
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)
        self.is_ready = True
        log.info("Engine started", extra={"extra_fields": {"event": "startup"}})

    def _handle_sigterm(self, signum, frame):
        """Handle SIGTERM/SIGINT: initiate graceful drain."""
        if self.is_draining:
            return  # Already draining
        self.is_draining = True
        log.info("Shutdown initiated, draining requests",
                 extra={"extra_fields": {"event": "drain_start", "timeout_s": self.drain_timeout}})

        # Run drain in background thread to not block signal handler
        threading.Thread(target=self._drain_and_shutdown, daemon=False).start()

    def _drain_and_shutdown(self):
        """Wait for in-flight requests, then shutdown."""
        # Wait for drain (scheduler will stop accepting new work when is_draining=True)
        self._drain_complete.wait(timeout=self.drain_timeout)

        log.info("Running shutdown callbacks",
                 extra={"extra_fields": {"event": "shutdown_callbacks",
                                         "count": len(self._shutdown_callbacks)}})

        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                log.error(f"Shutdown callback error: {e}",
                          extra={"extra_fields": {"event": "shutdown_error", "error": str(e)}})

        log.info("Shutdown complete", extra={"extra_fields": {"event": "shutdown_complete"}})

    def signal_drain_complete(self):
        """Called by scheduler when all in-flight requests are done."""
        self._drain_complete.set()

    def check_accepting_requests(self) -> bool:
        """Returns False during drain (server should return 503)."""
        return self.is_ready and not self.is_draining
