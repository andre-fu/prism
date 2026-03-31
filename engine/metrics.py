"""Metrics collection: histograms, counters, gauges for Prometheus scraping.

Tracks latency distributions, throughput rates, and GPU utilization.
"""

import time
import threading
import math
from collections import defaultdict


class Histogram:
    """Simple histogram with percentile calculation."""

    def __init__(self, name: str, max_samples: int = 10000):
        self.name = name
        self._values: list[float] = []
        self._max = max_samples
        self._lock = threading.Lock()

    def observe(self, value: float):
        with self._lock:
            self._values.append(value)
            if len(self._values) > self._max:
                self._values = self._values[-self._max:]

    def percentile(self, p: float) -> float:
        with self._lock:
            if not self._values:
                return 0
            sorted_vals = sorted(self._values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def count(self) -> int:
        return len(self._values)

    def mean(self) -> float:
        with self._lock:
            return sum(self._values) / len(self._values) if self._values else 0

    def reset(self):
        with self._lock:
            self._values.clear()


class Counter:
    """Thread-safe counter."""

    def __init__(self, name: str):
        self.name = name
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1):
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value


class Gauge:
    """Thread-safe gauge (can go up and down)."""

    def __init__(self, name: str):
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float):
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1):
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1):
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class MetricsCollector:
    """Central metrics collection for the engine."""

    def __init__(self):
        # Request latency histograms
        self.ttft = Histogram("prism_ttft_ms")
        self.tbt = Histogram("prism_tbt_ms")
        self.request_duration = Histogram("prism_request_duration_ms")
        self.prefill_duration = Histogram("prism_prefill_duration_ms")

        # Throughput counters
        self.requests_total = Counter("prism_requests_total")
        self.tokens_generated = Counter("prism_tokens_generated_total")
        self.errors_total = Counter("prism_errors_total")

        # Swap metrics
        self.swaps_total = Counter("prism_swaps_total")
        self.swap_duration = Histogram("prism_swap_duration_ms")
        self.prefetch_hits = Counter("prism_prefetch_hits_total")

        # Current state gauges
        self.active_requests = Gauge("prism_active_requests")
        self.pending_requests = Gauge("prism_pending_requests")
        self.loaded_models = Gauge("prism_loaded_models")

        # Per-model histograms
        self._model_ttft: dict[str, Histogram] = defaultdict(lambda: Histogram("model_ttft"))
        self._model_tbt: dict[str, Histogram] = defaultdict(lambda: Histogram("model_tbt"))

    def record_request(self, model: str, ttft_ms: float, tbt_ms: float,
                       duration_ms: float, tokens: int, error: bool = False):
        """Record a completed request."""
        self.requests_total.inc()
        self.tokens_generated.inc(tokens)
        self.ttft.observe(ttft_ms)
        if tbt_ms > 0:
            self.tbt.observe(tbt_ms)
        self.request_duration.observe(duration_ms)
        self._model_ttft[model].observe(ttft_ms)
        if tbt_ms > 0:
            self._model_tbt[model].observe(tbt_ms)
        if error:
            self.errors_total.inc()

    def record_swap(self, duration_ms: float, prefetch_hit: bool = False):
        self.swaps_total.inc()
        self.swap_duration.observe(duration_ms)
        if prefetch_hit:
            self.prefetch_hits.inc()

    def to_prometheus(self, gpu_stats: dict | None = None) -> str:
        """Generate Prometheus text format metrics."""
        lines = []

        # Counters
        lines.append(f"prism_requests_total {self.requests_total.value}")
        lines.append(f"prism_tokens_generated_total {self.tokens_generated.value}")
        lines.append(f"prism_errors_total {self.errors_total.value}")
        lines.append(f"prism_swaps_total {self.swaps_total.value}")
        lines.append(f"prism_prefetch_hits_total {self.prefetch_hits.value}")

        # Gauges
        lines.append(f"prism_active_requests {self.active_requests.value}")
        lines.append(f"prism_pending_requests {self.pending_requests.value}")
        lines.append(f"prism_loaded_models {self.loaded_models.value}")

        # Histograms (percentiles)
        for name, hist in [("ttft_ms", self.ttft), ("tbt_ms", self.tbt),
                           ("request_duration_ms", self.request_duration),
                           ("swap_duration_ms", self.swap_duration)]:
            if hist.count() > 0:
                lines.append(f'prism_{name}{{quantile="0.5"}} {hist.percentile(50):.1f}')
                lines.append(f'prism_{name}{{quantile="0.95"}} {hist.percentile(95):.1f}')
                lines.append(f'prism_{name}{{quantile="0.99"}} {hist.percentile(99):.1f}')
                lines.append(f"prism_{name}_count {hist.count()}")
                lines.append(f"prism_{name}_avg {hist.mean():.1f}")

        # Per-model metrics
        for model, hist in self._model_ttft.items():
            if hist.count() > 0:
                lines.append(f'prism_model_ttft_ms{{model="{model}",quantile="0.5"}} {hist.percentile(50):.1f}')
                lines.append(f'prism_model_ttft_ms{{model="{model}",quantile="0.95"}} {hist.percentile(95):.1f}')

        # GPU stats
        if gpu_stats:
            for gpu_id, stats in gpu_stats.items():
                lines.append(f'prism_gpu_memory_allocated_gb{{gpu="{gpu_id}"}} {stats.get("allocated_gb", 0):.2f}')
                lines.append(f'prism_gpu_memory_reserved_gb{{gpu="{gpu_id}"}} {stats.get("reserved_gb", 0):.2f}')
                if "utilization" in stats:
                    lines.append(f'prism_gpu_utilization{{gpu="{gpu_id}"}} {stats["utilization"]:.1f}')

        return "\n".join(lines) + "\n"


# Global metrics instance
metrics = MetricsCollector()
