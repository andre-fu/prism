"""Test metrics collection: histograms, counters, prometheus export."""

from engine.metrics import MetricsCollector, Histogram, Counter, Gauge


def test_histogram_percentiles():
    h = Histogram("test")
    for i in range(100):
        h.observe(float(i))
    assert h.percentile(50) == 50.0
    assert h.percentile(95) == 95.0
    assert h.count() == 100
    assert 49 < h.mean() < 51


def test_counter():
    c = Counter("test")
    c.inc()
    c.inc(5)
    assert c.value == 6


def test_gauge():
    g = Gauge("test")
    g.set(10)
    assert g.value == 10
    g.inc(5)
    assert g.value == 15
    g.dec(3)
    assert g.value == 12


def test_metrics_collector():
    mc = MetricsCollector()
    mc.record_request("model-a", ttft_ms=150, tbt_ms=7, duration_ms=2000, tokens=100)
    mc.record_request("model-b", ttft_ms=300, tbt_ms=12, duration_ms=5000, tokens=200)
    mc.record_swap(duration_ms=280, prefetch_hit=True)

    assert mc.requests_total.value == 2
    assert mc.tokens_generated.value == 300
    assert mc.swaps_total.value == 1
    assert mc.prefetch_hits.value == 1


def test_prometheus_export():
    mc = MetricsCollector()
    mc.record_request("model-a", ttft_ms=150, tbt_ms=7, duration_ms=2000, tokens=100)

    output = mc.to_prometheus(gpu_stats={0: {"allocated_gb": 15.2, "reserved_gb": 16.0}})
    assert "prism_requests_total 1" in output
    assert "prism_tokens_generated_total 100" in output
    assert 'prism_gpu_memory_allocated_gb{gpu="0"} 15.20' in output
    assert "prism_ttft_ms" in output
