"""
Standalone benchmark for measuring swap latency in detail.

Usage:
    python -m gpu_swap.benchmark --active llama70b --target mistral --n 5

Measures each phase: sleep, wake, first-token-latency after swap.
"""

import argparse
import json
import logging
import sys
import time

import requests

from gpu_swap.vllm_manager import (
    InstanceRegistry,
    InstanceState,
    suspend_instance,
    resume_instance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def measure_first_token(port: int, prompt: str = "Hello, my name is") -> float:
    """Send a request and measure time to first token."""
    t0 = time.time()
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        },
        timeout=60,
    )
    r.raise_for_status()
    return time.time() - t0


def run_benchmark(active_name: str, target_name: str, n_cycles: int = 3):
    registry = InstanceRegistry()
    active = registry.get(active_name)
    target = registry.get(target_name)

    if not active:
        raise ValueError(f"Instance '{active_name}' not found")
    if not target:
        raise ValueError(f"Instance '{target_name}' not found")

    results = []

    for i in range(n_cycles):
        cycle = {"cycle": i + 1}
        print(f"\n{'='*60}")
        print(f"Cycle {i+1}/{n_cycles}")
        print(f"{'='*60}")

        # Warm up: first token on active
        if active.state == InstanceState.SERVING:
            print(f"  First token (warm, {active_name})...", end=" ", flush=True)
            cycle["warm_first_token"] = measure_first_token(active.port)
            print(f"{cycle['warm_first_token']:.3f}s")

        # Suspend active
        print(f"  Suspending {active_name}...", end=" ", flush=True)
        cycle["suspend_time"] = suspend_instance(active_name)
        print(f"{cycle['suspend_time']:.2f}s")

        # Resume target
        print(f"  Resuming {target_name}...", end=" ", flush=True)
        cycle["resume_time"] = resume_instance(target_name)
        print(f"{cycle['resume_time']:.2f}s")

        # First token after swap (cold-ish)
        print(f"  First token (post-swap, {target_name})...", end=" ", flush=True)
        cycle["cold_first_token"] = measure_first_token(target.port)
        print(f"{cycle['cold_first_token']:.3f}s")

        cycle["total_swap"] = cycle["suspend_time"] + cycle["resume_time"]
        cycle["total_with_first_token"] = cycle["total_swap"] + cycle["cold_first_token"]

        results.append(cycle)

        # Swap back for next cycle
        print(f"  Swapping back {target_name} -> {active_name}...")
        suspend_instance(target_name)
        resume_instance(active_name)

    # Summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({n_cycles} cycles)")
    print(f"{'='*60}")

    for key in ["suspend_time", "resume_time", "total_swap", "cold_first_token", "total_with_first_token"]:
        vals = [r[key] for r in results if key in r]
        if vals:
            print(f"  {key:30s}  avg={sum(vals)/len(vals):.3f}s  min={min(vals):.3f}s  max={max(vals):.3f}s")

    # Dump raw results
    print(f"\nRaw results:")
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="gpu_swap benchmark")
    parser.add_argument("--active", required=True, help="Currently serving instance name")
    parser.add_argument("--target", required=True, help="Suspended instance to swap to")
    parser.add_argument("--n", type=int, default=3, help="Number of swap cycles")
    args = parser.parse_args()

    run_benchmark(args.active, args.target, args.n)


if __name__ == "__main__":
    main()
