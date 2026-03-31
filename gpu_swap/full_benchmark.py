"""
Comprehensive multi-GPU swap benchmark.

Tests:
- Sleep/wake latency at TP=4
- TTFT (time to first token) before/after swap
- KV cache behavior after swap
- Throughput (tokens/sec) before/after swap
- GPU memory tracking per phase
- NCCL teardown/rebuild overhead
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["VLLM_SERVER_DEV_MODE"] = "1"

# Patch PyTorch 2.10 + vLLM 0.18 compatibility
try:
    import torch._inductor.standalone_compile as _sc_module
    from torch._subclasses.fake_tensor import FakeTensorMode as _FTM
    if not hasattr(_sc_module, "FakeTensorMode"):
        _sc_module.FakeTensorMode = _FTM
except Exception:
    pass

# ─── Configuration ────────────────────────────────────────────────────
MODEL_A = os.environ.get("MODEL_A", "Qwen/Qwen2.5-14B-Instruct")
MODEL_B = os.environ.get("MODEL_B", "Qwen/Qwen2.5-7B-Instruct")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
PORT_A = 8000
PORT_B = 8001

PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain the theory of relativity in detail, covering both special and general relativity, their key equations, and practical applications.",
    "long": "Write a comprehensive tutorial on building a distributed systems from scratch. Cover: " * 10,
}


def gpu_memory():
    """Get per-GPU memory usage in MB."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpus = []
    for line in r.stdout.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        gpus.append({"gpu": int(parts[0]), "used_mb": int(parts[1]), "total_mb": int(parts[2])})
    return gpus


def wait_for_server(port, timeout=600):
    """Wait for vLLM server to be ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(3)
    return False


def launch_vllm(model, port, tp_size, name):
    """Launch a vLLM server and wait for it."""
    log_path = f"/lambda/nfs/inference-texas/gpu_swap/logs/{name}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp_size),
        "--port", str(port),
        "--enable-sleep-mode",
        "--no-enable-log-requests",
    ]
    print(f"  Launching {name}: {model} (TP={tp_size}, port={port})...")
    env = os.environ.copy()
    # Ensure sitecustomize.py patch is picked up by subprocess
    pythonpath = env.get("PYTHONPATH", "")
    patch_dir = "/lambda/nfs/inference-texas"
    if patch_dir not in pythonpath:
        env["PYTHONPATH"] = f"{patch_dir}:{pythonpath}" if pythonpath else patch_dir
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    print(f"  PID: {proc.pid}, waiting for ready...")

    if not wait_for_server(port, timeout=600):
        proc.kill()
        print(f"  FAILED! Check {log_path}")
        sys.exit(1)

    print(f"  {name} ready!")
    return proc


def measure_ttft(port, model, prompt, max_tokens=1):
    """Measure time to first token via streaming."""
    t0 = time.perf_counter()
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0},
        timeout=60,
    )
    r.raise_for_status()
    ttft = time.perf_counter() - t0
    return ttft


def measure_throughput(port, model, prompt, max_tokens=100):
    """Measure tokens/sec for generation."""
    t0 = time.perf_counter()
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0},
        timeout=120,
    )
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    data = r.json()
    n_tokens = data["usage"]["completion_tokens"]
    return {"elapsed": elapsed, "tokens": n_tokens, "tok_per_sec": n_tokens / elapsed if elapsed > 0 else 0}


def sleep_server(port):
    """Sleep a vLLM server, return elapsed time."""
    t0 = time.perf_counter()
    r = requests.post(f"http://localhost:{port}/sleep?level=1&mode=abort", timeout=120)
    r.raise_for_status()
    return time.perf_counter() - t0


def wake_server(port):
    """Wake a vLLM server, return elapsed time."""
    t0 = time.perf_counter()
    r = requests.post(f"http://localhost:{port}/wake_up", timeout=120)
    r.raise_for_status()
    return time.perf_counter() - t0


def is_sleeping(port):
    r = requests.get(f"http://localhost:{port}/is_sleeping", timeout=10)
    return r.json()["is_sleeping"]


def print_gpu_mem(label, mem):
    total_used = sum(g["used_mb"] for g in mem)
    details = ", ".join(f"GPU{g['gpu']}:{g['used_mb']}MB" for g in mem)
    print(f"  [{label}] GPU mem: {total_used} MB total ({details})")


def run_inference_suite(port, model, label):
    """Run a full inference benchmark suite."""
    print(f"\n  === Inference: {label} ===")
    results = {}

    # TTFT
    for pname, prompt in PROMPTS.items():
        ttft = measure_ttft(port, model, prompt)
        results[f"ttft_{pname}"] = ttft
        print(f"    TTFT ({pname}): {ttft*1000:.1f}ms")

    # Throughput
    tp = measure_throughput(port, model, PROMPTS["medium"], max_tokens=100)
    results["throughput_tok_s"] = tp["tok_per_sec"]
    results["throughput_elapsed"] = tp["elapsed"]
    results["throughput_tokens"] = tp["tokens"]
    print(f"    Throughput: {tp['tok_per_sec']:.1f} tok/s ({tp['tokens']} tokens in {tp['elapsed']:.2f}s)")

    # Multi-turn (test KV cache reuse)
    # First request fills KV cache
    t0 = time.perf_counter()
    r1 = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": "Tell me about machine learning.", "max_tokens": 50, "temperature": 0},
        timeout=60,
    )
    r1.raise_for_status()
    t_first = time.perf_counter() - t0

    # Second request with longer prompt (prefix should hit KV cache if enabled)
    t0 = time.perf_counter()
    r2 = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": "Tell me about machine learning. Now explain neural networks.", "max_tokens": 50, "temperature": 0},
        timeout=60,
    )
    r2.raise_for_status()
    t_second = time.perf_counter() - t0

    results["kv_first_req"] = t_first
    results["kv_second_req"] = t_second
    print(f"    KV cache test: req1={t_first:.3f}s, req2={t_second:.3f}s (prefix overlap)")

    return results


def main():
    print("=" * 70)
    print(f"FULL MULTI-GPU SWAP BENCHMARK")
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"TP size: {TP_SIZE}")
    print("=" * 70)

    all_results = {
        "config": {
            "model_a": MODEL_A, "model_b": MODEL_B, "tp_size": TP_SIZE,
        },
    }

    # ─── Phase 1: Launch Model A ──────────────────────────────────────
    print("\n[Phase 1] Launching Model A...")
    print_gpu_mem("before launch", gpu_memory())
    proc_a = launch_vllm(MODEL_A, PORT_A, TP_SIZE, "model_a")
    print_gpu_mem("after launch A", gpu_memory())

    # ─── Phase 2: Baseline inference on Model A ───────────────────────
    print("\n[Phase 2] Baseline inference on Model A (pre-swap)...")
    all_results["model_a_baseline"] = run_inference_suite(PORT_A, MODEL_A, "Model A baseline")
    print_gpu_mem("after inference A", gpu_memory())

    # ─── Phase 3: Sleep Model A, Launch Model B ───────────────────────
    print("\n[Phase 3] Sleeping Model A, launching Model B...")
    t_sleep_a = sleep_server(PORT_A)
    print(f"  Model A sleep: {t_sleep_a:.3f}s")
    assert is_sleeping(PORT_A), "Model A should be sleeping"
    print_gpu_mem("after sleep A", gpu_memory())
    all_results["sleep_a_time"] = t_sleep_a

    proc_b = launch_vllm(MODEL_B, PORT_B, TP_SIZE, "model_b")
    print_gpu_mem("after launch B", gpu_memory())

    # ─── Phase 4: Baseline inference on Model B ───────────────────────
    print("\n[Phase 4] Baseline inference on Model B...")
    all_results["model_b_baseline"] = run_inference_suite(PORT_B, MODEL_B, "Model B baseline")
    print_gpu_mem("after inference B", gpu_memory())

    # ─── Phase 5: Swap B→A (sleep B, wake A) ─────────────────────────
    print("\n[Phase 5] SWAP: Model B → Model A")
    t0_swap = time.perf_counter()

    t_sleep_b = sleep_server(PORT_B)
    print(f"  Sleep B: {t_sleep_b:.3f}s")
    print_gpu_mem("after sleep B", gpu_memory())

    t_wake_a = wake_server(PORT_A)
    print(f"  Wake A:  {t_wake_a:.3f}s")
    print_gpu_mem("after wake A", gpu_memory())

    t_swap_ba = time.perf_counter() - t0_swap
    print(f"  TOTAL SWAP B→A: {t_swap_ba:.3f}s")
    all_results["swap_ba"] = {"sleep_b": t_sleep_b, "wake_a": t_wake_a, "total": t_swap_ba}

    # ─── Phase 6: Post-swap inference on Model A ──────────────────────
    print("\n[Phase 6] Post-swap inference on Model A...")
    all_results["model_a_postswap"] = run_inference_suite(PORT_A, MODEL_A, "Model A post-swap")

    # ─── Phase 7: Swap A→B (sleep A, wake B) ─────────────────────────
    print("\n[Phase 7] SWAP: Model A → Model B")
    t0_swap = time.perf_counter()

    t_sleep_a2 = sleep_server(PORT_A)
    print(f"  Sleep A: {t_sleep_a2:.3f}s")
    print_gpu_mem("after sleep A", gpu_memory())

    t_wake_b = wake_server(PORT_B)
    print(f"  Wake B:  {t_wake_b:.3f}s")
    print_gpu_mem("after wake B", gpu_memory())

    t_swap_ab = time.perf_counter() - t0_swap
    print(f"  TOTAL SWAP A→B: {t_swap_ab:.3f}s")
    all_results["swap_ab"] = {"sleep_a": t_sleep_a2, "wake_b": t_wake_b, "total": t_swap_ab}

    # ─── Phase 8: Post-swap inference on Model B ──────────────────────
    print("\n[Phase 8] Post-swap inference on Model B...")
    all_results["model_b_postswap"] = run_inference_suite(PORT_B, MODEL_B, "Model B post-swap")

    # ─── Phase 9: Rapid swap cycles ───────────────────────────────────
    print("\n[Phase 9] Rapid swap cycles (5x)...")
    rapid_results = []
    for i in range(5):
        # B→A
        t0 = time.perf_counter()
        sleep_server(PORT_B)
        t_mid = time.perf_counter()
        wake_server(PORT_A)
        t1 = time.perf_counter()
        ttft = measure_ttft(PORT_A, MODEL_A, "Hello", max_tokens=1)
        t2 = time.perf_counter()

        cycle = {
            "direction": "B→A",
            "sleep": t_mid - t0,
            "wake": t1 - t_mid,
            "ttft": ttft,
            "total": t2 - t0,
        }
        rapid_results.append(cycle)
        print(f"  Cycle {i+1} B→A: sleep={cycle['sleep']:.3f}s wake={cycle['wake']:.3f}s ttft={cycle['ttft']*1000:.1f}ms total={cycle['total']:.3f}s")

        # A→B
        t0 = time.perf_counter()
        sleep_server(PORT_A)
        t_mid = time.perf_counter()
        wake_server(PORT_B)
        t1 = time.perf_counter()
        ttft = measure_ttft(PORT_B, MODEL_B, "Hello", max_tokens=1)
        t2 = time.perf_counter()

        cycle = {
            "direction": "A→B",
            "sleep": t_mid - t0,
            "wake": t1 - t_mid,
            "ttft": ttft,
            "total": t2 - t0,
        }
        rapid_results.append(cycle)
        print(f"  Cycle {i+1} A→B: sleep={cycle['sleep']:.3f}s wake={cycle['wake']:.3f}s ttft={cycle['ttft']*1000:.1f}ms total={cycle['total']:.3f}s")

    all_results["rapid_cycles"] = rapid_results

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    swap_times = [c["sleep"] + c["wake"] for c in rapid_results]
    ttfts = [c["ttft"] for c in rapid_results]
    totals = [c["total"] for c in rapid_results]

    print(f"\n  Models: {MODEL_A} / {MODEL_B} (TP={TP_SIZE})")
    print(f"\n  Swap latency (sleep+wake, 10 swaps):")
    print(f"    avg={sum(swap_times)/len(swap_times):.3f}s  min={min(swap_times):.3f}s  max={max(swap_times):.3f}s")
    print(f"\n  TTFT after swap:")
    print(f"    avg={sum(ttfts)/len(ttfts)*1000:.1f}ms  min={min(ttfts)*1000:.1f}ms  max={max(ttfts)*1000:.1f}ms")
    print(f"\n  Total swap + first token:")
    print(f"    avg={sum(totals)/len(totals):.3f}s  min={min(totals):.3f}s  max={max(totals):.3f}s")

    # Compare pre/post swap inference
    if "model_a_baseline" in all_results and "model_a_postswap" in all_results:
        pre = all_results["model_a_baseline"]
        post = all_results["model_a_postswap"]
        print(f"\n  Model A throughput: pre={pre['throughput_tok_s']:.1f} tok/s  post={post['throughput_tok_s']:.1f} tok/s")
    if "model_b_baseline" in all_results and "model_b_postswap" in all_results:
        pre = all_results["model_b_baseline"]
        post = all_results["model_b_postswap"]
        print(f"  Model B throughput: pre={pre['throughput_tok_s']:.1f} tok/s  post={post['throughput_tok_s']:.1f} tok/s")

    # Save results
    results_path = "/lambda/nfs/inference-texas/gpu_swap/benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Cleanup
    print("\n[Cleanup] Stopping servers...")
    proc_a.send_signal(signal.SIGTERM)
    proc_b.send_signal(signal.SIGTERM)
    time.sleep(3)
    proc_a.kill()
    proc_b.kill()
    print("Done.")


if __name__ == "__main__":
    main()
