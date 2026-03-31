"""
gpu_swap orchestrator — CLI for warm model swapping on multi-GPU clusters.

Usage:
    python -m gpu_swap.orchestrator launch --model meta-llama/Llama-3.1-70B --tp 4 --name llama70b
    python -m gpu_swap.orchestrator launch --model mistralai/Mistral-7B-v0.1 --tp 4 --name mistral --suspended
    python -m gpu_swap.orchestrator swap --to mistral
    python -m gpu_swap.orchestrator swap --to llama70b
    python -m gpu_swap.orchestrator status
    python -m gpu_swap.orchestrator stop --name llama70b
    python -m gpu_swap.orchestrator stop-all
"""

import argparse
import json
import logging
import sys
import time

from gpu_swap.vllm_manager import (
    InstanceRegistry,
    InstanceState,
    get_status,
    launch_instance,
    resume_instance,
    stop_instance,
    suspend_instance,
    swap_instances,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_launch(args):
    extra = args.extra.split() if args.extra else None
    inst = launch_instance(
        name=args.name,
        model=args.model,
        tp_size=args.tp,
        port=args.port,
        extra_args=extra,
        suspended=args.suspended,
    )
    print(f"Launched '{inst.name}' ({inst.model}) on port {inst.port} — {inst.state.value}")


def cmd_suspend(args):
    elapsed = suspend_instance(args.name)
    print(f"Suspended '{args.name}' in {elapsed:.2f}s")


def cmd_resume(args):
    elapsed = resume_instance(args.name)
    print(f"Resumed '{args.name}' in {elapsed:.2f}s")


def cmd_swap(args):
    registry = InstanceRegistry()

    # Find currently active instance
    active = registry.active()
    if not active:
        # Nothing active — just resume the target
        elapsed = resume_instance(args.to)
        print(f"Resumed '{args.to}' in {elapsed:.2f}s (no prior active instance)")
        return

    if active.name == args.to:
        print(f"'{args.to}' is already serving")
        return

    target = registry.get(args.to)
    if not target:
        raise ValueError(f"Instance '{args.to}' not found. Launch it first.")
    if target.state != InstanceState.SUSPENDED:
        raise ValueError(f"Instance '{args.to}' is {target.state}, not suspended")

    timings = swap_instances(active.name, args.to)
    print(f"Swapped '{active.name}' -> '{args.to}' in {timings['total']:.2f}s")
    print(f"  Suspend: {timings['suspend']:.2f}s")
    print(f"  Resume:  {timings['resume']:.2f}s")


def cmd_status(args):
    statuses = get_status()
    if not statuses:
        print("No instances registered.")
        return

    # Table header
    print(f"{'Name':<20} {'Model':<40} {'TP':<4} {'Port':<6} {'State':<12} {'PID':<8} {'Alive':<6}")
    print("-" * 96)
    for s in statuses:
        print(
            f"{s['name']:<20} {s['model']:<40} {s['tp_size']:<4} {s['port']:<6} "
            f"{s['state']:<12} {s.get('pid', '-')!s:<8} {s.get('alive', '-')!s:<6}"
        )


def cmd_stop(args):
    stop_instance(args.name)
    print(f"Stopped '{args.name}'")


def cmd_stop_all(args):
    registry = InstanceRegistry()
    for inst in registry.all():
        if inst.state != InstanceState.STOPPED:
            stop_instance(inst.name)
            print(f"Stopped '{inst.name}'")


def cmd_benchmark(args):
    """Run a swap benchmark: suspend and resume N times, report timings."""
    registry = InstanceRegistry()
    active = registry.active()
    target = registry.get(args.to)

    if not active:
        raise ValueError("No active instance to swap from")
    if not target:
        raise ValueError(f"Instance '{args.to}' not found")

    results = []
    for i in range(args.n):
        print(f"\n--- Swap {i+1}/{args.n}: {active.name} -> {args.to} ---")
        t = swap_instances(active.name, args.to)
        results.append(t)
        print(f"  Total: {t['total']:.2f}s  Suspend: {t['suspend']:.2f}s  Resume: {t['resume']:.2f}s")

        # Swap back
        if i < args.n - 1 or args.n == 1:
            print(f"\n--- Swap {i+1}/{args.n}: {args.to} -> {active.name} ---")
            t2 = swap_instances(args.to, active.name)
            results.append(t2)
            print(f"  Total: {t2['total']:.2f}s  Suspend: {t2['suspend']:.2f}s  Resume: {t2['resume']:.2f}s")

    # Summary
    totals = [r["total"] for r in results]
    suspends = [r["suspend"] for r in results]
    resumes = [r["resume"] for r in results]
    print(f"\n=== Benchmark Summary ({len(results)} swaps) ===")
    print(f"  Total:   avg={sum(totals)/len(totals):.2f}s  min={min(totals):.2f}s  max={max(totals):.2f}s")
    print(f"  Suspend: avg={sum(suspends)/len(suspends):.2f}s  min={min(suspends):.2f}s  max={max(suspends):.2f}s")
    print(f"  Resume:  avg={sum(resumes)/len(resumes):.2f}s  min={min(resumes):.2f}s  max={max(resumes):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="gpu_swap — Warm model swapping for multi-GPU vLLM")
    sub = parser.add_subparsers(dest="command", required=True)

    # launch
    p = sub.add_parser("launch", help="Launch a vLLM instance")
    p.add_argument("--name", required=True, help="Instance name")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--tp", type=int, default=4, help="Tensor parallel size (default: 4)")
    p.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    p.add_argument("--extra", type=str, default=None, help="Extra vllm serve args (quoted string)")
    p.add_argument("--suspended", action="store_true", help="Immediately sleep after launch")
    p.set_defaults(func=cmd_launch)

    # suspend
    p = sub.add_parser("suspend", help="Suspend a serving instance")
    p.add_argument("name", help="Instance name")
    p.set_defaults(func=cmd_suspend)

    # resume
    p = sub.add_parser("resume", help="Resume a suspended instance")
    p.add_argument("name", help="Instance name")
    p.set_defaults(func=cmd_resume)

    # swap
    p = sub.add_parser("swap", help="Swap to a different model")
    p.add_argument("--to", required=True, help="Target instance name")
    p.set_defaults(func=cmd_swap)

    # status
    p = sub.add_parser("status", help="Show all instances")
    p.set_defaults(func=cmd_status)

    # stop
    p = sub.add_parser("stop", help="Stop an instance")
    p.add_argument("--name", required=True, help="Instance name")
    p.set_defaults(func=cmd_stop)

    # stop-all
    p = sub.add_parser("stop-all", help="Stop all instances")
    p.set_defaults(func=cmd_stop_all)

    # benchmark
    p = sub.add_parser("benchmark", help="Benchmark swap latency")
    p.add_argument("--to", required=True, help="Target instance to swap to")
    p.add_argument("--n", type=int, default=3, help="Number of swap cycles (default: 3)")
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
