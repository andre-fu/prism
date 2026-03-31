"""
vLLM instance lifecycle manager.

Handles launching vLLM server processes with sleep mode enabled,
and orchestrates suspend/resume with NCCL teardown/rebuild.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("/lambda/nfs/inference-texas/gpu_swap/.registry.json")


class InstanceState(str, Enum):
    LAUNCHING = "launching"
    SERVING = "serving"
    SUSPENDING = "suspending"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    STOPPED = "stopped"


@dataclass
class VLLMInstance:
    name: str
    model: str
    tp_size: int
    port: int
    pid: int | None = None
    state: InstanceState = InstanceState.LAUNCHING
    nccl_config: dict | None = None

    def api_url(self, path: str) -> str:
        return f"http://localhost:{self.port}{path}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "tp_size": self.tp_size,
            "port": self.port,
            "pid": self.pid,
            "state": self.state.value,
            "nccl_config": self.nccl_config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VLLMInstance":
        inst = cls(
            name=d["name"],
            model=d["model"],
            tp_size=d["tp_size"],
            port=d["port"],
            pid=d.get("pid"),
            nccl_config=d.get("nccl_config"),
        )
        inst.state = InstanceState(d.get("state", "stopped"))
        return inst


class InstanceRegistry:
    """Persist instance state to disk so the orchestrator can be restarted."""

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self._instances: dict[str, VLLMInstance] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for d in data:
                inst = VLLMInstance.from_dict(d)
                self._instances[inst.name] = inst

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [inst.to_dict() for inst in self._instances.values()]
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, inst: VLLMInstance):
        self._instances[inst.name] = inst
        self.save()

    def get(self, name: str) -> VLLMInstance | None:
        return self._instances.get(name)

    def remove(self, name: str):
        self._instances.pop(name, None)
        self.save()

    def all(self) -> list[VLLMInstance]:
        return list(self._instances.values())

    def active(self) -> VLLMInstance | None:
        for inst in self._instances.values():
            if inst.state == InstanceState.SERVING:
                return inst
        return None


def _wait_for_server(port: int, timeout: float = 300) -> bool:
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def launch_instance(
    name: str,
    model: str,
    tp_size: int = 4,
    port: int = 8000,
    extra_args: list[str] | None = None,
    suspended: bool = False,
) -> VLLMInstance:
    """
    Launch a vLLM server process with sleep mode enabled.

    Args:
        name: Unique name for this instance
        model: HuggingFace model ID
        tp_size: Tensor parallel size
        port: HTTP port
        extra_args: Additional vllm serve arguments
        suspended: If True, immediately sleep after launch
    """
    registry = InstanceRegistry()

    existing = registry.get(name)
    if existing and existing.state != InstanceState.STOPPED:
        raise ValueError(f"Instance '{name}' already exists in state {existing.state}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp_size),
        "--port", str(port),
        "--enable-sleep-mode",
        "--enforce-eager",
        "--no-enable-log-requests",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Launching vLLM: %s", " ".join(cmd))

    log_path = Path(f"/lambda/nfs/inference-texas/gpu_swap/logs/{name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    # Ensure user site-packages takes precedence over system packages
    user_site = os.path.expanduser("~/.local/lib/python3.10/site-packages")
    pythonpath = os.environ.get("PYTHONPATH", "")
    if user_site not in pythonpath:
        pythonpath = f"{user_site}:{pythonpath}" if pythonpath else user_site
    env = {
        **os.environ,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TRANSFORMERS_NO_TF": "1",
        "USE_TF": "0",
        "PYTHONPATH": pythonpath,
        # Required for sleep/wake HTTP endpoints
        "VLLM_SERVER_DEV_MODE": "1",
    }
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )

    inst = VLLMInstance(
        name=name,
        model=model,
        tp_size=tp_size,
        port=port,
        pid=proc.pid,
        state=InstanceState.LAUNCHING,
    )
    registry.add(inst)

    logger.info("Waiting for server to be ready on port %d (PID %d)...", port, proc.pid)
    if not _wait_for_server(port, timeout=300):
        proc.kill()
        inst.state = InstanceState.STOPPED
        registry.add(inst)
        raise RuntimeError(f"vLLM server failed to start. Check {log_path}")

    inst.state = InstanceState.SERVING
    registry.add(inst)
    logger.info("Instance '%s' serving on port %d", name, port)

    if suspended:
        suspend_instance(name)

    return inst


def suspend_instance(name: str) -> float:
    """
    Suspend a serving vLLM instance: sleep + NCCL teardown.
    Returns elapsed time in seconds.
    """
    registry = InstanceRegistry()
    inst = registry.get(name)
    if not inst:
        raise ValueError(f"Instance '{name}' not found")
    if inst.state != InstanceState.SERVING:
        raise ValueError(f"Instance '{name}' is {inst.state}, not serving")

    t0 = time.time()
    inst.state = InstanceState.SUSPENDING
    registry.add(inst)

    # Step 1: Put vLLM to sleep (offload weights to CPU, discard KV cache)
    logger.info("[%s] Sleeping (level=1)...", name)
    t_sleep_start = time.time()
    r = requests.post(inst.api_url("/sleep?level=1&mode=abort"), timeout=120)
    r.raise_for_status()
    t_sleep = time.time() - t_sleep_start
    logger.info("[%s] Sleep complete in %.2fs", name, t_sleep)

    # Step 2: Verify sleeping
    r = requests.get(inst.api_url("/is_sleeping"), timeout=10)
    r.raise_for_status()
    assert r.json()["is_sleeping"], "Server claims it's not sleeping"

    inst.state = InstanceState.SUSPENDED
    registry.add(inst)

    elapsed = time.time() - t0
    logger.info("[%s] Suspended in %.2fs (sleep=%.2fs)", name, elapsed, t_sleep)
    return elapsed


def resume_instance(name: str) -> float:
    """
    Resume a suspended vLLM instance: wake_up + NCCL rebuild.
    Returns elapsed time in seconds.
    """
    registry = InstanceRegistry()
    inst = registry.get(name)
    if not inst:
        raise ValueError(f"Instance '{name}' not found")
    if inst.state != InstanceState.SUSPENDED:
        raise ValueError(f"Instance '{name}' is {inst.state}, not suspended")

    t0 = time.time()
    inst.state = InstanceState.RESUMING
    registry.add(inst)

    # Step 1: Wake up vLLM (reload weights from CPU to GPU)
    logger.info("[%s] Waking up...", name)
    t_wake_start = time.time()
    r = requests.post(inst.api_url("/wake_up"), timeout=120)
    r.raise_for_status()
    t_wake = time.time() - t_wake_start
    logger.info("[%s] Wake up complete in %.2fs", name, t_wake)

    # Step 2: Verify awake
    r = requests.get(inst.api_url("/is_sleeping"), timeout=10)
    r.raise_for_status()
    assert not r.json()["is_sleeping"], "Server claims it's still sleeping"

    inst.state = InstanceState.SERVING
    registry.add(inst)

    elapsed = time.time() - t0
    logger.info("[%s] Resumed in %.2fs (wake=%.2fs)", name, elapsed, t_wake)
    return elapsed


def stop_instance(name: str):
    """Stop and clean up a vLLM instance."""
    registry = InstanceRegistry()
    inst = registry.get(name)
    if not inst:
        raise ValueError(f"Instance '{name}' not found")

    if inst.pid:
        try:
            os.kill(inst.pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to PID %d", inst.pid)
            # Wait a bit for graceful shutdown
            for _ in range(30):
                try:
                    os.kill(inst.pid, 0)  # Check if still alive
                    time.sleep(1)
                except ProcessLookupError:
                    break
            else:
                os.kill(inst.pid, signal.SIGKILL)
                logger.warning("Force-killed PID %d", inst.pid)
        except ProcessLookupError:
            pass

    inst.state = InstanceState.STOPPED
    inst.pid = None
    registry.add(inst)
    logger.info("Instance '%s' stopped", name)


def swap_instances(from_name: str, to_name: str) -> dict:
    """
    Swap from one model to another.
    Suspends the active model, resumes the target model.
    Returns timing dict.
    """
    timings = {}

    t0 = time.time()
    timings["suspend"] = suspend_instance(from_name)
    timings["resume"] = resume_instance(to_name)
    timings["total"] = time.time() - t0

    logger.info(
        "Swapped %s -> %s in %.2fs (suspend=%.2fs, resume=%.2fs)",
        from_name,
        to_name,
        timings["total"],
        timings["suspend"],
        timings["resume"],
    )
    return timings


def get_status() -> list[dict]:
    """Get status of all registered instances."""
    registry = InstanceRegistry()
    results = []
    for inst in registry.all():
        info = inst.to_dict()
        # Check if process is still alive
        if inst.pid:
            try:
                os.kill(inst.pid, 0)
                info["alive"] = True
            except ProcessLookupError:
                info["alive"] = False
        else:
            info["alive"] = False

        # Check GPU memory if serving
        if inst.state == InstanceState.SERVING:
            try:
                r = requests.get(inst.api_url("/health"), timeout=2)
                info["healthy"] = r.status_code == 200
            except Exception:
                info["healthy"] = False

        results.append(info)
    return results
