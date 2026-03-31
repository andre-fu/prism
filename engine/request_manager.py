"""Request management: per-model queues, request lifecycle."""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class RequestState(Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"


@dataclass
class Request:
    """A single inference request."""
    id: int
    model_name: str
    prompt: str
    prompt_tokens: list[int] = field(default_factory=list)
    max_new_tokens: int = 100
    temperature: float = 0.0
    state: RequestState = RequestState.QUEUED
    seq_id: int = -1                 # KV cache sequence ID (assigned by scheduler)
    generated_tokens: list[int] = field(default_factory=list)
    arrival_time: float = 0.0
    start_time: float = 0.0         # When prefill started
    first_token_time: float = 0.0   # When first token was generated
    done_time: float = 0.0
    error: str | None = None
    # SLO + tenant fields
    tenant_id: str = "default"
    priority: int = 0               # Higher = more important
    slo_ttft_ms: float | None = None
    slo_tbt_ms: float | None = None

    @property
    def ttft(self) -> float:
        """Time to first token (seconds)."""
        if self.first_token_time > 0 and self.arrival_time > 0:
            return self.first_token_time - self.arrival_time
        return 0.0

    @property
    def total_time(self) -> float:
        if self.done_time > 0 and self.arrival_time > 0:
            return self.done_time - self.arrival_time
        return 0.0

    @property
    def tbt(self) -> float:
        """Average time between tokens (seconds)."""
        n = len(self.generated_tokens)
        if n <= 1 or self.first_token_time <= 0 or self.done_time <= 0:
            return 0.0
        return (self.done_time - self.first_token_time) / (n - 1)


class RequestManager:
    """Manages per-model request queues."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 0
        self._next_seq_id = 0
        # model_name -> deque of Request
        self._queues: dict[str, deque[Request]] = {}
        # request_id -> Request (for lookup)
        self._requests: dict[int, Request] = {}
        # model_name -> list of active (prefilling/decoding) requests
        self._active: dict[str, list[Request]] = {}

    def add_request(
        self,
        model_name: str,
        prompt: str,
        prompt_tokens: list[int],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        tenant_id: str = "default",
        priority: int = 0,
        slo_ttft_ms: float | None = None,
    ) -> Request:
        """Enqueue a new request. Returns the Request object."""
        with self._lock:
            req = Request(
                id=self._next_id,
                model_name=model_name,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                arrival_time=time.time(),
                tenant_id=tenant_id,
                priority=priority,
                slo_ttft_ms=slo_ttft_ms,
            )
            self._next_id += 1

            if model_name not in self._queues:
                self._queues[model_name] = deque()
            if model_name not in self._active:
                self._active[model_name] = []

            self._queues[model_name].append(req)
            self._requests[req.id] = req
            return req

    def allocate_seq_id(self) -> int:
        """Get a unique sequence ID for KV cache tracking."""
        with self._lock:
            sid = self._next_seq_id
            self._next_seq_id += 1
            return sid

    def pop_pending(self, model_name: str, max_count: int = 1) -> list[Request]:
        """Pop up to max_count pending requests from a model's queue."""
        with self._lock:
            queue = self._queues.get(model_name)
            if not queue:
                return []
            result = []
            for _ in range(min(max_count, len(queue))):
                req = queue.popleft()
                req.state = RequestState.PREFILLING
                req.start_time = time.time()
                req.seq_id = self._next_seq_id
                self._next_seq_id += 1
                self._active.setdefault(model_name, []).append(req)
                result.append(req)
            return result

    def get_active(self, model_name: str) -> list[Request]:
        """Get all active (prefilling/decoding) requests for a model."""
        with self._lock:
            return list(self._active.get(model_name, []))

    def complete_request(self, request_id: int):
        """Mark a request as done."""
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return
            req.state = RequestState.DONE
            req.done_time = time.time()
            active = self._active.get(req.model_name, [])
            self._active[req.model_name] = [r for r in active if r.id != request_id]

    def models_with_pending(self) -> list[str]:
        """Models that have queued requests waiting."""
        with self._lock:
            return [name for name, q in self._queues.items() if q]

    def models_with_active(self) -> list[str]:
        """Models that have in-flight requests (prefilling or decoding)."""
        with self._lock:
            return [name for name, reqs in self._active.items() if reqs]

    def pending_count(self, model_name: str) -> int:
        with self._lock:
            return len(self._queues.get(model_name, []))

    def active_count(self, model_name: str) -> int:
        with self._lock:
            return len(self._active.get(model_name, []))

    def total_pending(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._queues.values())

    def get_request(self, request_id: int) -> Request | None:
        return self._requests.get(request_id)

    def get_completed(self) -> list[Request]:
        """Get all completed requests (for draining results)."""
        with self._lock:
            return [r for r in self._requests.values() if r.state == RequestState.DONE]
