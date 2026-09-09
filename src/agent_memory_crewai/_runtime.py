"""Shared runtime for the Agent Memory CrewAI integration.

Both the tools and the automatic memory listener talk to Agent Memory through a
single :class:`AgentMemoryRuntime`. It owns the lazily built client and enforces
the reliability rules the integration promises:

* **Fail open.** Every Agent Memory call is wrapped. Failures are logged and degrade
  to an empty result, never raised into the agent or crew loop.
* **Circuit breaker.** After repeated failures, or any authentication error, the
  runtime disables itself for the rest of the process and stops calling Agent Memory.
* **Non-blocking writes.** Writes are handed to a background daemon thread so a
  crew never blocks on Agent Memory I/O.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
from typing import Any, Callable, Optional, Tuple

from .client import build_client, is_auth_error, agent_memory_errors, agent_memory_installed
from .config import AgentMemoryConfig

logger = logging.getLogger("agent_memory_crewai")

# Disable the runtime after this many consecutive failures.
FAILURE_THRESHOLD = 3
# Sentinel pushed onto the write queue to stop the worker.
_STOP = object()


class AgentMemoryRuntime:
    """A shared, fail-open wrapper around an Agent Memory client."""

    def __init__(self, config: AgentMemoryConfig, client: Any = None) -> None:
        self._config = config
        self._client = client
        self._errors: Tuple[type, ...] = agent_memory_errors()
        self._consecutive_failures = 0
        self._disabled = False
        self._lock = threading.Lock()
        self._write_q: "queue.Queue[Any]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

    # -- identity ------------------------------------------------------------

    @property
    def config(self) -> AgentMemoryConfig:
        return self._config

    @property
    def default_scope(self) -> Optional[str]:
        return self._config.default_scope

    # -- client --------------------------------------------------------------

    def client(self) -> Any:
        """Return the Agent Memory client, building it lazily. None if unavailable."""
        if self._disabled:
            return None
        if self._client is not None:
            return self._client
        if not self._config.is_configured():
            logger.warning(
                "Agent Memory is not configured (need endpoint, context and api_key); "
                "memory disabled."
            )
            self._disabled = True
            return None
        if not agent_memory_installed():
            logger.warning(
                "Agent Memory SDK not installed; run `pip install \"surrealdb[memory]>=3.0.0b8\"`. "
                "Memory disabled."
            )
            self._disabled = True
            return None
        try:
            self._client = build_client(self._config)
            self._errors = agent_memory_errors()
        except Exception as exc:  # pragma: no cover - depends on SDK/env
            logger.warning("Agent Memory client init failed; memory disabled: %s", exc)
            self._disabled = True
            self._client = None
        return self._client

    def is_available(self) -> bool:
        """Config- and dependency-only readiness check. No network calls."""
        if self._disabled:
            return False
        if self._client is not None:
            return True
        return self._config.is_configured() and agent_memory_installed()

    @property
    def disabled(self) -> bool:
        return self._disabled

    # -- circuit breaker -----------------------------------------------------

    def _record_ok(self) -> None:
        self._consecutive_failures = 0

    def _record_fail(self, where: str, exc: BaseException) -> None:
        if is_auth_error(exc):
            logger.warning(
                "Agent Memory auth error during %s; disabling memory: %s", where, exc
            )
            self._disabled = True
            return
        self._consecutive_failures += 1
        logger.warning(
            "AgentMemory %s failed (%d/%d): %s",
            where,
            self._consecutive_failures,
            FAILURE_THRESHOLD,
            exc,
        )
        if self._consecutive_failures >= FAILURE_THRESHOLD:
            logger.warning(
                "Agent Memory failure threshold reached; disabling memory for this process."
            )
            self._disabled = True

    # -- calls ---------------------------------------------------------------

    def call(self, where: str, fn: Callable[[Any], Any]) -> Tuple[bool, Any]:
        """Run ``fn(client)`` fail-open.

        Returns ``(ok, result)``. On any failure ``ok`` is False, ``result`` is
        None, and the failure is recorded against the circuit breaker.
        """
        client = self.client()
        if client is None:
            return False, None
        try:
            result = fn(client)
            self._record_ok()
            return True, result
        except self._errors as exc:  # type: ignore[misc]
            self._record_fail(where, exc)
            return False, None
        except Exception as exc:  # pragma: no cover - unexpected
            self._record_fail(where, exc)
            return False, None

    # -- background writes ---------------------------------------------------

    def submit_write(self, fn: Callable[[Any], Any]) -> None:
        """Queue ``fn(client)`` to run on the background writer thread."""
        if self._disabled and self._client is None:
            return
        self._ensure_worker()
        self._write_q.put(fn)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._write_loop, name="agent_memory-writer", daemon=True
            )
            self._worker.start()

    def _write_loop(self) -> None:
        while True:
            job = self._write_q.get()
            try:
                if job is _STOP:
                    return
                self.call("write", job)
            finally:
                self._write_q.task_done()

    def drain(self) -> None:
        """Block until queued background writes have been processed."""
        self._write_q.join()

    def close(self) -> None:
        """Flush pending writes, stop the worker, and close the client."""
        worker = self._worker
        if worker and worker.is_alive():
            self._write_q.put(_STOP)
            worker.join(timeout=5.0)
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of Agent Memory SDK response objects to plain JSON data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    # Pydantic v2 / v1 models.
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return to_jsonable(method())
            except Exception:
                pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "__dict__"):
        return {
            k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")
        }
    return str(obj)
