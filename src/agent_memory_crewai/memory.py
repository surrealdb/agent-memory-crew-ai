"""Automatic Agent Memory for CrewAI.

Two pieces:

* :class:`AgentMemory` is a small façade over a shared Agent Memory client. Use it
  directly for programmatic remember/recall, or to build the tool set.
* :class:`AgentMemoryListener` wires that memory into a crew through CrewAI's
  event bus so memory works without changing your agents or tasks:

    - before each task it recalls relevant memory and stashes it (readable on the
      memory object and logged when ``verbose`` is set),
    - after each task it writes the result back to Agent Memory on a background
      thread,
    - when the crew finishes it triggers a background consolidation.

Constructing a listener registers it on the global event bus, so a single
``AgentMemory(...).attach()`` call is enough to enable automatic memory.

Every handler is fail-open: an Agent Memory problem is logged and skipped, never
raised into the crew.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from crewai.events import (
    BaseEventListener,
    CrewKickoffCompletedEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
)

from ._runtime import AgentMemoryRuntime, to_jsonable
from .config import AgentMemoryConfig
from .tools import get_agent_memory_tools

logger = logging.getLogger("agent_memory_crewai")


class AgentMemory:
    """A façade over a shared Agent Memory client for use with CrewAI."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        default_scope: Optional[str] = None,
        top_k: Optional[int] = None,
        config: Optional[AgentMemoryConfig] = None,
        runtime: Optional[AgentMemoryRuntime] = None,
        client: Any = None,
    ) -> None:
        cfg = config or AgentMemoryConfig.from_env(
            endpoint=endpoint,
            context=context,
            api_key=api_key,
            default_scope=default_scope,
            top_k=top_k,
        )
        self._runtime = runtime or AgentMemoryRuntime(cfg, client=client)

    @property
    def runtime(self) -> AgentMemoryRuntime:
        return self._runtime

    def is_available(self) -> bool:
        """Config- and dependency-only readiness check. No network calls."""
        return self._runtime.is_available()

    # -- programmatic memory operations (fail-open) --------------------------

    def remember(self, text: str, *, scope: Optional[str] = None) -> Any:
        effective = scope or self._runtime.default_scope

        def _call(client: Any) -> Any:
            if effective:
                return client.remember(text, scopes=effective)
            return client.remember(text)

        _, result = self._runtime.call("remember", _call)
        return to_jsonable(result)

    def remember_many(
        self,
        items: List[Dict[str, str]],
        *,
        session_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Any:
        effective = scope or self._runtime.default_scope

        def _call(client: Any) -> Any:
            kwargs: Dict[str, Any] = {}
            if session_id:
                kwargs["session_id"] = session_id
            if effective:
                kwargs["scopes"] = effective
            return client.remember_many(items, **kwargs)

        _, result = self._runtime.call("remember_many", _call)
        return to_jsonable(result)

    def recall(
        self, query: str, *, k: Optional[int] = None, scope: Optional[str] = None
    ) -> Any:
        top_k = k or self._runtime.config.top_k
        lens = [scope or self._runtime.default_scope] if (scope or self._runtime.default_scope) else None

        def _call(client: Any) -> Any:
            if lens:
                return client.recall(query, k=top_k, lens=lens)
            return client.recall(query, k=top_k)

        _, result = self._runtime.call("recall", _call)
        return to_jsonable(result)

    def context(
        self, query: str, *, k: Optional[int] = None, scope: Optional[str] = None
    ) -> Any:
        top_k = k or self._runtime.config.top_k
        lens = [scope or self._runtime.default_scope] if (scope or self._runtime.default_scope) else None

        def _call(client: Any) -> Any:
            if lens:
                return client.query_context(query, k=top_k, lens=lens)
            return client.query_context(query, k=top_k)

        _, result = self._runtime.call("context", _call)
        return to_jsonable(result)

    def forget(self, query: str, *, purge: bool = False) -> Any:
        _, result = self._runtime.call(
            "forget", lambda client: client.forget(query, purge=bool(purge))
        )
        return to_jsonable(result)

    def reflect(self, query: str, *, persist: bool = False) -> Any:
        _, result = self._runtime.call(
            "reflect", lambda client: client.reflect(query, persist=bool(persist))
        )
        return to_jsonable(result)

    def consolidate(self) -> Any:
        _, result = self._runtime.call(
            "consolidate", lambda client: client.consolidate()
        )
        return to_jsonable(result)

    # -- wiring --------------------------------------------------------------

    def tools(self, *, scope: Optional[str] = None) -> List[Any]:
        """Return the Agent Memory tool set backed by this memory's client."""
        return get_agent_memory_tools(runtime=self._runtime, scope=scope)

    def listener(self, **kwargs: Any) -> "AgentMemoryListener":
        """Create (and register) an event listener for automatic memory."""
        return AgentMemoryListener(self, **kwargs)

    def attach(self, **kwargs: Any) -> "AgentMemoryListener":
        """Enable automatic memory. Alias for :meth:`listener`.

        Constructing the listener registers it on the CrewAI event bus, so the
        returned object is already active.
        """
        return self.listener(**kwargs)

    def close(self) -> None:
        """Flush pending background writes and close the client."""
        self._runtime.close()


class AgentMemoryListener(BaseEventListener):
    """Wires :class:`AgentMemory` into a crew through the CrewAI event bus."""

    def __init__(
        self,
        memory: AgentMemory,
        *,
        recall: bool = True,
        write: bool = True,
        consolidate: bool = True,
        scope: Optional[str] = None,
        session_id: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self._memory = memory
        self._recall = recall
        self._write = write
        self._consolidate = consolidate
        self._scope = scope
        self._session_id = session_id
        self.verbose = verbose
        # Recalled context from the most recent task start, for inspection.
        self.last_context: Optional[str] = None
        super().__init__()

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        @crewai_event_bus.on(TaskStartedEvent)
        def _on_task_started(source: Any, event: Any) -> None:  # noqa: ANN001
            if not self._recall:
                return
            query = _text_of(getattr(event, "context", None)) or _text_of(
                getattr(event, "task_name", None)
            )
            if not query:
                return
            try:
                hits = self._memory.recall(query, scope=self._scope)
            except Exception as exc:  # pragma: no cover - fail open
                logger.warning("Agent Memory auto-recall failed: %s", exc)
                return
            block = _format_block(hits)
            self.last_context = block
            if block and self.verbose:
                logger.info("Agent Memory recalled for task:\n%s", block)

        @crewai_event_bus.on(TaskCompletedEvent)
        def _on_task_completed(source: Any, event: Any) -> None:  # noqa: ANN001
            if not self._write:
                return
            output = getattr(event, "output", None)
            task_desc = _text_of(getattr(output, "description", None)) or _text_of(
                getattr(event, "task_name", None)
            )
            result = _text_of(getattr(output, "raw", None))
            items: List[Dict[str, str]] = []
            if task_desc:
                items.append({"role": "user", "content": task_desc})
            if result:
                items.append({"role": "assistant", "content": result})
            if not items:
                return
            # Write on the background thread so the crew never blocks.
            self._memory.runtime.submit_write(
                lambda client: _remember_many(
                    client, items, self._session_id, self._effective_scope()
                )
            )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def _on_kickoff_completed(source: Any, event: Any) -> None:  # noqa: ANN001
            if not self._consolidate:
                return
            self._memory.runtime.submit_write(lambda client: client.consolidate())

    def _effective_scope(self) -> Optional[str]:
        return self._scope or self._memory.runtime.default_scope


# -- helpers -----------------------------------------------------------------


def _remember_many(
    client: Any,
    items: List[Dict[str, str]],
    session_id: Optional[str],
    scope: Optional[str],
) -> Any:
    kwargs: Dict[str, Any] = {}
    if session_id:
        kwargs["session_id"] = session_id
    if scope:
        kwargs["scopes"] = scope
    return client.remember_many(items, **kwargs)


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _format_block(hits: Any) -> str:
    """Render recalled memories as a short plain-text block."""
    data = to_jsonable(hits)
    if isinstance(data, dict):
        for key in ("answer", "context", "summary", "text"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for key in ("memories", "results", "hits", "items", "matches"):
            val = data.get(key)
            if isinstance(val, list):
                data = val
                break
    lines = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")
            elif isinstance(item, dict):
                for key in ("text", "content", "summary", "fact", "statement"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        lines.append(f"- {val.strip()}")
                        break
    return "\n".join(lines)
