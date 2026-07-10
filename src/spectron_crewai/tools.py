"""CrewAI tools backed by SurrealDB Spectron.

Each tool is a :class:`crewai.tools.BaseTool` an agent can call directly:

* ``spectron_recall``   search memory (semantic, lexical, graph, temporal)
* ``spectron_remember`` store a durable fact
* ``spectron_context``  ask Spectron to synthesise an answer from memory
* ``spectron_forget``   supersede or hard-delete memories
* ``spectron_reflect``  derive higher-level insights
* ``spectron_upload``   ingest a document into knowledge memory

All tools share one :class:`~spectron_crewai._runtime.SpectronRuntime`, so they
inherit the same fail-open and circuit-breaker behaviour: a tool never raises
into the agent loop. On failure it returns a short JSON error string instead.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ._runtime import SpectronRuntime, to_jsonable
from .config import SpectronConfig


# -- argument schemas --------------------------------------------------------


class _RecallArgs(BaseModel):
    query: str = Field(description="What to search memory for.")
    k: Optional[int] = Field(
        default=None,
        description="Max number of memories to return. Defaults to the configured top_k.",
    )


class _RememberArgs(BaseModel):
    text: str = Field(description="The fact to remember. Prefer a concise, self-contained statement.")
    scope: Optional[str] = Field(
        default=None,
        description="Optional scope for the fact, for example 'user/tobie'. Defaults to the configured scope.",
    )


class _ContextArgs(BaseModel):
    query: str = Field(description="The question to answer from memory.")
    k: Optional[int] = Field(
        default=None,
        description="Max memories to consider. Defaults to the configured top_k.",
    )


class _ForgetArgs(BaseModel):
    query: str = Field(description="What to forget.")
    purge: bool = Field(
        default=False,
        description="Hard-delete instead of superseding. Defaults to false.",
    )


class _ReflectArgs(BaseModel):
    query: str = Field(description="What to reflect on.")
    persist: bool = Field(
        default=False,
        description="Persist the reflection back into memory. Defaults to false.",
    )


class _UploadArgs(BaseModel):
    path: str = Field(description="Local filesystem path to the document.")
    title: Optional[str] = Field(default=None, description="Optional human-readable title.")


# -- base tool ---------------------------------------------------------------


class _SpectronTool(BaseTool):
    """Common wiring shared by all Spectron tools."""

    _runtime: SpectronRuntime = PrivateAttr()
    _scope: Optional[str] = PrivateAttr(default=None)

    def __init__(self, runtime: SpectronRuntime, scope: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._runtime = runtime
        self._scope = scope if scope is not None else runtime.default_scope

    def _unavailable(self) -> str:
        return json.dumps({"error": "Spectron memory is unavailable.", "provider": "spectron"})

    def _ok(self, result: Any) -> str:
        return json.dumps({"success": True, "result": to_jsonable(result)}, default=str)


# -- concrete tools ----------------------------------------------------------


class SpectronRecallTool(_SpectronTool):
    name: str = "spectron_recall"
    description: str = (
        "Search long-term memory in Spectron for facts relevant to a query, "
        "ranked across semantic, lexical, graph and temporal signals. Use this "
        "to retrieve what is known about a person, project, or topic."
    )
    args_schema: type[BaseModel] = _RecallArgs

    def _run(self, query: str, k: Optional[int] = None) -> str:
        top_k = k or self._runtime.config.top_k
        lens = [self._scope] if self._scope else None

        def _call(client: Any) -> Any:
            if lens:
                return client.recall(query, k=top_k, lens=lens)
            return client.recall(query, k=top_k)

        ok, result = self._runtime.call("recall", _call)
        if not ok:
            return self._unavailable()
        return _format_memories(result) or self._ok(result)


class SpectronRememberTool(_SpectronTool):
    name: str = "spectron_remember"
    description: str = (
        "Store a durable fact in Spectron memory. Spectron versions facts "
        "tri-temporally and never overwrites history, so prefer clear, "
        "self-contained statements."
    )
    args_schema: type[BaseModel] = _RememberArgs

    def _run(self, text: str, scope: Optional[str] = None) -> str:
        effective = scope or self._scope

        def _call(client: Any) -> Any:
            if effective:
                return client.remember(text, scope=effective)
            return client.remember(text)

        ok, result = self._runtime.call("remember", _call)
        return self._ok(result) if ok else self._unavailable()


class SpectronContextTool(_SpectronTool):
    name: str = "spectron_context"
    description: str = (
        "Ask Spectron to synthesise an answer from memory for a question, rather "
        "than returning raw hits. Use when you want a summarised, reasoned view."
    )
    args_schema: type[BaseModel] = _ContextArgs

    def _run(self, query: str, k: Optional[int] = None) -> str:
        top_k = k or self._runtime.config.top_k
        lens = [self._scope] if self._scope else None

        def _call(client: Any) -> Any:
            if lens:
                return client.query_context(query, k=top_k, lens=lens)
            return client.query_context(query, k=top_k)

        ok, result = self._runtime.call("context", _call)
        if not ok:
            return self._unavailable()
        return _format_memories(result) or self._ok(result)


class SpectronForgetTool(_SpectronTool):
    name: str = "spectron_forget"
    description: str = (
        "Forget memories matching a query. By default this supersedes them "
        "(kept as history, marked no longer valid). Set purge=true to hard-delete."
    )
    args_schema: type[BaseModel] = _ForgetArgs

    def _run(self, query: str, purge: bool = False) -> str:
        ok, result = self._runtime.call(
            "forget", lambda client: client.forget(query, purge=bool(purge))
        )
        return self._ok(result) if ok else self._unavailable()


class SpectronReflectTool(_SpectronTool):
    name: str = "spectron_reflect"
    description: str = (
        "Run a reflection over memory to derive higher-level insights about a "
        "topic. Set persist=true to write the reflection back into memory."
    )
    args_schema: type[BaseModel] = _ReflectArgs

    def _run(self, query: str, persist: bool = False) -> str:
        ok, result = self._runtime.call(
            "reflect", lambda client: client.reflect(query, persist=bool(persist))
        )
        return self._ok(result) if ok else self._unavailable()


class SpectronUploadTool(_SpectronTool):
    name: str = "spectron_upload"
    description: str = (
        "Ingest a document from a local file path into Spectron's knowledge "
        "memory so its contents become recallable."
    )
    args_schema: type[BaseModel] = _UploadArgs

    def _run(self, path: str, title: Optional[str] = None) -> str:
        def _call(client: Any) -> Any:
            if title:
                return client.documents.upload(path, title=title)
            return client.documents.upload(path)

        ok, result = self._runtime.call("upload", _call)
        return self._ok(result) if ok else self._unavailable()


_TOOL_CLASSES = (
    SpectronRecallTool,
    SpectronRememberTool,
    SpectronContextTool,
    SpectronForgetTool,
    SpectronReflectTool,
    SpectronUploadTool,
)


# -- factories ---------------------------------------------------------------


def get_spectron_tools(
    *,
    scope: Optional[str] = None,
    config: Optional[SpectronConfig] = None,
    runtime: Optional[SpectronRuntime] = None,
    client: Any = None,
) -> List[BaseTool]:
    """Build the full set of Spectron tools sharing one runtime.

    Args:
        scope: Optional scope applied to reads (as a lens) and writes. Defaults
            to the configured ``default_scope``.
        config: Explicit config. Resolved from the environment when omitted.
        runtime: An existing runtime to reuse (all tools share it).
        client: A pre-built Spectron client (mainly for tests).

    Returns:
        A list of ``BaseTool`` instances to attach to a CrewAI agent.
    """
    rt = runtime or SpectronRuntime(config or SpectronConfig.from_env(), client=client)
    return [cls(rt, scope=scope) for cls in _TOOL_CLASSES]


def get_sessionized_spectron_tools(
    session_id: str,
    *,
    config: Optional[SpectronConfig] = None,
    runtime: Optional[SpectronRuntime] = None,
    client: Any = None,
) -> List[BaseTool]:
    """Build Spectron tools scoped to a single session or user.

    The ``session_id`` becomes the scope, so reads and writes made through these
    tools are isolated to that session (for example ``"user-123"``).
    """
    return get_spectron_tools(
        scope=session_id, config=config, runtime=runtime, client=client
    )


# -- formatting helpers ------------------------------------------------------


def _format_memories(resp: Any) -> str:
    """Turn a recall/context response into a compact, readable string.

    Returns an empty string when nothing usable is found, so callers can fall
    back to the raw JSON result.
    """
    data = to_jsonable(resp)

    if isinstance(data, dict):
        for key in ("answer", "context", "summary", "text"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    items = _extract_items(data)
    lines = []
    for item in items:
        text = _item_text(item)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def _extract_items(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("memories", "results", "hits", "items", "matches", "recall"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("text", "content", "summary", "fact", "statement", "value"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return json.dumps(item, default=str)
    return str(item)
