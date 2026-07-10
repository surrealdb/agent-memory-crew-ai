"""Tests for the Spectron CrewAI tools using a mock client."""

from __future__ import annotations

import json

from spectron_crewai import get_sessionized_spectron_tools, get_spectron_tools
from spectron_crewai._runtime import FAILURE_THRESHOLD, SpectronRuntime, to_jsonable
from spectron_crewai.config import SpectronConfig

from conftest import FakeResp, FakeSpectron


def _tools(config, fake, scope="user/tobie"):
    return get_spectron_tools(config=config, client=fake, scope=scope)


def _by_name(tools, name):
    return next(t for t in tools if t.name == name)


# -- schemas -----------------------------------------------------------------


def test_tool_names_and_schemas(config, fake):
    tools = _tools(config, fake)
    assert {t.name for t in tools} == {
        "spectron_recall",
        "spectron_remember",
        "spectron_context",
        "spectron_forget",
        "spectron_reflect",
        "spectron_upload",
    }
    for t in tools:
        assert t.args_schema is not None
        assert t.description


# -- dispatch and argument threading -----------------------------------------


def test_recall_threads_k_and_scope_lens(config, fake):
    out = _by_name(_tools(config, fake), "spectron_recall").run(query="who is tobie")
    assert "Tobie is CTO" in out
    assert ("recall", "who is tobie", 3, ["user/tobie"]) in fake.calls


def test_recall_explicit_k_overrides_top_k(config, fake):
    _by_name(_tools(config, fake), "spectron_recall").run(query="q", k=9)
    assert ("recall", "q", 9, ["user/tobie"]) in fake.calls


def test_remember_uses_default_scope(config, fake):
    _by_name(_tools(config, fake), "spectron_remember").run(text="Tobie is CTO")
    assert ("remember", "Tobie is CTO", "user/tobie") in fake.calls


def test_remember_scope_arg_overrides(config, fake):
    _by_name(_tools(config, fake), "spectron_remember").run(
        text="x", scope="team/eng"
    )
    assert ("remember", "x", "team/eng") in fake.calls


def test_context_uses_query_context(config, fake):
    out = _by_name(_tools(config, fake), "spectron_context").run(query="summarise")
    assert "prefers dark mode" in out
    assert any(c[0] == "query_context" for c in fake.calls)


def test_forget_threads_purge(config, fake):
    _by_name(_tools(config, fake), "spectron_forget").run(query="old", purge=True)
    assert ("forget", "old", True) in fake.calls


def test_reflect_threads_persist(config, fake):
    _by_name(_tools(config, fake), "spectron_reflect").run(query="week", persist=True)
    assert ("reflect", "week", True) in fake.calls


def test_upload_threads_title(config, fake):
    _by_name(_tools(config, fake), "spectron_upload").run(
        path="/tmp/h.pdf", title="Handbook"
    )
    assert ("upload", "/tmp/h.pdf", "Handbook") in fake.calls


def test_sessionized_tools_scope_to_session(config, fake):
    tools = get_sessionized_spectron_tools("user-123", config=config, client=fake)
    _by_name(tools, "spectron_remember").run(text="hi")
    assert ("remember", "hi", "user-123") in fake.calls


def test_no_scope_omits_lens_and_scope(fake):
    cfg = SpectronConfig(endpoint="e", context="c", api_key="k", top_k=5)
    tools = get_spectron_tools(config=cfg, client=fake, scope=None)
    _by_name(tools, "spectron_recall").run(query="q")
    _by_name(tools, "spectron_remember").run(text="t")
    assert ("recall", "q", 5, None) in fake.calls
    assert ("remember", "t", None) in fake.calls


# -- fail open ---------------------------------------------------------------


def test_tools_fail_open_returns_error_string(config):
    fake = FakeSpectron(fail=True)
    tools = _tools(config, fake)
    for name in ("spectron_recall", "spectron_remember", "spectron_context"):
        out = _by_name(tools, name).run(
            **({"query": "q"} if "remember" not in name else {"text": "t"})
        )
        assert "unavailable" in out.lower()


def test_circuit_breaker_disables_after_threshold(config):
    fake = FakeSpectron(fail=True)
    runtime = SpectronRuntime(config, client=fake)
    tools = get_spectron_tools(runtime=runtime, scope="user/tobie")
    recall = _by_name(tools, "spectron_recall")
    for _ in range(FAILURE_THRESHOLD):
        recall.run(query="q")
    assert runtime.disabled is True
    # Once disabled the client is not called again.
    before = len(fake.calls)
    recall.run(query="q")
    assert len(fake.calls) == before


# -- serialization -----------------------------------------------------------


def test_to_jsonable_variants():
    assert to_jsonable(FakeResp(a=1, b=[FakeResp(c=2)])) == {"a": 1, "b": [{"c": 2}]}
    assert to_jsonable({"x": (1, 2)}) == {"x": [1, 2]}
    assert to_jsonable("s") == "s"


def test_ok_result_is_json(config, fake):
    out = _by_name(_tools(config, fake), "spectron_remember").run(text="fact")
    parsed = json.loads(out)
    assert parsed["success"] is True
    assert parsed["result"] == {"stored": True}
