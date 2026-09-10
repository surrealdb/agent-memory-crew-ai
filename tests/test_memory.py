"""Tests for Agent Memory and the automatic-memory event listener."""

from __future__ import annotations

from crewai.events import (
    CrewKickoffCompletedEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    crewai_event_bus,
)
from crewai.tasks.task_output import TaskOutput

from agent_memory_crewai import AgentMemory

from conftest import FakeAgentMemory


def _memory(config, fake):
    return AgentMemory(config=config, client=fake)


def _task_output(description="Plan a trip for Tobie", raw="Day 1: Lisbon"):
    return TaskOutput(description=description, agent="Planner", raw=raw)


def _emit(event):
    """Emit and wait for the sync handler to finish (emit runs it in a pool)."""
    future = crewai_event_bus.emit("src", event)
    if future is not None:
        future.result(timeout=5)


# -- facade ------------------------------------------------------------------


def test_facade_threads_args(config, fake):
    mem = _memory(config, fake)
    mem.remember("Tobie is CTO")
    mem.remember_many([{"role": "user", "content": "hi"}], session_id="s1")
    mem.recall("role?")
    mem.context("summarise")
    mem.forget("old", purge=True)
    mem.reflect("week", persist=True)
    mem.consolidate()

    assert ("remember", "Tobie is CTO", "user/tobie") in fake.calls
    assert ("remember_many", ("user",), "s1", "user/tobie") in fake.calls
    assert ("recall", "role?", 3, ["user/tobie"]) in fake.calls
    assert ("query_context", "summarise", 3, ["user/tobie"]) in fake.calls
    assert ("forget", "old", True) in fake.calls
    assert ("reflect", "week", True) in fake.calls
    assert ("consolidate",) in fake.calls


def test_is_available(config, fake):
    assert _memory(config, fake).is_available() is True


# -- listener ----------------------------------------------------------------


def test_listener_recalls_on_task_start(config, fake):
    with crewai_event_bus.scoped_handlers():
        mem = _memory(config, fake)
        listener = mem.attach()
        _emit(TaskStartedEvent(context="what about tobie?"))
    assert any(c[0] == "recall" for c in fake.calls)
    assert "Tobie is CTO" in (listener.last_context or "")


def test_listener_writes_back_on_task_complete(config, fake):
    with crewai_event_bus.scoped_handlers():
        mem = _memory(config, fake)
        mem.attach(session_id="sess-1")
        _emit(TaskCompletedEvent(output=_task_output()))
        mem.runtime.drain()
    writes = [c for c in fake.calls if c[0] == "remember_many"]
    assert writes, "expected a background remember_many"
    assert writes[0][1] == ("user", "assistant")
    assert writes[0][2] == "sess-1"
    assert writes[0][3] == "user/tobie"


def test_listener_consolidates_on_kickoff_complete(config, fake):
    with crewai_event_bus.scoped_handlers():
        mem = _memory(config, fake)
        mem.attach()
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        mem.runtime.drain()
    assert any(c[0] == "consolidate" for c in fake.calls)


def test_listener_toggles_off(config, fake):
    with crewai_event_bus.scoped_handlers():
        mem = _memory(config, fake)
        mem.attach(recall=False, write=False, consolidate=False)
        _emit(TaskStartedEvent(context="q"))
        _emit(TaskCompletedEvent(output=_task_output()))
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        mem.runtime.drain()
    assert fake.calls == []


def test_listener_fails_open(config):
    fake = FakeAgentMemory(fail=True)
    with crewai_event_bus.scoped_handlers():
        mem = AgentMemory(config=config, client=fake)
        mem.attach(session_id="s")
        # None of these should raise despite the client failing.
        _emit(TaskStartedEvent(context="q"))
        _emit(TaskCompletedEvent(output=_task_output()))
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        mem.runtime.drain()
    # It attempted the calls but degraded gracefully.
    assert any(c[0] == "recall" for c in fake.calls)
