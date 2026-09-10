"""Show the automatic memory listener against a fake client. No credentials.

This drives the same event flow a real crew produces (task started, task
completed, crew finished) and prints what the Agent Memory listener does at
each step: recall before a task, write the result back after a task, and
consolidate when the crew finishes. It uses an in-memory fake client so it runs
with nothing but this package and CrewAI installed.

Run:

    python examples/auto_memory.py
"""

from __future__ import annotations

from crewai.events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    crewai_event_bus,
)
from crewai.tasks.task_output import TaskOutput

from agent_memory_crewai import AgentMemoryConfig, AgentMemory


class _Resp:
    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class FakeAgentMemory:
    """In-memory stand-in that records what the listener sends it."""

    def __init__(self):
        self._facts = []

    def recall(self, query, *, k=None, lens=None):
        return _Resp(memories=[{"text": f} for f in self._facts][: (k or 5)])

    def remember_many(self, items, *, session_id=None, scopes=None, **kwargs):
        for m in items:
            self._facts.append(f"[{m['role']}] {m['content']}")
        return _Resp(count=len(items))

    def consolidate(self, **kwargs):
        return _Resp(ok=True, facts=len(self._facts))


def _emit(event):
    """Emit an event and wait for the sync handler to finish."""
    future = crewai_event_bus.emit("example", event)
    if future is not None:
        future.result(timeout=5)


def main() -> None:
    config = AgentMemoryConfig(
        endpoint="https://demo.agent_memory.local",
        context="demo",
        api_key="sk-demo",
        default_scope="user/tobie",
    )
    fake = FakeAgentMemory()

    # In real use: AgentMemory(default_scope="user/tobie").attach()
    memory = AgentMemory(config=config, client=fake)

    # Seed a fact so the pre-task recall has something to return.
    memory.remember_many([{"role": "user", "content": "Tobie prefers window seats"}])

    # We hand-build events here to simulate a crew. Doing that inside
    # scoped_handlers keeps CrewAI's own listeners out of this demo, which a
    # real crew.kickoff() would satisfy with fully populated events. In your own
    # code you just call memory.attach() and run the crew as usual.
    with crewai_event_bus.scoped_handlers():
        listener = memory.attach(session_id="demo-session", verbose=True)

        _emit(CrewKickoffStartedEvent(crew_name="demo", inputs=None))

        print("== task starts: the listener recalls relevant memory ==")
        _emit(TaskStartedEvent(context="Plan a weekend trip for Tobie"))
        print("recalled context:\n" + (listener.last_context or "(none)"))

        print("\n== task completes: the listener writes the result back ==")
        output = TaskOutput(
            description="Plan a weekend trip for Tobie",
            agent="Travel Planning Specialist",
            raw="Day 1: arrive in Lisbon. Day 2: Sintra day trip.",
        )
        _emit(TaskCompletedEvent(output=output))

        print("\n== crew finishes: the listener consolidates memory ==")
        _emit(CrewKickoffCompletedEvent(crew_name="demo", output="done"))

    # Flush background writes before exit (a real app calls memory.close()).
    memory.close()
    print("\nfacts now on record:")
    for fact in fake._facts:
        print(" ", fact)


if __name__ == "__main__":
    main()
