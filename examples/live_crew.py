"""Run a real CrewAI crew with Spectron-backed automatic memory.

Requires:
  * pip install "spectron-crewai"   (pulls in crewai + surrealdb[spectron])
  * Spectron credentials in the environment:
        export SPECTRON_ENDPOINT="https://your-instance.spectron.dev"
        export SPECTRON_CONTEXT="my-context"
        export SPECTRON_API_KEY="..."
  * An LLM configured for CrewAI (for example OPENAI_API_KEY).

Run:

    python examples/live_crew.py

This writes to your Spectron context, so use a throwaway context if you do not
want the demo data to stick.
"""

from __future__ import annotations

import os
import sys

from crewai import Agent, Crew, Task

from spectron_crewai import SpectronMemory


def main() -> int:
    memory = SpectronMemory(default_scope="user/tobie")

    if not memory.is_available():
        print(
            "Spectron is not configured. Set SPECTRON_ENDPOINT / SPECTRON_CONTEXT / "
            'SPECTRON_API_KEY and `pip install "surrealdb[spectron]"`.',
            file=sys.stderr,
        )
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (or configure another CrewAI LLM) first.", file=sys.stderr)
        return 1

    # Enable automatic memory: recall before tasks, write back after,
    # consolidate when the crew finishes.
    memory.attach(verbose=True)

    agent = Agent(
        role="Personal Assistant",
        goal="Help Tobie while remembering what you learn about them",
        backstory=(
            "You keep durable notes about Tobie in long-term memory and use them "
            "to give consistent, personalised help."
        ),
        tools=memory.tools(),  # also expose explicit memory tools
        verbose=True,
    )

    task = Task(
        description=(
            "Tobie just told you they are moving to Lisbon and prefer morning "
            "meetings. Acknowledge this and store the durable facts."
        ),
        expected_output="A short acknowledgement.",
        agent=agent,
    )

    result = Crew(agents=[agent], tasks=[task]).kickoff()
    print("\nresult:\n", result)

    # Show that the facts are now recallable.
    print("\nrecall 'where does tobie live?':")
    print(memory.recall("where does tobie live?"))

    memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
