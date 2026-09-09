"""SurrealDB Agent Memory integration for CrewAI.

Two ways to give a crew persistent, provenance-first memory backed by
`Agent Memory <https://surrealdb.com/agent-memory>`_:

* **Tools** an agent calls explicitly::

      from agent_memory_crewai import get_agent_memory_tools
      agent = Agent(role="Analyst", tools=get_agent_memory_tools(scope="user/tobie"))

* **Automatic memory** wired into a crew through the event bus::

      from agent_memory_crewai import Agent Memory
      memory = Agent Memory(default_scope="user/tobie")
      memory.attach()   # recall before tasks, write back after, consolidate at end
"""

from __future__ import annotations

from .config import AgentMemoryConfig
from .memory import AgentMemory, AgentMemoryListener
from .tools import (
    AgentMemoryContextTool,
    AgentMemoryForgetTool,
    AgentMemoryRecallTool,
    AgentMemoryReflectTool,
    AgentMemoryRememberTool,
    AgentMemoryUploadTool,
    get_sessionized_agent_memory_tools,
    get_agent_memory_tools,
)

__all__ = [
    "AgentMemoryConfig",
    "AgentMemory",
    "AgentMemoryListener",
    "get_agent_memory_tools",
    "get_sessionized_agent_memory_tools",
    "AgentMemoryRecallTool",
    "AgentMemoryRememberTool",
    "AgentMemoryContextTool",
    "AgentMemoryForgetTool",
    "AgentMemoryReflectTool",
    "AgentMemoryUploadTool",
]

__version__ = "0.3.0"
