"""SurrealDB Spectron integration for CrewAI.

Two ways to give a crew persistent, provenance-first memory backed by
`Spectron <https://surrealdb.com/platform/spectron>`_:

* **Tools** an agent calls explicitly::

      from spectron_crewai import get_spectron_tools
      agent = Agent(role="Analyst", tools=get_spectron_tools(scope="user/tobie"))

* **Automatic memory** wired into a crew through the event bus::

      from spectron_crewai import SpectronMemory
      memory = SpectronMemory(default_scope="user/tobie")
      memory.attach()   # recall before tasks, write back after, consolidate at end
"""

from __future__ import annotations

from .config import SpectronConfig
from .memory import SpectronMemory, SpectronMemoryListener
from .tools import (
    SpectronContextTool,
    SpectronForgetTool,
    SpectronRecallTool,
    SpectronReflectTool,
    SpectronRememberTool,
    SpectronUploadTool,
    get_sessionized_spectron_tools,
    get_spectron_tools,
)

__all__ = [
    "SpectronConfig",
    "SpectronMemory",
    "SpectronMemoryListener",
    "get_spectron_tools",
    "get_sessionized_spectron_tools",
    "SpectronRecallTool",
    "SpectronRememberTool",
    "SpectronContextTool",
    "SpectronForgetTool",
    "SpectronReflectTool",
    "SpectronUploadTool",
]

__version__ = "0.2.0"
