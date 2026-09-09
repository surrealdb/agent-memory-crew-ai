# AgentMemory ⇄ CrewAI

Give your [CrewAI](https://www.crewai.com/) agents persistent, provenance-first
memory backed by [SurrealDB AgentMemory](https://surrealdb.com/agent-memory):
tri-temporal agent memory with semantic, lexical, graph and temporal recall.

This package offers two ways to use AgentMemory with CrewAI, and they work well
together:

- **Tools** an agent calls explicitly (recall, remember, context, forget,
  reflect, upload).
- **Automatic memory** that recalls relevant memory before each task, writes the
  result back after each task, and consolidates when the crew finishes, without
  changing your agents or tasks.

## Requirements

- Python 3.10+
- CrewAI 1.5+
- AgentMemory access (endpoint, context, API key).

## Install

```bash
pip install agent-memory-crew-ai
```

## Configure

Provide credentials through the environment. The API key is a secret and belongs
in a `.env` file, not in source.

```bash
export AGENT_MEMORY_ENDPOINT="https://your-instance.agent_memory.dev"
export AGENT_MEMORY_CONTEXT="my-context"
export AGENT_MEMORY_API_KEY="..."
# optional
export AGENT_MEMORY_DEFAULT_SCOPE="user/tobie"
export AGENT_MEMORY_TOP_K="5"
```

You can also pass any of these directly to `AgentMemoryMemory(...)` or
`AgentMemoryConfig(...)` instead of using the environment.

## Quickstart: tools

Attach the AgentMemory tools to an agent and let it decide when to use memory.

```python
from crewai import Agent, Task, Crew
from agent_memory_crewai import get_agent_memory_tools

agent = Agent(
    role="Research Analyst",
    goal="Answer questions using long-term memory",
    backstory="You recall what you have learned before and store new findings.",
    tools=get_agent_memory_tools(scope="user/tobie"),
    verbose=True,
)

task = Task(
    description="What do we know about Tobie's role? Store any new facts you learn.",
    expected_output="A short summary.",
    agent=agent,
)

Crew(agents=[agent], tasks=[task]).kickoff()
```

To isolate memory per user or session, use the sessionized factory:

```python
from agent_memory_crewai import get_sessionized_agent_memory_tools

tools = get_sessionized_agent_memory_tools("user-123")
```

## Quickstart: automatic memory

Enable automatic memory once and run your crew as usual. Recall happens before
each task, write-back after each task (on a background thread), and consolidation
when the crew finishes.

```python
from crewai import Agent, Task, Crew
from agent_memory_crewai import AgentMemoryMemory

memory = AgentMemoryMemory(default_scope="user/tobie")
memory.attach(verbose=True)   # registers the event listener

agent = Agent(
    role="Travel Planning Specialist",
    goal="Plan trips that respect the traveller's known preferences",
    backstory="You remember past trips and preferences.",
    tools=memory.tools(),     # optional: also expose explicit tools
)

task = Task(
    description="Plan a weekend trip for Tobie.",
    expected_output="A day-by-day plan.",
    agent=agent,
)

Crew(agents=[agent], tasks=[task]).kickoff()
memory.close()                # flush background writes on shutdown
```

`AgentMemoryMemory` is also usable directly:

```python
memory.remember("Tobie prefers window seats", scope="user/tobie")
hits = memory.recall("seat preference", scope="user/tobie")
answer = memory.context("What are Tobie's travel preferences?")
```

## Tools

| Tool | AgentMemory call | Purpose |
|---|---|---|
| `agent_memory_recall(query, k?)` | `recall` | Search memory (semantic, lexical, graph, temporal). |
| `agent_memory_remember(text, scope?)` | `remember` | Store a durable fact. |
| `agent_memory_context(query, k?)` | `query_context` | Synthesised answer from memory. |
| `agent_memory_forget(query, purge?)` | `forget` | Supersede (default) or hard-delete. |
| `agent_memory_reflect(query, persist?)` | `reflect` | Derive insights; optionally persist. |
| `agent_memory_upload(path, title?)` | `documents.upload` | Ingest a document into knowledge memory. |

## Configuration

| Setting | Env var | Default | Notes |
|---|---|---|---|
| `api_key` | `AGENT_MEMORY_API_KEY` | none | secret, required (keep it in `.env`) |
| `endpoint` | `AGENT_MEMORY_ENDPOINT` | none | required, origin with no trailing slash |
| `context` | `AGENT_MEMORY_CONTEXT` | none | required; AgentMemory pins a client to one context |
| `default_scope` | `AGENT_MEMORY_DEFAULT_SCOPE` | none | scope for writes and lens for reads, for example `user/tobie` |
| `top_k` | `AGENT_MEMORY_TOP_K` | `5` | memories recalled per query |
| `timeout` | `AGENT_MEMORY_TIMEOUT` | `30` | client timeout in seconds |
| `max_retries` | `AGENT_MEMORY_MAX_RETRIES` | `3` | client retry attempts |

## Reliability

The integration is built to never destabilise a crew:

- Writes run on a background daemon thread, so tasks never block on AgentMemory I/O.
- Every AgentMemory call is wrapped. Failures are logged and degrade to an empty or
  error result rather than raising into the agent or crew loop (fail open). A
  tool returns a short JSON error string instead of throwing.
- After repeated failures, or any authentication error, a circuit breaker
  disables memory for the rest of the process.

## A note on CrewAI memory backends

CrewAI's built-in `Memory` storage backend is embedding-centric: it embeds a
query locally and hands the storage layer a vector, never the query text.
AgentMemory is a text-native service that does its own embedding and multi-signal
ranking server-side, so it is exposed here as tools and an event-driven memory
layer rather than as a `StorageBackend`. This keeps AgentMemory's semantic, lexical,
graph and temporal recall intact.

## Development

```bash
pip install -e ".[dev]" crewai
pytest
```

## License

Apache-2.0
