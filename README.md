# Spectron ⇄ CrewAI

Give your [CrewAI](https://www.crewai.com/) agents persistent, provenance-first
memory backed by [SurrealDB Spectron](https://surrealdb.com/platform/spectron):
tri-temporal agent memory with semantic, lexical, graph and temporal recall.

This package offers two ways to use Spectron with CrewAI, and they work well
together:

- **Tools** an agent calls explicitly (recall, remember, context, forget,
  reflect, upload).
- **Automatic memory** that recalls relevant memory before each task, writes the
  result back after each task, and consolidates when the crew finishes, without
  changing your agents or tasks.

## Requirements

- Python 3.10+
- CrewAI 1.5+
- Spectron access (endpoint, context, API key).

## Install

```bash
pip install spectron-crew-ai
```

## Configure

Provide credentials through the environment. The API key is a secret and belongs
in a `.env` file, not in source.

```bash
export SPECTRON_ENDPOINT="https://your-instance.spectron.dev"
export SPECTRON_CONTEXT="my-context"
export SPECTRON_API_KEY="..."
# optional
export SPECTRON_DEFAULT_SCOPE="user/tobie"
export SPECTRON_TOP_K="5"
```

You can also pass any of these directly to `SpectronMemory(...)` or
`SpectronConfig(...)` instead of using the environment.

## Quickstart: tools

Attach the Spectron tools to an agent and let it decide when to use memory.

```python
from crewai import Agent, Task, Crew
from spectron_crewai import get_spectron_tools

agent = Agent(
    role="Research Analyst",
    goal="Answer questions using long-term memory",
    backstory="You recall what you have learned before and store new findings.",
    tools=get_spectron_tools(scope="user/tobie"),
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
from spectron_crewai import get_sessionized_spectron_tools

tools = get_sessionized_spectron_tools("user-123")
```

## Quickstart: automatic memory

Enable automatic memory once and run your crew as usual. Recall happens before
each task, write-back after each task (on a background thread), and consolidation
when the crew finishes.

```python
from crewai import Agent, Task, Crew
from spectron_crewai import SpectronMemory

memory = SpectronMemory(default_scope="user/tobie")
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

`SpectronMemory` is also usable directly:

```python
memory.remember("Tobie prefers window seats", scope="user/tobie")
hits = memory.recall("seat preference", scope="user/tobie")
answer = memory.context("What are Tobie's travel preferences?")
```

## Tools

| Tool | Spectron call | Purpose |
|---|---|---|
| `spectron_recall(query, k?)` | `recall` | Search memory (semantic, lexical, graph, temporal). |
| `spectron_remember(text, scope?)` | `remember` | Store a durable fact. |
| `spectron_context(query, k?)` | `query_context` | Synthesised answer from memory. |
| `spectron_forget(query, purge?)` | `forget` | Supersede (default) or hard-delete. |
| `spectron_reflect(query, persist?)` | `reflect` | Derive insights; optionally persist. |
| `spectron_upload(path, title?)` | `documents.upload` | Ingest a document into knowledge memory. |

## Configuration

| Setting | Env var | Default | Notes |
|---|---|---|---|
| `api_key` | `SPECTRON_API_KEY` | none | secret, required (keep it in `.env`) |
| `endpoint` | `SPECTRON_ENDPOINT` | none | required, origin with no trailing slash |
| `context` | `SPECTRON_CONTEXT` | none | required; Spectron pins a client to one context |
| `default_scope` | `SPECTRON_DEFAULT_SCOPE` | none | scope for writes and lens for reads, for example `user/tobie` |
| `top_k` | `SPECTRON_TOP_K` | `5` | memories recalled per query |
| `timeout` | `SPECTRON_TIMEOUT` | `30` | client timeout in seconds |
| `max_retries` | `SPECTRON_MAX_RETRIES` | `3` | client retry attempts |

## Reliability

The integration is built to never destabilise a crew:

- Writes run on a background daemon thread, so tasks never block on Spectron I/O.
- Every Spectron call is wrapped. Failures are logged and degrade to an empty or
  error result rather than raising into the agent or crew loop (fail open). A
  tool returns a short JSON error string instead of throwing.
- After repeated failures, or any authentication error, a circuit breaker
  disables memory for the rest of the process.

## A note on CrewAI memory backends

CrewAI's built-in `Memory` storage backend is embedding-centric: it embeds a
query locally and hands the storage layer a vector, never the query text.
Spectron is a text-native service that does its own embedding and multi-signal
ranking server-side, so it is exposed here as tools and an event-driven memory
layer rather than as a `StorageBackend`. This keeps Spectron's semantic, lexical,
graph and temporal recall intact.

## Development

```bash
pip install -e ".[dev]" crewai
pytest
```

## License

Apache-2.0
