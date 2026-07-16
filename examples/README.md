# Examples

| File | What it shows | Needs |
|---|---|---|
| [`quickstart_tools.py`](quickstart_tools.py) | The Spectron tools driven directly against a **fake in-memory client**, so you can see a remember then recall round-trip. | Nothing but this package and CrewAI. |
| [`auto_memory.py`](auto_memory.py) | The automatic memory listener reacting to the crew event flow (recall before a task, write back after, consolidate at the end), against a **fake client**. | Nothing but this package and CrewAI. |
| [`live_crew.py`](live_crew.py) | A **real crew** with Spectron-backed automatic memory. | `surrealdb>=3.0.0a2` + Spectron credentials + an LLM. |
| [`.env.example`](.env.example) | Sample environment and secrets. | none |

## Run the no-credentials demos

```bash
pip install -e . crewai
python examples/quickstart_tools.py
python examples/auto_memory.py
```

Both run with a fake client, so they touch no network and need no credentials.
They print each step so you can see exactly what the tools and the listener do.

## Run against real Spectron

```bash
pip install -e . "surrealdb>=3.0.0a2"
cp examples/.env.example .env    # fill in SPECTRON_* and OPENAI_API_KEY
set -a; . ./.env; set +a
python examples/live_crew.py
```

> `live_crew.py` writes to your Spectron context. Use a throwaway context if you
> do not want the demo data to persist.

See the top-level [README](../README.md) for full configuration.
