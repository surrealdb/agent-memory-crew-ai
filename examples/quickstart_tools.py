"""Drive the Spectron tools against a fake client. No credentials needed.

This shows exactly what happens when an agent calls the Spectron tools, using an
in-memory fake client so it runs anywhere with nothing but this package and
CrewAI installed. For the real thing against a Spectron instance and a live crew,
see ``live_crew.py``.

Run:

    python examples/quickstart_tools.py
"""

from __future__ import annotations

from spectron_crewai import SpectronConfig, get_spectron_tools


# --- a tiny fake Spectron client (mirrors the methods the tools call) --------


class _Resp:
    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class _Docs:
    def upload(self, path, *, title=None, **kwargs):
        return _Resp(document_id="doc:1", title=title or path)


class FakeSpectron:
    """In-memory stand-in that stores facts and echoes them back on recall."""

    def __init__(self):
        self._facts = []
        self.documents = _Docs()

    def remember(self, text, *, scope=None, **kwargs):
        self._facts.append(text)
        return _Resp(stored=True, scope=scope)

    def recall(self, query, *, k=None, lens=None):
        return _Resp(memories=[{"text": f} for f in self._facts][: (k or 5)])

    def query_context(self, query, *, k=None, lens=None):
        return _Resp(answer="; ".join(self._facts) or "nothing recalled yet")

    def forget(self, query, *, purge=False, **kwargs):
        self._facts.clear()
        return _Resp(forgotten=True, purge=purge)

    def reflect(self, query, *, persist=False, **kwargs):
        return _Resp(reflection=f"{len(self._facts)} facts on record", persisted=persist)


def main() -> None:
    config = SpectronConfig(
        endpoint="https://demo.spectron.local",
        context="demo",
        api_key="sk-demo",
        default_scope="user/tobie",
        top_k=5,
    )
    fake = FakeSpectron()

    # In real use, drop the client argument and let it resolve from the
    # environment: get_spectron_tools(scope="user/tobie").
    tools = get_spectron_tools(config=config, client=fake, scope="user/tobie")
    by_name = {t.name: t for t in tools}

    print("tools:", ", ".join(by_name))

    print("\nremember ->", by_name["spectron_remember"].run(text="Tobie was promoted to CTO"))
    print("remember ->", by_name["spectron_remember"].run(text="Tobie moved to Lisbon"))

    print("\nrecall ->\n" + by_name["spectron_recall"].run(query="what about tobie?"))
    print("\ncontext ->", by_name["spectron_context"].run(query="summarise what you know"))
    print("\nreflect ->", by_name["spectron_reflect"].run(query="this week", persist=True))

    print("\nTo use these on a real agent:")
    print("    from crewai import Agent")
    print("    agent = Agent(role='Analyst', tools=get_spectron_tools(scope='user/tobie'))")


if __name__ == "__main__":
    main()
