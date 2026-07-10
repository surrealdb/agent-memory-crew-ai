"""Shared fakes and fixtures for the Spectron CrewAI tests.

Nothing here touches the network or the real ``surrealdb[spectron]`` extra: a
fake client stands in for Spectron and is injected directly into the runtime.
"""

from __future__ import annotations

import pytest

from spectron_crewai.config import SpectronConfig


class FakeResp:
    """Stands in for a pydantic-style SDK response object."""

    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class FakeDocuments:
    def __init__(self, parent):
        self._parent = parent

    def upload(self, path, *, title=None, **kwargs):
        self._parent.calls.append(("upload", path, title))
        return FakeResp(document_id="doc:1", title=title or path)


class FakeSpectron:
    """Records calls and returns canned responses; can be told to fail."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.documents = FakeDocuments(self)

    def recall(self, query, *, k=None, lens=None):
        self.calls.append(("recall", query, k, lens))
        if self.fail:
            raise RuntimeError("boom in recall")
        return FakeResp(memories=[{"text": "Tobie is CTO"}, {"text": "Tobie likes dark mode"}])

    def query_context(self, query, *, k=None, lens=None):
        self.calls.append(("query_context", query, k, lens))
        if self.fail:
            raise RuntimeError("boom in query_context")
        return FakeResp(answer="Tobie is the CTO and prefers dark mode.")

    def remember(self, text, *, scope=None, **kwargs):
        self.calls.append(("remember", text, scope))
        if self.fail:
            raise RuntimeError("boom in remember")
        return FakeResp(stored=True)

    def remember_many(self, items, *, session_id=None, scope=None, **kwargs):
        self.calls.append(
            ("remember_many", tuple(m["role"] for m in items), session_id, scope)
        )
        if self.fail:
            raise RuntimeError("boom in remember_many")
        return FakeResp(count=len(items))

    def forget(self, query, *, purge=False, **kwargs):
        self.calls.append(("forget", query, purge))
        if self.fail:
            raise RuntimeError("boom in forget")
        return FakeResp(forgotten=1)

    def reflect(self, query, *, persist=False, **kwargs):
        self.calls.append(("reflect", query, persist))
        if self.fail:
            raise RuntimeError("boom in reflect")
        return FakeResp(reflection="things changed")

    def consolidate(self, **kwargs):
        self.calls.append(("consolidate",))
        if self.fail:
            raise RuntimeError("boom in consolidate")
        return FakeResp(ok=True)


@pytest.fixture
def config():
    return SpectronConfig(
        endpoint="https://example.spectron.dev",
        context="test-ctx",
        api_key="sk-test",
        default_scope="user/tobie",
        top_k=3,
    )


@pytest.fixture
def fake():
    return FakeSpectron()
