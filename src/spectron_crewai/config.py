"""Configuration for the Spectron CrewAI integration.

Every field can be passed explicitly to :class:`SpectronConfig` or left to
resolve from the environment. Explicit values win over the environment. The API
key is a secret and is expected to come from the environment (a ``.env`` file in
development), not from source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Environment variables read when a field is not set explicitly.
ENV_ENDPOINT = "SPECTRON_ENDPOINT"
ENV_CONTEXT = "SPECTRON_CONTEXT"
ENV_API_KEY = "SPECTRON_API_KEY"
ENV_DEFAULT_SCOPE = "SPECTRON_DEFAULT_SCOPE"
ENV_TOP_K = "SPECTRON_TOP_K"
ENV_TIMEOUT = "SPECTRON_TIMEOUT"
ENV_MAX_RETRIES = "SPECTRON_MAX_RETRIES"


@dataclass
class SpectronConfig:
    """Resolved settings for a Spectron client.

    Fields left as ``None`` fall back to their environment variable when
    :meth:`from_env` is used. ``endpoint``, ``context`` and ``api_key`` are the
    minimum needed to talk to Spectron.
    """

    endpoint: Optional[str] = None
    context: Optional[str] = None
    api_key: Optional[str] = None
    default_scope: Optional[str] = None
    top_k: int = 5
    timeout: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(
        cls,
        *,
        endpoint: Optional[str] = None,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        default_scope: Optional[str] = None,
        top_k: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> "SpectronConfig":
        """Build a config from explicit arguments, filling gaps from the environment."""
        return cls(
            endpoint=endpoint or os.environ.get(ENV_ENDPOINT) or None,
            context=context or os.environ.get(ENV_CONTEXT) or None,
            api_key=api_key or os.environ.get(ENV_API_KEY) or None,
            default_scope=default_scope or os.environ.get(ENV_DEFAULT_SCOPE) or None,
            top_k=_int(top_k if top_k is not None else os.environ.get(ENV_TOP_K), 5),
            timeout=_float(
                timeout if timeout is not None else os.environ.get(ENV_TIMEOUT), 30.0
            ),
            max_retries=_int(
                max_retries
                if max_retries is not None
                else os.environ.get(ENV_MAX_RETRIES),
                3,
            ),
        )

    def is_configured(self) -> bool:
        """True when the minimum needed to talk to Spectron is present."""
        return bool(self.endpoint and self.context and self.api_key)


def _int(value: object, fallback: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _float(value: object, fallback: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
