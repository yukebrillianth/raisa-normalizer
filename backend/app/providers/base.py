"""Abstract base classes for all pipeline providers.

Each provider exposes:
- ``process(...)`` — executes the provider's core logic (async).
- ``health()`` — returns a readiness status dict (async).
"""

from __future__ import annotations

import abc
from typing import Any


class STTProvider(abc.ABC):
    """Speech-to-text — audio bytes → transcript string."""

    @abc.abstractmethod
    async def process(self, audio: bytes) -> dict[str, Any]:  # noqa: D102
        ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:  # noqa: D102
        ...


class NormalizerProvider(abc.ABC):
    """Query normalisation — raw transcript → cleaned query."""

    @abc.abstractmethod
    async def process(self, transcript: str) -> str:  # noqa: D102
        ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:  # noqa: D102
        ...


class EmbeddingProvider(abc.ABC):
    """Text embedding — query string → dense vector."""

    @abc.abstractmethod
    async def process(self, text: str) -> list[float]:  # noqa: D102
        ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:  # noqa: D102
        ...


class RetrievalProvider(abc.ABC):
    """Vector retrieval — embedding vector → ranked candidates."""

    @abc.abstractmethod
    async def process(self, embedding: list[float]) -> list[dict[str, Any]]:  # noqa: D102
        ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:  # noqa: D102
        ...


class SelectionVerbalizerProvider(abc.ABC):
    """LLM selection + verbalisation — candidates → selected answer."""

    @abc.abstractmethod
    async def process(  # noqa: D102
        self, query: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:  # noqa: D102
        ...


class TTSProvider(abc.ABC):
    """Text-to-speech — text → audio file path or bytes."""

    @abc.abstractmethod
    async def process(self, text: str) -> str:  # noqa: D102
        ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:  # noqa: D102
        ...
