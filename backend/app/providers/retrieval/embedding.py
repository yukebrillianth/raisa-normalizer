"""BAAI/bge-m3 embedding provider via sentence-transformers.

Matches the embedding load and encode pattern from the evaluation notebook:
``_MConverter.eu_test_retrieval.md:176-180`` (model load) and ``:300-307`` (encoding).
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

from app.config import get_settings
from app.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class BGEEmbeddingProvider(EmbeddingProvider):
    """Encodes query text into dense vectors using ``BAAI/bge-m3``.

    The model is lazy-loaded on first ``process()`` call to keep startup fast.
    It is set to ``eval()`` mode after loading, matching the notebook pattern.
    """

    _instance: BGEEmbeddingProvider | None = None

    @classmethod
    def get_instance(cls) -> BGEEmbeddingProvider:
        """Return (or create) the process-wide singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name: str = settings.embedding_model_name
        self.device: str = settings.embedding_device
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        """Lazy-load the SentenceTransformer model once."""
        if self._model is None:
            logger.info(
                "Loading sentence embedding model: %s on %s ...",
                self.model_name,
                self.device,
            )
            sentence_transformers = import_module("sentence_transformers")
            SentenceTransformer = sentence_transformers.SentenceTransformer
            model = SentenceTransformer(self.model_name, device=self.device)
            model.eval()
            logger.info("Embedding model loaded successfully.")
            self._model = model
        assert self._model is not None
        return self._model

    async def process(self, text: str) -> list[float]:
        """Encode a single query string into a dense vector.

        Args:
            text: The query string to encode (e.g. a normalised question).

        Returns:
            A list of floats representing the 1024‑dimensional BGE‑M3 embedding.
        """
        model = self._ensure_model()
        vector = model.encode(text)
        return vector.tolist()

    async def health(self) -> dict[str, Any]:
        """Report embedding provider readiness."""
        return {
            "provider": "bge_m3",
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model is not None,
        }
