from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import settings


class EmbeddingProvider(ABC):
    """
    Interface for embedding providers.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Lightweight deterministic embedding provider for tests.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        values: list[float] = []
        seed = text.encode("utf-8")

        for i in range(self.dimensions):
            digest = hashlib.sha256(
                seed + str(i).encode("utf-8")
            ).digest()

            value = int.from_bytes(
                digest[:4],
                byteorder="big",
            ) / (2**32)

            values.append(value * 2.0 - 1.0)

        norm = math.sqrt(
            sum(value * value for value in values)
        )

        if norm == 0:
            return values

        return [value / norm for value in values]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Production embedding provider using Ollama's /api/embed endpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ):
        self.model = model or settings.embedding_model

        self.base_url = (
            base_url or settings.ollama_base_url
        ).rstrip("/")

        self.dimensions = (
            dimensions or settings.embedding_dimensions
        )

    def embed(self, text: str) -> list[float]:
        """
        Generate one embedding using Ollama.
        """

        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text,
            },
            timeout=120.0,
        )

        response.raise_for_status()

        data: dict[str, Any] = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                f"Ollama returned no embeddings. "
                f"Response: {data}"
            )

        embedding = embeddings[0]

        if len(embedding) != self.dimensions:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.dimensions}, "
                f"got {len(embedding)}"
            )

        return embedding

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts
        using one Ollama request.
        """

        if not texts:
            return []

        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=120.0,
        )

        response.raise_for_status()

        data: dict[str, Any] = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                f"Ollama returned no embeddings. "
                f"Response: {data}"
            )

        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedding count mismatch: "
                f"expected {len(texts)}, "
                f"got {len(embeddings)}"
            )

        for embedding in embeddings:
            if len(embedding) != self.dimensions:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {self.dimensions}, "
                    f"got {len(embedding)}"
                )

        return embeddings