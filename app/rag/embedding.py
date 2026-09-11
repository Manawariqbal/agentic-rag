from abc import ABC, abstractmethod
import hashlib
import math


class EmbeddingProvider(ABC):
    """
    Abstract interface for generating embeddings.

    The rest of the RAG system will depend on this interface,
    not directly on Ollama.
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
    Lightweight deterministic embedding provider.

    Used only for local development/testing.

    Later this will be replaced by OllamaEmbeddingProvider.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        vector = []

        for index in range(self.dimensions):

            digest = hashlib.sha256(
                f"{text}:{index}".encode("utf-8")
            ).hexdigest()

            value = int(digest[:8], 16) / 0xFFFFFFFF

            vector.append(value)

        return self._normalize(vector)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:

        magnitude = math.sqrt(
            sum(value * value for value in vector)
        )

        if magnitude == 0:
            return vector

        return [
            value / magnitude
            for value in vector
        ]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama-based embedding provider.

    This will be used on the cloud VM.

    Expected Ollama API:
        POST /api/embed
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> list[float]:

        import httpx

        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data["embeddings"][0]

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        import httpx

        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=120,
        )

        response.raise_for_status()

        return response.json()["embeddings"]