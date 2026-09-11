from abc import ABC, abstractmethod
from typing import Any

from app.ingestion.chunker import DocumentChunk


class VectorStore(ABC):
    """
    Abstract interface for vector storage.
    """

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """
        Store document chunks and their embeddings.

        embeddings can be provided by an external
        EmbeddingProvider. If omitted, the concrete
        implementation may generate them itself.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the vector store and return the most
        relevant documents.
        """
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """
    Temporary vector store for local development.

    Production:
        LlamaIndex + PostgreSQL + PGVector

    Local:
        InMemoryVectorStore
    """

    def __init__(self, embedding_provider):
        self.embedding_provider = embedding_provider
        self.documents: list[dict[str, Any]] = []

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:

        # -----------------------------------------
        # Generate embeddings if they were not
        # supplied by the caller.
        # -----------------------------------------

        if embeddings is None:
            embeddings = self.embedding_provider.embed_batch(
                [chunk.text for chunk in chunks]
            )

        # -----------------------------------------
        # Validate input
        # -----------------------------------------

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        # -----------------------------------------
        # Store chunks
        # -----------------------------------------

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            self.documents.append(
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # -----------------------------------------
        # Embed query
        # -----------------------------------------

        query_embedding = (
            self.embedding_provider.embed(query)
        )

        results = []

        # -----------------------------------------
        # Calculate similarity
        # -----------------------------------------

        for document in self.documents:

            score = self._cosine_similarity(
                query_embedding,
                document["embedding"],
            )

            results.append(
                {
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "score": score,
                }
            )

        # -----------------------------------------
        # Sort by similarity
        # -----------------------------------------

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        dot_product = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b,
            )
        )

        magnitude_a = sum(
            a * a
            for a in vector_a
        ) ** 0.5

        magnitude_b = sum(
            b * b
            for b in vector_b
        ) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )