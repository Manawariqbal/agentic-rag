from typing import Any

from app.config import settings
from app.rag.embedding import EmbeddingProvider
from app.rag.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        top_k: int | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

        self.top_k = (
            top_k
            if top_k is not None
            else settings.retrieval_top_k
        )

    def retrieve(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        results = self.vector_store.search(
            query=query,
            top_k=self.top_k,
        )

        return [
            {
                "text": result["text"],
                "score": result["score"],
                "metadata": result["metadata"],
            }
            for result in results
        ]