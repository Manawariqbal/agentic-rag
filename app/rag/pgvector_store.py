from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore

from app.config import settings
from app.ingestion.chunker import DocumentChunk
from app.rag.embedding import EmbeddingProvider


class PGVectorStoreAdapter:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        database: str | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        table_name: str = "rag_documents_1024",
        embed_dim: int | None = None,
    ):
        self.embedding_provider = embedding_provider

        self.vector_store = PGVectorStore.from_params(
            database=database or settings.postgres_database,
            host=host or settings.postgres_host,
            password=password or settings.postgres_password,
            port=port or settings.postgres_port,
            user=user or settings.postgres_user,
            table_name=table_name,
            embed_dim=embed_dim or settings.embedding_dimensions,
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        nodes = []

        for chunk, embedding in zip(chunks, embeddings):
            node = TextNode(
                text=chunk.text,
                metadata=chunk.metadata,
            )

            node.embedding = embedding
            nodes.append(node)

        self.vector_store.add(nodes)

    def create_index(self) -> VectorStoreIndex:
        return VectorStoreIndex.from_vector_store(
            self.vector_store
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Generate query embedding using the same
        # Ollama embedding model used during ingestion.
        query_embedding = self.embedding_provider.embed(query)

        query_obj = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
        )

        result = self.vector_store.query(query_obj)

        results = []

        for index, node_id in enumerate(result.ids):

            metadata = {}

            if result.nodes:
                node = result.nodes[index]
                text = node.get_content()
                metadata = node.metadata or {}
            else:
                text = ""

            score = 0.0

            if result.similarities:
                score = result.similarities[index]

            results.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "score": score,
                    "node_id": node_id,
                }
            )

        return results