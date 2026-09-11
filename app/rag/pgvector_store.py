from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore


class PGVectorStoreAdapter:
    """
    Adapter around LlamaIndex PGVectorStore.

    Stores:
    - chunk text
    - chunk metadata
    - embeddings
    """

    def __init__(
        self,
        database: str,
        host: str,
        port: int,
        user: str,
        password: str,
        table_name: str = "rag_documents",
        embed_dim: int = 384,
    ):
        self.vector_store = PGVectorStore.from_params(
            database=database,
            host=host,
            port=port,
            user=user,
            password=password,
            table_name=table_name,
            embed_dim=embed_dim,
        )

    def add_chunks(
        self,
        chunks,
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