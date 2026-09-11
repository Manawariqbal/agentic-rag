from app.config import settings
from app.ingestion.chunker import DocumentChunk
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter


def main():
    # Real Ollama embedding provider
    embedding_provider = OllamaEmbeddingProvider(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    # Production 1024-dimensional PGVector table
    store = PGVectorStoreAdapter(
        database=settings.postgres_database,
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        table_name="rag_documents_1024",
        embed_dim=settings.embedding_dimensions,
        embedding_provider=embedding_provider,
    )

    chunks = [
        DocumentChunk(
            text=(
                "Employees are entitled to 20 days of "
                "annual leave per year."
            ),
            metadata={
                "source": "leave_and_attendance_policy.pdf",
                "section": "Annual Leave",
                "chunk_index": 0,
            },
        ),
        DocumentChunk(
            text=(
                "Employees may carry forward up to 5 "
                "unused annual leave days."
            ),
            metadata={
                "source": "leave_and_attendance_policy.pdf",
                "section": "Carry Forward",
                "chunk_index": 0,
            },
        ),
    ]

    # Generate real 1024-dimensional embeddings
    embeddings = embedding_provider.embed_batch(
        [chunk.text for chunk in chunks]
    )

    print(f"Embedding dimension: {len(embeddings[0])}")

    # Store in PostgreSQL + pgvector
    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(
        "Successfully inserted chunks into "
        "PostgreSQL + pgvector."
    )


if __name__ == "__main__":
    main()