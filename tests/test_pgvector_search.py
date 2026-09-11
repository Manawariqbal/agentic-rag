from app.config import settings
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter


def main():
    embedding_provider = OllamaEmbeddingProvider(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

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

    query = "How many annual leave days do employees get?"

    print("\n" + "=" * 70)
    print("PGVECTOR SEARCH TEST")
    print("=" * 70)

    print(f"\nQuery: {query}")

    results = store.search(
        query=query,
        top_k=5,
    )

    print(f"Results: {len(results)}")

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print("\n" + "-" * 70)
        print(f"RESULT {index}")
        print("-" * 70)

        print(f"Score: {result['score']:.4f}")

        print(
            f"Source: "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section', 'Unknown')}"
        )

        print(
            f"Chunk: "
            f"{metadata.get('chunk_index', 'Unknown')}"
        )

        print("\nText:")
        print(result["text"][:1000])


if __name__ == "__main__":
    main()