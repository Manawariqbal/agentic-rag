from app.ingestion.chunker import ContextualChunker
from app.rag.embedding import MockEmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore


def main():

    content = """
# Annual Leave

Employees are entitled to 20 days of annual leave per year.

# Sick Leave

Employees are entitled to 10 days of sick leave per year.
"""

    chunker = ContextualChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.chunk(
        content=content,
        filename="leave_policy.pdf",
    )

    embedding_provider = MockEmbeddingProvider(
        dimensions=384
    )

    vector_store = InMemoryVectorStore(
        embedding_provider
    )

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="How many annual leave days do employees get?",
        top_k=2,
    )

    print(f"Results: {len(results)}")

    for result in results:

        print("\n" + "=" * 60)

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Metadata: {result['metadata']}"
        )

        print(
            f"Text:\n{result['text']}"
        )


if __name__ == "__main__":
    main()