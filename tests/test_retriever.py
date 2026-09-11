from app.ingestion.chunker import ContextualChunker
from app.rag.embedding import MockEmbeddingProvider
from app.rag.retriever import Retriever
from app.rag.vector_store import InMemoryVectorStore


def main():

    content = """
# Annual Leave

Employees are entitled to 20 days of annual leave per year.

# Sick Leave

Employees are entitled to 10 days of sick leave per year.

# Casual Leave

Employees are entitled to 6 days of casual leave per year.
"""

    # 1. Chunk documents
    chunker = ContextualChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.chunk(
        content=content,
        filename="leave_and_attendance_policy.pdf",
    )

    # 2. Embedding provider
    embedding_provider = MockEmbeddingProvider(
        dimensions=384
    )

    # 3. Vector store
    vector_store = InMemoryVectorStore(
        embedding_provider=embedding_provider
    )

    # 4. Index chunks
    vector_store.add_chunks(chunks)

    # 5. Retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # 6. Query
    results = retriever.retrieve(
        "How many annual leave days are available?"
    )

    print(f"\nRetrieved {len(results)} results\n")

    for i, result in enumerate(results, start=1):

        print("=" * 70)
        print(f"RESULT {i}")
        print("=" * 70)

        print(f"Score: {result['score']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"\n{result['text']}")


if __name__ == "__main__":
    main()