from app.ingestion.chunker import ContextualChunker
from app.rag.embedding import MockEmbeddingProvider
from app.rag.reranker import SimpleReranker
from app.rag.vector_store import InMemoryVectorStore
from app.rag.retriever import Retriever


def main():

    content = """
# Annual Leave

Employees are entitled to 20 days of annual leave per year.

# Sick Leave

Employees are entitled to 10 days of sick leave per year.

# Casual Leave

Employees are entitled to 6 days of casual leave per year.
"""

    # Chunk
    chunker = ContextualChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.chunk(
        content=content,
        filename="leave_policy.pdf",
    )

    # Embeddings
    embedding_provider = MockEmbeddingProvider(
        dimensions=384
    )

    # Vector store
    vector_store = InMemoryVectorStore(
        embedding_provider
    )

    vector_store.add_chunks(chunks)

    # Retrieve top 3
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        top_k=3,
    )

    query = "How many annual leave days are available?"

    retrieved_results = retriever.retrieve(query)

    print("\nBEFORE RERANKING")
    print("=" * 70)

    for result in retrieved_results:

        print(
            f"Score: {result['score']:.4f} | "
            f"{result['metadata']}"
        )

    # Rerank
    reranker = SimpleReranker()

    reranked_results = reranker.rerank(
        query=query,
        results=retrieved_results,
        top_k=2,
    )

    print("\nAFTER RERANKING")
    print("=" * 70)

    for result in reranked_results:

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Metadata: "
            f"{result['metadata']}"
        )

        print(
            f"Text:\n{result['text']}\n"
        )


if __name__ == "__main__":
    main()