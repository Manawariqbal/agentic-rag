from app.config import settings
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter
from app.rag.retriever import Retriever
from app.rag.reranker import SimpleReranker


def main():

    embedding_provider = OllamaEmbeddingProvider(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    vector_store = PGVectorStoreAdapter(
        embedding_provider=embedding_provider,
        database=settings.postgres_database,
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        table_name="rag_documents_1024",
        embed_dim=settings.embedding_dimensions,
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        top_k=10,
    )

    reranker = SimpleReranker()

    query = "How many annual leave days do employees get?"

    # Step 1: Vector retrieval
    retrieved_results = retriever.retrieve(query)

    print("\n" + "=" * 70)
    print("BEFORE RERANKING")
    print("=" * 70)

    for index, result in enumerate(retrieved_results, start=1):

        print(
            f"{index}. "
            f"{result['metadata'].get('section')} "
            f"| {result['score']:.4f}"
        )

    # Step 2: Reranking
    reranked_results = reranker.rerank(
        query=query,
        results=retrieved_results,
        top_k=3,
    )

    print("\n" + "=" * 70)
    print("AFTER RERANKING")
    print("=" * 70)

    for index, result in enumerate(
        reranked_results,
        start=1,
    ):

        print(
            f"{index}. "
            f"{result['metadata'].get('section')} "
            f"| "
            f"vector={result['score']:.4f} "
            f"| "
            f"rerank={result['rerank_score']:.4f}"
        )

        print(
            result["text"][:300]
            .replace("\n", " ")
        )


if __name__ == "__main__":
    main()