from app.config import settings
from app.agents.answer_agent import AnswerAgent
from app.rag.citations import CitationManager
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.llm import OllamaLLMProvider
from app.rag.pgvector_store import PGVectorStoreAdapter
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever


def main():

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    embedding_provider = OllamaEmbeddingProvider(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    # --------------------------------------------------
    # PGVector
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Retriever
    # --------------------------------------------------

    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        top_k=10,
    )

    # --------------------------------------------------
    # Reranker
    # --------------------------------------------------

    reranker = SimpleReranker()

    # --------------------------------------------------
    # Citation Manager
    # --------------------------------------------------

    citation_manager = CitationManager()

    # --------------------------------------------------
    # Real Ollama LLM
    # --------------------------------------------------

    llm = OllamaLLMProvider(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
    )

    answer_agent = AnswerAgent(
        llm=llm,
        citation_manager=citation_manager,
    )

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = (
        "How many annual leave days do employees get?"
    )

    print("\n" + "=" * 70)
    print("ANSWER AGENT TEST")
    print("=" * 70)

    print(f"\nQuery: {query}")

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retrieved_results = retriever.retrieve(
        query
    )

    print(
        f"\nRetrieved: "
        f"{len(retrieved_results)}"
    )

    # --------------------------------------------------
    # Reranking
    # --------------------------------------------------

    reranked_results = reranker.rerank(
        query=query,
        results=retrieved_results,
        top_k=3,
    )

    print(
        f"After reranking: "
        f"{len(reranked_results)}"
    )

    # --------------------------------------------------
    # Answer
    # --------------------------------------------------

    response = answer_agent.answer(
        query=query,
        results=reranked_results,
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(response.answer)

    print("\n" + "=" * 70)
    print("CITATIONS")
    print("=" * 70)

    for citation in response.citations:
        print(citation.display())


if __name__ == "__main__":
    main()