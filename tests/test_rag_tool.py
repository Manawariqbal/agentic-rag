from app.agents.rag_tool import RAGTool
from app.rag.citations import CitationManager
from app.rag.embedding import MockEmbeddingProvider
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever
from app.rag.vector_store import InMemoryVectorStore
from app.ingestion.chunker import DocumentChunk


def main():

    # -------------------------
    # Create test documents
    # -------------------------

    chunks = [
        DocumentChunk(
            text=(
                "Employees are entitled to 20 days "
                "of annual leave per year."
            ),
            metadata={
                "source": "leave_and_attendance_policy.pdf",
                "section": "Annual Leave",
                "chunk_index": 0,
            },
        ),
        DocumentChunk(
            text=(
                "Employees may carry forward up to "
                "5 unused annual leave days."
            ),
            metadata={
                "source": "leave_and_attendance_policy.pdf",
                "section": "Carry Forward",
                "chunk_index": 0,
            },
        ),
        DocumentChunk(
            text=(
                "Domestic hotel expenses are reimbursed "
                "up to INR 6000 per night excluding taxes."
            ),
            metadata={
                "source": "travel_and_expense_policy.pdf",
                "section": "Accommodation",
                "chunk_index": 0,
            },
        ),
    ]

    # -------------------------
    # Build RAG components
    # -------------------------

    embedding_provider = MockEmbeddingProvider(
        dimensions=384
    )

    vector_store = InMemoryVectorStore(
        embedding_provider=embedding_provider
    )

    vector_store.add_chunks(chunks)

    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        top_k=3,
    )

    reranker = SimpleReranker()

    citation_manager = CitationManager()

    # -------------------------
    # Create RAG tool
    # -------------------------

    rag_tool = RAGTool(
        retriever=retriever,
        reranker=reranker,
        citation_manager=citation_manager,
        rerank_top_k=3,
    )

    # -------------------------
    # Execute tool
    # -------------------------

    result = rag_tool.run(
        query="How many annual leave days do employees get?"
    )

    print("\nRAG TOOL RESULT:")
    print(result)


if __name__ == "__main__":
    main()