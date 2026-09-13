import pytest

from app.agents.rag_tool import RAGTool
from app.ingestion.chunker import DocumentChunk
from app.rag.citations import CitationManager
from app.rag.embedding import MockEmbeddingProvider
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever
from app.rag.vector_store import InMemoryVectorStore


@pytest.fixture
def rag_tool():
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

    return RAGTool(
        retriever=retriever,
        reranker=reranker,
        citation_manager=citation_manager,
        rerank_top_k=3,
    )


def test_rag_tool_returns_evidence(rag_tool):
    result = rag_tool.run(
        query="How many annual leave days do employees get?"
    )

    assert isinstance(result, str)
    assert result


def test_rag_tool_returns_relevant_annual_leave_content(rag_tool):
    result = rag_tool.run(
        query="How many annual leave days do employees get?"
    )

    assert "20 days" in result
    assert "annual leave" in result
    assert "leave_and_attendance_policy.pdf" in result
    assert "Annual Leave" in result


def test_rag_tool_generates_citations(rag_tool):
    rag_tool.run(
        query="How many annual leave days do employees get?"
    )

    citations = rag_tool.get_last_citations()

    assert citations
    assert len(citations) <= 3

    assert citations[0].source == (
        "leave_and_attendance_policy.pdf"
    )


def test_rag_tool_returns_citation_markers(rag_tool):
    result = rag_tool.run(
        query="How many annual leave days do employees get?"
    )

    citations = rag_tool.get_last_citations()

    assert citations

    citation_id = citations[0].citation_id

    assert f"[{citation_id}]" in result


def test_rag_tool_returns_sources_section(rag_tool):
    result = rag_tool.run(
        query="How many annual leave days do employees get?"
    )

    assert "Sources:" in result


def test_rag_tool_can_retrieve_travel_policy(rag_tool):
    result = rag_tool.run(
        query="What is the hotel reimbursement limit?"
    )

    assert "6000" in result
    assert "travel_and_expense_policy.pdf" in result
    assert "Accommodation" in result


def test_rag_tool_respects_rerank_top_k():
    chunks = [
        DocumentChunk(
            text=f"Policy document number {i} contains policy information.",
            metadata={
                "source": f"policy_{i}.pdf",
                "section": "Policy",
                "chunk_index": i,
            },
        )
        for i in range(5)
    ]

    embedding_provider = MockEmbeddingProvider(
        dimensions=384
    )

    vector_store = InMemoryVectorStore(
        embedding_provider=embedding_provider
    )

    vector_store.add_chunks(chunks)

    retriever = Retriever(
        vector_store=vector_store,
        top_k=5,
    )

    rag_tool = RAGTool(
        retriever=retriever,
        reranker=SimpleReranker(),
        citation_manager=CitationManager(),
        rerank_top_k=2,
    )

    rag_tool.run(
        query="policy information"
    )

    citations = rag_tool.get_last_citations()

    assert len(citations) <= 2


def test_rag_tool_handles_no_results():
    class EmptyRetriever:
        def retrieve(self, query):
            return []

    rag_tool = RAGTool(
        retriever=EmptyRetriever(),
        reranker=SimpleReranker(),
        citation_manager=CitationManager(),
        rerank_top_k=3,
    )

    result = rag_tool.run(
        query="Something completely unknown"
    )

    assert result == "No relevant information was found."
    assert rag_tool.get_last_citations() == []


def test_rag_tool_rejects_empty_query(rag_tool):
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        rag_tool.run(query="")


def test_rag_tool_replaces_previous_citations(rag_tool):
    first_result = rag_tool.run(
        query="How many annual leave days do employees get?"
    )

    first_citations = rag_tool.get_last_citations()

    assert first_citations
    assert "leave_and_attendance_policy.pdf" in first_result

    second_result = rag_tool.run(
        query="What is the hotel reimbursement limit?"
    )

    second_citations = rag_tool.get_last_citations()

    assert second_citations
    assert "travel_and_expense_policy.pdf" in second_result

    assert second_citations != first_citations