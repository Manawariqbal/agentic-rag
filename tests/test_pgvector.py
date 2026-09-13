import pytest

from app.config import settings
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter


@pytest.fixture
def embedding_provider():
    return OllamaEmbeddingProvider(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


@pytest.fixture
def vector_store(embedding_provider):
    return PGVectorStoreAdapter(
        database=settings.postgres_database,
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        table_name="rag_documents_1024",
        embed_dim=settings.embedding_dimensions,
        embedding_provider=embedding_provider,
    )


def test_pgvector_search_returns_results(vector_store):
    results = vector_store.search(
        query="How many annual leave days do employees get?",
        top_k=3,
    )

    assert results
    assert len(results) <= 3


def test_pgvector_search_result_structure(vector_store):
    results = vector_store.search(
        query="How many annual leave days do employees get?",
        top_k=3,
    )

    assert results

    result = results[0]

    assert "text" in result
    assert "metadata" in result
    assert "score" in result
    assert "node_id" in result

    assert isinstance(result["text"], str)
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["score"], (int, float))
    assert isinstance(result["node_id"], str)


def test_pgvector_finds_annual_leave_policy(vector_store):
    results = vector_store.search(
        query="How many annual leave days do employees get?",
        top_k=5,
    )

    assert results

    sources = [
        result["metadata"].get("source")
        for result in results
    ]

    assert "leave_and_attendance_policy.pdf" in sources


def test_pgvector_finds_travel_policy(vector_store):
    results = vector_store.search(
        query="What is the domestic hotel reimbursement limit?",
        top_k=5,
    )

    assert results

    sources = [
        result["metadata"].get("source")
        for result in results
    ]

    assert "travel_and_expense_policy.pdf" in sources


def test_pgvector_returns_similarity_scores(vector_store):
    results = vector_store.search(
        query="annual leave",
        top_k=3,
    )

    assert results

    for result in results:
        assert isinstance(result["score"], (int, float))
        assert result["score"] >= 0


def test_pgvector_respects_top_k(vector_store):
    results = vector_store.search(
        query="employee policy",
        top_k=2,
    )

    assert len(results) <= 2


def test_pgvector_rejects_empty_query(vector_store):
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        vector_store.search(
            query="",
            top_k=3,
        )


def test_pgvector_rejects_whitespace_query(vector_store):
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        vector_store.search(
            query="   ",
            top_k=3,
        )