import pytest

from app.ingestion.chunker import ContextualChunker, DocumentChunk
from app.rag.embedding import MockEmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore


def create_vector_store():
    embedding_provider = MockEmbeddingProvider(
        dimensions=384
    )

    vector_store = InMemoryVectorStore(
        embedding_provider=embedding_provider
    )

    return vector_store


def create_chunks():
    return [
        DocumentChunk(
            text=(
                "Employees are entitled to 20 days "
                "of annual leave per year."
            ),
            metadata={
                "source": "leave_policy.pdf",
                "section": "Annual Leave",
            },
        ),
        DocumentChunk(
            text=(
                "Employees are entitled to 10 days "
                "of sick leave per year."
            ),
            metadata={
                "source": "leave_policy.pdf",
                "section": "Sick Leave",
            },
        ),
    ]


def test_vector_store_starts_empty():
    vector_store = create_vector_store()

    assert vector_store.documents == []


def test_add_chunks_stores_documents():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    assert len(vector_store.documents) == 2


def test_add_chunks_stores_text():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    assert (
        vector_store.documents[0]["text"]
        == chunks[0].text
    )


def test_add_chunks_stores_metadata():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    assert (
        vector_store.documents[0]["metadata"]
        == chunks[0].metadata
    )


def test_add_chunks_generates_embeddings():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    for document in vector_store.documents:
        assert "embedding" in document
        assert len(document["embedding"]) == 384


def test_add_chunks_accepts_explicit_embeddings():
    vector_store = create_vector_store()
    chunks = create_chunks()

    embeddings = [
        [0.1] * 384,
        [0.2] * 384,
    ]

    vector_store.add_chunks(
        chunks,
        embeddings=embeddings,
    )

    assert (
        vector_store.documents[0]["embedding"]
        == embeddings[0]
    )

    assert (
        vector_store.documents[1]["embedding"]
        == embeddings[1]
    )


def test_add_chunks_rejects_mismatched_embeddings():
    vector_store = create_vector_store()
    chunks = create_chunks()

    embeddings = [
        [0.1] * 384,
    ]

    with pytest.raises(
        ValueError,
        match="Number of chunks and embeddings must match",
    ):
        vector_store.add_chunks(
            chunks,
            embeddings=embeddings,
        )


def test_search_returns_results():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="How many annual leave days do employees get?",
        top_k=2,
    )

    assert len(results) == 2


def test_search_returns_expected_structure():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="annual leave",
        top_k=2,
    )

    result = results[0]

    assert "text" in result
    assert "metadata" in result
    assert "score" in result


def test_search_returns_scores():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="annual leave",
        top_k=2,
    )

    for result in results:
        assert isinstance(result["score"], float)


def test_search_results_are_sorted_by_score():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="annual leave",
        top_k=2,
    )

    scores = [result["score"] for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_search_respects_top_k():
    vector_store = create_vector_store()

    chunks = [
        DocumentChunk(
            text=f"Document number {index} about annual leave.",
            metadata={
                "source": "leave_policy.pdf",
                "section": f"Section {index}",
            },
        )
        for index in range(5)
    ]

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="annual leave",
        top_k=3,
    )

    assert len(results) == 3


def test_search_with_top_k_one_returns_one_result():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="annual leave",
        top_k=1,
    )

    assert len(results) == 1


def test_search_rejects_empty_query():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        vector_store.search(
            query="",
            top_k=2,
        )


def test_search_rejects_whitespace_query():
    vector_store = create_vector_store()
    chunks = create_chunks()

    vector_store.add_chunks(chunks)

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        vector_store.search(
            query="   ",
            top_k=2,
        )


def test_search_empty_store_returns_empty_list():
    vector_store = create_vector_store()

    results = vector_store.search(
        query="annual leave",
        top_k=3,
    )

    assert results == []


def test_cosine_similarity_identical_vectors():
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [1.0, 2.0, 3.0]

    score = InMemoryVectorStore._cosine_similarity(
        vector_a,
        vector_b,
    )

    assert score == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]

    score = InMemoryVectorStore._cosine_similarity(
        vector_a,
        vector_b,
    )

    assert score == pytest.approx(0.0)


def test_cosine_similarity_handles_zero_vector():
    vector_a = [0.0, 0.0]
    vector_b = [1.0, 2.0]

    score = InMemoryVectorStore._cosine_similarity(
        vector_a,
        vector_b,
    )

    assert score == 0.0


def test_vector_store_works_with_contextual_chunker():
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

    vector_store = create_vector_store()

    vector_store.add_chunks(chunks)

    results = vector_store.search(
        query="How many annual leave days do employees get?",
        top_k=2,
    )

    assert len(results) == 2
    assert all("text" in result for result in results)
    assert all("metadata" in result for result in results)