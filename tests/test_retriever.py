import pytest

from app.rag.retriever import Retriever


class FakeVectorStore:
    def __init__(self, results=None):
        self.results = results or []
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k):
        self.last_query = query
        self.last_top_k = top_k
        return self.results


def test_retriever_returns_results():
    vector_store = FakeVectorStore(
        results=[
            {
                "text": "Employees receive 20 days of annual leave.",
                "score": 0.85,
                "metadata": {
                    "source": "leave_and_attendance_policy.pdf",
                    "section": "Entitlement",
                    "chunk_index": 0,
                },
            }
        ]
    )

    retriever = Retriever(
        vector_store=vector_store,
        top_k=10,
    )

    results = retriever.retrieve(
        "How many annual leave days do employees get?"
    )

    assert len(results) == 1

    assert results[0]["text"] == (
        "Employees receive 20 days of annual leave."
    )

    assert results[0]["score"] == 0.85

    assert results[0]["metadata"]["source"] == (
        "leave_and_attendance_policy.pdf"
    )

    assert results[0]["metadata"]["section"] == "Entitlement"


def test_retriever_passes_query_to_vector_store():
    vector_store = FakeVectorStore()

    retriever = Retriever(
        vector_store=vector_store,
        top_k=10,
    )

    query = "How many annual leave days do employees get?"

    retriever.retrieve(query)

    assert vector_store.last_query == query


def test_retriever_passes_top_k_to_vector_store():
    vector_store = FakeVectorStore()

    retriever = Retriever(
        vector_store=vector_store,
        top_k=5,
    )

    retriever.retrieve("annual leave")

    assert vector_store.last_top_k == 5


def test_retriever_uses_default_top_k_from_settings():
    vector_store = FakeVectorStore()

    retriever = Retriever(
        vector_store=vector_store,
    )

    retriever.retrieve("annual leave")

    assert vector_store.last_top_k == retriever.top_k


def test_retriever_respects_custom_top_k():
    vector_store = FakeVectorStore()

    retriever = Retriever(
        vector_store=vector_store,
        top_k=3,
    )

    retriever.retrieve("annual leave")

    assert retriever.top_k == 3
    assert vector_store.last_top_k == 3


def test_retriever_rejects_empty_query():
    vector_store = FakeVectorStore()

    retriever = Retriever(
        vector_store=vector_store,
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("")


def test_retriever_rejects_whitespace_query():
    vector_store = FakeVectorStore()

    retriever = Retriever(
        vector_store=vector_store,
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("   ")


def test_retriever_preserves_result_structure():
    vector_store = FakeVectorStore(
        results=[
            {
                "text": "Remote work is allowed.",
                "score": 0.72,
                "metadata": {
                    "source": "employee_handbook.pdf",
                    "section": "Remote Work",
                    "chunk_index": 4,
                },
                "extra_field": "ignored",
            }
        ]
    )

    retriever = Retriever(
        vector_store=vector_store,
        top_k=1,
    )

    results = retriever.retrieve("remote work")

    assert results == [
        {
            "text": "Remote work is allowed.",
            "score": 0.72,
            "metadata": {
                "source": "employee_handbook.pdf",
                "section": "Remote Work",
                "chunk_index": 4,
            },
        }
    ]


def test_retriever_returns_empty_list_when_no_results():
    vector_store = FakeVectorStore(results=[])

    retriever = Retriever(
        vector_store=vector_store,
        top_k=10,
    )

    results = retriever.retrieve("something unknown")

    assert results == []