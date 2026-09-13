from app.rag.reranker import SimpleReranker


def test_tokenize():
    tokens = SimpleReranker._tokenize(
        "Employees receive 20 days of Annual Leave."
    )

    assert "employees" in tokens
    assert "receive" in tokens
    assert "20" in tokens
    assert "days" in tokens
    assert "annual" in tokens
    assert "leave" in tokens


def test_tokenize_is_case_insensitive():
    tokens = SimpleReranker._tokenize(
        "Annual Leave ANNUAL leave"
    )

    assert tokens == {"annual", "leave"}


def test_tokenize_removes_punctuation():
    tokens = SimpleReranker._tokenize(
        "Annual, Leave! (20 days)."
    )

    assert tokens == {
        "annual",
        "leave",
        "20",
        "days",
    }


def test_rerank_calculates_lexical_score():
    reranker = SimpleReranker()

    results = [
        {
            "text": "Employees receive annual leave.",
            "score": 0.5,
            "metadata": {
                "source": "leave_policy.pdf",
                "section": "Entitlement",
            },
        },
        {
            "text": "Employees can work remotely.",
            "score": 0.5,
            "metadata": {
                "source": "employee_handbook.pdf",
                "section": "Remote Work",
            },
        },
    ]

    reranked = reranker.rerank(
        query="annual leave",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2

    # First result contains both query words.
    # lexical_score = 2 / 2 = 1.0
    # final_score = 0.7 * 1.0 + 0.3 * 0.5 = 0.85
    assert reranked[0]["rerank_score"] == 0.85


def test_rerank_orders_results_by_score():
    reranker = SimpleReranker()

    results = [
        {
            "text": "Employees can work remotely.",
            "score": 0.9,
            "metadata": {
                "section": "Remote Work",
            },
        },
        {
            "text": "Employees receive annual leave.",
            "score": 0.5,
            "metadata": {
                "section": "Entitlement",
            },
        },
    ]

    reranked = reranker.rerank(
        query="annual leave",
        results=results,
        top_k=2,
    )

    assert reranked[0]["metadata"]["section"] == "Entitlement"
    assert reranked[1]["metadata"]["section"] == "Remote Work"

    assert (
        reranked[0]["rerank_score"]
        > reranked[1]["rerank_score"]
    )


def test_rerank_respects_top_k():
    reranker = SimpleReranker()

    results = [
        {
            "text": "annual leave policy",
            "score": 0.5,
        },
        {
            "text": "remote work policy",
            "score": 0.5,
        },
        {
            "text": "travel expense policy",
            "score": 0.5,
        },
    ]

    reranked = reranker.rerank(
        query="policy",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2


def test_rerank_preserves_original_result_data():
    reranker = SimpleReranker()

    original = {
        "text": "Employees receive annual leave.",
        "score": 0.8,
        "metadata": {
            "source": "leave_policy.pdf",
            "section": "Entitlement",
            "chunk_index": 1,
        },
    }

    reranked = reranker.rerank(
        query="annual leave",
        results=[original],
        top_k=1,
    )

    result = reranked[0]

    assert result["text"] == original["text"]
    assert result["score"] == original["score"]
    assert result["metadata"] == original["metadata"]

    assert "rerank_score" in result


def test_rerank_with_empty_query():
    reranker = SimpleReranker()

    results = [
        {
            "text": "annual leave policy",
            "score": 0.8,
        }
    ]

    reranked = reranker.rerank(
        query="",
        results=results,
        top_k=1,
    )

    assert len(reranked) == 1

    # No query words means lexical score = 0.
    # final score = 0.7 * 0 + 0.3 * 0.8 = 0.24
    assert reranked[0]["rerank_score"] == 0.24


def test_rerank_with_empty_results():
    reranker = SimpleReranker()

    reranked = reranker.rerank(
        query="annual leave",
        results=[],
        top_k=3,
    )

    assert reranked == []