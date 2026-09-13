from app.rag.embedding import MockEmbeddingProvider


def test_mock_embedding_returns_vector():
    provider = MockEmbeddingProvider(
        dimensions=384
    )

    text = """
    Employees are entitled to 20 days
    of annual leave per year.
    """

    embedding = provider.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_mock_embedding_has_numeric_values():
    provider = MockEmbeddingProvider(
        dimensions=384
    )

    embedding = provider.embed(
        "Employees receive annual leave."
    )

    assert all(
        isinstance(value, (int, float))
        for value in embedding
    )


def test_mock_embedding_respects_dimensions():
    for dimensions in [10, 128, 384, 1024]:

        provider = MockEmbeddingProvider(
            dimensions=dimensions
        )

        embedding = provider.embed(
            "Test embedding"
        )

        assert len(embedding) == dimensions


def test_mock_embedding_is_deterministic():
    provider = MockEmbeddingProvider(
        dimensions=384
    )

    text = "Employees receive 20 days of annual leave."

    embedding_1 = provider.embed(text)
    embedding_2 = provider.embed(text)

    assert embedding_1 == embedding_2


def test_mock_embedding_different_text_produces_embedding():
    provider = MockEmbeddingProvider(
        dimensions=384
    )

    embedding_1 = provider.embed(
        "Employees receive annual leave."
    )

    embedding_2 = provider.embed(
        "Employees can work remotely."
    )

    assert embedding_1 != embedding_2


def test_mock_embedding_empty_text():
    provider = MockEmbeddingProvider(
        dimensions=384
    )

    embedding = provider.embed("")

    assert isinstance(embedding, list)
    assert len(embedding) == 384