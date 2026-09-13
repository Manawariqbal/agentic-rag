from app.rag.citations import CitationManager


def create_test_results():
    return [
        {
            "metadata": {
                "source": "leave_policy.pdf",
                "section": "Entitlement",
                "chunk_index": 0,
            }
        },
        {
            "metadata": {
                "source": "leave_policy.pdf",
                "section": "Entitlement",
                "chunk_index": 0,
            }
        },
        {
            "metadata": {
                "source": "leave_policy.pdf",
                "section": "Approval",
                "chunk_index": 0,
            }
        },
    ]


def test_build_citations():
    manager = CitationManager()

    citations = manager.build_citations(create_test_results())

    assert citations
    assert len(citations) == 2

    assert citations[0].source == "leave_policy.pdf"
    assert citations[0].section == "Entitlement"

    assert citations[1].source == "leave_policy.pdf"
    assert citations[1].section == "Approval"


def test_filter_used_citations():
    manager = CitationManager()

    citations = manager.build_citations(create_test_results())

    answer = "Employees receive 20 days of annual leave [1]."

    used = manager.filter_used_citations(
        answer=answer,
        citations=citations,
    )

    assert len(used) == 1
    assert used[0].source == "leave_policy.pdf"
    assert used[0].section == "Entitlement"


def test_validate_valid_citations():
    manager = CitationManager()

    citations = manager.build_citations(create_test_results())

    answer = "Employees receive 20 days of annual leave [1]."

    result = manager.validate_citations(
        answer=answer,
        citations=citations,
    )

    assert result is True


def test_validate_invalid_citations():
    manager = CitationManager()

    citations = manager.build_citations(create_test_results())

    answer = (
        "Employees receive 20 days [1]. "
        "The policy also says something else [99]."
    )

    result = manager.validate_citations(
        answer=answer,
        citations=citations,
    )

    assert result is False


def test_remove_invalid_citations():
    manager = CitationManager()

    citations = manager.build_citations(create_test_results())

    answer = (
        "Employees receive 20 days [1]. "
        "The policy also says something else [99]."
    )

    cleaned = manager.remove_invalid_citations(
        answer=answer,
        citations=citations,
    )

    assert "[1]" in cleaned
    assert "[99]" not in cleaned