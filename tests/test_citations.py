from app.rag.citations import CitationManager


def main():

    manager = CitationManager()

    results = [
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

    citations = manager.build_citations(results)

    print("\nCITATIONS")
    print("=" * 60)

    for citation in citations:
        print(citation.display())

    answer = (
        "Employees receive 20 days of annual leave [1]."
    )

    print("\nUSED CITATIONS")
    print("=" * 60)

    used = manager.filter_used_citations(
        answer=answer,
        citations=citations,
    )

    for citation in used:
        print(citation.display())

    print("\nVALIDATION")
    print("=" * 60)

    print(
        manager.validate_citations(
            answer=answer,
            citations=citations,
        )
    )

    invalid_answer = (
        "Employees receive 20 days [1]. "
        "The policy also says something else [99]."
    )

    print("\nINVALID CITATION CHECK")
    print("=" * 60)

    print(
        manager.validate_citations(
            answer=invalid_answer,
            citations=citations,
        )
    )

    cleaned = manager.remove_invalid_citations(
        answer=invalid_answer,
        citations=citations,
    )

    print("\nCLEANED ANSWER")
    print("=" * 60)
    print(cleaned)


if __name__ == "__main__":
    main()