from app.rag.citations import CitationManager


def main():

    results = [
        {
            "text": "Employees are entitled to 20 days...",
            "score": 0.91,
            "metadata": {
                "source": "leave_and_attendance_policy.pdf",
                "section": "Annual Leave",
                "chunk_index": 0,
            },
        },
        {
            "text": "Employees must submit expenses...",
            "score": 0.82,
            "metadata": {
                "source": "travel_and_expense_policy.pdf",
                "section": "Expense Submission",
                "chunk_index": 1,
            },
        },
    ]

    manager = CitationManager()

    citations = manager.build_citations(
        results
    )

    print("CITATIONS")
    print("=" * 60)

    print(
        manager.format_citations(citations)
    )


if __name__ == "__main__":
    main()