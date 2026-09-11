from app.agents.answer_agent import AnswerAgent
from app.rag.citations import CitationManager
from app.rag.llm import MockLLMProvider


def main():

    results = [
        {
            "text": (
                "Employees are entitled to 20 days "
                "of annual leave per year."
            ),
            "score": 0.91,
            "metadata": {
                "source": "leave_and_attendance_policy.pdf",
                "section": "Annual Leave",
                "chunk_index": 0,
            },
        },
        {
            "text": (
                "Up to 5 unused annual leave days "
                "may be carried forward."
            ),
            "score": 0.87,
            "metadata": {
                "source": "leave_and_attendance_policy.pdf",
                "section": "Carry Forward",
                "chunk_index": 0,
            },
        },
    ]

    agent = AnswerAgent(
        llm=MockLLMProvider(),
        citation_manager=CitationManager(),
    )

    response = agent.answer(
        query="How many annual leave days do employees get?",
        results=results,
    )

    print("\nANSWER:")
    print(response.answer)

    print("\nCITATIONS:")
    for citation in response.citations:
        print(citation.display())


if __name__ == "__main__":
    main()