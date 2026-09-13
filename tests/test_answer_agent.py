from app.agents.answer_agent import AnswerAgent, AnswerResponse
from app.rag.citations import CitationManager


class DummyLLM:
    def __init__(self, response="Dummy generated answer"):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt) -> str:
        self.last_prompt = prompt
        return self.response


def prompt_to_text(prompt) -> str:
    """
    Convert the OpenAIPrompt returned by build_rag_prompt()
    into searchable text for assertions.
    """
    messages = getattr(prompt, "messages", None)

    if messages is None:
        template = getattr(prompt, "_template", None)

        if isinstance(template, dict):
            messages = template.get("messages", [])

    if messages is None:
        return str(prompt)

    parts = []

    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
            parts.append(str(content))
        else:
            parts.append(str(message))

    return "\n".join(parts)


def create_answer_agent(response="Dummy generated answer"):
    llm = DummyLLM(response=response)
    citation_manager = CitationManager()

    agent = AnswerAgent(
        llm=llm,
        citation_manager=citation_manager,
    )

    return agent, llm


def sample_results():
    return [
        {
            "text": (
                "Eligible full-time employees receive 20 days "
                "of annual leave per calendar year."
            ),
            "score": 0.69,
            "rerank_score": 0.55,
            "metadata": {
                "source": "leave_and_attendance_policy.pdf",
                "section": "Entitlement",
            },
        },
        {
            "text": (
                "Annual leave requests require approval "
                "from the appropriate manager."
            ),
            "score": 0.58,
            "rerank_score": 0.52,
            "metadata": {
                "source": "leave_and_attendance_policy.pdf",
                "section": "Approval",
            },
        },
    ]


def test_answer_agent_returns_answer_response():
    agent, _ = create_answer_agent(
        response="Employees receive 20 days of annual leave."
    )

    response = agent.answer(
        query="How many annual leave days do employees get?",
        results=sample_results(),
    )

    assert isinstance(response, AnswerResponse)


def test_answer_agent_returns_llm_answer():
    expected_answer = (
        "Employees receive 20 days of annual leave per calendar year."
    )

    agent, _ = create_answer_agent(response=expected_answer)

    response = agent.answer(
        query="How many annual leave days do employees get?",
        results=sample_results(),
    )

    assert response.answer == expected_answer


def test_answer_agent_calls_llm():
    agent, llm = create_answer_agent(
        response="20 days of annual leave."
    )

    agent.answer(
        query="How many annual leave days do employees get?",
        results=sample_results(),
    )

    assert llm.last_prompt is not None


def test_answer_agent_prompt_contains_query():
    agent, llm = create_answer_agent()

    query = "How many annual leave days do employees get?"

    agent.answer(
        query=query,
        results=sample_results(),
    )

    prompt_text = prompt_to_text(llm.last_prompt)

    assert query in prompt_text


def test_answer_agent_prompt_contains_retrieved_context():
    agent, llm = create_answer_agent()

    agent.answer(
        query="How many annual leave days do employees get?",
        results=sample_results(),
    )

    prompt_text = prompt_to_text(llm.last_prompt)

    assert "20 days" in prompt_text
    assert "leave_and_attendance_policy.pdf" in prompt_text
    assert "Entitlement" in prompt_text
    assert "Approval" in prompt_text


def test_answer_agent_builds_citations():
    agent, _ = create_answer_agent(
        response="Employees receive 20 days of annual leave."
    )

    response = agent.answer(
        query="How many annual leave days do employees get?",
        results=sample_results(),
    )

    assert len(response.citations) == 2

    assert response.citations[0].source == (
        "leave_and_attendance_policy.pdf"
    )

    assert response.citations[0].section == "Entitlement"


def test_answer_agent_handles_empty_results():
    agent, llm = create_answer_agent(
        response="I don't have enough information to answer this question."
    )

    response = agent.answer(
        query="What is the employee stock option vesting policy?",
        results=[],
    )

    assert response.answer == (
        "I don't have enough information to answer this question."
    )

    assert response.citations == []

    prompt_text = prompt_to_text(llm.last_prompt)

    assert "No relevant documents were retrieved." in prompt_text


def test_answer_agent_builds_context_with_multiple_results():
    agent, llm = create_answer_agent()

    results = [
        {
            "text": "Employees receive 20 annual leave days.",
            "metadata": {
                "source": "leave_and_attendance_policy.pdf",
                "section": "Entitlement",
            },
        },
        {
            "text": "Employees may carry forward up to 5 unused days.",
            "metadata": {
                "source": "leave_and_attendance_policy.pdf",
                "section": "Carry Forward",
            },
        },
    ]

    agent.answer(
        query="What are the leave benefits?",
        results=results,
    )

    prompt_text = prompt_to_text(llm.last_prompt)

    assert "20 annual leave days" in prompt_text
    assert "5 unused days" in prompt_text
    assert "Entitlement" in prompt_text
    assert "Carry Forward" in prompt_text


def test_answer_agent_handles_missing_metadata():
    agent, llm = create_answer_agent()

    results = [
        {
            "text": "Some retrieved information.",
            "metadata": {},
        }
    ]

    agent.answer(
        query="What information is available?",
        results=results,
    )

    prompt_text = prompt_to_text(llm.last_prompt)

    assert "Some retrieved information." in prompt_text
    assert "Unknown source" in prompt_text
    assert "Unknown section" in prompt_text