from crewai import Process
from crewai.tools import BaseTool

from app.agents.crew import AgenticRAGCrew, CrewRunResult
from app.agents.research_agent import ResearchAgent
from app.agents.crew_answer_agent import CrewAnswerAgent


class DummyRAGTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = (
        "Search the enterprise knowledge base and return relevant evidence."
    )

    def _run(self, query: str) -> str:
        return "Dummy evidence"


def create_crew_wrapper():
    rag_tool = DummyRAGTool()

    research_agent = ResearchAgent(
        rag_tool=rag_tool
    )

    answer_agent = CrewAnswerAgent()

    crew_wrapper = AgenticRAGCrew(
        research_agent=research_agent,
        answer_agent=answer_agent,
        rag_tool=rag_tool,
    )

    return crew_wrapper


def test_crew_result_dataclass():
    result = CrewRunResult(
        answer="Employees receive 20 days of annual leave. [1]",
        citations=[],
        raw_output="raw output",
    )

    assert result.answer == (
        "Employees receive 20 days of annual leave. [1]"
    )
    assert result.citations == []
    assert result.raw_output == "raw output"


def test_crew_result_has_expected_fields():
    result = CrewRunResult(
        answer="test answer",
        citations=["citation"],
        raw_output="raw",
    )

    assert hasattr(result, "answer")
    assert hasattr(result, "citations")
    assert hasattr(result, "raw_output")


def test_create_crew_requires_grounded_evidence():
    crew_wrapper = create_crew_wrapper()

    evidence = """
[1]
Source: leave_and_attendance_policy.pdf
Section: Entitlement

Eligible full-time employees receive 20 days of annual leave
per calendar year.
"""

    crew = crew_wrapper.create_crew(
        query="How many annual leave days do employees get?",
        retrieved_evidence=evidence,
    )

    assert crew is not None
    assert len(crew.agents) == 2
    assert len(crew.tasks) == 2


def test_create_crew_uses_sequential_process():
    crew_wrapper = create_crew_wrapper()

    crew = crew_wrapper.create_crew(
        query="How many annual leave days do employees get?",
        retrieved_evidence="20 days of annual leave.",
    )

    assert crew.process == Process.sequential


def test_create_crew_contains_research_and_answer_agents():
    crew_wrapper = create_crew_wrapper()

    crew = crew_wrapper.create_crew(
        query="How many annual leave days do employees get?",
        retrieved_evidence="20 days of annual leave.",
    )

    assert crew_wrapper.research_agent.agent in crew.agents
    assert crew_wrapper.answer_agent.agent in crew.agents


def test_create_crew_contains_two_tasks():
    crew_wrapper = create_crew_wrapper()

    crew = crew_wrapper.create_crew(
        query="How many annual leave days do employees get?",
        retrieved_evidence="20 days of annual leave.",
    )

    assert len(crew.tasks) == 2


def test_create_crew_accepts_empty_evidence():
    crew_wrapper = create_crew_wrapper()

    crew = crew_wrapper.create_crew(
        query="What is the company?",
        retrieved_evidence="",
    )

    assert crew is not None
    assert len(crew.tasks) == 2