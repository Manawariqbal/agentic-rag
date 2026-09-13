from crewai.tools import BaseTool

from app.agents.research_agent import ResearchAgent


class DummyRAGTool(BaseTool):
    name: str = "knowledge_base_search"

    description: str = (
        "Search the enterprise knowledge base and "
        "return relevant evidence."
    )

    def _run(self, query: str) -> str:
        return "Dummy evidence"


def create_research_agent():
    rag_tool = DummyRAGTool()

    return ResearchAgent(
        rag_tool=rag_tool,
    )


def test_research_agent_is_created():
    research_agent = create_research_agent()

    assert research_agent.agent is not None


def test_research_agent_has_correct_role():
    research_agent = create_research_agent()

    assert research_agent.agent.role == (
        "Enterprise Knowledge Researcher"
    )


def test_research_agent_has_correct_goal():
    research_agent = create_research_agent()

    assert (
        research_agent.agent.goal
        == (
            "Retrieve accurate information from the enterprise "
            "knowledge base and provide evidence for the answer agent."
        )
    )


def test_research_agent_has_knowledge_base_tool():
    research_agent = create_research_agent()

    tools = research_agent.agent.tools

    assert len(tools) == 1
    assert tools[0].name == "knowledge_base_search"


def test_research_agent_disables_delegation():
    research_agent = create_research_agent()

    assert research_agent.agent.allow_delegation is False


def test_research_agent_has_single_iteration():
    research_agent = create_research_agent()

    assert research_agent.agent.max_iter == 1


def test_research_agent_has_grounding_instruction():
    research_agent = create_research_agent()

    backstory = research_agent.agent.backstory

    assert "knowledge_base_search" in backstory
    assert "evidence" in backstory
    assert "Do not invent information." in backstory