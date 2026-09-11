from dataclasses import dataclass

from crewai import Crew, Process

from app.agents.tasks import (
    create_research_task,
    create_answer_task,
)


@dataclass
class CrewRunResult:
    answer: str
    citations: list
    raw_output: str


class AgenticRAGCrew:

    def __init__(
        self,
        research_agent,
        answer_agent,
        rag_tool,
    ):
        self.research_agent = research_agent
        self.answer_agent = answer_agent
        self.rag_tool = rag_tool

    def create_crew(self, query: str):

        research_task = create_research_task(
            research_agent=self.research_agent.agent,
            query=query,
        )

        answer_task = create_answer_task(
            answer_agent=self.answer_agent.agent,
            query=query,
            research_task=research_task,
        )

        return Crew(
            agents=[
                self.research_agent.agent,
                self.answer_agent.agent,
            ],
            tasks=[
                research_task,
                answer_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

    def run(self, query: str) -> CrewRunResult:

        crew = self.create_crew(query)

        result = crew.kickoff()

        answer = str(result.raw)

        citations = self.rag_tool.get_last_citations()

        return CrewRunResult(
            answer=answer,
            citations=citations,
            raw_output=str(result),
        )