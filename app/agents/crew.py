from dataclasses import dataclass

from crewai import Crew, Process

from app.agents.tasks import (
    create_answer_task,
    create_research_task,
)
from app.observability.phoenix import get_tracer


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

    def create_crew(self, query):

        research_task = create_research_task(
            self.research_agent.agent,
            query,
        )

        answer_task = create_answer_task(
            self.answer_agent.agent,
            query,
            research_task,
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

    def run(self, query):

        tracer = get_tracer()

        with tracer.start_as_current_span(
            "agentic_rag.crew"
        ) as span:

            span.set_attribute(
                "crew.query",
                query,
            )

            span.set_attribute(
                "crew.process",
                "sequential",
            )

            span.set_attribute(
                "crew.agents",
                "research,answer",
            )

            crew = self.create_crew(query)

            result = crew.kickoff()

            answer = str(result.raw)

            span.set_attribute(
                "crew.output_length",
                len(answer),
            )

            span.set_attribute(
                "crew.result",
                "success",
            )

            return CrewRunResult(
                answer=answer,
                citations=self.rag_tool.get_last_citations(),
                raw_output=str(result),
            )