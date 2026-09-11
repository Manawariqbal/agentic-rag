from crewai import Crew, Process

from app.agents.tasks import (
    create_research_task,
    create_answer_task,
)


class AgenticRAGCrew:

    def __init__(
        self,
        research_agent,
        answer_agent,
    ):
        self.research_agent = research_agent
        self.answer_agent = answer_agent

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

        crew = Crew(
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

        return crew