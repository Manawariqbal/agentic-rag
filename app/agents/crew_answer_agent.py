from crewai import Agent, LLM

from app.config import settings


class CrewAnswerAgent:

    def __init__(self):

        llm = LLM(
            model=f"ollama/{settings.llm_model}",
            base_url=settings.ollama_base_url,
        )

        self.agent = Agent(

            role="Enterprise Answer Specialist",

            goal=(
                "Generate accurate and concise answers using only "
                "the research provided by the research agent."
            ),

            backstory=(
                "You are an enterprise knowledge assistant. "
                "You transform research findings into clear answers. "
                "You never invent information."
            ),

            llm=llm,

            verbose=True,

            max_iter=2,

            allow_delegation=False,
        )