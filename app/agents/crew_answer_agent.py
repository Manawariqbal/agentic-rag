from crewai import Agent, LLM


class CrewAnswerAgent:

    def __init__(self):

        llm = LLM(
            model="ollama/qwen3:8b",
            base_url="http://localhost:11434",
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