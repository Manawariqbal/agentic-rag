from crewai import Agent, LLM


class CrewAIAnswerAgent:

    def __init__(self):

        llm = LLM(
            model="ollama/qwen3:8b",
            base_url="http://localhost:11434",
        )

        self.agent = Agent(
            role="Enterprise Answer Specialist",

            goal=(
                "Generate accurate answers using only the research "
                "provided by the research agent."
            ),

            backstory=(
                "You are an enterprise knowledge assistant. "
                "You transform research findings into concise, "
                "accurate answers. You never invent information "
                "and preserve citation references."
            ),

            llm=llm,

            verbose=True,
        )