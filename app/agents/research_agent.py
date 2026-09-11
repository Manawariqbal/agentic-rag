from crewai import Agent, LLM


class ResearchAgent:

    def __init__(self, rag_tool):

        llm = LLM(
            model="ollama/qwen3:8b",
            base_url="http://localhost:11434",
        )

        self.agent = Agent(
            role="Enterprise Knowledge Researcher",

            goal=(
                "Find accurate information from the enterprise "
                "knowledge base and provide well-supported research "
                "for the answer agent."
            ),

            backstory=(
                "You are an enterprise research specialist. "
                "You search the company's knowledge base, identify "
                "the most relevant information, and preserve source "
                "references. You never invent information that is "
                "not present in the retrieved documents."
            ),

            tools=[rag_tool],

            llm=llm,

            verbose=True,
        )