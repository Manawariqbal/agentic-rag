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
                "Retrieve accurate information from the enterprise "
                "knowledge base and provide evidence for the answer agent."
            ),

            backstory=(
                "You are an enterprise research specialist. "
                "Use the knowledge_base_search tool to retrieve "
                "evidence from company documents. "
                "Do not invent information."
            ),

            tools=[rag_tool],

            llm=llm,

            verbose=True,

            max_iter=1,

            allow_delegation=False,
        )