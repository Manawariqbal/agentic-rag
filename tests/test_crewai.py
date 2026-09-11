from app.agents.rag_tool import RAGTool
from app.agents.research_agent import ResearchAgent
from app.agents.crew_answer_agent import CrewAnswerAgent
from app.agents.crew import AgenticRAGCrew

from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter
from app.rag.retriever import Retriever
from app.rag.reranker import SimpleReranker
from app.rag.citations import CitationManager


def main():

    print("=" * 70)
    print("CREWAI AGENTIC RAG TEST")
    print("=" * 70)

    embedding_provider = OllamaEmbeddingProvider()

    vector_store = PGVectorStoreAdapter(
        embedding_provider=embedding_provider,
        table_name="rag_documents_1024",
        embed_dim=1024,
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        top_k=10,
    )

    reranker = SimpleReranker()

    citation_manager = CitationManager()

    rag_tool = RAGTool(
    retriever=retriever,
    reranker=reranker,
    citation_manager=citation_manager,
    rerank_top_k=3,
    result_as_answer=True,
)

    research_agent = ResearchAgent(
        rag_tool=rag_tool
    )

    answer_agent = CrewAnswerAgent()

    crew = AgenticRAGCrew(
        research_agent=research_agent,
        answer_agent=answer_agent,
    )

    query = "How many annual leave days do employees get?"

    print("\nUSER QUESTION")
    print(query)

    print("\nSTARTING CREW...\n")

    result = crew.run(query)

    print("\n" + "=" * 70)
    print("FINAL CREW RESULT")
    print("=" * 70)

    print(result)


if __name__ == "__main__":
    main()