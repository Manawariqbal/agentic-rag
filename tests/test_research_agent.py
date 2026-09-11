from app.agents.rag_tool import RAGTool
from app.agents.research_agent import ResearchAgent

from app.rag.citations import CitationManager
from app.rag.embedding import MockEmbeddingProvider
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever
from app.rag.vector_store import InMemoryVectorStore
from app.ingestion.chunker import DocumentChunk


def main():

    # -------------------------
    # Test documents
    # -------------------------

    chunks = [
        DocumentChunk(
            text=(
                "Employees are entitled to 20 days "
                "of annual leave per year."
            ),
            metadata={
                "source": "leave_and_attendance_policy.pdf",
                "section": "Annual Leave",
                "chunk_index": 0,
            },
        ),
        DocumentChunk(
            text=(
                "Employees may carry forward up to "
                "5 unused annual leave days."
            ),
            metadata={
                "source": "leave_and_attendance_policy.pdf",
                "section": "Carry Forward",
                "chunk_index": 0,
            },
        ),
    ]

    # -------------------------
    # RAG components
    # -------------------------

    embedding_provider = MockEmbeddingProvider()

    vector_store = InMemoryVectorStore(
        embedding_provider
    )

    vector_store.add_chunks(chunks)

    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        top_k=3,
    )

    reranker = SimpleReranker()

    citation_manager = CitationManager()

    # -------------------------
    # RAG Tool
    # -------------------------

    rag_tool = RAGTool(
        retriever=retriever,
        reranker=reranker,
        citation_manager=citation_manager,
    )

    # -------------------------
    # Research Agent
    # -------------------------

    research_agent = ResearchAgent(
        rag_tool=rag_tool
    )

    print("\nRESEARCH AGENT CREATED")
    print("======================")

    print(
        f"Role: {research_agent.agent.role}"
    )

    print(
        f"Goal: {research_agent.agent.goal}"
    )

    print(
        f"Tools: {[tool.name for tool in research_agent.agent.tools]}"
    )

    print(
        f"LLM: {research_agent.agent.llm}"
    )

if __name__ == "__main__":
    main()