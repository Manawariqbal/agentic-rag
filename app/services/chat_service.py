from app.agents.crew import AgenticRAGCrew
from app.agents.crew_answer_agent import CrewAnswerAgent
from app.agents.rag_tool import RAGTool
from app.agents.research_agent import ResearchAgent
from app.agents.router_agent import RouterAgent

from app.memory.conversation_memory import ConversationMemory

from app.rag.citations import CitationManager
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever


class ChatService:

    def __init__(
        self,
        memory: ConversationMemory,
        router: RouterAgent,
        retriever: Retriever,
        reranker: SimpleReranker,
        citation_manager: CitationManager,
    ):

        self.memory = memory

        self.router = router

        self.retriever = retriever

        self.reranker = reranker

        self.citation_manager = citation_manager

    # ---------------------------------------------------------
    # Main Chat
    # ---------------------------------------------------------

    def chat(
        self,
        conversation_id: str,
        message: str,
    ):

        # -----------------------------------------------------
        # 1. Store user message
        # -----------------------------------------------------

        self.memory.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )

        # -----------------------------------------------------
        # 2. Route query
        # -----------------------------------------------------

        decision = self.router.route(message)

        # -----------------------------------------------------
        # 3. General question
        # -----------------------------------------------------

        if decision.route == "general":

            answer = (
                "This is a general question and does not require "
                "the enterprise knowledge base."
            )

            self.memory.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
            )

            return {
                "answer": answer,
                "route": "general",
                "reason": decision.reason,
                "citations": [],
            }

        # -----------------------------------------------------
        # 4. Create RAG Tool
        #
        # IMPORTANT:
        # Create it per request so citation state is isolated.
        # -----------------------------------------------------

        rag_tool = RAGTool(
            retriever=self.retriever,
            reranker=self.reranker,
            citation_manager=self.citation_manager,
            rerank_top_k=3,
        )

        # -----------------------------------------------------
        # 5. Create Research Agent
        # -----------------------------------------------------

        research_agent = ResearchAgent(
            rag_tool=rag_tool
        )

        # -----------------------------------------------------
        # 6. Create Answer Agent
        # -----------------------------------------------------

        answer_agent = CrewAnswerAgent()

        # -----------------------------------------------------
        # 7. Create Crew
        # -----------------------------------------------------

        crew = AgenticRAGCrew(
            research_agent=research_agent,
            answer_agent=answer_agent,
            rag_tool=rag_tool,
        )

        # -----------------------------------------------------
        # 8. Execute CrewAI
        # -----------------------------------------------------

        crew_result = crew.run(message)

        answer = crew_result.answer

        citations = crew_result.citations

        # -----------------------------------------------------
        # 9. Normalize citations
        # -----------------------------------------------------

        answer = self._normalize_citations(
            answer=answer,
            citations=citations,
        )

        # -----------------------------------------------------
        # 10. Keep only citations actually referenced
        # -----------------------------------------------------

        used_citations = (
            self.citation_manager.filter_used_citations(
                answer=answer,
                citations=citations,
            )
        )

        # -----------------------------------------------------
        # 11. Store assistant message
        # -----------------------------------------------------

        citation_strings = [
            citation.display()
            for citation in used_citations
        ]

        self.memory.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            citations=citation_strings,
        )

        # -----------------------------------------------------
        # 12. API response
        # -----------------------------------------------------

        return {
            "answer": answer,
            "route": "rag",
            "reason": decision.reason,
            "citations": used_citations,
        }

    # ---------------------------------------------------------
    # Citation normalization
    # ---------------------------------------------------------

    def _normalize_citations(
        self,
        answer: str,
        citations: list,
    ) -> str:

        if not citations:
            return answer

        # If model already used [1], [2], etc.,
        # preserve them.
        if self.citation_manager.validate_citations(
            answer,
            citations,
        ):
            return answer

        # Convert:
        #
        # (Source: leave_and_attendance_policy.pdf — Entitlement)
        #
        # into:
        #
        # [1]

        for citation in citations:

            source_text = (
                f"(Source: {citation.source} — "
                f"{citation.section})"
            )

            if source_text in answer:

                answer = answer.replace(
                    source_text,
                    f"[{citation.citation_id}]",
                )

        # Remove any invalid citation numbers.
        answer = (
            self.citation_manager.remove_invalid_citations(
                answer,
                citations,
            )
        )

        return answer