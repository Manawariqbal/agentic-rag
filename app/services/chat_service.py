from app.agents.crew import AgenticRAGCrew
from app.agents.crew_answer_agent import CrewAnswerAgent
from app.agents.rag_tool import RAGTool
from app.agents.research_agent import ResearchAgent
from app.agents.router_agent import RouterAgent

from app.memory.conversation_memory import ConversationMemory

from app.observability.phoenix import get_tracer

from app.rag.citations import CitationManager
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

        tracer = get_tracer()

        with tracer.start_as_current_span("chat") as span:

            span.set_attribute(
                "conversation.id",
                conversation_id,
            )

            span.set_attribute(
                "user.message",
                message,
            )

            span.set_attribute(
                "message.length",
                len(message),
            )

            # -------------------------------------------------
            # 1. Load previous conversation context
            # -------------------------------------------------

            previous_messages = self.memory.get_recent_messages(
                conversation_id=conversation_id,
                limit=6,
            )

            conversation_context = self._build_conversation_context(
                previous_messages
            )

            # -------------------------------------------------
            # 2. Store current user message
            # -------------------------------------------------

            self.memory.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message,
            )

            # -------------------------------------------------
            # 3. Context-aware routing
            # -------------------------------------------------

            decision = self.router.route(
                query=message,
                conversation_context=conversation_context,
            )

            span.set_attribute(
                "rag.route",
                decision.route,
            )

            span.set_attribute(
                "rag.route_reason",
                decision.reason,
            )

            span.set_attribute(
                "conversation.context_length",
                len(conversation_context),
            )

            # -------------------------------------------------
            # 4. General question
            # -------------------------------------------------

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

                span.set_attribute(
                    "rag.result",
                    "general",
                )

                span.set_attribute(
                    "response.length",
                    len(answer),
                )

                return {
                    "answer": answer,
                    "route": "general",
                    "reason": decision.reason,
                    "citations": [],
                }

            # -------------------------------------------------
            # 5. Build contextualized RAG query
            # -------------------------------------------------

            retrieval_query = self._build_retrieval_query(
                message=message,
                conversation_context=conversation_context,
            )

            span.set_attribute(
                "rag.retrieval_query",
                retrieval_query,
            )

            span.set_attribute(
                "rag.result",
                "rag",
            )

            # -------------------------------------------------
            # 6. Create RAG Tool
            #
            # IMPORTANT:
            # Create it per request so citation state is isolated.
            # -------------------------------------------------

            rag_tool = RAGTool(
                retriever=self.retriever,
                reranker=self.reranker,
                citation_manager=self.citation_manager,
                rerank_top_k=3,
            )

            # -------------------------------------------------
            # 7. Create Research Agent
            # -------------------------------------------------

            research_agent = ResearchAgent(
                rag_tool=rag_tool,
            )

            # -------------------------------------------------
            # 8. Create Answer Agent
            # -------------------------------------------------

            answer_agent = CrewAnswerAgent()

            # -------------------------------------------------
            # 9. Create Crew
            # -------------------------------------------------

            crew = AgenticRAGCrew(
                research_agent=research_agent,
                answer_agent=answer_agent,
                rag_tool=rag_tool,
            )

            # -------------------------------------------------
            # 10. Execute CrewAI
            #
            # Use the contextualized query so the RAG tool
            # receives enough information for follow-ups.
            # -------------------------------------------------

            crew_result = crew.run(
                retrieval_query
            )

            answer = crew_result.answer

            citations = crew_result.citations

            # -------------------------------------------------
            # 11. Normalize citations
            # -------------------------------------------------

            answer = self._normalize_citations(
                answer=answer,
                citations=citations,
            )

            # -------------------------------------------------
            # 12. Keep only citations actually referenced
            # -------------------------------------------------

            used_citations = (
                self.citation_manager.filter_used_citations(
                    answer=answer,
                    citations=citations,
                )
            )

            span.set_attribute(
                "rag.citation_count",
                len(used_citations),
            )

            # -------------------------------------------------
            # 13. Store assistant message
            # -------------------------------------------------

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

            # -------------------------------------------------
            # 14. Response metadata
            # -------------------------------------------------

            span.set_attribute(
                "response.length",
                len(answer),
            )

            span.set_attribute(
                "rag.result",
                "success",
            )

            # -------------------------------------------------
            # 15. API response
            # -------------------------------------------------

            return {
                "answer": answer,
                "route": "rag",
                "reason": decision.reason,
                "citations": used_citations,
            }

    # ---------------------------------------------------------
    # Conversation context
    # ---------------------------------------------------------

    def _build_conversation_context(
        self,
        messages,
    ) -> str:

        if not messages:
            return ""

        context_lines = []

        for message in messages:

            role = message.role.upper()

            context_lines.append(
                f"{role}: {message.content}"
            )

        return "\n".join(context_lines)

    # ---------------------------------------------------------
    # Contextualized retrieval query
    # ---------------------------------------------------------

    def _build_retrieval_query(
        self,
        message: str,
        conversation_context: str,
    ) -> str:

        if not conversation_context:
            return message

        # Identify whether this looks like a follow-up.
        query_lower = message.lower()

        follow_up_terms = {
            "those",
            "that",
            "it",
            "they",
            "them",
            "these",
            "this",
            "same",
            "previous",
            "above",
            "earlier",
        }

        is_follow_up = any(
            term in query_lower.split()
            for term in follow_up_terms
        )

        if not is_follow_up:
            return message

        # Use recent conversation context together with the
        # current question. This gives the embedding model
        # enough semantic information to retrieve the correct
        # enterprise document.
        return (
            "Conversation context:\n"
            f"{conversation_context}\n\n"
            "Current user question:\n"
            f"{message}"
        )

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