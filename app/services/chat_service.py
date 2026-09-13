from app.agents.crew import AgenticRAGCrew
from app.agents.crew_answer_agent import CrewAnswerAgent
from app.agents.rag_tool import RAGTool
from app.agents.research_agent import ResearchAgent
from app.agents.router_agent import RouterAgent

from app.config import settings

from app.memory.conversation_memory import ConversationMemory

from app.observability.phoenix import get_tracer

from app.rag.citations import CitationManager
from app.rag.evidence_gate import EvidenceGate
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever

import json


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

        self.evidence_gate = EvidenceGate(
            threshold=getattr(
                settings,
                "rag_relevance_threshold",
                0.48,
            )
        )

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
            # 5. Build retrieval query
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
            # 6. DETERMINISTIC RETRIEVAL
            # -------------------------------------------------

            retrieved_results = self.retriever.retrieve(
                retrieval_query
            )

            span.set_attribute(
                "rag.retrieved_count",
                len(retrieved_results),
            )

            # -------------------------------------------------
            # 7. DETERMINISTIC RERANKING
            # -------------------------------------------------

            reranked_results = self.reranker.rerank(
                query=retrieval_query,
                results=retrieved_results,
                top_k=3,
            )

            span.set_attribute(
                "rag.reranked_count",
                len(reranked_results),
            )

            top_score = 0.0

            if reranked_results:
                top_score = float(
                    reranked_results[0].get(
                        "rerank_score",
                        0.0,
                    )
                )

            span.set_attribute(
                "rag.top_rerank_score",
                top_score,
            )

            # -------------------------------------------------
            # 8. EVIDENCE GATE
            # -------------------------------------------------

            gate_result = self.evidence_gate.evaluate(
                reranked_results
            )

            span.set_attribute(
                "rag.evidence_sufficient",
                gate_result.sufficient,
            )

            span.set_attribute(
                "rag.evidence_threshold",
                self.evidence_gate.threshold,
            )

            span.set_attribute(
                "rag.evidence_gate_reason",
                gate_result.reason,
            )

            # -------------------------------------------------
            # 9. ABSTAIN WHEN EVIDENCE IS INSUFFICIENT
            # -------------------------------------------------

            if not gate_result.sufficient:

                answer = (
                    "I don't have enough information in the "
                    "available company policies to answer this "
                    "question."
                )

                self.memory.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer,
                    citations=[],
                )

                span.set_attribute(
                    "rag.result",
                    "insufficient_evidence",
                )

                span.set_attribute(
                    "response.length",
                    len(answer),
                )

                return {
                    "answer": answer,
                    "route": "rag",
                    "reason": (
                        "Insufficient evidence in the "
                        "enterprise knowledge base."
                    ),
                    "citations": [],
                }

            # -------------------------------------------------
            # 10. Build deterministic citation set
            #
            # Citations now come from the deterministic retrieval
            # results instead of relying on the Research Agent
            # to call the RAG tool.
            # -------------------------------------------------

            citations = self.citation_manager.build_citations(
                reranked_results
            )

            span.set_attribute(
                "rag.citation_count",
                len(citations),
            )

            # -------------------------------------------------
            # 11. Serialize retrieved evidence
            #
            # This is the source of truth that is passed into
            # the CrewAI Research Agent.
            # -------------------------------------------------

            retrieved_evidence = json.dumps(
                reranked_results,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

            span.set_attribute(
                "rag.evidence_length",
                len(retrieved_evidence),
            )

            # -------------------------------------------------
            # 12. Create RAG Tool
            #
            # Kept available for the Crew architecture and
            # future agentic retrieval workflows.
            # -------------------------------------------------

            rag_tool = RAGTool(
                retriever=self.retriever,
                reranker=self.reranker,
                citation_manager=self.citation_manager,
                rerank_top_k=3,
            )

            # -------------------------------------------------
            # 13. Create Research Agent
            # -------------------------------------------------

            research_agent = ResearchAgent(
                rag_tool=rag_tool,
            )

            # -------------------------------------------------
            # 14. Create Answer Agent
            # -------------------------------------------------

            answer_agent = CrewAnswerAgent()

            # -------------------------------------------------
            # 15. Create Crew
            # -------------------------------------------------

            crew = AgenticRAGCrew(
                research_agent=research_agent,
                answer_agent=answer_agent,
                rag_tool=rag_tool,
            )

            # -------------------------------------------------
            # 16. Execute CrewAI
            #
            # IMPORTANT:
            # The deterministic retrieved evidence is explicitly
            # passed into CrewAI.
            # -------------------------------------------------

            crew_result = crew.run(
                query=retrieval_query,
                retrieved_evidence=retrieved_evidence,
            )

            answer = crew_result.answer

            # -------------------------------------------------
            # 17. Normalize citations
            # -------------------------------------------------

            answer = self._normalize_citations(
                answer=answer,
                citations=citations,
            )

            # -------------------------------------------------
            # 18. Keep only citations actually referenced
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
            # 19. Store assistant message
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
            # 20. Response metadata
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
            # 21. API response
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
    # Retrieval query
    # ---------------------------------------------------------

    def _build_retrieval_query(
        self,
        message: str,
        conversation_context: str,
    ) -> str:
        """
        Build the query used for vector retrieval.

        The current user message is used directly for retrieval.
        Conversation context remains available separately for
        conversation-aware answer generation.
        """

        return message.strip()

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