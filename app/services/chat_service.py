from app.agents.router_agent import RouterAgent
from app.memory.conversation_memory import ConversationMemory
from app.agents.answer_agent import AnswerAgent

from app.rag.citations import CitationManager
from app.rag.llm import MockLLMProvider
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever

from app.observability.phoenix import PhoenixTracer


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

        # ------------------------------------------------
        # Phoenix tracer
        # ------------------------------------------------

        self.tracer = PhoenixTracer(
            project_name="agentic-rag"
        )

        # ------------------------------------------------
        # Answer Agent
        # ------------------------------------------------

        self.answer_agent = AnswerAgent(
            llm=MockLLMProvider(),
            citation_manager=citation_manager,
        )

    def chat(
        self,
        conversation_id: str,
        message: str,
    ):

        # =================================================
        # 1. Store user message
        # =================================================

        self.memory.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )

        # =================================================
        # 2. Route query
        # =================================================

        with self.tracer.trace(
            "query_routing",
            {
                "conversation_id": conversation_id,
                "query": message,
            },
        ):

            decision = self.router.route(message)

        # =================================================
        # 3. General query
        # =================================================

        if decision.route == "general":

            answer = (
                "This is a general question and does not "
                "require the enterprise knowledge base."
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

        # =================================================
        # 4. Retrieval
        # =================================================

        with self.tracer.trace(
            "retrieval",
            {
                "conversation_id": conversation_id,
                "query": message,
            },
        ):

            retrieved_results = self.retriever.retrieve(
                message
            )

        # Record retrieval information

        self.tracer.record(
            "retrieval_results",
            {
                "query": message,
                "result_count": len(retrieved_results),
            },
        )

        # =================================================
        # 5. Reranking
        # =================================================

        with self.tracer.trace(
            "reranking",
            {
                "conversation_id": conversation_id,
                "query": message,
                "retrieved_count": len(retrieved_results),
            },
        ):

            reranked_results = self.reranker.rerank(
                query=message,
                results=retrieved_results,
                top_k=3,
            )

        # Record reranking information

        self.tracer.record(
            "reranking_results",
            {
                "query": message,
                "result_count": len(reranked_results),
            },
        )

        # =================================================
        # 6. Generate answer
        # =================================================

        with self.tracer.trace(
            "answer_generation",
            {
                "conversation_id": conversation_id,
                "query": message,
                "context_count": len(reranked_results),
            },
        ):

            response = self.answer_agent.answer(
                query=message,
                results=reranked_results,
            )

        # =================================================
        # 7. Convert citations
        # =================================================

        citations = [
            {
                "citation_id": citation.citation_id,
                "source": citation.source,
                "section": citation.section,
                "chunk_index": citation.chunk_index,
            }
            for citation in response.citations
        ]

        citation_strings = [
            citation.display()
            for citation in response.citations
        ]

        # =================================================
        # 8. Store assistant response
        # =================================================

        self.memory.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.answer,
            citations=citation_strings,
        )

        # =================================================
        # 9. Record final response
        # =================================================

        self.tracer.record(
            "chat_response",
            {
                "conversation_id": conversation_id,
                "route": "rag",
                "citation_count": len(citations),
            },
        )

        # =================================================
        # 10. Return API response
        # =================================================

        return {
            "answer": response.answer,
            "route": "rag",
            "reason": decision.reason,
            "citations": citations,
        }