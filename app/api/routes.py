from fastapi import APIRouter

from app.agents.router_agent import RouterAgent

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
)

from app.memory.conversation_memory import (
    ConversationMemory,
)

from app.rag.citations import CitationManager
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever

from app.services.chat_service import ChatService


router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


# ==========================================================
# RAG Components
# ==========================================================

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

memory = ConversationMemory()

router_agent = RouterAgent()


# ==========================================================
# Chat Service
# ==========================================================

chat_service = ChatService(

    memory=memory,

    router=router_agent,

    retriever=retriever,

    reranker=reranker,

    citation_manager=citation_manager,
)


# ==========================================================
# Chat Endpoint
# ==========================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    result = chat_service.chat(

        conversation_id=request.conversation_id,

        message=request.message,
    )

    citations = [
        CitationResponse(
            citation_id=c.citation_id,
            source=c.source,
            section=c.section,
            chunk_index=c.chunk_index,
        )
        for c in result["citations"]
    ]

    return ChatResponse(

        conversation_id=request.conversation_id,

        answer=result["answer"],

        route=result["route"],

        reason=result["reason"],

        citations=citations,
    )