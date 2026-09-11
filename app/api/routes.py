from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.agents.router_agent import RouterAgent
from app.ingestion.chunker import ContextualChunker
from app.ingestion.parser import DocumentParser
from app.ingestion.pipeline import IngestionPipeline
from app.memory.conversation_memory import ConversationMemory
from app.rag.citations import CitationManager
from app.rag.embedding import MockEmbeddingProvider
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever
from app.rag.vector_store import InMemoryVectorStore
from app.services.chat_service import ChatService


router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


# -----------------------------
# RAG Components
# -----------------------------

embedding_provider = MockEmbeddingProvider(
    dimensions=384
)

vector_store = InMemoryVectorStore(
    embedding_provider=embedding_provider
)

parser = DocumentParser()

chunker = ContextualChunker(
    chunk_size=800,
    chunk_overlap=100,
)

ingestion_pipeline = IngestionPipeline(
    parser=parser,
    chunker=chunker,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
)


# -----------------------------
# Ingest Knowledge Base
# -----------------------------

try:
    ingestion_results = ingestion_pipeline.ingest_directory(
        "documents"
    )

    print("\nKnowledge Base Ingestion")
    print("========================")

    for filename, count in ingestion_results.items():
        print(f"{filename}: {count} chunks")

    print(
        f"Total chunks: {len(vector_store.documents)}"
    )

except Exception as e:
    print(f"Knowledge base ingestion failed: {e}")


# -----------------------------
# RAG Components
# -----------------------------

retriever = Retriever(
    vector_store=vector_store,
    embedding_provider=embedding_provider,
    top_k=3,
)

reranker = SimpleReranker()

citation_manager = CitationManager()


# -----------------------------
# Memory
# -----------------------------

memory = ConversationMemory()


# -----------------------------
# Router
# -----------------------------

router_agent = RouterAgent()


# -----------------------------
# Chat Service
# -----------------------------

chat_service = ChatService(
    memory=memory,
    router=router_agent,
    retriever=retriever,
    reranker=reranker,
    citation_manager=citation_manager,
)


# -----------------------------
# API Endpoint
# -----------------------------

@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    result = chat_service.chat(
        conversation_id=request.conversation_id,
        message=request.message,
    )

    return ChatResponse(
        conversation_id=request.conversation_id,
        answer=result["answer"],
        citations=result["citations"],
    )