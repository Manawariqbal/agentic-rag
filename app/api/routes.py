import json
import re
import uuid

from fastapi import APIRouter, HTTPException

from app.agents.router_agent import RouterAgent
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatResponseMessage,
    OpenAIChoice,
    OpenAIModel,
    OpenAIModelList,
)
from app.memory.conversation_memory import ConversationMemory
from app.rag.citations import CitationManager
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever
from app.services.chat_service import ChatService


# ---------------------------------------------------------------------------
# Shared Agentic RAG services
# ---------------------------------------------------------------------------

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

chat_service = ChatService(
    memory=memory,
    router=router_agent,
    retriever=retriever,
    reranker=reranker,
    citation_manager=citation_manager,
)


# ---------------------------------------------------------------------------
# Native Agentic RAG API
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
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


# ---------------------------------------------------------------------------
# OpenAI-compatible API
# ---------------------------------------------------------------------------

openai_router = APIRouter(
    prefix="/v1",
    tags=["openai-compatible"],
)


@openai_router.get(
    "/models",
    response_model=OpenAIModelList,
)
def list_models():
    return OpenAIModelList(
        data=[
            OpenAIModel(
                id="agentic-rag",
                owned_by="agentic-rag",
            )
        ]
    )


# ---------------------------------------------------------------------------
# OpenWebUI title-generation helpers
# ---------------------------------------------------------------------------

def is_title_generation_request(message: str) -> bool:
    """
    Detect OpenWebUI's internal title-generation request.

    OpenWebUI sends prompts containing text such as:
    "Generate a concise title summarizing the chat history."

    These requests should not invoke CrewAI or RAG.
    """

    if not message:
        return False

    normalized = message.lower()

    return (
        "generate a concise title summarizing the chat history"
        in normalized
        or (
            "generate a concise title" in normalized
            and "chat history" in normalized
        )
    )


def extract_title_source(message: str) -> str:
    """
    Extract the original USER question from OpenWebUI's
    title-generation prompt.
    """

    match = re.search(
        r"USER:\s*(.+?)(?:\n|$)",
        message,
        flags=re.IGNORECASE,
    )

    if match:
        question = match.group(1).strip()

        if question:
            return question

    lines = [
        line.strip()
        for line in message.splitlines()
        if line.strip()
    ]

    for line in lines:
        if not line.startswith(("#", "-", "{", "}")):
            return line[:120]

    return "Enterprise Knowledge"


def generate_local_title(message: str) -> str:
    """
    Generate a short deterministic title locally.

    We don't call Ollama or CrewAI because title generation
    is a UI operation, not an Agentic RAG operation.
    """

    question = extract_title_source(message)

    question = re.sub(
        r"[^a-zA-Z0-9\s\-]",
        "",
        question,
    )

    words = question.split()

    if not words:
        return "Enterprise Knowledge"

    stop_words = {
        "how",
        "what",
        "when",
        "where",
        "why",
        "which",
        "is",
        "are",
        "do",
        "does",
        "can",
        "could",
        "would",
        "should",
        "the",
        "a",
        "an",
    }

    meaningful_words = [
        word
        for word in words
        if word.lower() not in stop_words
    ]

    if meaningful_words:
        words = meaningful_words

    title = " ".join(words[:4])

    if not title:
        title = "Enterprise Knowledge"

    return title[:60]


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions
# ---------------------------------------------------------------------------

@openai_router.post(
    "/chat/completions",
    response_model=OpenAIChatResponse,
)
def openai_chat_completions(
    request: OpenAIChatRequest,
):
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty",
        )

    # Find the latest user message.
    user_message = None

    for message in reversed(request.messages):
        if (
            message.role == "user"
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            user_message = message.content.strip()
            break

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="No user message found",
        )

    # -----------------------------------------------------------------------
    # OpenWebUI title generation
    # -----------------------------------------------------------------------

    if is_title_generation_request(user_message):

        title = generate_local_title(user_message)

        title_content = json.dumps(
            {"title": title},
            ensure_ascii=False,
        )

        response_id = (
            "chatcmpl-"
            + uuid.uuid4().hex
        )

        return OpenAIChatResponse(
            id=response_id,
            object="chat.completion",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChatResponseMessage(
                        role="assistant",
                        content=title_content,
                    ),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

    # -----------------------------------------------------------------------
    # Normal Agentic RAG request
    # -----------------------------------------------------------------------

    conversation_id = (
    request.conversation_id
    or request.chat_id
    or request.session_id
    or request.id
    or str(uuid.uuid4())
)

    try:
        result = chat_service.chat(
            conversation_id=conversation_id,
            message=user_message,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agentic RAG error: {str(exc)}",
        ) from exc

    answer = result["answer"]

    # -----------------------------------------------------------------------
    # Add citation sources for OpenWebUI
    # -----------------------------------------------------------------------

    citations = result.get("citations", [])

    if citations:
        source_lines = [
            "",
            "Sources:",
        ]

        for citation in citations:
            source_lines.append(
                f"[{citation.citation_id}] "
                f"{citation.source} — "
                f"{citation.section}"
            )

        answer = (
            answer
            + "\n"
            + "\n".join(source_lines)
        )

    response_id = (
        "chatcmpl-"
        + uuid.uuid4().hex
    )

    return OpenAIChatResponse(
        id=response_id,
        object="chat.completion",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIChatResponseMessage(
                    role="assistant",
                    content=answer,
                ),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

router_api = router