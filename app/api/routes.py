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
# Agentic RAG services
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
# Native API
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post("/chat", response_model=ChatResponse)
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
# OpenAI-compatible API for OpenWebUI
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
# OpenWebUI internal request detection
# ---------------------------------------------------------------------------

def is_title_generation_request(message: str) -> bool:

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


def is_follow_up_generation_request(message: str) -> bool:

    if not message:
        return False

    normalized = message.lower()

    return (
        "suggest 3-5 relevant follow-up questions"
        in normalized
        or "suggest 3-5 relevant follow up questions"
        in normalized
        or (
            "follow-up questions or prompts"
            in normalized
            and "chat history" in normalized
        )
    )


def is_tag_generation_request(message: str) -> bool:

    if not message:
        return False

    normalized = message.lower()

    return (
        "generate 1-3 broad tags"
        in normalized
        or "generate 1-3 broad tags categorizing"
        in normalized
        or (
            "broad tags categorizing the main themes"
            in normalized
            and "chat history" in normalized
        )
    )


def is_openwebui_metadata_request(message: str) -> bool:

    return (
        is_title_generation_request(message)
        or is_follow_up_generation_request(message)
        or is_tag_generation_request(message)
    )


# ---------------------------------------------------------------------------
# Local OpenWebUI metadata handlers
# ---------------------------------------------------------------------------

def extract_title_source(message: str) -> str:

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


def metadata_response(content: str) -> OpenAIChatResponse:

    response_id = "chatcmpl-" + uuid.uuid4().hex

    return OpenAIChatResponse(
        id=response_id,
        object="chat.completion",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIChatResponseMessage(
                    role="assistant",
                    content=content,
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

    # Find the latest actual string user message.
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
    # OpenWebUI internal requests
    #
    # IMPORTANT:
    # These must NEVER enter ChatService / CrewAI / RAG.
    # -----------------------------------------------------------------------

    if is_title_generation_request(user_message):

        title = generate_local_title(user_message)

        return metadata_response(
            json.dumps(
                {"title": title},
                ensure_ascii=False,
            )
        )

    if is_follow_up_generation_request(user_message):

        return metadata_response(
            json.dumps(
                {"follow_ups": []},
                ensure_ascii=False,
            )
        )

    if is_tag_generation_request(user_message):

        return metadata_response(
            json.dumps(
                {"tags": ["General"]},
                ensure_ascii=False,
            )
        )

    # -----------------------------------------------------------------------
    # Genuine user question
    # -----------------------------------------------------------------------

    conversation_id = (
        request.conversation_id
        or request.chat_id
        or request.session_id
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

    citations = result.get(
        "citations",
        [],
    )

    # OpenWebUI displays the source information as part of the answer.
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

    response_id = "chatcmpl-" + uuid.uuid4().hex

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


router_api = router