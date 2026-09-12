from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str
    message: str = Field(..., min_length=1)


class CitationResponse(BaseModel):
    citation_id: int
    source: str
    section: str
    chunk_index: int


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    route: str
    reason: str
    citations: list[CitationResponse] = Field(default_factory=list)


# ============================================================
# OpenAI-compatible API schemas
# ============================================================


class OpenAIMessage(BaseModel):
    role: str
    content: Any = None


class OpenAIChatRequest(BaseModel):
    model: str = "agentic-rag"

    messages: list[OpenAIMessage]

    temperature: float | None = 0.0

    stream: bool = False

    # Native Agentic RAG conversation identifier
    conversation_id: str | None = None

    # OpenWebUI identifiers
    chat_id: str | None = None
    session_id: str | None = None

    # OpenAI/OpenWebUI request identifier
    id: str | None = None


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "agentic-rag"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


class OpenAIChatResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIChatResponseMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    id: str

    object: str = "chat.completion"

    choices: list[OpenAIChoice]

    usage: OpenAIUsage = Field(
        default_factory=OpenAIUsage
    )