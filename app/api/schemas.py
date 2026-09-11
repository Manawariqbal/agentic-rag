from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str = Field(
        ...,
        description="Unique conversation identifier.",
    )

    message: str = Field(
        ...,
        min_length=1,
        description="User's message.",
    )


class CitationResponse(BaseModel):
    citation_id: int
    source: str
    section: str
    chunk_index: int


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[CitationResponse] = []