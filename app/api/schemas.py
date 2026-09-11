from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    conversation_id: str

    message: str = Field(
        ...,
        min_length=1,
    )


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

    citations: list[CitationResponse] = Field(
        default_factory=list
    )