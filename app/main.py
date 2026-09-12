import os

# Disable CrewAI's automatic telemetry before importing CrewAI.
os.environ.setdefault(
    "CREWAI_DISABLE_TELEMETRY",
    "true",
)

from fastapi import FastAPI

from app.config import settings
from app.observability.phoenix import init_phoenix
from app.api.routes import router, openai_router


# ---------------------------------------------------------------------------
# Phoenix observability
# ---------------------------------------------------------------------------

init_phoenix()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Agentic RAG Backend",
)


# Existing Agentic RAG API
app.include_router(router)


# OpenAI-compatible API for OpenWebUI
app.include_router(openai_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
    }