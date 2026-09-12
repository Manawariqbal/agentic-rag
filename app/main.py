import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from fastapi import FastAPI

from app.config import settings
from app.observability.phoenix import init_phoenix


# Initialize Phoenix BEFORE importing application components
init_phoenix()

from app.api.routes import router


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Agentic RAG Backend",
)

app.include_router(router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
    }