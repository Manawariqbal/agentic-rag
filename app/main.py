from fastapi import FastAPI

from app.api.routes import router
from app.config import settings


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