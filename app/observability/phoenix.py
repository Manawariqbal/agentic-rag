from opentelemetry import trace
from phoenix.otel import register

from app.config import settings


_tracer = None


def init_phoenix():
    global _tracer

    if not settings.phoenix_enabled:
        return None

    tracer_provider = register(
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_endpoint,
        auto_instrument=False,
    )

    _tracer = trace.get_tracer("agentic-rag")

    return tracer_provider


def get_tracer():
    global _tracer

    if _tracer is None:
        _tracer = trace.get_tracer("agentic-rag")

    return _tracer