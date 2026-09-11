from contextlib import contextmanager
from typing import Any


class PhoenixTracer:

    def __init__(
        self,
        project_name: str = "agentic-rag",
        enabled: bool = True,
    ):
        self.project_name = project_name
        self.enabled = enabled

    @contextmanager
    def trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Generic tracing context.

        Later this will create an actual Phoenix span.
        """

        if not self.enabled:
            yield None
            return

        trace_data = {
            "name": name,
            "project": self.project_name,
            "metadata": metadata or {},
        }

        print(f"[TRACE START] {name}")

        try:
            yield trace_data

        finally:
            print(f"[TRACE END] {name}")

    def record(
        self,
        name: str,
        data: dict[str, Any],
    ):
        """
        Record an event that can later be sent to Phoenix.
        """

        if not self.enabled:
            return

        print(
            f"[TRACE] {name}: {data}"
        )