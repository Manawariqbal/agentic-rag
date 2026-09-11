from dataclasses import dataclass
from typing import Any


@dataclass
class Citation:
    citation_id: int
    source: str
    section: str
    chunk_index: int

    def display(self) -> str:
        return (
            f"[{self.citation_id}] "
            f"{self.source} — {self.section}"
        )


class CitationManager:

    def build_citations(
        self,
        results: list[dict[str, Any]],
    ) -> list[Citation]:

        citations = []

        for index, result in enumerate(results, start=1):

            metadata = result.get("metadata", {})

            citation = Citation(
                citation_id=index,
                source=metadata.get(
                    "source",
                    "Unknown source",
                ),
                section=metadata.get(
                    "section",
                    "Unknown section",
                ),
                chunk_index=metadata.get(
                    "chunk_index",
                    0,
                ),
            )

            citations.append(citation)

        return citations

    def format_citations(
        self,
        citations: list[Citation],
    ) -> str:

        if not citations:
            return "No sources available."

        return "\n".join(
            citation.display()
            for citation in citations
        )