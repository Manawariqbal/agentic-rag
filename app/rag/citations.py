from dataclasses import dataclass
import re


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
        results: list[dict],
    ) -> list[Citation]:
        """
        Build deterministic citations from retrieval results.

        Duplicate source + section + chunk combinations
        are removed.
        """

        citations = []
        seen = set()

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "Unknown source",
            )

            section = metadata.get(
                "section",
                "Unknown section",
            )

            chunk_index = metadata.get(
                "chunk_index",
                0,
            )

            citation_key = (
                source,
                section,
                chunk_index,
            )

            # Skip duplicate source/section/chunk
            if citation_key in seen:
                continue

            seen.add(citation_key)

            citations.append(
                Citation(
                    citation_id=len(citations) + 1,
                    source=source,
                    section=section,
                    chunk_index=chunk_index,
                )
            )

        return citations

    def filter_used_citations(
        self,
        answer: str,
        citations: list[Citation],
    ) -> list[Citation]:
        """
        Return only citations explicitly referenced
        by the generated answer.
        """

        used_ids = {
            int(match)
            for match in re.findall(
                r"\[(\d+)\]",
                answer,
            )
        }

        return [
            citation
            for citation in citations
            if citation.citation_id in used_ids
        ]

    def validate_citations(
        self,
        answer: str,
        citations: list[Citation],
    ) -> bool:
        """
        Check whether every citation referenced by the
        generated answer exists in the available citations.
        """

        used_ids = {
            int(match)
            for match in re.findall(
                r"\[(\d+)\]",
                answer,
            )
        }

        valid_ids = {
            citation.citation_id
            for citation in citations
        }

        return used_ids.issubset(valid_ids)

    def remove_invalid_citations(
        self,
        answer: str,
        citations: list[Citation],
    ) -> str:
        """
        Remove citation markers that do not exist
        in the available citation list.
        """

        valid_ids = {
            citation.citation_id
            for citation in citations
        }

        def replace(match):

            citation_id = int(
                match.group(1)
            )

            if citation_id in valid_ids:
                return match.group(0)

            return ""

        return re.sub(
            r"\[(\d+)\]",
            replace,
            answer,
        )

    def format_citations(
        self,
        citations: list[Citation],
    ) -> str:
        """
        Format citations for display.
        """

        if not citations:
            return "No sources available."

        return "\n".join(
            citation.display()
            for citation in citations
        )