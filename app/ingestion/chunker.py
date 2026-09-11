from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentChunk:
    text: str
    metadata: dict[str, Any]


class ContextualChunker:
    """
    Creates chunks enriched with document and section context.

    The contextualized text is what we will eventually embed.
    Metadata is retained separately for filtering and citations.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        content: str,
        filename: str,
    ) -> list[DocumentChunk]:

        sections = self._split_sections(content)

        chunks = []

        for section_title, section_content in sections:

            raw_chunks = self._split_text(section_content)

            for index, chunk_text in enumerate(raw_chunks):

                contextual_text = self._build_context(
                    filename=filename,
                    section=section_title,
                    chunk=chunk_text,
                )

                metadata = {
                    "source": filename,
                    "section": section_title,
                    "chunk_index": index,
                }

                chunks.append(
                    DocumentChunk(
                        text=contextual_text,
                        metadata=metadata,
                    )
                )

        return chunks

    def _split_sections(
        self,
        content: str,
    ) -> list[tuple[str, str]]:

        sections = []

        current_section = "General"
        current_content = []

        for line in content.splitlines():

            if line.startswith("#"):

                if current_content:
                    sections.append(
                        (
                            current_section,
                            "\n".join(current_content).strip(),
                        )
                    )

                current_section = line.lstrip("#").strip()
                current_content = []

            else:
                current_content.append(line)

        if current_content:
            sections.append(
                (
                    current_section,
                    "\n".join(current_content).strip(),
                )
            )

        return sections

    def _split_text(
        self,
        text: str,
    ) -> list[str]:

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        return chunks

    def _build_context(
        self,
        filename: str,
        section: str,
        chunk: str,
    ) -> str:

        return (
            f"Document: {filename}\n"
            f"Section: {section}\n"
            f"Context: This section contains information "
            f"from the {section} section of the document.\n\n"
            f"{chunk}"
        )