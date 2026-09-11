from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.rag.citations import CitationManager
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever


class RAGToolInput(BaseModel):
    query: str = Field(
        ...,
        description="The user's question to search in the enterprise knowledge base.",
    )


class RAGTool(BaseTool):

    name: str = "knowledge_base_search"

    description: str = (
        "Search the enterprise knowledge base and return relevant "
        "information with source and citation references."
    )

    args_schema: type[BaseModel] = RAGToolInput

    # CrewAI should terminate the research task
    # after this tool produces the evidence.
    result_as_answer: bool = True

    def __init__(
        self,
        retriever: Retriever,
        reranker: SimpleReranker,
        citation_manager: CitationManager,
        rerank_top_k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._retriever = retriever
        self._reranker = reranker
        self._citation_manager = citation_manager
        self._rerank_top_k = rerank_top_k

        # Populated on every _run().
        self._last_citations = []

    def _run(self, query: str) -> str:

        retrieved_results = self._retriever.retrieve(query)

        if not retrieved_results:
            self._last_citations = []
            return "No relevant information was found."

        reranked_results = self._reranker.rerank(
            query=query,
            results=retrieved_results,
            top_k=self._rerank_top_k,
        )

        citations = self._citation_manager.build_citations(
            reranked_results
        )

        self._last_citations = citations

        citation_lookup = {
            (
                citation.source,
                citation.section,
                citation.chunk_index,
            ): citation
            for citation in citations
        }

        response_parts = []
        seen = set()

        for result in reranked_results:

            metadata = result.get("metadata", {})

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

            key = (
                source,
                section,
                chunk_index,
            )

            if key in seen:
                continue

            seen.add(key)

            citation = citation_lookup.get(key)

            if citation is None:
                continue

            response_parts.append(
                f"""
[{citation.citation_id}]
Source: {source}
Section: {section}

Content:
{result.get("text", "")}
"""
            )

        response_parts.append(
            "\nSources:\n"
            + self._citation_manager.format_citations(
                citations
            )
        )

        return "\n".join(response_parts)

    def get_last_citations(self):
        return self._last_citations