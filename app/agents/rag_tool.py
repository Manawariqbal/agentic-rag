from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.observability.phoenix import get_tracer
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

        tracer = get_tracer()

        with tracer.start_as_current_span("rag.retrieval") as span:

            span.set_attribute("rag.query", query)
            span.set_attribute("rag.retrieval_top_k", 10)
            span.set_attribute(
                "rag.rerank_top_k",
                self._rerank_top_k,
            )

            # -------------------------
            # Retrieval
            # -------------------------
            retrieved_results = self._retriever.retrieve(query)

            span.set_attribute(
                "rag.retrieved_count",
                len(retrieved_results),
            )

            if not retrieved_results:

                self._last_citations = []

                span.set_attribute(
                    "rag.result",
                    "no_results",
                )

                return "No relevant information was found."

            # -------------------------
            # Reranking
            # -------------------------
            reranked_results = self._reranker.rerank(
                query=query,
                results=retrieved_results,
                top_k=self._rerank_top_k,
            )

            span.set_attribute(
                "rag.reranked_count",
                len(reranked_results),
            )

            if reranked_results:

                span.set_attribute(
                    "rag.top_score",
                    float(
                        reranked_results[0].get(
                            "rerank_score",
                            0.0,
                        )
                    ),
                )

            # -------------------------
            # Result metadata
            # -------------------------
            for index, result in enumerate(reranked_results):

                metadata = result.get("metadata", {})

                span.set_attribute(
                    f"rag.result.{index}.source",
                    metadata.get("source", ""),
                )

                span.set_attribute(
                    f"rag.result.{index}.section",
                    metadata.get("section", ""),
                )

                span.set_attribute(
                    f"rag.result.{index}.score",
                    float(result.get("score", 0.0)),
                )

                span.set_attribute(
                    f"rag.result.{index}.rerank_score",
                    float(
                        result.get(
                            "rerank_score",
                            0.0,
                        )
                    ),
                )

            # -------------------------
            # Citations
            # -------------------------
            citations = self._citation_manager.build_citations(
                reranked_results
            )

            self._last_citations = citations

            span.set_attribute(
                "rag.citation_count",
                len(citations),
            )

            # -------------------------
            # Existing response logic
            # -------------------------
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

            span.set_attribute(
                "rag.result",
                "success",
            )

            return "\n".join(response_parts)

    def get_last_citations(self):
        return self._last_citations