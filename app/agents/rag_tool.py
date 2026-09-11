from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.rag.citations import CitationManager
from app.rag.reranker import SimpleReranker
from app.rag.retriever import Retriever


class RAGToolInput(BaseModel):
    query: str = Field(
        ...,
        description="The user's question to search in the knowledge base.",
    )


class RAGTool(BaseTool):
    name: str = "knowledge_base_search"

    description: str = (
        "Search the enterprise knowledge base and return "
        "the most relevant documents with citations."
    )

    args_schema: type[BaseModel] = RAGToolInput

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

    def _run(self, query: str) -> str:

        # 1. Retrieve candidates
        retrieved_results = self._retriever.retrieve(query)

        if not retrieved_results:
            return "No relevant information was found."

        # 2. Rerank candidates
        reranked_results = self._reranker.rerank(
            query=query,
            results=retrieved_results,
            top_k=self._rerank_top_k,
        )

        # 3. Build citations
        citations = self._citation_manager.build_citations(
            reranked_results
        )

        # 4. Build tool response
        response_parts = []

        for index, result in enumerate(
            reranked_results,
            start=1,
        ):
            metadata = result.get("metadata", {})

            response_parts.append(
                f"""
[{index}]
Source: {metadata.get("source", "Unknown")}
Section: {metadata.get("section", "Unknown")}
Score: {result.get("rerank_score", 0.0):.4f}

Content:
{result.get("text", "")}
"""
            )

        response_parts.append(
            "\nSources:\n"
            + self._citation_manager.format_citations(citations)
        )

        return "\n".join(response_parts)