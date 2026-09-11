from dataclasses import dataclass

from app.rag.citations import CitationManager
from app.rag.prompts import build_rag_prompt
from app.rag.llm import LLMProvider


@dataclass
class AnswerResponse:
    answer: str
    citations: list


class AnswerAgent:

    def __init__(
        self,
        llm: LLMProvider,
        citation_manager: CitationManager,
    ):
        self.llm = llm
        self.citation_manager = citation_manager

    def answer(
        self,
        query: str,
        results: list[dict],
    ) -> AnswerResponse:

        context = self._build_context(results)

        prompt = build_rag_prompt(
            query=query,
            context=context,
        )

        answer = self.llm.generate(prompt)

        citations = self.citation_manager.build_citations(results)

        return AnswerResponse(
            answer=answer,
            citations=citations,
        )

    @staticmethod
    def _build_context(results: list[dict]) -> str:

        if not results:
            return "No relevant documents were retrieved."

        context_parts = []

        for index, result in enumerate(results, start=1):

            metadata = result.get("metadata", {})

            source = metadata.get(
                "source",
                "Unknown source",
            )

            section = metadata.get(
                "section",
                "Unknown section",
            )

            text = result.get("text", "")

            context_parts.append(
                f"""
[{index}]
Source: {source}
Section: {section}

{text}
"""
            )

        return "\n".join(context_parts)