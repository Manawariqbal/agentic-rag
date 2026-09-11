from abc import ABC, abstractmethod
from typing import Any
import re


class Reranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class SimpleReranker(Reranker):
    """
    Lightweight lexical reranker for local development.

    This is NOT the production reranker.
    It allows us to validate the reranking pipeline
    without downloading a model.
    """

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:

        query_words = self._tokenize(query)

        reranked = []

        for result in results:

            document_words = self._tokenize(
                result["text"]
            )

            if not query_words:
                lexical_score = 0.0
            else:
                overlap = (
                    query_words.intersection(document_words)
                )

                lexical_score = (
                    len(overlap) / len(query_words)
                )

            original_score = result.get(
                "score",
                0.0,
            )

            final_score = (
                0.7 * lexical_score
                + 0.3 * original_score
            )

            reranked.append(
                {
                    **result,
                    "rerank_score": final_score,
                }
            )

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:

        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
        )