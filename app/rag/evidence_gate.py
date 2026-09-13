from dataclasses import dataclass
from typing import Any


@dataclass
class EvidenceGateResult:
    sufficient: bool
    score: float
    results: list[dict[str, Any]]
    reason: str


class EvidenceGate:
    def __init__(self, threshold: float = 0.48):
        self.threshold = threshold

    def evaluate(
        self,
        reranked_results: list[dict[str, Any]],
    ) -> EvidenceGateResult:

        if not reranked_results:
            return EvidenceGateResult(
                sufficient=False,
                score=0.0,
                results=[],
                reason="No relevant evidence was retrieved.",
            )

        top_result = reranked_results[0]

        score = float(
            top_result.get(
                "rerank_score",
                0.0,
            )
        )

        sufficient = score >= self.threshold

        if sufficient:
            reason = (
                f"Top rerank score {score:.4f} is above "
                f"the evidence threshold {self.threshold:.4f}."
            )
        else:
            reason = (
                f"Top rerank score {score:.4f} is below "
                f"the evidence threshold {self.threshold:.4f}."
            )

        return EvidenceGateResult(
            sufficient=sufficient,
            score=score,
            results=reranked_results,
            reason=reason,
        )