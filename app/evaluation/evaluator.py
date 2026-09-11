from dataclasses import dataclass


@dataclass
class EvaluationResult:
    question: str
    answer: str
    ground_truth: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class RAGASEvaluator:

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> EvaluationResult:

        # Placeholder until Ragas is connected.

        return EvaluationResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
        )