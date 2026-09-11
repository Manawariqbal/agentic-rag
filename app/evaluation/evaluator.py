from dataclasses import dataclass

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)


@dataclass
class EvaluationResult:
    question: str
    answer: str
    ground_truth: str
    contexts: list[str]

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

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )

        dataset = [sample]

        result = evaluate(
            dataset,
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ],
        )

        scores = result.to_pandas().iloc[0]

        return EvaluationResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("response_relevancy"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
        )