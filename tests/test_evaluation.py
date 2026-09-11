from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import RAGASEvaluator


def main():
    evaluator = RAGASEvaluator()

    for item in EVALUATION_DATASET:
        result = evaluator.evaluate(
            question=item["question"],
            answer=item["ground_truth"],
            contexts=[item["ground_truth"]],
            ground_truth=item["ground_truth"],
        )

        print("\n" + "=" * 70)
        print(f"Question: {result.question}")
        print(f"Answer: {result.answer}")
        print(f"Ground Truth: {result.ground_truth}")
        print(f"Contexts: {result.contexts}")


if __name__ == "__main__":
    main()