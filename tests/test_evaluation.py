import pytest

from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import (
    EvaluationResult,
    OllamaRagasEmbeddings,
    OllamaRagasLLM,
    RAGASEvaluator,
)


def test_evaluation_dataset_contains_expected_questions():
    assert len(EVALUATION_DATASET) == 4

    questions = [item["question"] for item in EVALUATION_DATASET]

    assert "How many annual leave days do employees get?" in questions
    assert "How many annual leave days can be carried forward?" in questions
    assert "What is the domestic hotel reimbursement limit?" in questions
    assert "How quickly must employees submit expenses?" in questions


def test_evaluation_dataset_items_have_required_fields():
    for item in EVALUATION_DATASET:
        assert "question" in item
        assert "ground_truth" in item
        assert item["question"]
        assert item["ground_truth"]


def test_evaluation_result_defaults():
    result = EvaluationResult(
        question="Test question",
        answer="Test answer",
        ground_truth="Expected answer",
        contexts=["Test context"],
    )

    assert result.question == "Test question"
    assert result.answer == "Test answer"
    assert result.ground_truth == "Expected answer"
    assert result.contexts == ["Test context"]

    assert result.faithfulness is None
    assert result.answer_relevancy is None
    assert result.context_precision is None
    assert result.context_recall is None


def test_ollama_ragas_llm_configuration():
    llm = OllamaRagasLLM(
        model="qwen3:8b",
        base_url="http://localhost:11434",
    )

    assert llm.model == "qwen3:8b"
    assert llm.base_url == "http://localhost:11434"


def test_ollama_ragas_llm_strips_trailing_slash():
    llm = OllamaRagasLLM(
        model="qwen3:8b",
        base_url="http://localhost:11434/",
    )

    assert llm.base_url == "http://localhost:11434"


def test_ollama_ragas_embeddings_configuration():
    embeddings = OllamaRagasEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url="http://localhost:11434",
    )

    assert embeddings.provider.model == "qwen3-embedding:0.6b"
    assert embeddings.provider.base_url == "http://localhost:11434"


def test_ragas_evaluator_configuration():
    evaluator = RAGASEvaluator(
        model="qwen3:8b",
        embedding_model="qwen3-embedding:0.6b",
        base_url="http://localhost:11434",
    )

    assert evaluator.llm.model == "qwen3:8b"
    assert evaluator.llm.base_url == "http://localhost:11434"

    assert evaluator.embeddings.provider.model == "qwen3-embedding:0.6b"
    assert evaluator.embeddings.provider.base_url == "http://localhost:11434"


@pytest.mark.parametrize(
    "dataset_item",
    EVALUATION_DATASET,
)
def test_evaluation_dataset_ground_truth_is_non_empty(dataset_item):
    assert isinstance(dataset_item["ground_truth"], str)
    assert dataset_item["ground_truth"].strip()


def test_evaluator_returns_evaluation_result(monkeypatch):
    class FakeMetricResult:
        def __init__(self, value):
            self.value = value

    class FakeFaithfulness:
        def __init__(self, llm):
            assert llm is evaluator.llm

        async def ascore(self, **kwargs):
            assert kwargs["user_input"] == "Test question"
            assert kwargs["response"] == "Test answer"
            assert kwargs["retrieved_contexts"] == ["Test context"]
            return FakeMetricResult(0.90)

    class FakeAnswerRelevancy:
        def __init__(self, llm, embeddings):
            assert llm is evaluator.llm
            assert embeddings is evaluator.embeddings

        async def ascore(self, **kwargs):
            assert kwargs["user_input"] == "Test question"
            assert kwargs["response"] == "Test answer"
            return FakeMetricResult(0.85)

    class FakeContextPrecision:
        def __init__(self, llm):
            assert llm is evaluator.llm

        async def ascore(self, **kwargs):
            assert kwargs["user_input"] == "Test question"
            assert kwargs["retrieved_contexts"] == ["Test context"]
            assert kwargs["reference"] == "Expected answer"
            return FakeMetricResult(0.80)

    class FakeContextRecall:
        def __init__(self, llm):
            assert llm is evaluator.llm

        async def ascore(self, **kwargs):
            assert kwargs["user_input"] == "Test question"
            assert kwargs["retrieved_contexts"] == ["Test context"]
            assert kwargs["reference"] == "Expected answer"
            return FakeMetricResult(0.75)

    monkeypatch.setattr(
        "app.evaluation.evaluator.Faithfulness",
        FakeFaithfulness,
    )
    monkeypatch.setattr(
        "app.evaluation.evaluator.AnswerRelevancy",
        FakeAnswerRelevancy,
    )
    monkeypatch.setattr(
        "app.evaluation.evaluator.ContextPrecision",
        FakeContextPrecision,
    )
    monkeypatch.setattr(
        "app.evaluation.evaluator.ContextRecall",
        FakeContextRecall,
    )

    evaluator = RAGASEvaluator()

    result = evaluator.evaluate(
        question="Test question",
        answer="Test answer",
        contexts=["Test context"],
        ground_truth="Expected answer",
    )

    assert isinstance(result, EvaluationResult)

    assert result.question == "Test question"
    assert result.answer == "Test answer"
    assert result.ground_truth == "Expected answer"
    assert result.contexts == ["Test context"]

    assert result.faithfulness == pytest.approx(0.90)
    assert result.answer_relevancy == pytest.approx(0.85)
    assert result.context_precision == pytest.approx(0.80)
    assert result.context_recall == pytest.approx(0.75)
