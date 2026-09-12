# app/evaluation/evaluator.py

from dataclasses import dataclass
import asyncio
import json

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from app.rag.embedding import OllamaEmbeddingProvider


# ============================================================
# EVALUATION RESULT
# ============================================================

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


# ============================================================
# OLLAMA RAGAS LLM ADAPTER
# ============================================================

class OllamaRagasLLM(InstructorBaseRagasLLM):

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 600.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --------------------------------------------------------
    # Build prompt for structured output
    # --------------------------------------------------------

    def _build_prompt(
        self,
        prompt: str,
        response_model,
    ) -> str:

        schema = response_model.model_json_schema()

        return f"""
You are an evaluation component.

Follow the instruction below.

{prompt}

IMPORTANT OUTPUT RULES:

1. Return ONLY one valid JSON object.
2. Do NOT use Markdown.
3. Do NOT use ```json.
4. Do NOT include explanations before or after the JSON.
5. Do NOT include reasoning.
6. Do NOT include <think>...</think>.
7. Do NOT include any text outside the JSON object.
8. The JSON must conform exactly to this schema:

{json.dumps(schema, indent=2)}

Return ONLY the JSON object.
"""

    # --------------------------------------------------------
    # Extract JSON from Ollama output
    # --------------------------------------------------------

    def _extract_json(
        self,
        content: str,
    ) -> str:

        if not content:

            raise ValueError(
                "Ollama returned an empty response."
            )

        content = content.strip()

        # ----------------------------------------------------
        # Remove Qwen thinking block
        # ----------------------------------------------------

        if "<think>" in content:

            if "</think>" in content:

                content = content.split(
                    "</think>",
                    1,
                )[1].strip()

            else:

                content = content.split(
                    "<think>",
                    1,
                )[0].strip()

        # ----------------------------------------------------
        # Remove markdown fences
        # ----------------------------------------------------

        if content.startswith("```"):

            lines = content.splitlines()

            # Remove opening ```
            if lines:
                lines = lines[1:]

            # Remove closing ```
            if (
                lines
                and lines[-1].strip().startswith("```")
            ):
                lines = lines[:-1]

            content = "\n".join(
                lines
            ).strip()

        # ----------------------------------------------------
        # Find JSON object inside response
        # ----------------------------------------------------

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1 and end > start:

            content = content[
                start:end + 1
            ]

        return content.strip()

    # --------------------------------------------------------
    # Generate structured response
    # --------------------------------------------------------

    def _generate_sync(
        self,
        prompt,
        response_model,
    ):

        import httpx

        schema = response_model.model_json_schema()

        formatted_prompt = self._build_prompt(
            prompt,
            response_model,
        )

        # ----------------------------------------------------
        # Ollama API
        # ----------------------------------------------------

        response = httpx.post(
            f"{self.base_url}/api/chat",

            json={
                "model": self.model,

                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt,
                    }
                ],

                "stream": False,

                # Ollama structured output
                "format": schema,

                "options": {
                    # Deterministic judge
                    "temperature": 0,

                    # Enough tokens for RAGAS responses
                    "num_predict": 2048,
                },

                # Disable Qwen3 reasoning
                # so we get clean JSON
                "think": False,
            },

            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        # ----------------------------------------------------
        # Extract message
        # ----------------------------------------------------

        message = data.get(
            "message",
            {},
        )

        content = message.get(
            "content",
            "",
        )

        # ----------------------------------------------------
        # Empty response
        # ----------------------------------------------------

        if not content:

            raise RuntimeError(
                "Ollama returned an empty content field.\n\n"
                "Full response:\n"
                f"{json.dumps(data, indent=2)}"
            )

        # ----------------------------------------------------
        # Extract JSON
        # ----------------------------------------------------

        json_content = self._extract_json(
            content
        )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            parsed = json.loads(
                json_content
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Ollama returned invalid JSON.\n\n"

                "RAW RESPONSE:\n"
                f"{content}\n\n"

                "EXTRACTED JSON:\n"
                f"{json_content}\n\n"

                "EXPECTED SCHEMA:\n"
                f"{json.dumps(schema, indent=2)}"
            ) from exc

        # ----------------------------------------------------
        # Validate response model
        # ----------------------------------------------------

        try:

            return response_model.model_validate(
                parsed
            )

        except Exception as exc:

            raise RuntimeError(
                "Ollama JSON does not match the "
                "expected RAGAS response model.\n\n"

                "JSON:\n"
                f"{json.dumps(parsed, indent=2)}"
            ) from exc

    # --------------------------------------------------------
    # Synchronous RAGAS interface
    # --------------------------------------------------------

    def generate(
        self,
        prompt,
        response_model,
    ):

        return self._generate_sync(
            prompt,
            response_model,
        )

    # --------------------------------------------------------
    # Asynchronous RAGAS interface
    # --------------------------------------------------------

    async def agenerate(
        self,
        prompt,
        response_model,
    ):

        return await asyncio.to_thread(
            self._generate_sync,
            prompt,
            response_model,
        )


# ============================================================
# OLLAMA RAGAS EMBEDDING ADAPTER
# ============================================================

class OllamaRagasEmbeddings(BaseRagasEmbedding):

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
    ):

        super().__init__()

        self.provider = OllamaEmbeddingProvider(
            model=model,
            base_url=base_url,
        )

    # --------------------------------------------------------
    # Modern RAGAS embedding interface
    # --------------------------------------------------------

    def embed_text(
        self,
        text: str,
        **kwargs,
    ) -> list[float]:

        return self.provider.embed(
            text
        )

    # --------------------------------------------------------
    # Async embedding
    # --------------------------------------------------------

    async def aembed_text(
        self,
        text: str,
        **kwargs,
    ) -> list[float]:

        return await asyncio.to_thread(
            self.provider.embed,
            text,
        )


# ============================================================
# RAGAS EVALUATOR
# ============================================================

class RAGASEvaluator:

    def __init__(
        self,
        model: str = "qwen3:8b",
        embedding_model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
    ):

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.llm = OllamaRagasLLM(
            model=model,
            base_url=base_url,
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.embeddings = OllamaRagasEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )

    # ========================================================
    # ASYNC EVALUATION
    # ========================================================

    async def _evaluate_async(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ):

        # ====================================================
        # 1. FAITHFULNESS
        # ====================================================

        faithfulness_metric = Faithfulness(
            llm=self.llm,
        )

        faithfulness_result = (
            await faithfulness_metric.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
        )

        faithfulness_score = float(
            faithfulness_result.value
        )

        # ====================================================
        # 2. ANSWER RELEVANCY
        # ====================================================

        answer_relevancy_metric = AnswerRelevancy(
            llm=self.llm,
            embeddings=self.embeddings,
        )

        answer_relevancy_result = (
            await answer_relevancy_metric.ascore(
                user_input=question,
                response=answer,
            )
        )

        answer_relevancy_score = float(
            answer_relevancy_result.value
        )

        # ====================================================
        # 3. CONTEXT PRECISION
        # ====================================================

        context_precision_metric = ContextPrecision(
            llm=self.llm,
        )

        context_precision_result = (
            await context_precision_metric.ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

        context_precision_score = float(
            context_precision_result.value
        )

        # ====================================================
        # 4. CONTEXT RECALL
        # ====================================================

        context_recall_metric = ContextRecall(
            llm=self.llm,
        )

        context_recall_result = (
            await context_recall_metric.ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

        context_recall_score = float(
            context_recall_result.value
        )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return EvaluationResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,

            faithfulness=faithfulness_score,

            answer_relevancy=answer_relevancy_score,

            context_precision=context_precision_score,

            context_recall=context_recall_score,
        )

    # ========================================================
    # PUBLIC SYNC METHOD
    # ========================================================

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ):

        return asyncio.run(
            self._evaluate_async(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        )