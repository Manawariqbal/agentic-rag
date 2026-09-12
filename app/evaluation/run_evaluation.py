from app.config import settings

from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import RAGASEvaluator

from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter
from app.rag.retriever import Retriever
from app.rag.reranker import SimpleReranker
from app.rag.llm import OllamaLLMProvider


def build_rag_components():

    # -------------------------
    # Embeddings
    # -------------------------
    embedding_provider = OllamaEmbeddingProvider(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    # -------------------------
    # PGVector
    # -------------------------
    vector_store = PGVectorStoreAdapter(
        embedding_provider=embedding_provider,
        table_name="rag_documents_1024",
        embed_dim=settings.embedding_dimensions,
    )

    # -------------------------
    # Retrieval
    # -------------------------
    retriever = Retriever(
        vector_store=vector_store,
        top_k=settings.retrieval_top_k,
    )

    # -------------------------
    # Reranking
    # -------------------------
    reranker = SimpleReranker()

    # -------------------------
    # Answer LLM
    # -------------------------
    llm = OllamaLLMProvider(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
    )

    return retriever, reranker, llm


def generate_answer(llm, question, contexts):

    context_text = "\n\n".join(
        f"[{index + 1}]\n{context}"
        for index, context in enumerate(contexts)
    )

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the question using ONLY the provided context.

Question:
{question}

Context:
{context_text}

Rules:
- Do not invent facts.
- If the answer is not present in the context, say so.
- Be concise and factual.
"""

    return llm.generate(prompt)


def main():

    retriever, reranker, llm = build_rag_components()

    evaluator = RAGASEvaluator(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
    )

    results = []

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)

    for index, item in enumerate(EVALUATION_DATASET, start=1):

        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{index}/{len(EVALUATION_DATASET)}]")
        print(f"Question: {question}")

        # -------------------------
        # Retrieval
        # -------------------------
        retrieved = retriever.retrieve(question)

        print(f"Retrieved: {len(retrieved)}")

        # -------------------------
        # Reranking
        # -------------------------
        reranked = reranker.rerank(
            query=question,
            results=retrieved,
            top_k=settings.rerank_top_k,
        )

        contexts = [
            result["text"]
            for result in reranked
        ]

        print(f"Reranked contexts: {len(contexts)}")

        # -------------------------
        # Generate answer
        # -------------------------
        answer = generate_answer(
            llm=llm,
            question=question,
            contexts=contexts,
        )

        print(f"Answer: {answer}")

        # -------------------------
        # RAGAS
        # -------------------------
        print("Running RAGAS metrics...")

        evaluation = evaluator.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        print(
            f"Faithfulness:      {evaluation.faithfulness}"
        )
        print(
            f"Answer Relevancy:  {evaluation.answer_relevancy}"
        )
        print(
            f"Context Precision: {evaluation.context_precision}"
        )
        print(
            f"Context Recall:    {evaluation.context_recall}"
        )

        results.append(evaluation)

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    for metric in metric_names:

        values = [
            getattr(result, metric)
            for result in results
            if getattr(result, metric) is not None
        ]

        if values:
            average = sum(values) / len(values)

            print(
                f"{metric:20s}: "
                f"{average:.4f}"
            )


if __name__ == "__main__":
    main()