from app.observability.prompts import PromptManager


SYSTEM_PROMPT = """
You are a helpful enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent information.
2. If the context does not contain the answer, say:
   "I don't have enough information in the provided documents."
3. Keep the answer concise and factual.
4. Use citation numbers such as [1], [2] to reference
   relevant sources.
5. Do not create citations that do not exist.
"""


prompt_manager = PromptManager()

prompt_manager.register(
    name="rag_answer",
    template=SYSTEM_PROMPT,
)


def build_rag_prompt(
    query: str,
    context: str,
) -> str:

    prompt = prompt_manager.get(
        name="rag_answer"
    )

    return f"""
{prompt.template}

User Question:
{query}

Retrieved Context:
{context}

Answer the question using the retrieved context.
Include citations such as [1], [2] where appropriate.
"""