from app.observability.prompts import PromptManager


PROMPT_NAME = "rag_answer"


RAG_PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": """
You are a helpful enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent facts.
2. If the context does not contain the answer, clearly say that the available documents do not provide enough information.
3. Be concise and factual.
4. Use [1], [2], etc. for citations.
5. Never create a citation that does not exist.
6. Do not mention internal retrieval mechanics.
""",
    },
    {
        "role": "user",
        "content": """
User Question:
{{query}}

Retrieved Context:
{{context}}

Answer using the retrieved context.
Include citations [1], [2] where appropriate.
""",
    },
]


def create_rag_prompt():
    """
    Create a new version of the RAG answer prompt in Phoenix.
    """

    manager = PromptManager()

    return manager.create(
        name=PROMPT_NAME,
        messages=RAG_PROMPT_MESSAGES,
        model_name="qwen3:8b",
        description="Enterprise Agentic RAG answer prompt",
    )


def get_rag_prompt():
    """
    Retrieve the current RAG answer prompt from Phoenix.
    """

    manager = PromptManager()

    return manager.get(
        name=PROMPT_NAME
    )


def format_rag_prompt(
    query: str,
    context: str,
):
    """
    Retrieve the Phoenix-managed prompt and
    inject the runtime query and retrieved context.
    """

    manager = PromptManager()

    prompt = manager.get(
        name=PROMPT_NAME
    )

    return manager.format(
        prompt,
        {
            "query": query,
            "context": context,
        },
    )