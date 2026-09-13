from crewai import Task

from app.rag.rag_prompt import get_rag_prompt


def create_research_task(
    research_agent,
    query,
    retrieved_evidence="",
):
    """
    Create the Research Task.

    The deterministic retrieval pipeline in ChatService performs
    retrieval, reranking, and evidence gating before CrewAI runs.

    When retrieved_evidence is provided, the Research Agent must
    use that evidence as its source of truth instead of performing
    an independent knowledge-base search.
    """

    if retrieved_evidence:
        research_description = f"""
Review the retrieved evidence below for the following user question.

User Question:
{query}

================ RETRIEVED EVIDENCE ================

{retrieved_evidence}

======================================================

Your job is to extract the answer from the retrieved evidence.

IMPORTANT RULES:
- Use ONLY the retrieved evidence provided above.
- Do NOT perform another knowledge-base search.
- Do NOT use your own knowledge.
- Do NOT invent facts.
- Do NOT invent document names.
- Do NOT invent section names.
- Do NOT invent page numbers.
- Do NOT invent citations.
- If the retrieved evidence does not contain the answer, explicitly
  state that the evidence is insufficient.

Return:
- relevant facts
- source document
- section
- citation references
"""
    else:
        # Backward-compatible fallback.
        research_description = f"""
Retrieve evidence for the following user question:

User Question:
{query}

Use the knowledge_base_search tool.

Return the retrieved evidence with:
- relevant facts
- source document
- section
- citation references

Do not invent information.
"""

    return Task(
        description=research_description,
        expected_output=(
            "Evidence retrieved from the enterprise knowledge base, "
            "including relevant facts, source, section and citations."
        ),
        agent=research_agent,
    )


def create_answer_task(answer_agent, query, research_task):
    """
    Create the Answer Task using the prompt managed by Phoenix.
    """

    # Retrieve the current prompt from Phoenix.
    phoenix_prompt = get_rag_prompt()

    # Phoenix SDK 3.5.0 stores the chat template in _template.
    messages = phoenix_prompt._template["messages"]

    # Extract the system instructions.
    system_message = next(
        (
            message["content"]
            for message in messages
            if message["role"] == "system"
        ),
        None,
    )

    if system_message is None:
        raise ValueError(
            "Phoenix RAG prompt does not contain a system message."
        )

    return Task(
        description=f"""
Follow the Phoenix-managed RAG answer instructions below.

================ PHOENIX MANAGED PROMPT ================

{system_message}

=========================================================

User Question:
{query}

The previous research task contains the retrieved evidence.

Use ONLY the research result from the previous task as your
retrieved context.

Important:
- Do not perform another knowledge-base search.
- Produce only the final natural-language answer.
- Preserve valid citation markers such as [1] and [2].
- Do not output filenames.
- Do not output a separate Sources section.
""",
        expected_output="""
A concise natural-language answer with citation markers.

Example:

Eligible full-time employees receive 20 days of annual leave
per calendar year. [1]

Do not include a separate Sources section.
Do not include filenames.
""",
        agent=answer_agent,
        context=[research_task],
    )