from crewai import Task

from app.rag.rag_prompt import get_rag_prompt


def create_research_task(research_agent, query):
    return Task(
        description=f"""
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
""",
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