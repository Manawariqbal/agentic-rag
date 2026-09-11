from crewai import Task


def create_research_task(
    research_agent,
    query: str,
):

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

        expected_output="""
Evidence retrieved from the enterprise knowledge base,
including relevant facts, source, section and citations.
""",

        agent=research_agent,
    )


def create_answer_task(
    answer_agent,
    query: str,
    research_task,
):

    return Task(
        description=f"""
Answer this user question:

{query}

Use ONLY the research result from the previous task.

Requirements:

1. Answer directly and concisely.

2. Use only information supported by the research.

3. Do not invent facts.

4. Preserve citation references such as [1] and [2].

5. Place citations immediately after the factual statement
   they support.

6. Do NOT write source names in the answer.

7. Do NOT write:
   "Source: ..."
   "(Source: ...)"
   "According to leave_and_attendance_policy.pdf..."

8. The final answer should contain only the natural-language
   answer plus citation markers.

9. If the research does not contain enough information,
   say that the available documents do not provide enough
   information.

10. Do not perform another knowledge-base search.
""",

        expected_output="""
A concise natural-language answer with citation markers.

Example:

Eligible full-time employees receive 20 days of annual leave
per calendar year. [1]

Do not include a separate Sources section.
Do not include filenames or source descriptions in the answer.
""",

        agent=answer_agent,

        context=[research_task],
    )