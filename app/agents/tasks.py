from crewai import Task


def create_research_task(
    research_agent,
    query: str,
):

    return Task(
        description=f"""
Research the following user question using the
enterprise knowledge base:

User Question:
{query}

Instructions:

1. Use the knowledge_base_search tool.
2. Retrieve information relevant to the question.
3. Prefer information directly supported by the documents.
4. Preserve source and section information.
5. Do not invent facts.
6. If the knowledge base does not contain enough information,
   clearly state that.
""",

        expected_output="""
A concise research result containing:

- Relevant facts
- Supporting document information
- Section information
- Citation references
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
Answer the user's question:

{query}

Use the research produced by the research agent.

Instructions:

1. Use only information supported by the research.
2. Do not invent facts.
3. Give a concise and useful answer.
4. Preserve citation references such as [1], [2].
5. If the research does not contain enough information,
   say that the available documents do not provide enough
   information.
""",

        expected_output="""
A clear final answer to the user's question with
appropriate citation references.
""",

        agent=answer_agent,

        context=[research_task],
    )