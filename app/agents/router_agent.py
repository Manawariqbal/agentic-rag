from dataclasses import dataclass


@dataclass
class RouteDecision:
    route: str
    reason: str


class RouterAgent:
    """
    Decides whether a user query requires retrieval
    from the enterprise knowledge base.
    """

    def route(self, query: str) -> RouteDecision:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Temporary deterministic routing.
        # Later this will be an actual CrewAI/LLM agent.
        knowledge_keywords = {
            "policy",
            "leave",
            "attendance",
            "holiday",
            "expense",
            "travel",
            "hotel",
            "reimbursement",
            "benefit",
            "employee",
            "remote",
            "annual leave",
            "sick leave",
        }

        query_lower = query.lower()

        requires_rag = any(
            keyword in query_lower
            for keyword in knowledge_keywords
        )

        if requires_rag:
            return RouteDecision(
                route="rag",
                reason="Query requires information from the knowledge base.",
            )

        return RouteDecision(
            route="general",
            reason="Query does not appear to require the knowledge base.",
        )