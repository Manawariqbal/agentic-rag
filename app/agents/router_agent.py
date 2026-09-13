from dataclasses import dataclass


@dataclass
class RouteDecision:
    route: str
    reason: str


class RouterAgent:
    """
    Decides whether a user query requires retrieval
    from the enterprise knowledge base.

    The router is conversation-aware so follow-up questions
    such as "Can I carry some of those days?" can use the
    previous conversation context.
    """

    def route(
        self,
        query: str,
        conversation_context: str = "",
    ) -> RouteDecision:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        knowledge_keywords = {
            # General enterprise / employee terms
            "policy",
            "employee",
            "benefit",
            "responsibilities",

            # Leave and attendance
            "leave",
            "attendance",
            "holiday",
            "annual leave",
            "sick leave",
            "casual leave",
            "carry forward",
            "carry-forward",
            "carry",
            "carried forward",
            "carryover",
            "unused days",
            "unused leave",
            "entitlement",
            "approval",

            # Travel and expenses
            "expense",
            "expenses",
            "travel",
            "hotel",
            "reimbursement",
            "meal",
            "meals",
            "claim",
            "domestic",
            "economy",
            "business class",

            # Employee handbook
            "remote",
            "remote work",
            "security",
            "performance",
            "conduct",
            "working hours",
            "work hours",
        }

        query_lower = query.lower()
        context_lower = conversation_context.lower()

        # ---------------------------------------------------------
        # Check the current question.
        # ---------------------------------------------------------

        current_requires_rag = any(
            keyword in query_lower
            for keyword in knowledge_keywords
        )

        if current_requires_rag:
            return RouteDecision(
                route="rag",
                reason="Query requires information from the knowledge base.",
            )

        # ---------------------------------------------------------
        # Follow-up question detection.
        # ---------------------------------------------------------

        follow_up_terms = {
            "those",
            "that",
            "it",
            "they",
            "them",
            "these",
            "this",
            "same",
            "previous",
            "above",
            "earlier",
        }

        query_words = set(
            query_lower.replace("?", "").replace(",", "").split()
        )

        contains_follow_up_reference = bool(
            query_words.intersection(follow_up_terms)
        )

        context_requires_rag = any(
            keyword in context_lower
            for keyword in knowledge_keywords
        )

        if contains_follow_up_reference and context_requires_rag:
            return RouteDecision(
                route="rag",
                reason=(
                    "Follow-up question depends on enterprise "
                    "knowledge from the conversation context."
                ),
            )

        # ---------------------------------------------------------
        # General question.
        # ---------------------------------------------------------

        return RouteDecision(
            route="general",
            reason="Query does not appear to require the knowledge base.",
        )