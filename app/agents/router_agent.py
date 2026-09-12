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
            "casual leave",
            "carry forward",
            "carry-forward",
            "carry",
            "entitlement",
            "approval",
        }

        query_lower = query.lower()
        context_lower = conversation_context.lower()

        # Check the current question.
        current_requires_rag = any(
            keyword in query_lower
            for keyword in knowledge_keywords
        )

        if current_requires_rag:
            return RouteDecision(
                route="rag",
                reason="Query requires information from the knowledge base.",
            )

        # ------------------------------------------------------------------
        # Follow-up question detection.
        #
        # Questions containing references such as:
        #   those
        #   that
        #   it
        #   they
        #   them
        #   these
        #   previous
        #
        # may depend on earlier conversation context.
        # ------------------------------------------------------------------

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

        contains_follow_up_reference = any(
            re_word in query_lower.split()
            for re_word in follow_up_terms
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

        return RouteDecision(
            route="general",
            reason="Query does not appear to require the knowledge base.",
        )