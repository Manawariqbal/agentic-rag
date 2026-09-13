from app.agents.router_agent import RouterAgent


def test_router_routes_annual_leave_question_to_rag():
    router = RouterAgent()

    decision = router.route(
        "How many annual leave days do employees get?"
    )

    assert decision.route == "rag"


def test_router_routes_travel_policy_question_to_rag():
    router = RouterAgent()

    decision = router.route(
        "What is the hotel reimbursement limit?"
    )

    assert decision.route == "rag"


def test_router_routes_general_question_to_general():
    router = RouterAgent()

    decision = router.route(
        "What is the capital of France?"
    )

    assert decision.route == "general"


def test_router_returns_reason():
    router = RouterAgent()

    decision = router.route(
        "How many annual leave days do employees get?"
    )

    assert decision.reason
    assert isinstance(decision.reason, str)


def test_router_handles_leave_keywords():
    router = RouterAgent()

    queries = [
        "How many sick leaves are available?",
        "What is the casual leave policy?",
        "Can unused leave be carried forward?",
        "What is the attendance policy?",
    ]

    for query in queries:
        decision = router.route(query)

        assert decision.route == "rag"


def test_router_handles_travel_and_expense_keywords():
    router = RouterAgent()

    queries = [
        "What is the domestic hotel limit?",
        "How much can I claim for meals?",
        "What is the travel reimbursement policy?",
        "When should I submit expenses?",
    ]

    for query in queries:
        decision = router.route(query)

        assert decision.route == "rag"


def test_router_handles_general_questions():
    router = RouterAgent()

    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is the capital of France?",
    ]

    for query in queries:
        decision = router.route(query)

        assert decision.route == "general"