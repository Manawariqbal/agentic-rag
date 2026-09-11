from app.agents.router_agent import RouterAgent


def main():

    router = RouterAgent()

    queries = [
        "How many annual leave days do employees get?",
        "What is the hotel reimbursement limit?",
        "What is the capital of France?",
    ]

    for query in queries:

        decision = router.route(query)

        print(f"\nQuery: {query}")
        print(f"Route: {decision.route}")
        print(f"Reason: {decision.reason}")


if __name__ == "__main__":
    main()