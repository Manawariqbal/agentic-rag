from app.memory.conversation_memory import ConversationMemory


def main():

    memory = ConversationMemory()

    conversation_id = "conversation-001"

    # User message
    memory.add_message(
        conversation_id=conversation_id,
        role="user",
        content="How many annual leave days do I get?",
    )

    # Assistant message
    memory.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=(
            "Employees are entitled to 20 days "
            "of annual leave per year."
        ),
        citations=[
            "[1] leave_and_attendance_policy.pdf — Annual Leave"
        ],
    )

    # Another user message
    memory.add_message(
        conversation_id=conversation_id,
        role="user",
        content="Can I carry unused days forward?",
    )

    messages = memory.get_messages(
        conversation_id
    )

    print("\nCONVERSATION")
    print("============")

    for message in messages:

        print(f"\nRole: {message.role}")
        print(f"Content: {message.content}")

        if message.citations:
            print("Citations:")
            for citation in message.citations:
                print(f"  {citation}")


if __name__ == "__main__":
    main()