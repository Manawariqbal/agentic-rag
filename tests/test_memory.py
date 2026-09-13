import uuid

from app.memory.conversation_memory import ConversationMemory


def unique_conversation_id():
    return f"pytest-{uuid.uuid4()}"


def test_add_and_get_messages():
    memory = ConversationMemory()
    conversation_id = unique_conversation_id()

    try:
        memory.add_message(
            conversation_id=conversation_id,
            role="user",
            content="How many annual leave days do I get?",
        )

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

        messages = memory.get_messages(conversation_id)

        assert len(messages) == 2

        assert messages[0].role == "user"
        assert messages[0].content == (
            "How many annual leave days do I get?"
        )

        assert messages[1].role == "assistant"
        assert "20 days" in messages[1].content

        assert messages[1].citations == [
            "[1] leave_and_attendance_policy.pdf — Annual Leave"
        ]

    finally:
        memory.clear(conversation_id)


def test_get_recent_messages():
    memory = ConversationMemory()
    conversation_id = unique_conversation_id()

    try:
        for i in range(5):
            memory.add_message(
                conversation_id=conversation_id,
                role="user",
                content=f"Message {i}",
            )

        messages = memory.get_recent_messages(
            conversation_id=conversation_id,
            limit=3,
        )

        assert len(messages) == 3

        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"
        assert messages[2].content == "Message 4"

    finally:
        memory.clear(conversation_id)


def test_clear_conversation():
    memory = ConversationMemory()
    conversation_id = unique_conversation_id()

    memory.add_message(
        conversation_id=conversation_id,
        role="user",
        content="Test message",
    )

    messages_before = memory.get_messages(conversation_id)

    assert len(messages_before) == 1

    memory.clear(conversation_id)

    messages_after = memory.get_messages(conversation_id)

    assert messages_after == []


def test_empty_conversation():
    memory = ConversationMemory()
    conversation_id = unique_conversation_id()

    try:
        messages = memory.get_messages(conversation_id)

        assert messages == []

    finally:
        memory.clear(conversation_id)