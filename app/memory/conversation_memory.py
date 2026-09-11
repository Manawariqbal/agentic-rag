from app.memory.models import Conversation, Message


class ConversationMemory:

    def __init__(self):
        self.conversations: dict[str, Conversation] = {}

    def create_conversation(
        self,
        conversation_id: str,
    ) -> Conversation:

        if conversation_id in self.conversations:
            return self.conversations[conversation_id]

        conversation = Conversation(
            conversation_id=conversation_id
        )

        self.conversations[conversation_id] = conversation

        return conversation

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[str] | None = None,
    ):

        conversation = self.create_conversation(
            conversation_id
        )

        message = Message(
            role=role,
            content=content,
            citations=citations or [],
        )

        conversation.messages.append(message)

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[Message]:

        conversation = self.conversations.get(
            conversation_id
        )

        if conversation is None:
            return []

        return conversation.messages

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> list[Message]:

        messages = self.get_messages(
            conversation_id
        )

        return messages[-limit:]

    def clear(
        self,
        conversation_id: str,
    ):

        self.conversations.pop(
            conversation_id,
            None,
        )