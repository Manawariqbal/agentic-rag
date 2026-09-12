import json

from app.memory.database import get_connection
from app.memory.models import Conversation, Message


class ConversationMemory:

    def create_conversation(
        self,
        conversation_id: str,
    ) -> Conversation:

        with get_connection() as connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    INSERT INTO conversations (conversation_id)
                    VALUES (%s)
                    ON CONFLICT (conversation_id)
                    DO NOTHING;
                    """,
                    (conversation_id,),
                )

            connection.commit()

        return Conversation(
            conversation_id=conversation_id
        )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[str] | None = None,
    ):

        # Make sure the conversation exists.
        self.create_conversation(
            conversation_id
        )

        citations = citations or []

        with get_connection() as connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    INSERT INTO messages (
                        conversation_id,
                        role,
                        content,
                        citations
                    )
                    VALUES (%s, %s, %s, %s::jsonb);
                    """,
                    (
                        conversation_id,
                        role,
                        content,
                        json.dumps(citations),
                    ),
                )

            connection.commit()

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[Message]:

        with get_connection() as connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        role,
                        content,
                        citations,
                        timestamp
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY timestamp ASC, id ASC;
                    """,
                    (conversation_id,),
                )

                rows = cursor.fetchall()

        return [
            Message(
                role=row[0],
                content=row[1],
                citations=row[2] or [],
                timestamp=row[3],
            )
            for row in rows
        ]

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> list[Message]:

        if limit <= 0:
            return []

        messages = self.get_messages(
            conversation_id
        )

        return messages[-limit:]

    def clear(
        self,
        conversation_id: str,
    ):

        with get_connection() as connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    DELETE FROM conversations
                    WHERE conversation_id = %s;
                    """,
                    (conversation_id,),
                )

            connection.commit()