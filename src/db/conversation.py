from psycopg2.extras import Json
from typing import List, Dict, Optional
from src.db.connection import get_scoped_connection


class Conversation:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string


    def create_conversation(self, user_id: str, title: str = "Nouvelle discussion") -> str:
        with get_scoped_connection(self.connection_string, user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id;",
                    (user_id, title)
                )
                conv_id = cur.fetchone()[0]
                conn.commit()
                return str(conv_id)

    def add_message(self, conversation_id: str, role: str, content: str, sources: Optional[list] = None):
        with get_scoped_connection(self.connection_string, user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, sources)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (conversation_id, role, content, Json(sources) if sources else None)
                )
                conn.commit()

    def list_conversations(self, user_id: str) -> List[Dict]:
        with get_scoped_connection(self.connection_string, user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title, created_at FROM conversations WHERE user_id = %s ORDER BY created_at DESC;",
                    (user_id,)
                )
                return [
                    {"id": str(row[0]), "title": row[1], "created_at": row[2]}
                    for row in cur.fetchall()
                ]

    def load_messages(self, conversation_id: str, user_id: str) -> List[Dict]:
        with get_scoped_connection(self.connection_string, user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT m.role, m.content, m.sources FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.conversation_id = %s AND c.user_id = %s
                    ORDER BY m.created_at;
                    """,
                    (conversation_id, user_id)
                )
                return [
                    {"role": row[0], "content": row[1], "sources": row[2] or []}
                    for row in cur.fetchall()
                ]

    def conversation_belongs_to_user(self, conversation_id: str, user_id: str) -> bool:
        with get_scoped_connection(self.connection_string, user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s;",
                    (conversation_id, user_id)
                )
                return cur.fetchone() is not None

    def delete_conversation(self, conversation_id: str, user_id: str):
        with get_scoped_connection(self.connection_string, user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversations WHERE id = %s AND user_id = %s;",
                    (conversation_id, user_id)
                )
                conn.commit()