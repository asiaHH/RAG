import psycopg2
import bcrypt
from typing import List, Dict, Optional


class Users:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._init_db()

    def _init_db(self):
        """
        Initializes the database by creating the users table if it doesn't exist.
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        email TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        is_admin BOOLEAN NOT NULL DEFAULT false,
                        created_at TIMESTAMP DEFAULT now()
                    );
                """)
                conn.commit()

    def create_user(self, email: str, password: str, is_admin: bool = False) -> str:
        """
        Creates a new user with the given email, password, and admin status.
        """
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (email, password_hash, is_admin)
                    VALUES (%s, %s, %s) RETURNING id;
                    """,
                    (email, password_hash, is_admin)
                )
                user_id = cur.fetchone()[0]
                conn.commit()
                return str(user_id)

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        Retrieves a user by their email address.
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, password_hash, is_admin FROM users WHERE email = %s;",
                    (email,)
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "email": row[1],
                    "password_hash": row[2],
                    "is_admin": row[3],
                }

    def verify_password(self, email: str, password: str) -> Optional[Dict]:
        """
        Check the email and password. Returns the user (without the hash) if valid, otherwise None.
        """
        user = self.get_user_by_email(email)
        if user is None:
            return None
        if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            user.pop("password_hash")
            return user
        return None

    def list_users(self) -> List[Dict]:
        """
        Reserved for admins — lists all users without exposing the hashes.
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, is_admin, created_at FROM users ORDER BY created_at DESC;"
                )
                return [
                    {"id": str(row[0]), "email": row[1], "is_admin": row[2], "created_at": row[3]}
                    for row in cur.fetchall()
                ]

    def delete_user(self, user_id: str):
        """
        Deletes a user by their ID. Reserved for admins.
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
                conn.commit()