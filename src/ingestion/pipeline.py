from src.config import CONNECTION_STRING, embeddings, PSYCOPG2_CONNECTION_STRING
from langchain_community.vectorstores import PGVector
import psycopg2

vector_store = None

def init_vector_store():
    """
    Initialize and return a PGVector instance.
    """
    global vector_store
    try:
        vector_store = PGVector(
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
            collection_name="test_collection",
            pre_delete_collection=False
        )
        print("Connexion PGVector successful")
        return vector_store
    except Exception as e:
        print(f"Error PGVector: {e}")
        return None

def clear_user_collection(user_id: str) -> bool:
    """
    Delete only the documents and catalog entries belonging to a given user.
    """
    global vector_store
    
    try:
        with psycopg2.connect(PSYCOPG2_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_catalog WHERE user_id = %s;", (user_id,))
                cur.execute(
                    "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'user_id' = %s;",
                    (user_id,)
                )
                conn.commit()
        print(f"Documents de l'utilisateur {user_id} supprimés.")
        return True
    except Exception as e:
        print(f"Erreur lors du vidage du catalogue: {e}")
        return False