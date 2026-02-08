from src.config import CONNECTION_STRING, embeddings
from langchain_community.vectorstores import PGVector

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