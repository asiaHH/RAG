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

def clear_collection():
    """
    Completely empty the vector collection and the documents catalog.
    Deletes all documents, embeddings and entries from the catalog.
    """
    global vector_store
    
    # empty the document_catalog table
    try:
        with psycopg2.connect(PSYCOPG2_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_catalog;")
                conn.commit()
        print("Catalogue des documents vidé.")
    except Exception as e:
        print(f"Erreur lors du vidage du catalogue: {e}")
        return False
    
    # empty the PGVector tables
    if vector_store is None:
        vector_store = init_vector_store()
    
    try:
        # delete all documents from the collection
        vector_store.delete_collection()
        print("Collection vectorielle vidée.")
        
        # reinitialize the instance to force recreation
        vector_store = None
        return True
    except Exception as e:
        print(f"Erreur lors du vidage de la collection vectorielle: {e}")
        return False