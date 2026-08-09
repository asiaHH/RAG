from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
import psycopg2
import re
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from src.config import PSYCOPG2_CONNECTION_STRING

def get_retriever(vector_store, k: int = 5):
    """
    Returns the retriever for the RAG, reusable for evaluation.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


class PostgresBM25Retriever(BaseRetriever):
    """
    Ask directly the BM25 index (pg_search) created on langchain_pg_embedding.
    No corpus in memory: the lexical search is done on the Postgres side.
    """
    k: int = 5

    @staticmethod
    def _sanitize_query(query: str) -> str:
        # Some retrievers may wrap the query in a field-specifier like "document:(...)",
        # and PostgreSQL's BM25 parser rejects punctuation-heavy strings.
        if isinstance(query, str) and query.lower().startswith("document:"):
            query = query.split(":", 1)[1]

        query = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿœŒæÆ\s]+", " ", query)
        return " ".join(query.split())

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        sanitized_query = self._sanitize_query(query)

        with psycopg2.connect(PSYCOPG2_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document, cmetadata, paradedb.score(uuid) AS score
                    FROM langchain_pg_embedding
                    WHERE document @@@ %s
                    ORDER BY score DESC
                    LIMIT %s;
                    """,
                    (sanitized_query, self.k)
                )
                rows = cur.fetchall()

        return [
            Document(page_content=row[0], metadata=row[1] or {})
            for row in rows
        ]


def get_hybrid_retriever(vector_store, k: int = 5):
    """
    Hybrid retriever : BM25 (pg_search, in database) + vector similarity (pgvector),
    merged by RRF via EnsembleRetriever.
    """
    semantic_retriever = get_retriever(vector_store, k=k)
    bm25_retriever = PostgresBM25Retriever(k=k)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.0, 1.0],  # à ajuster selon l'éval
    )

def generate_response(vector_store, question, retriever=None):
    """
    Generate a response to a question using the vector store and a language model.
    :param vector_store: The vector store containing the indexed documents
    :param question: The question to be answered
    :param retriever: The retriever to use (optional)
    :return: A dictionary containing the answer and the sources used
    """
    try:
        chat_model = ChatMistralAI(model="open-mistral-7b", temperature=0.2)

        prompt=ChatPromptTemplate.from_template("""
        Tu es un assistant qui répond à partir des documents fournis.
        Réponds toujours si l'information peut être déduite raisonnablement du contexte.
        Si tu es certain qu'elle n'est pas présente, dis "Information non disponible".
            
            Contexte: {context}
            Question: {input}
            """)
        
        document_chain= create_stuff_documents_chain(
            llm=chat_model,
            prompt=prompt
        )

        active_retriever = retriever if retriever is not None else get_hybrid_retriever(vector_store)

        
        retrieval_chain = create_retrieval_chain(
             retriever=active_retriever,
            combine_docs_chain=document_chain
        )

        result = retrieval_chain.invoke({"input": question})
        return {"answer": result["answer"], "sources": result.get("context", [])}
    
    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise e


