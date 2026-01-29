import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector

load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@localhost:5432/vector_db"
embeddings = MistralAIEmbeddings(model="mistral-embed")

try:
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="test_collection"
    )
    print("Connexion automatique à la base vectorielle réussie.")
except Exception as e:
    vector_store = None
    print(f"Base vide ou non trouvée : {e}")

# Cette fonction sera appelée par l'API pour "nourrir" la base
def ingest_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name="test_collection"
    )
    return vector_store

# Cette fonction sera appelée par l'API pour répondre
def generate_response(vector_store, question):
    try:
        chat_model = ChatMistralAI(model="open-mistral-7b", temperature=0.7)

        prompt=ChatPromptTemplate.from_template("""
            Tu es un assistant qui répond uniquement à partir du contexte fourni.
            
            Contexte: {context}
            Question: {input}
            """)
        document_chain= create_stuff_documents_chain(
            llm=chat_model,
            prompt=prompt
        )
        retrieval_chain = create_retrieval_chain(
            retriever=vector_store.as_retriever(),
            combine_docs_chain=document_chain
        )

        result = retrieval_chain.invoke({"input": question})
        return result["answer"]
    except Exception as e:
        print(f"Erreur dans generate_response: {e}")
        raise e

# Ce bloc ne s'exécute QUE si tu lances "python rag.py" directement
if __name__ == "__main__":
    v_store = ingest_pdf("data/rag_pdf.pdf")
    res = generate_response(v_store, "Fais un résumé court.")
    print(res)

"""
Un user envoie un pdf via streamlit, streamlit envoie le fichier
à l'api fastapi, fastapi appelle ingest_pdf de rag.py et rag;py decoupe
le pdf, demande les embeddings a mistral et stocke dans docker. 
"""