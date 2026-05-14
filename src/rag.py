from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate

def get_retriever(vector_store, k: int = 5):
    """
    Returns the retriever for the RAG, reusable for evaluation.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def generate_response(vector_store, question):
    """
    Generate a response to a question using the vector store and a language model.
    :param vector_store: The vector store containing the indexed documents
    :param question: The question to be answered
    :return: A dictionary containing the answer and the sources used
    """
    try:
        chat_model = ChatMistralAI(model="open-mistral-7b", temperature=0.7)

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
        
        retrieval_chain = create_retrieval_chain(
            retriever=get_retriever(vector_store),
            combine_docs_chain=document_chain
        )

        result = retrieval_chain.invoke({"input": question})
        return {"answer": result["answer"], "sources": result.get("context", [])}
    
    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise e


