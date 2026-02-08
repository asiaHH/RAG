from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate


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
            Tu es un assistant qui répond uniquement à partir du contexte fourni. Si tu ne connais pas la réponse, dit que tu ne sais pas.
            
            Contexte: {context}
            Question: {input}
            """)
        
        document_chain= create_stuff_documents_chain(
            llm=chat_model,
            prompt=prompt
        )
        
        retrieval_chain = create_retrieval_chain(
            retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            combine_docs_chain=document_chain
        )

        result = retrieval_chain.invoke({"input": question})
        answer = result["answer"]
        sources = result.get("context", []) 
        return {"answer": answer, "sources": sources}
    
    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise e


