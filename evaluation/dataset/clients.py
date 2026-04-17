import os
from langchain_mistralai import ChatMistralAI
from evaluation.dataset.models import Chunk


class RealLLMClient:
    def __init__(self):
        # Reuse the same model as in src.rag.py for consistency
        self.llm = ChatMistralAI(
            model="open-mistral-7b",  # Same model as generate_response()
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.7,  # Same temperature
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content


class RealRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 5) -> list[Chunk]:
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        return [Chunk(id=doc.metadata.get("source_id", "unknown"), text=doc.page_content) for doc in docs]

