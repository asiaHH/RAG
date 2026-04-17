import os
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = ChatGoogleGenerativeAI(
            model=model,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
        )
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content
    
    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content
    
    def get_model_name(self) -> str:
        return f"Gemini ({self.model.model})"

THRESHOLDS = {
    # Retrieval
    "contextual_precision": 0.7,   # les chunks récupérés sont-ils pertinents ?
    "contextual_recall": 0.7,      # les chunks couvrent-ils la réponse attendue ?
    "contextual_relevancy": 0.6,   # chaque chunk est-il utile à la question ?
    # Génération
    "faithfulness": 0.8,           # la réponse ne sort-elle pas des chunks ?
    "answer_relevancy": 0.75,      # la réponse répond-elle à la question ?
}

GEMINI_JUDGE = GeminiJudge()

    
        
        