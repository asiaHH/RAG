import os
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "gemini-2.5-pro"): 
        self.model = ChatGoogleGenerativeAI(
            model=model,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
            model_kwargs={
                "generation_config": {
                    "thinking_config": {"thinking_budget": 0},
                    "response_mime_type": "application/json"
                }
            }
        )
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        result = self.model.invoke(prompt)
        return result.content
    
    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content
    
    def get_model_name(self) -> str:
        return f"Gemini ({self.model.model})"

THRESHOLDS = {
    "faithfulness": 0.8,          
    "answer_relevancy": 0.75,    
}

GEMINI_JUDGE = GeminiJudge()

    
        
        