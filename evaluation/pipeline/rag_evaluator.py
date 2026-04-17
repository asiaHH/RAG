import json
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from evaluation.metrics.retrieval_metrics import get_retrieval_metrics
from evaluation.metrics.generation_metrics import get_generation_metrics
from src.rag import generate_response


class RAGEvaluator:
    """
    Evaluate a RAG system, from retrieval to generation, using DeepEval metrics.
    The evaluation process includes:
    1. For each question in the dataset, retrieve relevant chunks from the vector store.
    2. Generate a response based on the retrieved chunks.
    3. Evaluate the retrieval step with contextual metrics and the generation step with faithfulness and relevancy metrics.
    4. Return a comprehensive report of the evaluation results."""
    def __init__(self, vector_store: Any):
        """
        vector_store : PGVector instance avec RAG déjà indexé
        """
        self.vector_store = vector_store
        self.retrieval_metrics = get_retrieval_metrics()
        self.generation_metrics = get_generation_metrics()

    def _build_test_case(self, item: dict) -> LLMTestCase:
        """
        Pour chaque question du dataset :
        1. Lance le retriever -> récupère les chunks
        2. Lance generate_response() -> génère la réponse
        3. Emballe tout dans un LLMTestCase DeepEval
        """
        question = item["input"]

        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)
        retrieval_context = [doc.page_content for doc in retrieved_docs]

        response = generate_response(self.vector_store, question)
        actual_output = response.get("answer", "")

        return LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=item.get("expected_output", ""),
            retrieval_context=retrieval_context,
            context=retrieval_context,
        )

    def run(self, dataset_path: str | None = None) -> dict:
        """
        Lance l'évaluation complète et retourne un résumé des scores.
        """
        #Chargement du dataset
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "dataset" / "dataset.json"

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        print(f"\n Évaluation sur {len(dataset)} questions...\n")

        test_cases = []
        for item in dataset:
            print(f"  → Traitement : {item['id']} — {item['input'][:60]}...")
            try:
                tc = self._build_test_case(item)
                test_cases.append(tc)
            except Exception as e:
                print(f"  ✗ Erreur sur {item['id']}: {e}")

        all_metrics = self.retrieval_metrics + self.generation_metrics

        #modif pour pas exploser le quota gemini
        #results = evaluate(test_cases, all_metrics)
        results = evaluate(
            test_cases, 
            all_metrics, 
            run_async=False,  # Désactive le parallélisme
            throttle_value=10 # Attend 10 secondes entre chaque appel
        )
        return results