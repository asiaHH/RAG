import json
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from evaluation.metrics.generation_metrics import get_generation_metrics
from src.rag import generate_response


class GenerationEvaluator:
    """
    Évalue la génération du système RAG avec DeepEval + Gemini.
    
    Le processus d'évaluation :
    1. Pour chaque question du dataset, récupérer les chunks pertinents du vector store.
    2. Générer une réponse basée sur les chunks récupérés.
    3. Évaluer la génération avec les métriques Faithfulness et Answer Relevancy.
    4. Retourner un rapport des résultats.
    """
    def __init__(self, vector_store: Any):
        """
        vector_store : PGVector instance avec RAG déjà indexé
        """
        self.vector_store = vector_store
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
            dataset_path = Path(__file__).parent.parent / "dataset" / "generated_dataset_ratio_0.7.json"

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

        all_metrics = self.generation_metrics

        # Évaluation classique avec DeepEval + Gemini
        results = evaluate(test_cases, all_metrics)
        return results