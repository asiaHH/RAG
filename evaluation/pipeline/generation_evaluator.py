import json
import math
import time
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
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
        is_relevant = item.get("is_relevant", True)

        response = generate_response(self.vector_store, question)
        actual_output = response.get("answer", "")
        retrieval_context = [doc.page_content for doc in response.get("sources", [])]    

        # Pour les questions négatives, l'expected_output est "je ne sais pas"
        # Pour les positives, on garde la réponse du dataset
        if is_relevant:
            expected_output = item.get("expected_output", "")
        else:
            expected_output = "Information non disponible dans le corpus."

        return LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
            context=retrieval_context,
        )

    def run(self, dataset: list | None = None,  dataset_path: str | None = None) -> dict:
        """
        Lance l'évaluation complète et retourne un résumé des scores.
        dataset : liste de questions à évaluer
        dataset_path : chemin vers le fichier JSON du dataset (utilisé si dataset=None)
        """
        # return results
        if dataset is None:
            if dataset_path is None:
                dataset_path = Path(__file__).parent.parent / "dataset" / "generated_dataset_V2_ratio_0.7.json"
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

        n_positive = sum(1 for item in dataset if item.get("is_relevant", True))
        n_negative = len(dataset) - n_positive
        print(f"\nÉvaluation génération sur {len(dataset)} questions "
              f"({n_positive} pertinentes, {n_negative} hors-corpus)...\n")

        test_cases = []
        for item in dataset:
            label = "✓" if item.get("is_relevant", True) else "✗ hors-corpus"
            print(f"  [{label}] {item['id']} — {item['input'][:60]}...")
            try:
                tc = self._build_test_case(item)
                test_cases.append(tc)
            except Exception as e:
                print(f"  ✗ Erreur sur {item['id']}: {e}")
        
        # Évaluation par batch
        batch_size = 5
        total_batches = math.ceil(len(test_cases) / batch_size)
        all_test_results = []

        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"\nBatch {batch_num}/{total_batches} "
                f"(questions {i+1} à {min(i+batch_size, len(test_cases))})...")

            try:
                result = evaluate(
                    batch,
                    self.generation_metrics,
                    async_config=AsyncConfig(max_concurrent=3, throttle_value=1)
                )
                all_test_results.extend(result.test_results)
                print(f"  ✓ Batch {batch_num} terminé")
            except Exception as e:
                print(f"  ✗ Batch {batch_num} échoué : {e}")
                # On continue avec le batch suivant plutôt que de tout crasher

            # Pause inter-batch sauf après le dernier
            if i + batch_size < len(test_cases):
                print(f"  → Pause 5s avant le batch suivant...")
                time.sleep(5)
        # on agrege les scores par metriques sur l'ens des n questions du dataset
        scor_by_metrics = {}
        for test_result in all_test_results:
            for metric_data in test_result.metrics_data:
                name = metric_data.name
                score= metric_data.score or 0.0
                if name not in scor_by_metrics:
                    scor_by_metrics[name] = []
                scor_by_metrics[name].append(score)


        #on calcule les moyennes globales
        aggregated = {}
        for name, scores in scor_by_metrics.items():
            aggregated[name] = {
                "mean": sum(scores) / len(scores),
                "scores": scores,
                "n": len(scores)
            }
            print(f"\n{name} : {aggregated[name]['mean']:.4f} (sur {aggregated[name]['n']} questions)")
        print(f"\nÉvaluation terminée : {len(all_test_results)}/{total_batches} batches réussis")

        final_report = {
            "summary": aggregated,
            "details": all_test_results
        }
        return final_report