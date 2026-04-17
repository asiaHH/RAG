import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_vector_store():
    """
    Initialize and return the vector store
    """
    from src.ingestion.pipeline import init_vector_store
    
    vector_store = init_vector_store()
    if vector_store is None:
        console.print(" Erreur : Impossible de se connecter au vector store")
        sys.exit(1)
    
    console.print("Vector store chargé avec succès")
    return vector_store


def print_results(results):
    """
    Format and print the evaluation results in a readable way, with a summary of scores and pass/fail status for each metric.
    """
    console.print(Panel("[bold]Résultats de l'évaluation RAG[/bold]", style="blue"))
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Métrique", style="white", width=28)
    table.add_column("Catégorie", style="dim", width=12)
    table.add_column("Score moyen", justify="center", width=14)
    table.add_column("Seuil", justify="center", width=8)
    table.add_column("Statut", justify="center", width=10)

    category_map = {
        "ContextualPrecisionMetric": "Retrieval",
        "ContextualRecallMetric": "Retrieval",
        "ContextualRelevancyMetric": "Retrieval",
        "FaithfulnessMetric": "Génération", # a garder elle verifie si le rag hallucine pas
        "AnswerRelevancyMetric": "Génération",# a garder verifie si la reponse repond bien a la question
    }
    threshold_map = {
        "ContextualPrecisionMetric": 0.7,
        "ContextualRecallMetric": 0.7,
        "ContextualRelevancyMetric": 0.6,
        "FaithfulnessMetric": 0.8,
        "AnswerRelevancyMetric": 0.75,
    }

    scores_by_category = {"Retrieval": [], "Génération": []}

    for test_result in results.test_results:
        for metric_data in test_result.metrics_data:
            name = metric_data.name
            score = metric_data.score or 0.0
            threshold = threshold_map.get(name, 0.7)
            passed = score >= threshold
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            category = category_map.get(name, "—")

            table.add_row(
                name.replace("Metric", ""),
                category,
                f"{score:.2f}",
                f"{threshold:.2f}",
                status,
            )

            if category in scores_by_category:
                scores_by_category[category].append(score)

    console.print(table) 

    console.print()
    summary = Table(show_header=True, header_style="bold magenta")
    summary.add_column("Catégorie", width=14)
    summary.add_column("Score moyen", justify="center", width=14)

    for category, scores in scores_by_category.items():
        if scores:
            avg = sum(scores) / len(scores)
            color = "green" if avg >= 0.7 else "red"
            summary.add_row(category, f"[{color}]{avg:.2f}[/{color}]")

    console.print(Panel(summary, title="Résumé par catégorie", style="magenta"))

def main():
    parser = argparse.ArgumentParser(description="Évaluation du système RAG")
    parser.add_argument("--dataset", type=str, default=None, help="Chemin vers le fichier dataset JSON")
    args = parser.parse_args()

    console.print("\nDémarrage de l'évaluation RAG")
    
    vector_store = load_vector_store()
    console.print()

    from evaluation.pipeline.rag_evaluator import RAGEvaluator
    evaluator = RAGEvaluator(vector_store=vector_store)

    results = evaluator.run(dataset_path=args.dataset)
    print_results(results)
    
    console.print("Évaluation terminée!")


if __name__ == "__main__":
    main()