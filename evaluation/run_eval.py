import argparse
import json
import sys
import time
from pathlib import Path
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


def load_dataset(dataset_path: str | None = None):
    """Charger le dataset depuis un fichier JSON."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "dataset" / "generated_dataset_ratio_0.7.json"
    else:
        dataset_path = Path(dataset_path)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retrieval_evaluation(vector_store, dataset, args):
    """
    Évaluer le retrieval avec des métriques Python pures.
    """
    from evaluation.metrics.retrieval_metrics import RetrievalMetricsCalculator
    
    # Initialiser le calculateur
    k = args.k or 5
    calculator = RetrievalMetricsCalculator(k=k)
    
    # Définir les métriques à calculer
    available_metrics = ["precision", "recall", "mrr", "hit_rate"]
    if args.metrics:
        metrics_to_compute = [m.strip().lower() for m in args.metrics.split(",")]
        metrics_to_compute = [m for m in metrics_to_compute if m in available_metrics]
    else:
        metrics_to_compute = available_metrics
    
    console.print(f"\n[bold]Évaluation Retrieval (K={k})[/bold]")
    console.print(f"Métriques : {', '.join(metrics_to_compute)}")
    console.print(f"Dataset : {len(dataset)} questions\n")
    
    # Préparer les données pour le calcul batch
    queries_results = []
    
    #### A LA PLACE utiliser le vrai retriever ICI 
    ###########
    #########
    ########
    #######
    #retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    for i, item in enumerate(dataset):
        question = item["input"]
        # Le chunk pertinent est dans assigned_chunk_id
        relevant_chunk = item.get("assigned_chunk_id", "")
        
        # Normaliser le chemin pertinent pour correspondre au format des métadonnées
        # Enlever le chemin absolu si présent (n'importe quelle machine)
        if "/" in relevant_chunk:
            relevant_chunk = relevant_chunk.split("/")[-1]
        # Enlever le préfixe "data/" si présent pour uniformiser
        if relevant_chunk.startswith("data/"):
            relevant_chunk = relevant_chunk[5:]
        
        # Récupérer les chunks
        retrieved_docs = retriever.invoke(question)
        
        # Extraire les IDs des chunks récupérés depuis les métadonnées
        retrieved_chunk_ids = []
        for doc in retrieved_docs:
            # Utiliser 'source' qui contient le chemin relatif (ex: "data/xxx.pdf")
            source = doc.metadata.get("source", "")
            # Enlever le préfixe "data/" pour uniformiser et garder juste le nom du fichier
            if source.startswith("data/"):
                source = source[5:]
            # Garder juste le nom du fichier (pas le chemin complet)
            if "/" in source:
                source = source.split("/")[-1]
            retrieved_chunk_ids.append(source)
        
        relevant_chunks = [relevant_chunk] if relevant_chunk else []
        
        queries_results.append((retrieved_chunk_ids, relevant_chunks))
        
        if (i + 1) % 10 == 0:
            console.print(f"  → {i + 1}/{len(dataset)} requêtes traitées...")
    
    # Calculer les métriques
    console.print("\nCalcul des métriques...")
    report = calculator.compute_batch(queries_results, metrics_to_compute)
    
    # Afficher les résultats
    console.print(Panel("[bold]Résultats Retrieval[/bold]", style="cyan"))
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Métrique", style="white", width=20)
    table.add_column("Score", justify="center", width=12)
    table.add_column("K", justify="center", width=6)
    table.add_column("Queries", justify="center", width=10)
    
    metric_display_names = {
        "precision": "Precision@K",
        "recall": "Recall@K",
        "mrr": "MRR",
        "hit_rate": "Hit Rate@K",
    }
    
    for metric in metrics_to_compute:
        score = getattr(report, f"{metric}_at_k", None) or getattr(report, metric, None)
        if score is not None:
            display_name = metric_display_names.get(metric, metric)
            color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            table.add_row(display_name, f"[{color}]{score:.4f}[/{color}]", str(k), str(report.num_queries))
    
    console.print(table)
    
    # Optionnel: sauvegarder en JSON
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]✓ Résultats sauvegardés dans {output_path}[/green]")
    
    return report


def run_generation_evaluation(vector_store, dataset, args):
    """
    Évaluer la génération avec Deepeval + Gemini.
    """
    from evaluation.pipeline.generation_evaluator import GenerationEvaluator
    
    console.print("\n[bold]Évaluation Génération (Deepeval + Gemini)[/bold]")
    console.print(f"Dataset : {len(dataset)} questions\n")
    
    evaluator = GenerationEvaluator(vector_store=vector_store)
    
    # Deepeval throttle est déjà configuré dans generation_evaluator.py
    results = evaluator.run(dataset_path=None)
    
    print_results(results)
    
    return results


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
    parser = argparse.ArgumentParser(
        description="Évaluation du système RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Évaluation retrieval uniquement avec toutes les métriques
  python -m evaluation.run_eval --retrieval

  # Évaluation retrieval avec métriques spécifiques
  python -m evaluation.run_eval --retrieval --metrics precision,recall --k 10

  # Évaluation génération uniquement (Deepeval + Gemini)
  python -m evaluation.run_eval --generation

  # Les deux évaluations
  python -m evaluation.run_eval --retrieval --generation

  # Avec output JSON
  python -m evaluation.run_eval --retrieval --output results.json
        """
    )
    
    # Arguments généraux
    parser.add_argument("--dataset", type=str, default=None, help="Chemin vers le fichier dataset JSON")
    parser.add_argument("--output", type=str, default=None, help="Chemin pour sauvegarder les résultats en JSON")
    
    # Type d'évaluation
    parser.add_argument("--retrieval", action="store_true", help="Évaluer le retrieval (métriques Python pures)")
    parser.add_argument("--generation", action="store_true", help="Évaluer la génération (Deepeval + Gemini)")
    
    # Paramètres retrieval
    parser.add_argument("--k", type=int, default=5, help="Nombre de chunks à récupérer (défaut: 5)")
    parser.add_argument(
        "--metrics", type=str, default=None,
        help="Métriques à calculer (séparées par virgule): precision,recall,mrr,hit_rate. Par défaut: toutes"
    )
    
    args = parser.parse_args()
    
    # Par défaut, lancer les deux si aucun flag n'est spécifié
    run_both = not args.retrieval and not args.generation
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]              ÉVALUATION DU SYSTÈME RAG[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Charger le vector store
    vector_store = load_vector_store()
    console.print()
    
    # Charger le dataset
    dataset = load_dataset(args.dataset)
    console.print(f"Dataset chargé : {len(dataset)} questions\n")
    
    # Évaluation retrieval
    if args.retrieval or run_both:
        retrieval_report = run_retrieval_evaluation(vector_store, dataset, args)
    
    # Évaluation génération
    if args.generation or run_both:
        generation_results = run_generation_evaluation(vector_store, dataset, args)
    
    console.print("\n[bold green]✓ Évaluation terminée![/bold green]")


if __name__ == "__main__":
    main()