"""
Module d'évaluation du retrieval avec des métriques calculées en Python pur.
Pas d'appels API nécessaires - calculs locaux.
"""
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalResult:
    """Résultat d'une métrique de retrieval."""
    metric_name: str
    score: float
    k: Optional[int] = None
    details: str = ""


@dataclass
class RetrievalMetricsReport:
    """Rapport complet des métriques de retrieval."""
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    hit_rate_at_k: Optional[float] = None
    
    # Métadonnées
    k_value: int = 5
    num_queries: int = 0
    
    def to_dict(self) -> dict:
        """Convertir le rapport en dictionnaire."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "hit_rate_at_k": self.hit_rate_at_k,
            "k": self.k_value,
            "num_queries": self.num_queries,
        }


class RetrievalMetricsCalculator:
    """
    Classe pour calculer les métriques de retrieval.
    
    Métriques supportées :
    - Precision@K : proportion de documents pertinents parmi les K premiers
    - Recall@K : proportion de documents pertinents récupérés parmi les K premiers
    - MRR (Mean Reciprocal Rank) : rang moyen du premier résultat pertinent
    - Hit Rate@K : au moins un résultat pertinent dans les K premiers
    """
    
    def __init__(self, k: int = 5):
        """
        Initialiser le calculateur.
        
        Args:
            k: Nombre de documents à considérer pour les métriques
        """
        self.k = k
        self._results: list[RetrievalResult] = []
    
    def precision_at_k(self, retrieved_chunks: list[str], relevant_chunks: list[str], k: Optional[int] = None) -> float:
        """
        Compute Precision@K.
        
        Args:
            retrieved_chunks: Liste des chunks récupérés (ordonnés par pertinence)
            relevant_chunks: Liste des chunks pertinents (vérité terrain)
            k: Override du k par défaut
        
        Returns:
            Score Precision@K entre 0 et 1
        """
        k = k or self.k
        if k == 0 or not retrieved_chunks:
            return 0.0
        
        retrieved_k = set(retrieved_chunks[:k])
        relevant_set = set(relevant_chunks)
        
        return len(retrieved_k & relevant_set) / k
    
    def recall_at_k(self, retrieved_chunks: list[str], relevant_chunks: list[str], k: Optional[int] = None) -> float:
        """
        Compute Recall@K.
        
        Args:
            retrieved_chunks: Liste des chunks récupérés
            relevant_chunks: Liste des chunks pertinents
            k: Override du k par défaut
        
        Returns:
            Score Recall@K entre 0 et 1
        """
        k = k or self.k
        if not relevant_chunks:
            return 0.0
        
        retrieved_k = set(retrieved_chunks[:k])
        relevant_set = set(relevant_chunks)
        
        return len(retrieved_k & relevant_set) / len(relevant_set)
    
    def mean_reciprocal_rank(self, retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
        """
        Compute MRR (Mean Reciprocal Rank).
        
        Args:
            retrieved_chunks: Liste des chunks récupérés
            relevant_chunks: Liste des chunks pertinents
        
        Returns:
            Score MRR entre 0 et 1
        """
        relevant_set = set(relevant_chunks)
        
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk in relevant_set:
                return 1.0 / rank
        return 0.0
    
    
    def hit_rate_at_k(self, retrieved_chunks: list[str], relevant_chunks: list[str], k: Optional[int] = None) -> float:
        """
        Compute Hit Rate@K.
        
        Args:
            retrieved_chunks: Liste des chunks récupérés
            relevant_chunks: Liste des chunks pertinents
            k: Override du k par défaut
        
        Returns:
            Score Hit Rate@K entre 0 et 1
        """
        k = k or self.k
        if not relevant_chunks:
            return 0.0
        
        retrieved_k = set(retrieved_chunks[:k])
        relevant_set = set(relevant_chunks)
        
        return 1.0 if bool(retrieved_k & relevant_set) else 0.0
    
    def compute_all(
        self,
        retrieved_chunks: list[str],
        relevant_chunks: list[str],
        metrics: Optional[list[str]] = None
    ) -> dict:
        """
        Calculer toutes les métriques demandées.
        
        Args:
            retrieved_chunks: Liste des chunks récupérés
            relevant_chunks: Liste des chunks pertinents
            metrics: Liste des métriques à calculer. Si None, toutes.
                     Options: ["precision", "recall", "mrr", "hit_rate"]
        
        Returns:
            Dict avec les scores de chaque métrique
        """
        available_metrics = {
            "precision": self.precision_at_k,
            "recall": self.recall_at_k,
            "mrr": self.mean_reciprocal_rank,
            "hit_rate": self.hit_rate_at_k,
        }
        
        if metrics is None:
            metrics = list(available_metrics.keys())
        
        results = {}
        for metric_name in metrics:
            if metric_name in available_metrics:
                func = available_metrics[metric_name]
                # Appeler sans k pour les métriques qui n'en ont pas besoin
                if metric_name in ["mrr"]:
                    results[metric_name] = func(retrieved_chunks, relevant_chunks)
                else:
                    results[metric_name] = func(retrieved_chunks, relevant_chunks)
        
        return results
    
    def compute_batch(
        self,
        queries_results: list[tuple[list[str], list[str]]],
        metrics: Optional[list[str]] = None
    ) -> RetrievalMetricsReport:
        """
        Calculer les métriques sur un batch de requêtes.
        
        Args:
            queries_results: Liste de tuples (retrieved_chunks, relevant_chunks)
            metrics: Liste des métriques à calculer
        
        Returns:
            RetrievalMetricsReport avec les moyennes
        """
        if not queries_results:
            return RetrievalMetricsReport(k_value=self.k, num_queries=0)
        
        all_results = {
            "precision": [],
            "recall": [],
            "mrr": [],
            "hit_rate": [],
        }
        
        for retrieved, relevant in queries_results:
            scores = self.compute_all(retrieved, relevant, metrics)
            for key, value in scores.items():
                all_results[key].append(value)
        
        # Calculer les moyennes
        report = RetrievalMetricsReport(
            k_value=self.k,
            num_queries=len(queries_results),
            precision_at_k=sum(all_results["precision"]) / len(all_results["precision"]) if all_results["precision"] else None,
            recall_at_k=sum(all_results["recall"]) / len(all_results["recall"]) if all_results["recall"] else None,
            mrr=sum(all_results["mrr"]) / len(all_results["mrr"]) if all_results["mrr"] else None,
            hit_rate_at_k=sum(all_results["hit_rate"]) / len(all_results["hit_rate"]) if all_results["hit_rate"] else None,
        )
        
        return report


# Alias pour compatibilité
RetrievalMetrics = RetrievalMetricsCalculator