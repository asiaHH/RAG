from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class QuestionType(str, Enum):
    """
    Enum to represent different types of questions for RAG evaluation.
    """
    FACTUAL      = "factual"
    REASONING    = "reasoning"
    SUMMARY      = "summary"
    UNANSWERABLE = "unanswerable"
    MULTI_HOP    = "multi_hop"

GOOD_DIMENSIONS: dict[str, list[str]] = {
    "style":     ["formelle", "neutre", "professionnelle"],
    "clarity":   ["claire", "compréhensible", "sans ambiguïté"],
    "precision": ["précise", "spécifique", "bien ciblée"],
    "structure": ["bien structurée", "logique", "directe"],
    "length":    ["courte (moins de 10 mots)", "moyenne (10-20 mots)", "longue et détaillée (plus de 20 mots)"],
}

BAD_DIMENSIONS: dict[str, list[str]] = {
    "style":     ["informelle", "familière", "approximative"],
    "clarity":   ["vague", "floue", "confuse"],
    "precision": ["générale", "ambiguë", "hors-sujet"],
    "structure": ["mal structurée", "désorganisée", "incomplète"],
    "length":    ["très courte (1-3 mots)", "trop longue et répétitive (30+ mots)", "fragmentée"],
}

DEFAULT_POSITIVE_RATIO = 0.80
DEFAULT_N_QUESTIONS    = 100
DEFAULT_TOP_K_RETRIEVAL = 5

@dataclass
class Chunk:
    """Represent a retrieved chunk of information."""
    id:       str
    text:     str
    metadata: dict = field(default_factory=dict)

@dataclass
class QAPair:
    """
    Pair a question with its expected answer and relevant metadata for RAG evaluation.
    """
    question:          str
    answer:            str
    source_chunk_id:   str
    is_relevant:       bool
    assigned_chunk_id: str
    question_type:     str
    dimensions:        dict
    round_trip_passed: Optional[bool] = None