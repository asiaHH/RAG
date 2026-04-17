from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from evaluation.config import GEMINI_JUDGE, THRESHOLDS


def get_retrieval_metrics():
    """
    Return a list of metrics to evaluate the retrieval step in RAG, using GeminiJudge as the evaluation model.

    - ContextualPrecision  : among the retrieved chunks, how many are truly useful ?
    - ContextualRecall     : do the chunks cover everything needed to answer the question ?
    - ContextualRelevancy  : is each individual chunk relevant to the question ?
    """
    return [
        ContextualPrecisionMetric(
            threshold=THRESHOLDS["contextual_precision"],
            model=GEMINI_JUDGE,
            include_reason=True, 
        ),
        ContextualRecallMetric(
            threshold=THRESHOLDS["contextual_recall"],
            model=GEMINI_JUDGE,
            include_reason=True,
        ),
        ContextualRelevancyMetric(
            threshold=THRESHOLDS["contextual_relevancy"],
            model=GEMINI_JUDGE,
            include_reason=True,
        ),
    ]