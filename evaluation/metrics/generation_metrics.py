from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from evaluation.config import GEMINI_JUDGE, THRESHOLDS


def get_generation_metrics():
    """
    - Faithfulness    : the answer is fully grounded in the retrieved chunks (hallucination)
    - AnswerRelevancy : does the answer actually address the question asked ?
    """
    return [
        FaithfulnessMetric(
            threshold=THRESHOLDS["faithfulness"],
            model=GEMINI_JUDGE,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLDS["answer_relevancy"],
            model=GEMINI_JUDGE,
            include_reason=True,
        ),
    ]