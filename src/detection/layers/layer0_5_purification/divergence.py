"""Score divergence analysis for adversarial detection.

Compares detection scores between original and purified image variants.
Large divergence indicates adversarial perturbation was present.
"""

import numpy as np


def analyze_divergence(
    original_score: float,
    purified_scores: dict[str, float],
    threshold: float = 0.20,
) -> dict:
    """Analyze score divergence between original and purified variants.

    Args:
        original_score: Score from running detector on original image.
        purified_scores: Map of variant name to score on that variant.
        threshold: Divergence threshold for flagging adversarial perturbation.

    Returns:
        Dict with divergence analysis results.
    """
    if not purified_scores:
        return {
            "adversarial_detected": False,
            "max_divergence": 0.0,
            "recommended_score": original_score,
        }

    scores = list(purified_scores.values())
    divergences = [abs(s - original_score) for s in scores]
    max_divergence = max(divergences)
    mean_divergence = float(np.mean(divergences))

    adversarial_detected = max_divergence > threshold

    # If adversarial, use the maximum score across all variants
    # (most conservative — hardest to fool)
    if adversarial_detected:
        recommended_score = max(max(scores), original_score)
    else:
        recommended_score = original_score

    return {
        "adversarial_detected": adversarial_detected,
        "max_divergence": round(max_divergence, 4),
        "mean_divergence": round(mean_divergence, 4),
        "recommended_score": round(recommended_score, 4),
        "original_score": round(original_score, 4),
        "purified_scores": {k: round(v, 4) for k, v in purified_scores.items()},
    }
