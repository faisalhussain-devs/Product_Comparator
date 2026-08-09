import numpy as np

from .config import (
    ASPECT_RELIABILITY,
    CONFIDENCE_ALPHA,
    CONFIDENCE_BETA,
    CONFIDENCE_GAMMA,
    CONFIDENCE_PHI,
)

def robust_sign(x: np.ndarray, eps: float = 1e-6):
    s = np.zeros_like(x)
    s[x > eps] = 1
    s[x < -eps] = -1
    return s

def weighted_sentiment_score(
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-8) -> np.ndarray:
    """
    S_j: weighted belief score
    """
    if aspect_mask.sum() == 0:
        return 0.0

    num = np.sum(sentiments * aspect_mask * weights, axis=0)
    den = np.sum(weights * aspect_mask, axis=0) + eps
    return (num / den)


def g_conf(conf: np.ndarray, alpha: float):
    # conf = |logit_pos - logit_neg|
    return 1.0 - np.exp(-conf / alpha)

def h_useful(u: np.ndarray, beta: float = 1.0):
    return np.log1p(beta * u)

def build_weights(
    sent_conf: np.ndarray,   # (N, A)
    usefulness: np.ndarray,  # (N,)
    alpha: float,
    beta: float):
    return g_conf(sent_conf, alpha) * h_useful(usefulness[:, None], beta)

def build_weights_norm(
    sent_conf: np.ndarray,
    usefulness: np.ndarray,
    alpha: float,
    beta: float,
):
    sent_conf_median = np.median(sent_conf)
    usefulness_median = np.median(usefulness)

    return (
        g_conf(sent_conf / (sent_conf_median + 1e-8), alpha)
        * h_useful(
            usefulness[:, None] / (usefulness_median + 1e-8),
            beta
        )
    )

def final_confidence(
    usefulness: np.ndarray, #(n, 1)
    aspect_mask: np.ndarray, #(n, 29)
    sentiments: np.ndarray, #(n, 29)
    weights: np.ndarray, #(n, 29)
    aspect_reliability: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0):
    """
Final confidence combines three independent signals:

1. Aspect-specific sentiment reliability.
2. Weighted agreement among relevant reviews.
3. Evidence volume, with diminishing returns.

Confidence is intentionally NOT penalized by a separate
dispersion term because disagreement is already captured
by the agreement term.
"""
    weighted_sentiments = sentiments * aspect_mask * weights
    support = np.sum(weighted_sentiments, axis=0)
    total_weight = np.sum(np.abs(weighted_sentiments), axis=0) + 1e-8
    agreement = np.abs(support) / total_weight
    usefulness = usefulness[:, None]
    effective_volume = np.sum(aspect_mask * h_useful(usefulness, beta), axis=0)

    volume_term = 1.0 - np.exp(-effective_volume / gamma)
    reliability_floor = 0.65
    reliability_exponent = 0.8

    sentiment_reliability = reliability_floor + (1 - reliability_floor) * (aspect_reliability ** reliability_exponent)
    confidence = sentiment_reliability * agreement * volume_term
    return confidence

def calc_confidence(
    sent_conf: np.ndarray,
    usefulness: np.ndarray,
    sent_preds: np.ndarray,
    pred_sub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate final sentiment and confidence for every sub-aspect.

    Returns:
        sentiments:
            Final aggregated sentiment (-1, 0, +1).

        confidence:
            Final confidence score on a 0-100 scale.
    """

    weights_norm = build_weights_norm(
        sent_conf=sent_conf,
        usefulness=usefulness,
        alpha=CONFIDENCE_ALPHA,
        beta=CONFIDENCE_BETA,
    )

    weights = build_weights(
        sent_conf=sent_conf,
        usefulness=usefulness,
        alpha=CONFIDENCE_ALPHA,
        beta=CONFIDENCE_BETA,
    )

    sentiment_scores = weighted_sentiment_score(
        sentiments=sent_preds,
        aspect_mask=pred_sub,
        weights=weights_norm,
    )

    sentiments = robust_sign(sentiment_scores)

    raw_confidence = final_confidence(
        usefulness=usefulness,
        aspect_mask=pred_sub,
        sentiments=sentiments,
        weights=weights,
        aspect_reliability=ASPECT_RELIABILITY,
        beta=CONFIDENCE_BETA,
        gamma=CONFIDENCE_GAMMA,
    )

    confidence = 100 * (raw_confidence ** CONFIDENCE_PHI)
    return sentiments, confidence