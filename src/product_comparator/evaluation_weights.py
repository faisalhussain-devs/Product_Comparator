import numpy as np
from typing import Dict, List, Callable, Tuple
from collections import defaultdict
import pandas as pd

def sign(x: float, eps: float = 1e-6) -> int:
    if x > eps:
        return 1
    elif x < -eps:
        return -1
    return 0


def raw_sentiment_score(sentiments: np.ndarray, aspect_mask: np.ndarray) -> float:
    """
    R_j: raw people sentiment
    """
    num = np.sum(sentiments * aspect_mask, axis=0)
    den = np.sum(aspect_mask, axis=0) + 1e-8
    return num / den


def weighted_sentiment_score(
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    weights: np.ndarray,
    aspect_reliability: float,
    eps: float = 1e-8) -> float:
    """
    S_j: weighted belief score
    """
    if aspect_mask.sum() == 0:
        return 0.0

    num = np.sum(sentiments * aspect_mask * weights, axis=0)
    den = np.sum(weights * aspect_mask, axis=0) + eps
    return aspect_reliability * (num / den)


def evaluate_noise_robustness(
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    weights: np.ndarray,
    aspect_reliability: float,
    drop_rates: List[float] = [0.2, 0.3, 0.4],
    n_trials: int = 100,
) -> Dict[str, float]:
    """
    Randomly drop p% of sentences and measure variance
    """
    results = {}

    for p in drop_rates:
        raw_scores = []
        weighted_scores = []

        for _ in range(n_trials):
            keep = np.random.rand(len(sentiments)) > p
            raw_scores.append(
                raw_sentiment_score(
                    sentiments[keep],
                    aspect_mask[keep]))
            
            weighted_scores.append(
                weighted_sentiment_score(
                    sentiments[keep],
                    aspect_mask[keep],
                    weights[keep],
                    aspect_reliability))

        results[f"raw_var_p{int(p*100)}"] = np.var(np.stack(raw_scores), axis=0)
        results[f"weighted_var_p{int(p*100)}"] = np.var(np.stack(weighted_scores), axis=0)

    return results


def inject_adversarial_noise(
    sentiments: np.ndarray,        # (N, 29)
    aspect_mask: np.ndarray,       # (N, 29)
    weights: np.ndarray,        # (N, 29)
    n_noise: int,
    n_aspects: int = 29,
    max_active_aspects: int = 4):
    """
    Aspect-aware adversarial noise injection.
    """
    # Random sentiment per aspect
    noise_sentiments = np.random.choice([-1, 1], size=(n_noise, n_aspects))

    # Sparse aspect activation
    noise_aspect_mask = np.zeros((n_noise, n_aspects))
    for i in range(n_noise):
        k = np.random.randint(1, max_active_aspects + 1)
        idx = np.random.choice(n_aspects, size=k, replace=False)
        noise_aspect_mask[i, idx] = 1.0

    # Low confidence (aspect-level)
    noise_weights = np.random.uniform(0.01, 0.05, size=(n_noise, n_aspects))

    return (np.concatenate([sentiments, noise_sentiments], axis=0),
        np.concatenate([aspect_mask, noise_aspect_mask], axis=0),
        np.concatenate([weights, noise_weights], axis=0))


def evaluate_adversarial_noise(
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    weights: np.ndarray,
    aspect_reliability: float,
    n_noise: int = 50) -> Dict[str, float]:
    """
    Measure drift under adversarial noise
    """
    clean_raw = raw_sentiment_score(sentiments, aspect_mask)
    clean_weighted = weighted_sentiment_score(sentiments, aspect_mask, weights, aspect_reliability)

    ns, nm, nw = inject_adversarial_noise(sentiments, aspect_mask, weights, n_noise)

    noisy_raw = raw_sentiment_score(ns, nm)
    noisy_weighted = weighted_sentiment_score(ns, nm, nw, aspect_reliability)

    return {
        "raw_drift": abs(clean_raw - noisy_raw),
        "weighted_drift": abs(clean_weighted - noisy_weighted),
    }


def evaluate_agreement_consistency(
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    weights: np.ndarray,
    aspect_reliability: float,
) -> Dict[str, float]:
    """
    Compare sign(R_j) vs sign(S_j)
    """
    Rj = raw_sentiment_score(sentiments, aspect_mask)
    Sj = weighted_sentiment_score(
        sentiments, aspect_mask, weights, aspect_reliability)

    return {
        "raw_score": Rj,
        "weighted_score": Sj,
        "disagreement_magnitude": abs(Rj - Sj),
    }


def evaluate_cross_source_consistency(
    sources: Dict[str, Dict],
    aspect_reliability: float,
) -> Dict[str, float]:
    """
    sources[source_name] = {
        "sentiments": np.array,
        "aspect_mask": np.array,
        "weights": np.array
    }
    """
    scores = {}

    for src, data in sources.items():
        scores[src] = weighted_sentiment_score(
            data["sentiments"],
            data["aspect_mask"],
            data["weights"],
            aspect_reliability,
        )

    signs = [sign(v) for v in scores.values()]

    return {
        "scores": scores,
        "signs": signs,
        "agreement_rate": sum(s == signs[0] for s in signs) / len(signs),
    }


def select_best_weighting(
    stability_results: Dict[str, float],
    drift_results: Dict[str, float],
    agreement_ok: bool,
) -> bool:
    """
    Honest selection rule:
    - Lower variance
    - Lower adversarial drift
    - Sign preserved
    """
    return (
        stability_results["weighted_var_p30"] <
        stability_results["raw_var_p30"]
        and drift_results["weighted_drift"] <
        drift_results["raw_drift"]
        and agreement_ok
    )

def g_conf(conf: np.ndarray, alpha: float = 1.0):
    return np.power(conf, alpha)

def h_useful(u: np.ndarray, beta: float = 1.0):
    return np.log1p(beta * u)

def build_weights(
    sent_conf: np.ndarray,   # (N, A)
    usefulness: np.ndarray,  # (N,)
    alpha: float,
    beta: float):
    return g_conf(sent_conf, alpha) * h_useful(usefulness[:, None], beta)


def evaluate_configuration(
    config_id: str,
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    sent_conf: np.ndarray,
    usefulness: np.ndarray,
    aspect_reliability: np.ndarray,
    alpha: float,
    beta: float):
    """
    Runs all evaluations for ONE configuration.
    Returns a dict (one row in DataFrame).
    """

    weights = build_weights(sent_conf, usefulness, alpha, beta)
    stability = evaluate_noise_robustness(sentiments, aspect_mask, weights, aspect_reliability)
    drift = evaluate_adversarial_noise(sentiments, aspect_mask, weights, aspect_reliability)
    agreement = evaluate_agreement_consistency(sentiments, aspect_mask, weights, aspect_reliability)

    result = {
        "config_id": config_id, "alpha": alpha, "beta": beta,

        # stability
        **stability,

        # drift
        **drift,

        # agreement
        "disagreement_mag": agreement["disagreement_magnitude"]}

    return result


def run_evaluation_grid(
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    sent_conf: np.ndarray,
    usefulness: np.ndarray,
    aspect_reliability: np.ndarray,
    alphas: List[float],
    betas: List[float]):
    rows = []

    for a in alphas:
        for b in betas:
            cfg_id = f"a{a}_b{b}"

            row = evaluate_configuration(
                cfg_id,
                sentiments,
                aspect_mask,
                sent_conf,
                usefulness,
                aspect_reliability,
                alpha=a,
                beta=b)

            rows.append(row)

    return pd.DataFrame(rows)
