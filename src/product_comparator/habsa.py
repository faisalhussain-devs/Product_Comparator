import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities."""
    return 1.0 / (1.0 + np.exp(-x))


def sentiment_from_logits(
    sent_logits: np.ndarray,
    temperature: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert two-class sentiment logits into:
        - sentiment: +1 or -1
        - confidence: bounded confidence score
    """
    delta = sent_logits[..., 1] - sent_logits[..., 0]

    sentiments = np.ones(delta.shape, dtype=np.int64)
    sentiments[delta < 0] = -1

    confidence = sigmoid(np.abs(delta)) / temperature

    return sentiments, confidence


def run_absa_batches(
    session,
    tokenized_inputs: dict,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run HABSA ONNX inference in batches."""

    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    num_samples = len(input_ids)

    top_logits = []
    sub_logits = []
    comp_logits = []
    sent_logits = []

    for start in range(0, num_samples, batch_size):
        end = start + batch_size

        batch_inputs = {
            "input_ids": input_ids[start:end].astype(np.int64),
            "attention_mask": attention_mask[start:end].astype(np.int64),
        }

        outputs = session.run(None, batch_inputs)

        top_logits.append(outputs[0])
        sub_logits.append(outputs[1])
        comp_logits.append(outputs[2])
        sent_logits.append(outputs[3])

    return (
        np.concatenate(top_logits, axis=0),
        np.concatenate(sub_logits, axis=0),
        np.concatenate(comp_logits, axis=0),
        np.concatenate(sent_logits, axis=0),
    )


def predict_aspects(
    top_to_sub_dense: np.ndarray,
    top_logits: np.ndarray,
    sub_logits: np.ndarray,
    comp_logits: np.ndarray,
    sent_logits: np.ndarray,
    top_thresholds: np.ndarray,
    sub_thresholds: np.ndarray,
    comparison_threshold: float = 0.7,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Convert HABSA logits into hierarchical aspect,
    comparison, and sentiment predictions.
    """

    p_top = sigmoid(top_logits)
    p_sub = sigmoid(sub_logits)
    p_comp = sigmoid(comp_logits)

    pred_top = (
        p_top > top_thresholds
    ).astype(np.float32)

    # Only allow sub-aspects belonging to predicted top aspects.
    mask = np.matmul(
        pred_top,
        top_to_sub_dense,
    )

    mask = np.clip(mask, 0.0, 1.0)

    # If no top aspect was predicted, allow all sub-aspects.
    no_top = (
        np.sum(pred_top, axis=1, keepdims=True) == 0
    )

    mask = mask + no_top * (1.0 - mask)

    masked_sub_probabilities = p_sub * mask

    pred_sub = (
        masked_sub_probabilities > sub_thresholds
    ).astype(np.float32)

    pred_comp = (
        (p_comp > comparison_threshold) * pred_sub
    ).astype(np.float32)

    sentiments, sentiment_confidence = sentiment_from_logits(
        sent_logits
    )

    return (
        pred_top,
        pred_sub,
        pred_comp,
        sentiments,
        sentiment_confidence,
    )