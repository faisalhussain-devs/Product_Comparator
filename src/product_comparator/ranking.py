import numpy as np


def run_rank_batches(
    session,
    inputs: dict,
    batch_size: int = 128,
) -> np.ndarray:
    """Run the review-ranking ONNX model in batches."""

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    scores = []

    for start in range(0, len(input_ids), batch_size):
        end = start + batch_size

        batch_inputs = {
            "input_ids": input_ids[start:end].astype(np.int64),
            "attention_mask": attention_mask[start:end].astype(np.int64),
        }

        outputs = session.run(None, batch_inputs)
        scores.append(outputs[0][:, 0])

    return np.concatenate(scores, axis=0)


def select_reviews(
    ranking_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Select reviews whose ranking score exceeds the threshold."""
    return ranking_scores > threshold