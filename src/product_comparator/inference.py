import numpy as np

from .config import (
    ABSA_BATCH_SIZE,
    COMPARISON_THRESHOLD,
    MAX_ABSA_SEQUENCE_LENGTH,
    MAX_RANK_SEQUENCE_LENGTH,
    REVIEW_USEFULNESS_THRESHOLD,
    SUB_LABELS,
    SUB_THRESHOLDS,
    TOP_LABELS,
    TOP_THRESHOLDS,
    TOP_TO_SPECS,
    TOP_TO_SUB_PATH,
    RANK_BATCH_SIZE,
    TOP_REVIEWS_COUNT,
    SUB_ASPECT_REVIEWS_COUNT,
)
from .models import load_ranker, load_absa
from .ranking import run_rank_batches
from .habsa import run_absa_batches, predict_aspects
from .confidence import calc_confidence
from .relevance import product_relevance


class ProductComparator:
    def __init__(self):
        self.ranker = load_ranker()
        self.absa = load_absa()
        self.top_to_sub_dense = np.load(TOP_TO_SUB_PATH)

    def analyze(
        self,
        product_name: str,
        reviews: list[dict],
        specifications: dict,
        product_type: str = "Latest Smartphones",
    ) -> dict:

        valid_reviews = [review for review in reviews if isinstance(review, dict)and isinstance(review.get("text"), str) and review["text"].strip()]
        if not valid_reviews:
            raise ValueError("Product contains no valid reviews.")
        
        relevance_scores = np.asarray([product_relevance(review["text"], product_name) for review in valid_reviews])
        review_texts = [f"Main product: {product_name}. Review: {review['text']}" for review in valid_reviews]

        rank_inputs = self.ranker.tokenizer(
            review_texts,
            padding=True,
            truncation=True,
            max_length=MAX_RANK_SEQUENCE_LENGTH,
            return_tensors="np",
        )
        rank_scores = run_rank_batches(
            self.ranker.session,
            rank_inputs,
            batch_size=RANK_BATCH_SIZE,
        )

        relevant_mask = relevance_scores > 0.4
        useful_mask = rank_scores > 7.0
        evidence_mask = relevant_mask & useful_mask

        if np.sum(evidence_mask) <= 10:
            return {
                "product_info": {
                    "product_name": product_name,
                    "product_type": product_type,
                },
                "product_summary": [],
                "status": "insufficient_review_evidence",
            }

        selected_reviews = [valid_reviews[i] for i in np.where(evidence_mask)[0]]
        selected_texts = [f"Main product: {product_name}. Review: {review['text']}" for review in selected_reviews]

        absa_inputs = self.absa.tokenizer(
            selected_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_ABSA_SEQUENCE_LENGTH,
            return_tensors="np",
        )

        top_logits, sub_logits, comp_logits, sent_logits = run_absa_batches(
            self.absa.session,
            absa_inputs,
            batch_size=ABSA_BATCH_SIZE,
        )

        pred_top, pred_sub, pred_comp, sent_preds, sent_conf = predict_aspects(
            self.top_to_sub_dense,
            top_logits,
            sub_logits,
            comp_logits,
            sent_logits,
            TOP_THRESHOLDS,
            SUB_THRESHOLDS,
            comparison_threshold=COMPARISON_THRESHOLD,
        )

        usefulness = rank_scores[evidence_mask]
        sentiments, confidence = calc_confidence(
            sent_conf,
            usefulness,
            sent_preds,
            pred_sub,
        )

        top_indices = np.where(evidence_mask)[0]
        top_indices = top_indices[np.argsort(rank_scores[top_indices])[::-1]][:TOP_REVIEWS_COUNT]

        top_reviews = [
            {
                "text": valid_reviews[i]["text"],
                "usefulness": float(rank_scores[i]),
            }
            for i in top_indices
        ]

        product_summary = []

        for top_index, top_name in TOP_LABELS.items():
            aspect_specs = {
                spec_name: specifications[spec_name]
                for spec_name in TOP_TO_SPECS[top_index]
                if spec_name in specifications
            }

            product_summary.append({
                "top_aspect_name": top_name,
                "sub_aspects": [],
                "specifications": aspect_specs,
            })

        for sub_index, sub_confidence in enumerate(confidence):
            if sub_confidence == 0.0:
                continue

            active_reviews = (
                (pred_sub[:, sub_index] == 1.0)
                & (usefulness > REVIEW_USEFULNESS_THRESHOLD)
            )

            review_indices = np.where(active_reviews)[0]
            review_indices = review_indices[
                np.argsort(usefulness[review_indices])[::-1]
            ]

            supporting_reviews = [
                {
                    "text": selected_reviews[i]["text"],
                    "usefulness": float(usefulness[i]),
                }
                for i in review_indices[:SUB_ASPECT_REVIEWS_COUNT]
            ]

            active_top_indices = np.where(
                self.top_to_sub_dense[:, sub_index] == 1
            )[0]

            if len(active_top_indices) == 0:
                continue

            active_top = active_top_indices[0]

            product_summary[active_top]["sub_aspects"].append({
                "sub_aspect_name": SUB_LABELS[sub_index],
                "sub_aspect_sentiment": (
                    "positive" if sentiments[sub_index] > 0 else "negative"
                ),
                "sub_aspect_confidence": round(float(sub_confidence)),
                "supporting_reviews": supporting_reviews,
            })

        return {
            "product_info": {
                "product_name": product_name,
                "product_type": product_type,
            },
            "top_reviews": top_reviews,
            "product_summary": product_summary,
        }