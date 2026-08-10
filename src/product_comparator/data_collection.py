import sys
from pathlib import Path

from .config import DROP_COLUMNS
from .preprocessing import preprocess_reddit_reviews, preprocess_products
from .reddit_collector import fetch_and_preprocess_product_reviews
from .inference import ProductComparator


def run_pipeline_for_product(
    product_name: str,
    specifications: dict = None,
    use_live_reddit_api: bool = True,
):
    """
    Data collection and model pipeline for a target product:
    1. Calls Reddit API to fetch posts & comments for product_name.
    2. Runs reference preprocessing pipeline on Reddit data.
    3. Runs ProductComparator model on preprocessed dataset.
    """
    specifications = specifications or {}

    print(f"=== Starting Data Collection Pipeline for Product: '{product_name}' ===")

    if use_live_reddit_api:
        print(f"[1/3] Calling Reddit API to collect data for '{product_name}'...")
        result = fetch_and_preprocess_product_reviews(product_name=product_name)
        review_data = result["reviews"]
        print(f"[2/3] Preprocessing complete. Extracted {len(review_data)} valid review items (>15 words).")
    else:
        print("[1/3] Using sample offline dataset...")
        base_dir = Path(__file__).resolve().parents[2]
        reviews_file = base_dir / "example_data" / "Comparator.reviews.json"
        specifications_file = base_dir / "example_data" / "Comparator.specifications.json"

        processed = preprocess_reddit_reviews(str(reviews_file))
        review_data = processed[0]["text"] if processed else []

        if not specifications and specifications_file.exists():
            spec_data = preprocess_products(str(specifications_file), DROP_COLUMNS)
            if spec_data:
                specifications = spec_data[0].get("specifications", {})
                product_name = spec_data[0].get("name", product_name)

    if not review_data:
        print(f"[Warning] No valid reviews found for '{product_name}'.")
        return {"status": "no_reviews_found"}

    print("[3/3] Running ProductComparator model inference on Reddit dataset...")
    model = ProductComparator()
    analysis_result = model.analyze(
        product_name=product_name,
        reviews=review_data,
        specifications=specifications,
    )

    print("=== Pipeline Execution Finished Successfully ===")
    return analysis_result


if __name__ == "__main__":
    target_product = sys.argv[1] if len(sys.argv) > 1 else "iPhone 15 Pro"
    res = run_pipeline_for_product(product_name=target_product, use_live_reddit_api=True)
    print(res)