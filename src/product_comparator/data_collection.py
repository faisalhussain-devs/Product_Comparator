import sys
from pathlib import Path

from .config import DROP_COLUMNS
from .preprocessing import preprocess_reddit_reviews, preprocess_products
from .reddit_collector import fetch_and_preprocess_product_reviews
from .gsmarena_collector import fetch_and_preprocess_product_specs
from .inference import ProductComparator


def run_pipeline_for_product(
    product_name: str,
    specifications: dict = None,
    use_live_apis: bool = True,
):
    """
    Complete Data Collection & Model Inference Pipeline for a Product:
    1. Calls Reddit API to fetch & preprocess product reviews/comments.
    2. Calls GSMArena API/fetcher to get & flatten product specifications.
    3. Feeds preprocessed reviews AND preprocessed specifications to ProductComparator inference model.
    """
    print(f" Starting Data Collection Pipeline for Product: '{product_name}' ")

    if use_live_apis:
        print(f"[1/4] Calling Reddit API to collect reviews for '{product_name}'...")
        reddit_result = fetch_and_preprocess_product_reviews(product_name=product_name)
        review_data = reddit_result["reviews"]
        print(f" -> Extracted {len(review_data)} preprocessed review items .")

        print(f"[2/4] Calling GSMArena API to collect specifications for '{product_name}'...")
        if specifications is None:
            specifications = fetch_and_preprocess_product_specs(product_name=product_name)
        print(f" -> Extracted {len(specifications)} preprocessed specification keys.")
    else:
        print("[1/4] Using sample offline dataset...")
        base_dir = Path(__file__).resolve().parents[2]
        reviews_file = base_dir / "example_data" / "Comparator.reviews.json"
        specifications_file = base_dir / "example_data" / "Comparator.specifications.json"

        processed = preprocess_reddit_reviews(str(reviews_file))
        review_data = processed[0]["text"] if processed else []

        if specifications is None and specifications_file.exists():
            spec_data = preprocess_products(str(specifications_file), DROP_COLUMNS)
            if spec_data:
                specifications = spec_data[0].get("specifications", {})
                product_name = spec_data[0].get("name", product_name)

    if not review_data:
        print(f"[Warning] No valid review data found for '{product_name}'.")
        return {"status": "no_reviews_found"}

    print("[3/4] Running ProductComparator model inference on Reddit reviews + GSMArena specifications...")
    model = ProductComparator()
    analysis_result = model.analyze(
        product_name=product_name,
        reviews=review_data,
        specifications=specifications or {},
    )

    print("[4/4] Pipeline Execution Finished Successfully")
    return analysis_result


if __name__ == "__main__":
    target_product = sys.argv[1] if len(sys.argv) > 1 else "iPhone 15 Pro"
    res = run_pipeline_for_product(product_name=target_product, use_live_apis=True)
    print(res)