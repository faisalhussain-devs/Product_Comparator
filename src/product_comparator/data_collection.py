import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .config import DROP_COLUMNS
from .preprocessing import preprocess_reddit_reviews_dict, drop_specification_keys, flatten_data
from .reddit_collector import fetch_and_preprocess_product_reviews
from .gsmarena_collector import fetch_and_preprocess_product_specs
from .dataset_cache import DatasetCache
from .inference import ProductComparator


def run_pipeline_for_product(
    product_name: str,
    specifications: dict = None,
    force_live: bool = False,
):
    """
    Complete Data Collection & Model Inference Pipeline:
    1. [Cache First]: Check O(1) products_corpus index (product_index.json) for pre-packaged benchmark product data.
    2. [Live API Fallback]: If not in cache or force_live=True, fetch live Reddit reviews & GSMArena specifications.
    3. [Preprocessing]: Preprocess review text & flatten GSMArena specifications.
    4. [Model Inference]: Run ProductComparator model on preprocessed dataset.
    """
    print(f"=== Starting Data Collection Pipeline for Product: '{product_name}' ===")

    cache = DatasetCache()
    cached_entry = None if force_live else cache.find_product(product_name)

    if cached_entry:
        matched_name = cached_entry.get("name", product_name)
        print(f"[Step 1/4] [Cache Hit] Found packaged dataset for '{matched_name}' (ID: {cached_entry.get('id')}).")
        
        # Preprocess cached reviews & specifications
        raw_reviews = cached_entry.get("reviews", {})
        processed_list = preprocess_reddit_reviews_dict([{"product": {"$oid": cached_entry.get("id")}, "name": matched_name, **raw_reviews}])
        review_data = processed_list[0]["text"] if processed_list else []

        if specifications is None:
            raw_specs = cached_entry.get("specifications", {})
            if raw_specs:
                flat_specs = flatten_data(raw_specs)
                specs_obj = {"specifications": flat_specs}
                cleaned_specs = drop_specification_keys(specs_obj, drop_keys=DROP_COLUMNS)
                specifications = cleaned_specs.get("specifications", {})
            else:
                specifications = {}

        product_name = matched_name

    else:
        print(f"[Step 1/4] [Cache Miss] Product '{product_name}' not in cache. Calling Live APIs...")
        
        print(f"[2/4] Calling Reddit API for '{product_name}'...")
        reddit_result = fetch_and_preprocess_product_reviews(product_name=product_name)
        review_data = reddit_result["reviews"]
        print(f" -> Extracted {len(review_data)} preprocessed review items (>15 words).")

        print(f"[2/4] Calling GSMArena API for '{product_name}'...")
        if specifications is None:
            specifications = fetch_and_preprocess_product_specs(product_name=product_name)
        print(f" -> Extracted {len(specifications)} preprocessed specification keys.")

        # Save live fetched data to cache for future runs
        if reddit_result.get("raw_data"):
            cache.add_to_cache(
                product_name=product_name,
                reviews=reddit_result["raw_data"],
                specifications={},
            )

    if not review_data:
        print(f"[Warning] No valid review data found for '{product_name}'.")
        return {"status": "no_reviews_found"}

    print(f"[Step 3/4] Preprocessing finished. Reviews: {len(review_data)}, Specs: {len(specifications or {})} fields.")
    print("[Step 4/4] Running ProductComparator model inference...")
    model = ProductComparator()
    analysis_result = model.analyze(
        product_name=product_name,
        reviews=review_data,
        specifications=specifications or {},
    )

    print("=== Pipeline Execution Finished Successfully ===")
    return analysis_result


if __name__ == "__main__":
    target_product = sys.argv[1] if len(sys.argv) > 1 else "iPhone 15 Pro"
    res = run_pipeline_for_product(product_name=target_product)
    print(res)