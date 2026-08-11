#!/usr/bin/env python3
"""
Reddit API & GSMArena API Product Data Collection & Model Pipeline Script

This script fetches both Reddit reviews and GSMArena specifications for a product name,
runs the reference preprocessing pipeline on both datasets, and feeds them into the
ProductComparator aspect-based sentiment and ranking model.

Usage Examples:
    # 1. Fetch Reddit reviews + GSMArena specs and run through the model
    python fetch_reddit_product_data.py --product "iPhone 15 Pro" --run-model

    # 2. Fetch data and save raw/preprocessed JSON output
    python fetch_reddit_product_data.py --product "Samsung Galaxy S24 Ultra" --output data.json --save-model-output model_out.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src directory to PYTHONPATH
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.product_comparator.reddit_collector import RedditDataCollector
from src.product_comparator.gsmarena_collector import GSMArenaCollector, fetch_and_preprocess_product_specs
from src.product_comparator.preprocessing import preprocess_reddit_reviews_dict
from src.product_comparator.inference import ProductComparator


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Reddit reviews & GSMArena specs, preprocess datasets, and run model inference."
    )
    parser.add_argument(
        "-p", "--product",
        type=str,
        help="Name of the product to search (e.g., 'iPhone 15 Pro', 'Sony WH-1000XM5')."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output filepath to save collected Reddit data and preprocessed reviews/specs (.json)."
    )
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Run ProductComparator inference on the preprocessed dataset."
    )
    parser.add_argument(
        "--save-model-output",
        type=str,
        help="Output filepath to save model analysis results (.json)."
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=20,
        help="Maximum number of Reddit posts to search for (default: 20)."
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=20,
        help="Maximum comments to extract per post (default: 20)."
    )

    args = parser.parse_args()

    product_name = args.product
    if not product_name:
        product_name = input("Enter product name: ").strip()

    if not product_name:
        print("Error: Product name cannot be empty.")
        sys.exit(1)

    print(f"\n=========================================================================")
    print(f"   Reddit & GSMArena Data Collection & Product Comparator Pipeline      ")
    print(f"   Target Product: '{product_name}'                                     ")
    print(f"=========================================================================\n")

    # 1. Fetch raw Reddit reviews
    print(f"[Step 1/4] Fetching Reddit posts and comments for '{product_name}'...")
    reddit_collector = RedditDataCollector()
    raw_reddit = reddit_collector.fetch_product_data(
        product_name=product_name,
        max_posts=args.max_posts,
        max_comments_per_post=args.max_comments,
    )
    total_comments = len(raw_reddit.get("comments", []))
    total_reviews = len(raw_reddit.get("review_texts", []))
    print(f" -> Fetched {total_reviews} submission entries and {total_comments} comments from Reddit.")

    # 2. Fetch raw GSMArena specifications
    print(f"\n[Step 2/4] Fetching GSMArena product specifications for '{product_name}'...")
    gsm_collector = GSMArenaCollector()
    raw_specs = gsm_collector.fetch_product_specs(product_name=product_name)
    print(f" -> Fetched GSMArena device entry: '{raw_specs.get('name', product_name)}'")

    # 3. Preprocess datasets using reference pipeline
    print(f"\n[Step 3/4] Preprocessing raw Reddit reviews & GSMArena specifications...")
    processed_reviews_list = preprocess_reddit_reviews_dict([raw_reddit])
    preprocessed_reviews = processed_reviews_list[0]["text"] if processed_reviews_list else []
    
    preprocessed_specs = fetch_and_preprocess_product_specs(product_name=product_name, collector=gsm_collector)

    print(f" -> Preprocessing finished.")
    print(f"    * Valid Review Texts (>15 words): {len(preprocessed_reviews)}")
    print(f"    * Preprocessed Specification Keys: {len(preprocessed_specs)}")

    # Save output if requested
    combined_result = {
        "product_name": product_name,
        "product_id": raw_reddit["product"]["$oid"],
        "preprocessed_review_count": len(preprocessed_reviews),
        "preprocessed_reviews": preprocessed_reviews,
        "preprocessed_specifications": preprocessed_specs,
        "raw_reddit_data": raw_reddit,
        "raw_gsmarena_specs": raw_specs,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, indent=2, ensure_ascii=False)
        print(f" -> Saved collected & preprocessed dataset to: {out_path}")

    # 4. Model Inference
    model_analysis = None
    if args.run_model or args.save_model_output:
        if not preprocessed_reviews:
            print("\n[Step 4/4] Cannot run model inference: No preprocessed review evidence found.")
        else:
            print(f"\n[Step 4/4] Running ProductComparator inference on reviews & GSMArena specifications...")
            try:
                comparator = ProductComparator()
                model_analysis = comparator.analyze(
                    product_name=product_name,
                    reviews=preprocessed_reviews,
                    specifications=preprocessed_specs,
                )
                print(" -> Model inference successfully completed!")
                print(f" -> Top Useful Reviews Count: {len(model_analysis.get('top_reviews', []))}")
                print(f" -> Aspect Summary Sections: {len(model_analysis.get('product_summary', []))}")

                # Print breakdown of specs mapped into aspect sections
                specs_mapped_count = sum(
                    len(section.get("specifications", {}))
                    for section in model_analysis.get("product_summary", [])
                )
                print(f" -> GSMArena specifications mapped to aspect sections: {specs_mapped_count} fields.")

                if args.save_model_output:
                    model_out_path = Path(args.save_model_output)
                    model_out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(model_out_path, "w", encoding="utf-8") as f:
                        json.dump(model_analysis, f, indent=2, ensure_ascii=False)
                    print(f" -> Saved model analysis results to: {model_out_path}")

            except Exception as e:
                print(f" -> Error during model inference: {e}")

    print("\n[Done] Combined pipeline execution complete!\n")
    return {
        "dataset": combined_result,
        "model_analysis": model_analysis,
    }


if __name__ == "__main__":
    main()

