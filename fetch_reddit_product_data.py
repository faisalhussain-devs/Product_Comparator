#!/usr/bin/env python3
"""
Reddit API Product Data Collection & Model Pipeline Script

This script fetches Reddit data for a given product name (e.g. entered via website or CLI),
runs the reference preprocessing pipeline, and feeds the resulting dataset into the
ProductComparator aspect-based sentiment and ranking model.

Usage Examples:
    # 1. Fetch Reddit data for a product and save to JSON
    python fetch_reddit_product_data.py --product "iPhone 15 Pro" --output iphone15_reddit_data.json

    # 2. Fetch Reddit data, preprocess, and run through the model
    python fetch_reddit_product_data.py --product "Samsung Galaxy S24 Ultra" --run-model

    # 3. Interactive mode (prompts for product name if not provided)
    python fetch_reddit_product_data.py
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

from product_comparator.reddit_collector import RedditDataCollector, fetch_and_preprocess_product_reviews
from product_comparator.preprocessing import preprocess_reddit_reviews_dict
from product_comparator.inference import ProductComparator


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Reddit product data, preprocess using the reference pipeline, and run model inference."
    )
    parser.add_argument(
        "-p", "--product",
        type=str,
        help="Name of the product to fetch Reddit data for (e.g., 'iPhone 15 Pro', 'Sony WH-1000XM5')."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output filepath to save collected Reddit data and preprocessed reviews (.json)."
    )
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Run ProductComparator inference on the preprocessed Reddit data."
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
        product_name = input("Enter product name to search on Reddit: ").strip()

    if not product_name:
        print("Error: Product name cannot be empty.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"   Reddit API Product Collector & Model Pipeline       ")
    print(f"   Target Product: '{product_name}'                   ")
    print(f"=======================================================\n")

    # 1. Fetch raw product data from Reddit API
    print(f"[Step 1/3] Fetching Reddit posts and comments for '{product_name}'...")
    collector = RedditDataCollector()
    raw_data = collector.fetch_product_data(
        product_name=product_name,
        max_posts=args.max_posts,
        max_comments_per_post=args.max_comments,
    )

    total_comments = len(raw_data.get("comments", []))
    total_reviews = len(raw_data.get("review_texts", []))
    print(f" -> Fetched {total_reviews} submission text entries and {total_comments} comments from Reddit.")

    # 2. Preprocess fetched data using reference pipeline
    print(f"\n[Step 2/3] Preprocessing raw Reddit data using reference preprocessing pipeline...")
    processed_products = preprocess_reddit_reviews_dict([raw_data])
    
    if not processed_products or not processed_products[0].get("text"):
        print(" -> Warning: No review entries met the preprocessing criteria (>15 words after cleaning).")
        preprocessed_reviews = []
    else:
        preprocessed_reviews = processed_products[0]["text"]

    print(f" -> Preprocessing finished. Extracted {len(preprocessed_reviews)} valid review items.")

    # Save output if requested
    combined_result = {
        "product_name": product_name,
        "product_id": raw_data["product"]["$oid"],
        "preprocessed_review_count": len(preprocessed_reviews),
        "preprocessed_reviews": preprocessed_reviews,
        "raw_reddit_data": raw_data,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, indent=2, ensure_ascii=False)
        print(f" -> Saved collected & preprocessed data to: {out_path}")

    # 3. Model Inference (if requested or by default if --run-model set)
    model_analysis = None
    if args.run_model or args.save_model_output:
        if not preprocessed_reviews:
            print("\n[Step 3/3] Cannot run model inference: No preprocessed review evidence found.")
        else:
            print(f"\n[Step 3/3] Loading ProductComparator model and running inference on {len(preprocessed_reviews)} review items...")
            try:
                comparator = ProductComparator()
                model_analysis = comparator.analyze(
                    product_name=product_name,
                    reviews=preprocessed_reviews,
                    specifications={},  # Specs can be passed here if available
                )
                print(" -> Model inference successfully completed!")
                print(f" -> Top Useful Reviews Count: {len(model_analysis.get('top_reviews', []))}")
                print(f" -> Aspect Summary Sections: {len(model_analysis.get('product_summary', []))}")

                # Print summary of top aspects found
                aspects_found = [
                    aspect["top_aspect_name"]
                    for aspect in model_analysis.get("product_summary", [])
                    if aspect.get("sub_aspects")
                ]
                if aspects_found:
                    print(f" -> Aspects identified with confidence: {', '.join(aspects_found)}")
                else:
                    print(" -> No sub-aspects exceeded threshold criteria.")

                if args.save_model_output:
                    model_out_path = Path(args.save_model_output)
                    model_out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(model_out_path, "w", encoding="utf-8") as f:
                        json.dump(model_analysis, f, indent=2, ensure_ascii=False)
                    print(f" -> Saved model analysis results to: {model_out_path}")

            except Exception as e:
                print(f" -> Error during model inference: {e}")

    print("\n[Done] Pipeline complete!\n")
    return {
        "reddit_data": combined_result,
        "model_analysis": model_analysis,
    }


if __name__ == "__main__":
    main()
