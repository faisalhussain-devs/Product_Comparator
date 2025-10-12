import json
from pathlib import Path
from typing import Dict

PROMPT_TEMPLATE = """
You are an expert AI Product Analyst. Your task is to analyze a single customer review.

**Task Definition:**
1.  **Score Usefulness:** Assign a `usefulness_score` on a floating-point scale from 0.0 to 10.0. A more detailed and specific review is more useful. If the review is not in English, the score must be 0.0.
2.  **Determine Threshold:** Provide your personal `usefulness_threshold` score for this single review. A review scoring below this threshold is generally not useful.

**Input Format:**
You will be provided a JSON object with a unique `id` and `review_text`.

**Output Format:**
Your response MUST be a single, valid JSON object with TWO top-level keys:
1.  `usefulness_threshold`: A single float number for this review.
2.  `review_analysis`: An object containing:
    - `id`: The original review ID.
    - `usefulness_score`: The calculated score (float).

Do not include any other text or explanations outside of this JSON structure.

**Analyze the following review:**
{review_json}
"""


def run_gemini_pipeline() -> Dict[str, str]:
    """Reads product data and compiles all reviews/comments into a single dictionary."""
    text = {}
    data_path = Path(r"E:\Product Comparator\data\reddit_product_site_data\data_json\product_data_cleaned.json")
    
    if not data_path.exists():
        print(f"Error: The file at {data_path} was not found.")
        return {}

    with open(data_path, "r", encoding="utf-8") as f:
        try:
            sample_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file at {data_path}.")
            return {}
            
    for product in sample_data:
        product_id = product.get("product_id")
        if not product_id:
            continue
            
        reviews = product.get("reviews", [])
        comments = product.get("comments_texts", [])
        
        j_last = len(reviews)
        
        for j, review_text in enumerate(reviews):
            if len(review_text.split()) > 8:
                text[f"{product_id}_{j}"] = review_text
                
        for k, comment_text in enumerate(comments):
            if len(comment_text.split()) > 8:
                text[f"{product_id}_{j_last + k}"] = comment_text
                
    return text

def get_youtube_sentences(yt_path):
    text = {}
    if not yt_path.exists():
        print(f"Error: The file at {yt_path} was not found.")
        return {}

    with open(yt_path, "r", encoding="utf-8") as f:
        try:
            youtube_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file at {yt_path}.")
            return {}
    for i, transcript in enumerate(youtube_data):
        id = transcript["product"]
        for j, review_text in enumerate(transcript["transcript"]):
            if len(review_text.split()) > 8:
                text[f"yt_{id}_{i}_{j}"] = review_text
    return text


def create_gemini_batch_file(
    reviews_dict: Dict[str, str],
    output_filepath: str,
    model_name: str = "models/gemini-1.5-flash-latest",
) -> None:
    """Creates a JSONL file for the Gemini Batch API from a dictionary of reviews."""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for review_id, review_text in reviews_dict.items():
            clean_review = str(review_text).strip()
            if not clean_review:
                continue
            
            review_input_dict = {
                "id": review_id,
                "review_text": clean_review
            }
            review_input_json_string = json.dumps(review_input_dict)
            full_prompt_text = PROMPT_TEMPLATE.format(review_json=review_input_json_string)

            json_line_data = {
                "request": {
                        "contents": [{
                        "role": "user",
                        "parts": [{"text": full_prompt_text}]
                    }],
                    "generation_config": {
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json" 
                    }
                }
            }
            f.write(json.dumps(json_line_data) + "\n")

    print(f"Successfully created batch file at: {output_filepath}")

if __name__ == "__main__":
    final_reviews = run_gemini_pipeline()
    yt_path = Path("E:\Product Comparator\data\youtube_data\youtube_data.json")
    yt_script = get_youtube_sentences(yt_path)

    final_reviews |= yt_script
    batch_file_path = Path(r"E:\Product Comparator\data\batch_api_data.jsonl")
    batch_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(final_reviews)} reviews to process.")
    create_gemini_batch_file(
        reviews_dict=final_reviews,
        output_filepath=str(batch_file_path) # Convert Path object to string for open()
    )