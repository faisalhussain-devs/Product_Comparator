import json
from pathlib import Path


def run_gemini_pipeline() :
    """Reads product data and compiles all reviews/comments into a single dictionary."""
    text = {}
    data_path = Path(r"E:\Product Comparator\data_full_length\product_data_cleaned_full_length.json")
    
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


def create_gemini_batch_file(reviews_dict, output_filepath):
    review_input = []
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for review_id, review_text in reviews_dict.items():
            clean_review = str(review_text).strip()
            if not clean_review:
                continue
            
            review_input.append({
                "id": review_id,
                "review_text": clean_review
            })
        review_input_json_string = json.dumps(review_input, indent=4)
        f.write(review_input_json_string)

    print(f"Successfully created batch file at: {output_filepath}")

if __name__ == "__main__":
    final_reviews = run_gemini_pipeline()
    batch_file_path = Path(r"E:\Product Comparator\data_full_length.json")
    batch_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(final_reviews)} reviews to process.")
    create_gemini_batch_file(
        reviews_dict=final_reviews,
        output_filepath=str(batch_file_path) # Convert Path object to string for open()
    )