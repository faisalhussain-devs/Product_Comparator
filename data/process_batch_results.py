import json
from pathlib import Path

def parse_and_summarize_batch_results(results_filepath: Path, output_json_path: Path) -> None:
    """
    Parses a Gemini batch API output file, saves the data to a JSON file,
    and calculates the total token usage.

    Args:
        results_filepath (Path): The path to your .jsonl output file from Vertex AI.
        output_json_path (Path): The path where the output .json file will be saved.
    """
    all_results = []
    c = 0

    print(f"Starting to process file: {results_filepath}")

    with open(results_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                full_line_json = json.loads(line)
                if full_line_json.get("status") not in (None, "", "OK", "success"):
                    continue

                response = full_line_json.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    continue

                response_text = parts[0]["text"]
                result_data = json.loads(response_text)

                analysis = result_data.get("review_analysis", {})
                usefulness_score = analysis.get("usefulness_score")
                usefulness_threshold = result_data.get("usefulness_threshold")

                prompt_text = full_line_json["request"]["contents"][0]["parts"][0]["text"]
                json_start_index = prompt_text.find('{"id":')
                if json_start_index == -1:
                    continue

                original_review_json_str = prompt_text[json_start_index:]
                original_review_data = json.loads(original_review_json_str)
                review_id = original_review_data.get("id")
                review_text = original_review_data.get("review_text")
                if usefulness_score >= 8:
                    c += 1
                    all_results.append({
                        "id": review_id,
                        "review_text": review_text,
                        "usefullness_score":usefulness_score
                    })

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                continue
    print(c)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print("\n--- Processing Complete ---")
    print(f"Sucessfully saved results reviews to: {output_json_path}")


if __name__ == "__main__":
    results_file = Path(r"E:\Product Comparator\data\output_reviews_score_prediction-model-2025-10-12T00_16_45.439014Z_predictions.jsonl")
    output_json_file = Path(r"E:\Product Comparator\data\final_labelled_reviews.json")

    if results_file.exists():
        parse_and_summarize_batch_results(results_file, output_json_file)
    else:
        print(f"Error: Results file not found. Please check the path: {results_file}")