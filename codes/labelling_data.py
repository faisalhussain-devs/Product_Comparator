import google.generativeai as genai
import os
import time
import json
import random
from tqdm import tqdm
import pandas as pd
import sqlite3
import numpy as np


API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-pro-latest"
REQUESTS_PER_MINUTE = 5
SECONDS_TO_WAIT = 60 / REQUESTS_PER_MINUTE
BATCH_SIZE = 15
MAX_ITERATIONS = 10  
MIN_ITERATIONS = 5   
CONVERGENCE_THRESHOLD = 0.25 
THRESHOLD = 0
NON_CONVERGENCE_PENALTY = 0.5
DB_PATH = "your_database_name.db"
TABLE_NAME = "reviews_table"
OUTPUT_FILE = "labeled_reviews_final.jsonl"
FAILED_BATCHES_FILE = "failed_batches.jsonl"


PROMPT_TEMPLATE = """
You are an expert AI Product Analyst. Your task is to analyze a batch of customer reviews for a single product. For each review, you will score its usefulness, classify its type, and finally, provide an overall usefulness threshold for the batch.

**Task Definition:**
1.  **Score Usefulness:** Assign a `usefulness_score` on a **floating-point scale from 0.0 to 10.0 based on how useful the review is for analyzing a product. The better the review is the higher the score**.
2.  **Determine Threshold:** After analyzing all reviews, provide a single `usefulness_threshold` score. Reviews scoring below this are generally not useful.

**Input Format:**
You will be provided a JSON array of review objects. Each object has a unique `id` and `review_text`.

**Output Format:**
Your response MUST be a single, valid JSON object with TWO top-level keys:
1.  `usefulness_threshold`: A single float number.
2.  `review_analysis`: An array of objects, where each object contains:
    - `id`: The original review ID.
    - `usefulness_score`: The calculated score (float).

Do not include any other text, greetings, or explanations outside of this JSON structure.

**Analyze the following batch of reviews:**
{reviews_json}
"""


def get_reviews_from_db(db_path, table_name):
    try:
        conn = sqlite3.connect(db_path)
        # Your table must have columns named 'id' and 'review_text'
        df = pd.read_sql_query(f"SELECT id, review_text FROM {table_name}", conn)
        conn.close()
        # Convert to dict: {id: text}
        reviews_dict = dict(zip(df['id'], df['review_text']))
        return reviews_dict
        
    except Exception as e:
        print(f"Error connecting to or reading from the database: {e}")
        return []
    
def append_to_jsonl(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def has_converged(scores, tol=0.1):
    if not scores:  
        return False, None
    recent = scores[len(scores)//2:]  
    std = float(np.std(recent))
    return std < tol, std

def bootstrap_mean_ci(scores, n_boot=500, ci=0.95, min_samples=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return None, None, None, None
    mean_observed = float(scores.mean())

    if n < min_samples:
        return mean_observed, None, None, None

    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        boots[i] = sample.mean()

    lower_pct = (1.0 - ci) / 2.0 * 100
    upper_pct = (1.0 + ci) / 2.0 * 100
    lo = float(np.percentile(boots, lower_pct))
    hi = float(np.percentile(boots, upper_pct))
    return mean_observed, lo, hi, hi - lo

def confidence_score(scores_map, n_boot=500, ci=0.95, min_samples=3, cap_percentile=95, target_n=5):
    results = {}
    widths = []
    for rid, data in scores_map.items():
        _, _, _, width = bootstrap_mean_ci(data['scores'], n_boot=n_boot, ci=ci, min_samples=min_samples)
        results[rid] = {"width": width, "n": len(data['scores'])}
        if width is not None:
            widths.append(width)
    if widths:
        cap = float(np.percentile(widths, cap_percentile))
        if cap == 0:
            cap = max(widths)
    else:
        cap = 1.0

    for rid, info in results.items():
        width = info["width"]
        n = info["n"]
        if width is None:
            info["confidence"] = None
            continue
        normalized = 1.0 - min(width / cap, 1.0)
        if scores_map[rid]["converged"] == 1:  
            factor = 1.0 / (1.0 + 0.1 * (n/target_n - 1))
        else:  
            factor = NON_CONVERGENCE_PENALTY

        conf_score = 10.0 * normalized * factor 
        scores_map[rid]["confidence_score"] = float(conf_score)
    return scores_map

def validate_gemini_output(input_ids: list ,raw_output: str) -> dict:
    output_ids = set()
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if "review_analysis" not in parsed:
        raise ValueError("Missing 'review_analysis' in output")
    if "usefulness_threshold" not in parsed:
        raise ValueError("Missing 'usefulness_threshold' in output")

    threshold = parsed["usefulness_threshold"]
    if not isinstance(threshold, (int, float)):
        raise ValueError("'usefulness_threshold' must be a number")
    
    if not (0 <= threshold <= 10):
            raise ValueError("'usefulness_threshold' must be between 0 and 10")

    reviews = parsed["review_analysis"]
    if not isinstance(reviews, list):
        raise ValueError("'review_analysis' must be a list")

    for i, item in enumerate(reviews):
        if not isinstance(item, dict):
            raise ValueError(f"Review at index {i} is not a dictionary")

        if "id" not in item:
            raise ValueError(f"Review {i} missing 'id'")
        if not isinstance(item["id"], (str, int)):
            raise ValueError(f"Review {i} 'id' must be str or int")

        if "usefulness_score" not in item:
            raise ValueError(f"Review {i} missing 'usefulness_score'")
        score = item["usefulness_score"]
        if not isinstance(score, (int, float)):
            raise ValueError(f"Review {i} 'usefulness_score' must be a number")
        if not (0 <= score <= 10):
            raise ValueError(f"Review {i} 'usefulness_score' must be between 0 and 10")
        
        output_ids.add(str(item["id"]))    

    if output_ids != set(input_ids):
        raise ValueError(f"ID mismatch! Expected {input_ids}, got {output_ids}")
    return parsed

def labelling_data():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return
    iter_threshold = np.zeros(MAX_ITERATIONS)
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    all_reviews = get_reviews_from_db(DB_PATH, TABLE_NAME)
    all_reviews_copy = all_reviews.copy()
    if not all_reviews:
        print("No reviews to process. Exiting.")
        return
    
    scores_data = {
        id: {'scores': [], 'converged': 0}
        for id in all_reviews
    }

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Starting Iteration {iteration + 1}/{MAX_ITERATIONS} ---")
        all_reviews_ = all_reviews_copy.copy()
        all_review_ids = list(all_reviews_.keys())
        random.shuffle(all_review_ids)
        review_batches_ids = [all_review_ids[i:i + BATCH_SIZE] for i in range(0, len(all_review_ids), BATCH_SIZE)]
        progress_bar = tqdm(review_batches_ids, desc=f"Iter {iteration + 1}")
        for batch_ids in progress_bar:
            try:
                batch = [{"id": id, "review_text": all_reviews_[id]} for id in batch_ids]
                reviews_json_str = json.dumps(batch, indent=2)
                prompt = PROMPT_TEMPLATE.format(reviews_json=reviews_json_str)
                response = model.generate_content(prompt)
                cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
                result_json = validate_gemini_output(batch_ids, cleaned_response)
                labeled_data = result_json.get("review_analysis", [])
                iter_threshold[iteration] += result_json.get("usefulness_threshold")/len(review_batches_ids)

                for item in labeled_data:
                    review_id = item['id']
                    if review_id in scores_data:
                        scores_data[review_id]['scores'].append(item['usefulness_score'])
                        
            except Exception as e:
                print(f"\nAn error occurred while processing a batch: {e}")
                append_to_jsonl([{"error": str(e), "batch_data": batch}], FAILED_BATCHES_FILE)
            
            finally:
                time.sleep(SECONDS_TO_WAIT)
        
        if iteration >= 5:
            mean_std = 0
            c = 0
            for id, data in all_reviews_.items():
                converged, std_per_sample = has_converged(scores_data[id]['scores'], CONVERGENCE_THRESHOLD)
                if std_per_sample is not None:
                    mean_std += std_per_sample
                    c += 1
                if converged:
                    scores_data[id]['converged'] = 1
                    all_reviews_copy.pop(id)
            print(f"Iteration {iteration+1} complete. Std of later half: {mean_std/c:.6f}")

            if len(all_reviews_copy) == 0:
                print(f"\nScores have converged after {iteration + 1} iterations. Stopping early.")
                break

    print("\nAll iterations complete. Saving final results...")

    THRESHOLD = min(iter_threshold) 
    scores_data = confidence_score(scores_data)
    final_results = []
    for review_id, data in scores_data.items():
        scores = data.get('scores', [])
        final_results.append({
            "id": review_id,
            "text": all_reviews[review_id],
            "usefulness_score": np.mean(scores[len(scores)//2:]) if scores else None,
            "confidence_score": data.get('confidence_score')
        })
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
    print(f"Processing finished. Final stabilized labels saved to '{OUTPUT_FILE}'.")

labelling_data()