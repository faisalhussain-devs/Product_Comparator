import json
import random
from tqdm import tqdm
from pathlib import Path
from textwrap import dedent
from string import Template
from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    ThinkingConfig,
    Tool,
    ToolCodeExecution,
)
import os

MODEL_NAME = None
BATCH_SIZE = None
INPUT_FILE = None
OUTPUT_FILE = None
FAILED_BATCHES_FILE = None
PROMPT_TEMPLATE = None
GENERATION_CONFIG = None
SAFETY_SETTINGS = None


def init_config():
    global MODEL_NAME, BATCH_SIZE, INPUT_FILE, OUTPUT_FILE, FAILED_BATCHES_FILE, PROMPT_TEMPLATE, GENERATION_CONFIG, PROJECT_ID, LOCATION, SAFETY_SETTINGS
    MODEL_NAME = "gemini-2.5-flash"
    PROJECT_ID = "product-comparator-474516" 
    LOCATION = "global"
    if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
        PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    BATCH_SIZE = 1 
    
    INPUT_FILE = Path(r"data_full_length/high_confidence_reviews.json")
    OUTPUT_FILE = "labeled_reviews_final.jsonl"
    FAILED_BATCHES_FILE = "failed_batches.jsonl"

    PROMPT_TEMPLATE = """You are an **expert comparative product review annotator** specializing in structured 
                                      aspect-based sentiment labeling for AI training data. --- ### 
                                      OBJECTIVE From the following review(s), extract **products**, **aspects**, and **comparative sentiments** in **strict JSON format**. --- ### CORE TASKS 1. Identify the **main product** (the one being reviewed or implied to be the reviewer’s focus). 2. Identify any **compared products** (alternatives or references to other models). 3. For each product: - Extract all **aspects** (features, qualities, or experiences) discussed. - Each aspect must include: - "name" → concise aspect (e.g., "Battery life", "Display quality") - "category" → one of: ["Pricing/Value", "Performance", "Camera", "Battery", "Display", "Build Quality", "Software Experience", "Product Feature", "Other"] - "sentiment" → "positive", "neutral", or "negative" - "compared_to" → list of product names being compared, if any 4. Maintain **directional consistency**: - If Product A is said to be "better" or "cheaper" than Product B → A’s sentiment = "positive", B’s = "negative". --- ### QUALITY CONTROL RULES #### 1. **Relevance Filter** - **Ignore or exclude** reviews that: - Are unrelated to a product or product experience. - Do not discuss any features, performance, or comparisons. - Are jokes, off-topic, spam, or lack product-specific content. - If a review is irrelevant, **do not output it at all**. #### 2. **ID–Product Consistency** - Each id follows the pattern "brand_product_name-<digits>_<digits>". - The **main_product name** must **match or align** with the product name derived from the ID. - Example: ID "samsung_galaxy_tab_s10_ultra-13362_136" → main_product must include "Samsung Galaxy Tab S10 Ultra". - If there is a mismatch or uncertain mapping → **skip the review** (do not output). #### 3. **Content Sufficiency** - Include only reviews with **at least one meaningful aspect or comparison**. - Skip reviews that only describe unrelated experiences or lack evaluative content. --- ### OUTPUT REQUIREMENTS - Output **one valid JSON object** only. - Do **not** include "review_text" again. - Must include: - "id" → string - "main_product" → string - "products" → list of product objects - Each product has: "name", "role", "aspects" - Each aspect has: "name", "category", "sentiment", "compared_to" - "role" can be "reviewed" or "compared". - "compared_to" can be empty if not applicable. - Keep identical key ordering and schema as the canonical example. --- ### EXAMPLE **Input Review:** Gonna get the A35 or Note 13 Pro instead, cheaper, better value for money, and has a card slot. **Output JSON:** { "id": "samsung_galaxy_s24_fe-13262_921", "main_product": "Samsung Galaxy S24 FE", "products": [ { "name": "Samsung Galaxy S24 FE", "role": "reviewed", "aspects": [ {"name": "Price", "category": "Pricing/Value", "sentiment": "negative", "compared_to": ["Samsung Galaxy A35", "Redmi Note 13 Pro"]}, {"name": "Value for money", "category": "Pricing/Value", "sentiment": "negative", "compared_to": ["Samsung Galaxy A35", "Redmi Note 13 Pro"]}, {"name": "Card slot", "category": "Product Feature", "sentiment": "negative", "compared_to": ["Samsung Galaxy A35", "Redmi Note 13 Pro"]} ] }, { "name": "Samsung Galaxy A35", "role": "compared", "aspects": [ {"name": "Price", "category": "Pricing/Value", "sentiment": "positive"}, {"name": "Value for money", "category": "Pricing/Value", "sentiment": "positive"}, {"name": "Card slot", "category": "Product Feature", "sentiment": "positive"} ] }, { "name": "Redmi Note 13 Pro", "role": "compared", "aspects": [ {"name": "Price", "category": "Pricing/Value", "sentiment": "positive"}, {"name": "Value for money", "category": "Pricing/Value", "sentiment": "positive"}, 
                                      {"name": "Card slot", "category": "Product Feature", "sentiment": "positive"} ] } ] } --- ### FINAL INSTRUCTION Return only one **valid JSON object** that passes all above filters. If the review does not qualify (irrelevant, ID mismatch, or lacks product aspects), **omit it entirely** — do not produce an empty JSON or error message. Now process and label the following review(s)"""
    SAFETY_SETTINGS = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
    ]


def validate_json_structure(data):
    """Checks if Gemini output matches required schema."""
    required_keys = {"id", "main_product", "products"}
    if not isinstance(data, dict):
        return False
    if not all(k in data for k in required_keys):
        return False
    if not isinstance(data["products"], list):
        return False
    for product in data["products"]:
        if not isinstance(product, dict):
            return False
        if "name" not in product or "role" not in product or "aspects" not in product:
            return False
        if not isinstance(product["aspects"], list):
            return False
    return True


def append_to_jsonl(data_list, filename):
    """Appends data safely to JSONL file."""
    with open(filename, "a", encoding="utf-8") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def labelling_data(all_reviews):
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    processed_ids = set()

    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming: checking already processed reviews in {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
        print(f"Found {len(processed_ids)} already processed reviews. Skipping them.")

    all_review_ids = [rid for rid in all_reviews.keys() if rid not in processed_ids]
    if not all_review_ids:
        print("All reviews already processed. Exiting.")
        return

    random.shuffle(all_review_ids)
    print(len(all_review_ids))
    review_batches_ids = [
        all_review_ids[i:i + BATCH_SIZE] for i in range(0, len(all_review_ids), BATCH_SIZE)
    ]

    print(f"Starting processing {len(all_review_ids)} reviews ({len(review_batches_ids)} batches)")
    
    progress_bar = tqdm(review_batches_ids, desc="Labeling")
    for batch_ids in progress_bar:
        batch_for_prompt = [{"id": rid, "review_text": all_reviews[rid]} for rid in batch_ids]
        reviews_json_str = json.dumps(batch_for_prompt, indent=2)
        success = False
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=reviews_json_str, 
                    config=GenerateContentConfig(
                        system_instruction=PROMPT_TEMPLATE,
                        safety_settings=SAFETY_SETTINGS,))
            cleaned_response = (
                response.text.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            parsed = json.loads(cleaned_response)
            if validate_json_structure(parsed):
                parsed["text"] = all_reviews[parsed["id"]]
                append_to_jsonl([parsed], OUTPUT_FILE)
                success = True
            else:
                append_to_jsonl(
                    [{"error": "Invalid schema", "data": parsed, "batch": batch_for_prompt}],
                    FAILED_BATCHES_FILE,
                )
                success = True
        except Exception as e:
            print(f" failed: {e}")
        if not success:
            append_to_jsonl(
                [{"error": "error", "batch_data": batch_for_prompt}],
                FAILED_BATCHES_FILE,
            )

    print(f"\n All processing finished.\nSaved results → {OUTPUT_FILE}\nFailed batches → {FAILED_BATCHES_FILE}")


def run_gemini_pipeline():
    if not INPUT_FILE.exists():
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        try:
            reviews = json.load(f)
            print(len(reviews))
        except json.JSONDecodeError:
            print("Could not decode input JSON file.")
            return
    all_reviews = {r["id"]: r["review_text"] for r in reviews}
    labelling_data(all_reviews)

if __name__ == "__main__":
    init_config()
    run_gemini_pipeline()
