import json
import pandas as pd
from pathlib import Path

# -----------------------------
# Input / Output paths
# -----------------------------
INPUT_PATH = Path(r"E:\Product Comparator\aspect_analysis\cleaned_aspect_reviews_8-9_new.json")
INPUT_PATH2 = Path(r"aspect_analysis/cleaned_aspect_reviews_9-10.json")
OUTPUT_PATH = Path(r"E:\Product Comparator\\final_aspa_data_with_sentiments.json")

# -----------------------------
# Load both files
# -----------------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(INPUT_PATH2, "r", encoding="utf-8") as f:
    data_1 = json.load(f)

data = data + data_1

# -----------------------------
# Initialize counters and containers
# -----------------------------
c_product, c_text = 0, 0
inputs, labels, all_labels = [], [], []

# -----------------------------
# Process each review
# -----------------------------
for sample in data:
    review_text = sample.get("review_text", "").strip()
    if not review_text:
        review_text = sample.get("text", "").strip()

    main_product = sample.get("main_product", "").strip()
    sample_id = sample.get("id", "")

    if not review_text:
        print(f"Skipping incomplete record (no text): {sample_id}")
        c_text += 1
        continue
    if not main_product:
        print(f"Skipping incomplete record (no product): {sample_id}")
        c_product += 1
        continue

    sentiment_dict = {}  # new dictionary for sentiment mapping

    try:
        for product in sample.get("products", []):
            product_name = product.get("name", "")
            role = product.get("role", "").lower()

            for asp in product.get("aspects", []):
                category = asp.get("category", "").strip()
                aspect = asp.get("name", "").strip()
                sentiment = asp.get("sentiment", "").strip().lower()
                compared_to = asp.get("compared_to", [])

                if not category or not aspect or not sentiment:
                    continue

                # Build base label key
                base_label = f"{category}|{aspect}"
                if role == "compared":
                    base_label = f"compared|{product_name}|{base_label}"

                # Add to sentiment dictionary
                if compared_to:
                    for _ in compared_to:
                        sentiment_dict[base_label] = sentiment
                else:
                    sentiment_dict[base_label] = sentiment

        # If no aspects were found, skip
        if not sentiment_dict:
            continue

        model_input = f"Main product: {main_product}. Review: {review_text}"
        inputs.append(model_input)
        labels.append(sentiment_dict)
        all_labels.extend(list(sentiment_dict.keys()))

    except Exception as e:
        print(f"Error parsing sample {sample_id}: {e}")

# -----------------------------
# Create final DataFrame
# -----------------------------
df = pd.DataFrame({
    "input": inputs,
    "sentiments": labels
})

# Remove rows with no sentiment mapping
df = df[df["sentiments"].map(len) > 0].reset_index(drop=True)

# -----------------------------
# Save as JSON
# -----------------------------
data_list = df.to_dict(orient="records")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"\n✅ Preprocessed {len(df)} samples saved to {OUTPUT_PATH}")
print("🟡 Missing text:", c_text)
print("🟡 Missing product:", c_product)
print("🟢 Example output:\n", json.dumps(data_list[:2], indent=2, ensure_ascii=False))
