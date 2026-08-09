import json
import pandas as pd
from pathlib import Path

INPUT_PATH = Path(r"E:\Product Comparator\aspect_analysis\cleaned_aspect_reviews_8-9_new.json")
OUTPUT_PATH = Path(r"E:\Product Comparator\aspect_analysis\final_aspa_data.json")
INPUT_PATH2 = Path(r"aspect_analysis/cleaned_aspect_reviews_9-10.json")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
laa = []
with open(INPUT_PATH2, "r", encoding="utf-8") as f:
    data_1 = json.load(f)
data = data+data_1
c_product = 0
c_text = 0
inputs, labels = [], []

for sample in data:
    review_text = sample.get("review_text", "").strip()
    if not review_text:
        review_text = sample.get("text", "").strip()
    main_product = sample.get("main_product", "").strip()
    sample_id = sample.get("id", "")
    
    if not review_text :
        print(f"Skipping incomplete record: {sample_id}")
        c_text += 1
        continue
    if not main_product:
        c_product += 1
        print(f"Skipping incomplete record: {sample_id}")
        continue
    
    label_set = set()
    
    try:
        for product in sample.get("products", []):
            product_name = product.get("name", "")
            role = product.get("role", "").lower()  # 'reviewed' or 'compared'
            
            for asp in product.get("aspects", []):
                category = asp.get("category", "").strip()
                aspect = asp.get("name", "").strip()
                compared_to = asp.get("compared_to", [])
                
                if not category or not aspect:
                    continue

                base_label = f"{category}|{aspect}"
                
                if role == "compared":
                    base_label = f"compared|{product_name}|{base_label}"
                else:
                    base_label = f"{base_label}"

                if compared_to:
                    for target in compared_to:
                        label_set.add(f"{base_label}")
                else:
                    label_set.add(base_label)
                    
        model_input = f"Main product: {main_product}. Review: {review_text}"
        inputs.append(model_input)
        labels.append(list(label_set))
        laa.extend(list(label_set))
        
    except Exception as e:
        print(f"Error parsing sample {sample_id}: {e}")

print(len(laa), laa[:10])
df = pd.DataFrame({
    "input": inputs,
    "label": labels
})

# Remove rows with empty labels
df = df[df["label"].map(len) > 0].reset_index(drop=True)

data_list = df.to_dict(orient="records")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f" Preprocessed {len(df)} samples saved to {OUTPUT_PATH}")
print("Missing_text", c_text)
print("Missing_product", c_product)

