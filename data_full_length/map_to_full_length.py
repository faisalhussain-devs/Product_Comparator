import pandas as pd
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm  # progress bar

full_reviews = pd.read_json("data_full_length/data_full_length.json")
broken_sentences = pd.read_json(Path(r"E:\Product Comparator\data\final_labelled_reviews.json"))

full_reviews = full_reviews

full_reviews["base_product_id"] = full_reviews["id"].apply(
    lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x
)
full_groups = full_reviews.groupby("base_product_id")

model = SentenceTransformer('all-MiniLM-L6-v2')

product_embeddings_cache = {}

mapped = []

for _, row in tqdm(broken_sentences.iterrows(), total=len(broken_sentences), desc="Mapping short → full reviews"):
    base_id = "_".join(row["id"].split("_")[:-1])
    if base_id not in full_groups.groups:
        mapped.append({
            "product_id": row["id"],
            "short_text": row["review_text"],
            "usefullness_score": row["usefullness_score"],
            "matched_review": None,
            "similarity": None
        })
        continue

    product_reviews = full_groups.get_group(base_id)
    full_texts = product_reviews["review_text"].tolist()

    if base_id not in product_embeddings_cache:
        product_embeddings_cache[base_id] = model.encode(full_texts, convert_to_tensor=True)

    full_embs = product_embeddings_cache[base_id]
    short_emb = model.encode(row["review_text"], convert_to_tensor=True)

    cosine_scores = util.cos_sim(short_emb, full_embs)[0]
    best_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_idx].item()

    if best_score > 0.78:
        matched_review = full_texts[best_idx]
    else:
        matched_review = None 

    mapped.append({
        "product_id": row["id"],
        "short_text": row["review_text"],
        "usefullness_score": row["usefullness_score"],
        "matched_review": matched_review,
        "similarity": round(best_score, 3) if matched_review else None
    })

output_path = Path(r"E:\Product Comparator\data_full_length\map_to_full_review.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(mapped, f, indent=4, ensure_ascii=False)

print(f"\n Mapping complete. Saved {len(mapped)} entries to {output_path}")
