import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = Path(r"E:\Product Comparator\final_aspa_data_merged_with_sentiments.json")
OUTPUT_PATH = Path(r"E:\Product Comparator\final_aspa_data_hierarchical_with_sentiments.json")

# -------------------------------
# Load data
# -------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------
# Extract all top clusters and sub-clusters
# -------------------------------
all_top_clusters = set()
all_sub_clusters = set()

for review in data:
    for lbl in review["label"]:
        if "|" in lbl:
            top, sub = lbl.split("|", 1)
            all_top_clusters.add(top.strip().lower())
            all_sub_clusters.add(sub.strip().lower())

# -------------------------------
# Create ID mappings
# -------------------------------
top_cluster2id = {top: i for i, top in enumerate(sorted(all_top_clusters))}
sub_cluster2id = {sub: i for i, sub in enumerate(sorted(all_sub_clusters))}

print(f"Total top clusters: {len(all_top_clusters)}")
print(f"Total sub-clusters: {len(all_sub_clusters)}")

# -------------------------------
# Transform data into hierarchical format
# -------------------------------

def sentiment_to_num(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'neutral':
        return 0
    elif sentiment == 'negative':
        return -1
    
hierarchical_data = []

for review in data:
    text = review.get("input", "")
    label_dict = review.get("label", {})

    # Initialize multi-hot vectors
    top_vector = np.zeros(len(top_cluster2id), dtype=int)
    sub_vector = np.zeros(len(sub_cluster2id), dtype=int)
    top_to_sub_ids = defaultdict(list)
    sentiments_by_sub_id = {}

    for lbl, sentiment in label_dict.items():
        if "|" not in lbl:
            continue
        
        top, sub = lbl.split("|", 1)
        top = top.strip().lower()
        sub = sub.strip().lower()
        
        if top not in top_cluster2id or sub not in sub_cluster2id:
            continue

        top_id = top_cluster2id[top]
        sub_id = sub_cluster2id[sub]

        # Mark activation
        top_vector[top_id] = 1
        sub_vector[sub_id] = 1
        top_to_sub_ids[top_id].append(sub_id)
        sentiments_by_sub_id[sub_id] = sentiment_to_num(sentiment)

    hierarchical_data.append({
        "text": text,
        "top_cluster_ids": top_vector.tolist(),
        "sub_cluster_ids": sub_vector.tolist(),
        "top_to_sub_ids": {str(k): v for k, v in top_to_sub_ids.items()},
        "sentiments": sentiments_by_sub_id
    })

# -------------------------------
# Save transformed data
# -------------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(hierarchical_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Hierarchical training data with sentiments saved to: {OUTPUT_PATH}")
print(f"📊 Samples processed: {len(hierarchical_data)}")
