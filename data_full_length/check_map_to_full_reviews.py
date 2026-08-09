from pathlib import Path
import json

c_null = 0
c_confidence = 0
c_8 = 0

with open(Path("data_full_length/map_to_full_review.json"), 'r', encoding='utf-8') as f:
    list_long_reviews = json.load(f)

output_file = []
cache = set()

for i in list_long_reviews:
    usefullness_score = i.get("usefullness_score", 0)
    sim = i.get("similarity", None)
    matched_review = i.get("matched_review", None)
    short_text = i.get("short_text", "").strip()

    if usefullness_score >= 8 and usefullness_score <9:
        c_8 += 1

        if sim is None:
            c_null += 1
        elif sim < 0.78:
            c_confidence += 1

        review_text = matched_review.strip() if matched_review else short_text

        already_present = any(short_text in r for r in cache)
        already_added = review_text in cache

        if not already_added and not already_present:
            output_file.append({
                "id": i["product_id"],
                "review_text": review_text,
            })
            cache.add(review_text)

print("Less confident ones (score ≤ 0.78):", c_confidence)
print("NULL ones:", c_null)
print("Total reviews with usefulness ≥ 9:", c_8)
print("Final count (after deduplication):", len(output_file))

output_path = Path(r"data_full_length/high_confidence_reviews_8.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_file, f, indent=4, ensure_ascii=False)

print(f"\n Saved {len(output_file)} clean, deduplicated reviews to {output_path}")
