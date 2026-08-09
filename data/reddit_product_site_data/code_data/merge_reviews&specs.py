import json
import orjson

def merge_specs_reviews(specs_file, reviews_file, output_file):
    """
    Merge product specifications and reviews into a single JSON file.

    Parameters
    ----------
    specs_file : str
        Path to the JSON file containing product specifications.
    reviews_file : str
        Path to the JSON file containing product reviews.
    output_file : str
        Path to save the merged JSON file.

    Process
    -------
    1. Load specifications and reviews data from JSON files.
    2. Create a lookup dictionary for specifications keyed by product _id.
    3. Iterate over reviews and append them to the correct product entry.
    4. Convert the lookup dictionary into a list for final merged output.
    5. Write the merged JSON file using orjson (faster than builtin json).

    Output
    ------
    A JSON file where each record looks like:
    {
        "product_id": "...",
        "name": "...",
        "specifications": {...},
        "reviews": [
            {"review_texts": [...], "comments": [...]},
            ...
        ]
    }
    """
    with open(specs_file, "r", encoding="utf-8") as f:
        specs_data = json.load(f)
    with open(reviews_file, "r", encoding="utf-8") as f:
        reviews_data = json.load(f)

    specs_lookup = {
        str(item["_id"]): {
            "product_id": item.get("key"),
            "name": item.get("name"),
            "specifications": item.get("specification", {}),
            "reviews": []
        }
        for item in specs_data
    }

    for review in reviews_data:
        product_ref = str(review.get("product"))
        if product_ref in specs_lookup:
            specs_lookup[product_ref]["reviews"].append({
                "review_texts": review.get("review_texts", []),
                "comments": review.get("comments", []),
            })

    merged_data = list(specs_lookup.values())

    with open(output_file, "wb") as f:
        f.write(orjson.dumps(merged_data, option=orjson.OPT_INDENT_2))


merge_specs_reviews("E:\Product Comparator\data\Comparator.specifications.json", "E:\Product Comparator\data\Comparator.reviews.json", "product_data.json")
