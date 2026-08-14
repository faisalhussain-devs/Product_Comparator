import json
import re
from pathlib import Path


def clean_comment(comment: dict) -> dict:
    """Clean comment dict to keep only body/content and upvote counts, removing user IDs and authors."""
    if not isinstance(comment, dict):
        return {}

    body = comment.get("body") or comment.get("text") or ""
    ups = comment.get("ups")
    if ups is None:
        ups = comment.get("upvotes", 0)

    cleaned = {
        "body": body,
        "ups": ups if isinstance(ups, int) else 0,
    }

    post_title = comment.get("post_title")
    if post_title:
        cleaned["post_title"] = post_title

    return cleaned


def normalize_key(name: str) -> str:
    """Normalize string for consistent hash lookup."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def generate_aliases(name: str, key: str = "") -> list[str]:
    """Generate common search aliases and popular variants for a product name."""
    aliases = set()
    norm_name = normalize_key(name)
    if norm_name:
        aliases.add(norm_name)

    if key:
        norm_key = normalize_key(key)
        if norm_key:
            aliases.add(norm_key)

    # Strip brand prefixes ("apple ", "samsung ", "google ", "sony ", "oneplus ", "xiaomi ")
    brand_prefix_pattern = r"^(apple|samsung|google|sony|oneplus|xiaomi|motorola|realme|vivo|oppo|asus)\s+"
    without_brand = re.sub(brand_prefix_pattern, "", norm_name)
    if without_brand and without_brand != norm_name:
        aliases.add(without_brand)

    # Handle iPhone specific shorthand variants (e.g., "iphone 17 pro max" -> "17 pro max", "17pm", "pro max 17")
    if "iphone" in norm_name:
        no_iphone = norm_name.replace("iphone", "").strip()
        if no_iphone:
            aliases.add(no_iphone)

        numbers = re.findall(r"\d+", norm_name)
        if numbers:
            num = numbers[0]
            if "pro max" in norm_name:
                aliases.add(f"{num} pro max")
                aliases.add(f"{num}pm")
                aliases.add(f"pro max {num}")
            elif "pro" in norm_name:
                aliases.add(f"{num} pro")
                aliases.add(f"pro {num}")
                aliases.add(f"{num}p")
            elif "plus" in norm_name:
                aliases.add(f"{num} plus")
                aliases.add(f"plus {num}")
            elif "mini" in norm_name:
                aliases.add(f"{num} mini")
                aliases.add(f"mini {num}")

    # Handle Pixel / Galaxy specific shorthand
    if "galaxy" in norm_name or "pixel" in norm_name:
        no_brand = re.sub(r"^(samsung|google)\s+", "", norm_name)
        if no_brand:
            aliases.add(no_brand)

    return sorted(list(aliases))


def build_corpus():
    base_dir = Path(__file__).resolve().parent
    example_dir = base_dir / "example_data"
    reviews1_file = example_dir / "Comparator.reviews.json"
    reviews2_file = example_dir / "Comparator.review2.json"
    specs_file = example_dir / "Comparator.specifications.json"

    reviews_sources = []
    if reviews1_file.exists():
        with open(reviews1_file, "r", encoding="utf-8") as f:
            reviews_sources.append(json.load(f))
    if reviews2_file.exists():
        with open(reviews2_file, "r", encoding="utf-8") as f:
            reviews_sources.append(json.load(f))

    specs_by_oid = {}
    specs_by_name = {}
    specs_by_key = {}
    if specs_file.exists():
        with open(specs_file, "r", encoding="utf-8") as f:
            specs_data = json.load(f)
            for item in specs_data:
                oid = item.get("_id", {}).get("$oid")
                key = item.get("key")
                name = item.get("name")
                if oid:
                    specs_by_oid[oid] = item
                if key:
                    specs_by_key[key] = item
                if name:
                    specs_by_name[normalize_key(name)] = item

    products_corpus = []
    product_index = {}

    for source in reviews_sources:
        for r_item in source:
            prod_obj = r_item.get("product")
            prod_name_field = r_item.get("product_name") or r_item.get("name")

            oid = prod_obj.get("$oid") if isinstance(prod_obj, dict) else prod_obj

            matched_spec = None
            if oid:
                matched_spec = specs_by_oid.get(oid)
            if not matched_spec and prod_name_field:
                matched_spec = specs_by_name.get(normalize_key(prod_name_field))
            if not matched_spec and isinstance(prod_obj, str):
                matched_spec = specs_by_key.get(prod_obj)

            if matched_spec:
                product_name = matched_spec.get("name") or prod_name_field or f"Product_{oid}"
                spec_key = matched_spec.get("key", "")
                spec_dict = matched_spec
            elif prod_name_field:
                product_name = prod_name_field
                spec_key = normalize_key(prod_name_field).replace(" ", "_")
                spec_dict = {}
            else:
                product_name = f"Product_{oid}" if oid else f"Product_{len(products_corpus)}"
                spec_key = oid or ""
                spec_dict = {}

            raw_comments = r_item.get("comments", [])
            cleaned_comments = [clean_comment(c) for c in raw_comments if isinstance(c, dict)]

            corpus_entry = {
                "id": spec_key or oid or f"prod_{len(products_corpus)}",
                "name": product_name,
                "reviews": {
                    "comments": cleaned_comments,
                    "review_texts": r_item.get("review_texts", []),
                },
                "specifications": spec_dict,
            }

            corpus_idx = len(products_corpus)
            products_corpus.append(corpus_entry)

            # Generate all search aliases for hash indexing
            aliases = generate_aliases(product_name, spec_key)
            for alias in aliases:
                if alias not in product_index:
                    product_index[alias] = corpus_idx

    # Save packaged corpus files
    corpus_path = example_dir / "products_corpus.json"
    index_path = example_dir / "product_index.json"

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(products_corpus, f, indent=2, ensure_ascii=False)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(product_index, f, indent=2, ensure_ascii=False)

    print(f"Successfully built dataset corpus:")
    print(f" - Total packaged products: {len(products_corpus)} -> saved to {corpus_path.name}")
    print(f" - Hash index search aliases: {len(product_index)} -> saved to {index_path.name}")


if __name__ == "__main__":
    build_corpus()
