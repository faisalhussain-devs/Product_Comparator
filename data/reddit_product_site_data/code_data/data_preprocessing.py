import json
import pandas as pd
import re

def clean_value(value: str) -> str:
    """Clean and normalize string values in one go."""
    if not isinstance(value, str):
        return value

    value = re.sub(r'https?://\S+|www\.\S+', '', value)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-C
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    value = emoji_pattern.sub('', value)
    value = re.sub(r'\"|\'', '', value)
    value = re.sub(r'\\?/', ',', value)
    value = re.sub(r'(\\n|\n)+', ', ', value)
    value = re.sub(r'\s*,\s*', ', ', value)
    value = re.sub(r',\s*,+', ', ', value)
    value = re.sub(r'\s+', ' ', value).strip()

    return value


def preprocess_sentences(text):
    processed_chunks = []
    for sentence in text:  
        cleaned_chunk = []
        raw_sentences = [s.strip() for s in re.split(r'[.!?*#•]', sentence) if s.strip()]
        for raw_sentence in raw_sentences:
            raw_sentence = clean_value(raw_sentence)
            cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', raw_sentence)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned_chunk.append(cleaned)
        processed_chunks.extend(cleaned_chunk)
    return processed_chunks


def drop_specification_keys(data, drop_keys):
    """Remove specified keys from the 'specifications' dictionary of each item."""
    specs = data.get("specifications")
    if isinstance(specs, dict):
        for key in drop_keys:
            specs.pop(key, None)
    return data


def flatten_data(product, sep="_"):
    """
    Flatten product specifications:
    - Remove everything except 'More Specifications'
    - Rename 'More Specifications' → 'specifications'
    - Flatten title/data pairs recursively
    Clean nested 'reviews' and 'comments' structures:
      - 'reviews' → list of review_text strings
      - 'comments' → two lists: comment bodies & ups
    """

    flat = {}

    def process_specs(specs, parent=""):
        """Recursively flatten title/data pairs into key-value mapping."""
        for entry in specs:
            title = entry.get("title", "").strip().replace(" ", "_")
            data = entry.get("data")

            if isinstance(data, list) and all(isinstance(x, dict) and "title" in x and "data" in x for x in data):
                process_specs(data, parent=f"{parent}{sep}{title}")
            else:
                key = f"{parent}{sep}{title}" if parent else title
                if isinstance(data, list):
                    value = " ".join(str(x).strip() for x in data)
                else:
                    value = data
                flat[key.strip("_")] = clean_value(value)

    if "specifications" in product and isinstance(product["specifications"], dict):
        more_specs = product["specifications"].get("more_specification", [])
        process_specs(more_specs)

    return flat


def preprocess_products(input_file, output_file, drop_columns=None):
    """Preprocess merged product JSON with custom flattening logic."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_data = []
    for product in data:
        product["specifications"] = flatten_data(product)
        reviews = product.get("reviews", [])
        review_texts = []

        # Sometimes reviews = [{"review_texts": [list of sentences]}]
        for r in reviews:
            if isinstance(r, dict):
                texts = r.get("review_texts")
                comments = r.get("comments", [])
                comment_texts, comment_ups = [], []
                if isinstance(texts, list):
                    review_texts.extend([clean_value(t.strip()) for t in texts if isinstance(t, str)])
                elif isinstance(texts, str):
                    review_texts.append(texts.strip())
            elif isinstance(r, str):
                review_texts.append(r.strip())
        
            for c in comments:
                if isinstance(c, dict):
                    body = c.get("body")
                    ups = c.get("ups")
                    if isinstance(body, str):
                        comment_texts.append(clean_value(body.strip()))
                        comment_ups.append(ups if isinstance(ups, int) else 0)

        product["comments_texts"] = preprocess_sentences(comment_texts)
        product["comments_ups"] = comment_ups
        product["reviews"] = preprocess_sentences(review_texts)
        if drop_columns:
            drop_specification_keys(product, drop_keys=drop_columns)
        flat_data.append(product)

    df = pd.DataFrame(flat_data)
    if output_file.endswith(".json"):
        df.to_json(output_file, orient="records", indent=2, force_ascii=False)
    elif output_file.endswith(".csv"):
        df.to_csv(output_file, index=False, encoding="utf-8")
    else:
        raise ValueError("Output file must be .json or .csv")

    print(f"Preprocessed data saved to {output_file}")

DROP_COLUMNS = [
 'Network_2G_bands', 'Network_3G_bands', 'Network_4G_bands', 
 'Network_5G_bands', 'Network_Speed', 'Launch_Status', 
 'Body_SIM', 'Platform_Chipset', 'Comms_WLAN', 
 'Comms_Bluetooth', 'Comms_Positioning', 'Comms_NFC', 
 'Comms_Radio', 'Comms_USB', 'Misc_Price', 
 'EU_LABEL_Energy', 'EU_LABEL_Battery', 'EU_LABEL_Free_fall', 
 'EU_LABEL_Repairability', 'Misc_SAR', 'Misc_SAR_EU'
]


if __name__ == "__main__":
    input_path =r"E:\Product Comparator\data\reddit_product_site_data\data_json\product_data.json"
    output_path = r"E:\Product Comparator\data\reddit_product_site_data\data_json\product_data_cleaned.json"
    preprocess_products(input_path, output_path, drop_columns=DROP_COLUMNS)
    print(f"Cleaned data saved to {output_path}")