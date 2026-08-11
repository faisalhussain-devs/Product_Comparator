import json
import re

def clean_value(value: str) -> str:
    if not isinstance(value, str):
        return value

    value = re.sub(r'https?://\S+|www\.\S+', '', value)

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
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
    value = re.sub(r'^[^a-zA-Z0-9]+', '', value)
    value = re.sub(r'\s+', ' ', value).strip()

    return value


def preprocess_reddit_reviews_dict(data):
    if isinstance(data, dict):
        data = [data]

    products = []

    for product in data:
        texts = []
        product_obj = product.get("product")
        if isinstance(product_obj, dict):
            product_id = product_obj.get("$oid")
        else:
            product_id = product_obj

        reviews = product.get("review_texts", [])
        comments = product.get("comments", [])

        for comment in comments:
            if isinstance(comment, dict):
                body = comment.get("body")
                ups = comment.get("ups")

                if isinstance(body, str):
                    text = clean_value(body.strip())
                    if len(text.split()) > 15:
                        texts.append({
                            "text": text,
                            "likes": ups if isinstance(ups, int) else 0,
                        })

        for review in reviews:
            if isinstance(review, str):
                text = clean_value(review.strip())
                if len(text.split()) > 15:
                    texts.append({"text": text})

        if texts:
            products.append({
                "id": product_id,
                "text": texts
            })

    return products


def preprocess_reddit_reviews(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return preprocess_reddit_reviews_dict(data)


def drop_specification_keys(data, drop_keys):
    specs = data.get("specifications")

    if isinstance(specs, dict):
        for key in drop_keys:
            specs.pop(key, None)

    return data


def flatten_data(product, sep="_"):
    flat = {}

    if not isinstance(product, dict):
        return flat

    def process_specs(specs, parent=""):
        for entry in specs:
            title = entry.get("title", "").strip().replace(" ", "_")
            data = entry.get("data")

            if (
                isinstance(data, list)
                and all(
                    isinstance(x, dict)
                    and "title" in x
                    and "data" in x
                    for x in data
                )
            ):
                process_specs(data, parent=f"{parent}{sep}{title}")
            else:
                key = f"{parent}{sep}{title}" if parent else title
                value = (
                    " ".join(str(x).strip() for x in data)
                    if isinstance(data, list)
                    else data
                )
                flat[key.strip("_")] = value

    if "specification" in product and isinstance(product["specification"], dict):
        more_specs = product["specification"].get("more_specification", [])
        process_specs(more_specs)

    return flat


def preprocess_products(input_file, drop_columns=None):
    specs = {}
    flat_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for product in data:
        product_id = product.get("_id").get("$oid")
        specs["specifications"] = flatten_data(product)

        if drop_columns is not None:
            flat_data.append({
                "id": product_id,
                "specifications": drop_specification_keys(specs, drop_keys=drop_columns)["specifications"],
                "name": product["name"]
            })

    return flat_data