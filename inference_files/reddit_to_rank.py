import json
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
    value = re.sub(r'^[^a-zA-Z0-9]+', '', value)
    value = re.sub(r'\s+', ' ', value).strip()

    return value


def preprocess_reddit_reviews(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for product in data:
        reviews = product.get("review_texts", [])
        comments = product.get("comments", [])
        for c in comments:
            if isinstance(c, dict):
                body = c.get("body")
                ups = c.get("ups")
                if isinstance(body, str):
                    text = clean_value(body.strip())
                    if len(text.split()) > 15:
                        texts.append({"text": text, "likes": ups if isinstance(ups, int) else 0})

        for review in reviews:
            if isinstance(review, str):
                text = clean_value(review.strip())
                if len(text.split()) > 15:
                    texts.append({"text": text})

    return texts


if __name__ == "__main__":
    input_path =r"" # Path to the reddit json file containing reviews and its comments
    output = preprocess_reddit_reviews(input_path)