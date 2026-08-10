import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def product_tokens(product_name: str) -> list[str]:
    return normalize_text(product_name).split()


def token_weight(token: str) -> float:
    if token.isdigit():
        return 0.1

    if re.fullmatch(r"[a-z]*\d+[a-z]*", token):
        return 0.2

    if len(token) <= 2:
        return 0.2

    return 1.0


def product_match_score(
    text: str,
    product_name: str,
) -> float:
    normalized_text = normalize_text(text)
    normalized_name = normalize_text(product_name)

    if not normalized_text or not normalized_name:
        return 0.0

    if normalized_name in normalized_text:
        return 1.0

    tokens = product_tokens(product_name)
    matched = set(normalized_text.split()) & set(tokens)

    if not matched:
        return 0.0

    total_weight = sum(token_weight(token) for token in tokens)
    matched_weight = sum(token_weight(token) for token in matched)

    coverage = matched_weight / total_weight

    if coverage >= 0.75:
        return 0.9

    if coverage >= 0.5:
        return 0.7

    if coverage >= 0.3:
        return 0.45

    if coverage > 0:
        return 0.2

    return 0.0


def product_relevance(
    review_text: str,
    product_name: str,
    context: str = "",
) -> float:
    review_score = product_match_score(review_text, product_name)

    if context:
        context_score = product_match_score(context, product_name)
        return max(review_score, context_score)

    return review_score