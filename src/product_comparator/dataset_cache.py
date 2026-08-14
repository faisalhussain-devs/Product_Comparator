import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List


def normalize_query(name: str) -> str:
    """Normalize product search string for hash lookup."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


class DatasetCache:
    """
    Evaluation Corpus & Dataset Cache Manager.
    Uses O(1) hash indexing (product_index.json) pointing to products_corpus.json entries.
    """

    def __init__(self, example_dir: Optional[Path] = None):
        if example_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
            example_dir = base_dir / "example_data"

        self.example_dir = example_dir
        self.corpus_path = example_dir / "products_corpus.json"
        self.index_path = example_dir / "product_index.json"

        self.corpus: List[Dict[str, Any]] = []
        self.index: Dict[str, int] = {}
        self._load_cache()

    def _load_cache(self):
        """Loads corpus and index files, building them if missing."""
        if not self.corpus_path.exists() or not self.index_path.exists():
            try:
                from example_data.build_dataset_corpus import build_corpus
                build_corpus()
            except Exception as e:
                print(f"[DatasetCache] Warning: Could not auto-build corpus: {e}")

        if self.corpus_path.exists():
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    self.corpus = json.load(f)
            except Exception as e:
                print(f"[DatasetCache] Error loading products_corpus.json: {e}")

        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
            except Exception as e:
                print(f"[DatasetCache] Error loading product_index.json: {e}")

    def find_product(self, product_name: str) -> Optional[Dict[str, Any]]:
        """
        O(1) Hash Lookup: Finds product in corpus matching query string or common aliases.
        """
        if not product_name:
            return None

        norm = normalize_query(product_name)

        # 1. Direct O(1) index lookup
        if norm in self.index:
            idx = self.index[norm]
            if 0 <= idx < len(self.corpus):
                return self.corpus[idx]

        # 2. Substring fallback matching across indexed names & IDs
        for idx, item in enumerate(self.corpus):
            item_norm_name = normalize_query(item.get("name", ""))
            item_norm_id = normalize_query(item.get("id", ""))

            if norm and (norm in item_norm_name or item_norm_name in norm or norm in item_norm_id):
                return item

        return None

    def add_to_cache(
        self,
        product_name: str,
        reviews: Dict[str, Any],
        specifications: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Saves a newly fetched live product to the packaged dataset corpus."""
        product_id = normalize_query(product_name).replace(" ", "_")

        raw_comments = reviews.get("comments", []) if isinstance(reviews, dict) else []
        cleaned_comments = []
        for c in raw_comments:
            if isinstance(c, dict):
                cleaned_comments.append({
                    "body": c.get("body") or c.get("text") or "",
                    "ups": c.get("ups") if isinstance(c.get("ups"), int) else c.get("upvotes", 0),
                })

        clean_reviews = {
            "comments": cleaned_comments,
            "review_texts": reviews.get("review_texts", []) if isinstance(reviews, dict) else [],
        }

        new_entry = {
            "id": product_id,
            "name": product_name,
            "reviews": clean_reviews,
            "specifications": specifications or {},
        }

        corpus_idx = len(self.corpus)
        self.corpus.append(new_entry)


        # Add hash index keys
        norm = normalize_query(product_name)
        if norm:
            self.index[norm] = corpus_idx
        self.index[product_id] = corpus_idx

        # Persist updated corpus and index files
        try:
            with open(self.corpus_path, "w", encoding="utf-8") as f:
                json.dump(self.corpus, f, indent=2, ensure_ascii=False)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            print(f"[DatasetCache] Cached new product '{product_name}' at corpus index #{corpus_idx}.")
        except Exception as e:
            print(f"[DatasetCache] Error persisting updated cache: {e}")

        return new_entry
