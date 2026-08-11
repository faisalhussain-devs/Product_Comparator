import json
import re
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import DROP_COLUMNS
from .preprocessing import flatten_data, drop_specification_keys


class GSMArenaHTMLParser(HTMLParser):
    """HTML Parser to extract specification tables from GSMArena spec pages."""

    def __init__(self):
        super().__init__()
        self.more_specification = []
        self.current_section = None
        self.current_sub_title = None
        self.current_sub_data = []
        self.in_section_title = False
        self.in_sub_title = False
        self.in_sub_data = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "th":
            self.in_section_title = True
        elif tag == "td":
            cls = attrs_dict.get("class", "")
            if "ttl" in cls:
                self.in_sub_title = True
            elif "nfo" in cls:
                self.in_sub_data = True

    def handle_endtag(self, tag):
        if tag == "th":
            self.in_section_title = False
        elif tag == "td":
            if self.in_sub_title:
                self.in_sub_title = False
            elif self.in_sub_data:
                self.in_sub_data = False
                if self.current_section and self.current_sub_title:
                    sub_entry = {
                        "title": self.current_sub_title.strip(),
                        "data": [" ".join(self.current_sub_data).strip()],
                    }
                    self.current_section["data"].append(sub_entry)
                self.current_sub_title = None
                self.current_sub_data = []

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return

        if self.in_section_title:
            self.current_section = {"title": text, "data": []}
            self.more_specification.append(self.current_section)
        elif self.in_sub_title:
            self.current_sub_title = text
        elif self.in_sub_data:
            self.current_sub_data.append(text)


class GSMArenaCollector:
    """
    Collector class to fetch and parse product specifications from GSMArena.
    Includes web search/scraping and fallback local search.
    """

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        self.base_url = "https://www.gsmarena.com"

    def search_product(self, product_name: str) -> Optional[str]:
        """Search GSMArena for product and return detail page URL string."""
        query_encoded = urllib.parse.quote(product_name)
        search_url = f"{self.base_url}/results.php3?sQuickSearch=0&sName={query_encoded}"

        req = urllib.request.Request(search_url, headers={"User-Agent": self.user_agent})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    html = resp.read().decode("utf-8", errors="ignore")
                    # Search for detail page links: <a href="apple_iphone_15_pro-12557.php">
                    matches = re.findall(r'href="([a-zA-Z0-9_-]+-\d+\.php)"', html)
                    if matches:
                        return f"{self.base_url}/{matches[0]}"
        except Exception as e:
            print(f"[GSMArenaCollector] Search request issue: {e}")

        return None

    def fetch_specs_from_url(self, spec_url: str, product_name: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse specifications from GSMArena device detail page URL."""
        req = urllib.request.Request(spec_url, headers={"User-Agent": self.user_agent})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    html = resp.read().decode("utf-8", errors="ignore")
                    parser = GSMArenaHTMLParser()
                    parser.feed(html)

                    if parser.more_specification:
                        key = spec_url.split("/")[-1].replace(".php", "").split("-")[0]
                        return {
                            "_id": {"$oid": "000000000000000000000000"},
                            "key": key,
                            "name": product_name,
                            "specification": {
                                "key": key,
                                "more_specification": parser.more_specification,
                            },
                        }
        except Exception as e:
            print(f"[GSMArenaCollector] Spec fetch error from {spec_url}: {e}")

        return None

    def fetch_product_specs(self, product_name: str) -> Dict[str, Any]:
        """
        Fetch raw product specifications for product_name.
        Tries GSMArena online search first, then matches local specifications DB or constructs defaults.
        """
        # 1. Try web search & scrape from GSMArena
        spec_url = self.search_product(product_name)
        if spec_url:
            raw_specs = self.fetch_specs_from_url(spec_url, product_name)
            if raw_specs:
                return raw_specs

def fetch_and_preprocess_product_specs(
    product_name: str,
    collector: Optional[GSMArenaCollector] = None,
    drop_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function: Fetches raw GSMArena product specs and flattens them for the model pipeline.
    Returns dictionary of flat specification key-values.
    """
    if collector is None:
        collector = GSMArenaCollector()

    if drop_columns is None:
        drop_columns = DROP_COLUMNS

    raw_specs = collector.fetch_product_specs(product_name)
    if not raw_specs:
        print(
            f"[GSMArenaCollector] No specifications found "
            f"for '{product_name}'."
        )
        return {}
    flat_specs = flatten_data(raw_specs)

    specs_obj = {"specifications": flat_specs}
    cleaned_specs = drop_specification_keys(specs_obj, drop_keys=drop_columns)

    return cleaned_specs.get("specifications", {})
