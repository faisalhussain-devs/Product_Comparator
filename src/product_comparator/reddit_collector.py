import os
import json
import hashlib
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from .config import REDDIT_MAX_POSTS, REDDIT_MAX_COMMENTS_PER_POST
import praw
from .preprocessing import preprocess_reddit_reviews_dict


DEFAULT_SUBREDDITS = [
    "smartphones",
    "gadgets",
    "apple",
    "Android",
    "technology",
    "headphones",
    "laptops",
    "BuyItForLife",
]


class RedditDataCollector:
    """
    Collector class to fetch product reviews and comments from Reddit.
    Supports PRAW (with API credentials) as well as a public endpoint fallback.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = (
            user_agent
            or os.getenv("REDDIT_USER_AGENT")
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        self.reddit = None
        if self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
            except Exception as e:
                print(f"[RedditCollector] Warning: Failed to initialize PRAW: {e}. Falling back to public JSON API.")

    def _generate_product_oid(self, product_name: str) -> str:
        """Generate a 24-character hex string representing a MongoDB ObjectId."""
        hash_hex = hashlib.md5(product_name.lower().encode("utf-8")).hexdigest()
        return hash_hex[:24]

    def fetch_via_praw(
        self,
        product_name: str,
        max_posts: int = REDDIT_MAX_POSTS,
        max_comments_per_post: int = REDDIT_MAX_COMMENTS_PER_POST,
        subreddits: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch posts and comments using PRAW."""
        query = f'"{product_name}" review OR experience OR thoughts'
        sub_str = "+".join(subreddits) if subreddits else "all"

        review_texts = []
        comments = []

        try:
            subreddit = self.reddit.subreddit(sub_str)
            search_results = subreddit.search(query, sort="relevance", limit=max_posts)

            for post in search_results:
                # Include post selftext if substantial
                if post.selftext and len(post.selftext.strip()) > 30:
                    review_texts.append(post.title + ". " + post.selftext)
                elif post.title:
                    review_texts.append(post.title)

                # Fetch comments
                post.comments.replace_more(limit=0)
                comment_count = 0
                for comment in post.comments:
                    if comment_count >= max_comments_per_post:
                        break
                    if hasattr(comment, "body") and comment.body and len(comment.body.strip()) > 30:
                        comments.append({
                            "id": comment.id,
                            "body": post.title+". "+comment.body,
                            "ups": getattr(comment, "score", 0) or getattr(comment, "ups", 0),
                        })
                        comment_count += 1
        except Exception as e:
            print(f"[RedditCollector] Error fetching via PRAW: {e}")

        return {
            "product": {"$oid": self._generate_product_oid(product_name)},
            "name": product_name,
            "comments": comments,
            "review_texts": review_texts,
        }

    def fetch_via_public_api(
        self,
        product_name: str,
        max_posts: int = REDDIT_MAX_POSTS,
        max_comments_per_post: int = REDDIT_MAX_COMMENTS_PER_POST,
    ) -> Dict[str, Any]:
        """Fallback method fetching public Reddit search JSON when PRAW keys are unavailable."""
        query_encoded = urllib.parse.quote(f"{product_name} review OR experience OR thoughts")
        url = f"https://www.reddit.com/search.json?q={query_encoded}&sort=relevance&limit={max_posts}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent},
        )

        review_texts = []
        comments = []

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    posts = data.get("data", {}).get("children", [])

                    for p in posts:
                        pdata = p.get("data", {})
                        title = pdata.get("title", "")
                        selftext = pdata.get("selftext", "")
                        num_comments = pdata.get("num_comments", 0)

                        if selftext and len(selftext.strip()) > 30:
                            review_texts.append(f"{title}. {selftext}")
                        elif title:
                            review_texts.append(title)

                        # Attempt to fetch post comments via permalink if comments exist
                        permalink = pdata.get("permalink")
                        if permalink and num_comments > 0:
                            comment_url = f"https://www.reddit.com{permalink}.json?limit={max_comments_per_post}"
                            c_req = urllib.request.Request(comment_url, headers={"User-Agent": self.user_agent})
                            try:
                                with urllib.request.urlopen(c_req, timeout=5) as c_resp:
                                    if c_resp.status == 200:
                                        c_data = json.loads(c_resp.read().decode("utf-8"))
                                        if isinstance(c_data, list) and len(c_data) > 1:
                                            comm_children = c_data[1].get("data", {}).get("children", [])
                                            for c_item in comm_children[:max_comments_per_post]:
                                                c_body = c_item.get("data", {}).get("body")
                                                c_ups = c_item.get("data", {}).get("ups", 0)
                                                if c_body and len(c_body.strip()) > 20:
                                                    comments.append({
                                                        "id": c_item.get("data", {}).get("id", ""),
                                                        "body": title + ". " + c_body,
                                                        "ups": c_ups,
                                                    })
                            except Exception:
                                pass
        except Exception as e:
            print(f"[RedditCollector] Warning: Public Reddit API fetch encountered issue: {e}")

        return {
            "product": {"$oid": self._generate_product_oid(product_name)},
            "name": product_name,
            "comments": comments,
            "review_texts": review_texts,
        }

    def fetch_product_data(
        self,
        product_name: str,
        max_posts: int = REDDIT_MAX_POSTS,
        max_comments_per_post: int = REDDIT_MAX_COMMENTS_PER_POST,
        subreddits: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch raw product reviews and comments from Reddit.
        Tries PRAW first if credentials exist, otherwise falls back to public API.
        """
        if self.reddit is not None:
            data = self.fetch_via_praw(
                product_name=product_name,
                max_posts=max_posts,
                max_comments_per_post=max_comments_per_post,
                subreddits=subreddits or DEFAULT_SUBREDDITS,
            )
            if data["comments"] or data["review_texts"]:
                return data

        data = self.fetch_via_public_api(
            product_name=product_name,
            max_posts=max_posts,
            max_comments_per_post=max_comments_per_post,
        )

        if data["comments"] or data["review_texts"]:
            return data

        # If live API fetch returned empty (e.g. 403 blocked or offline), use sample benchmark dataset
        print(
            f"[RedditCollector] Live Reddit API fetch returned no results "
            f"for '{product_name}'."
        )
        return None

def fetch_and_preprocess_product_reviews(
    product_name: str,
    collector: Optional[RedditDataCollector] = None,
    max_posts: int = REDDIT_MAX_POSTS,
    max_comments_per_post: int = REDDIT_MAX_COMMENTS_PER_POST,
) -> Dict[str, Any]:
    """
    Convenience function: Fetches raw Reddit data for product_name and preprocesses it.
    Returns model-ready list of reviews: [{"text": "...", "likes": 10}, ...]
    """
    if collector is None:
        collector = RedditDataCollector()

    raw_data = collector.fetch_product_data(
        product_name=product_name,
        max_posts=max_posts,
        max_comments_per_post=max_comments_per_post,
    )
    if raw_data:
        processed_list = preprocess_reddit_reviews_dict([raw_data])
        reviews = processed_list[0]["text"] if processed_list else []
    else:
        reviews = []

    return {
        "product_id": raw_data["product"]["$oid"] if raw_data else None,
        "product_name": product_name,
        "reviews": reviews,
        "raw_data": raw_data,
    }
