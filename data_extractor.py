import requests
from config import BLUESKY_HANDLE, BLUESKY_PASSWORD
from preprocessing import TextPreprocessor
from datetime import datetime, timedelta


class BlueskyDataExtractor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.base_url = "https://bsky.social/xrpc"
        self.headers = {}

    def authenticate(self):
        url = f"{self.base_url}/com.atproto.server.createSession"
        payload = {
            "identifier": BLUESKY_HANDLE,
            "password": BLUESKY_PASSWORD
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        access_jwt = data["accessJwt"]
        self.headers = {"Authorization": f"Bearer {access_jwt}"}

    def fetch_posts_by_keywords(self, keywords, max_results):
        self.authenticate()
        all_posts = []
        search_url = f"{self.base_url}/app.bsky.feed.searchPosts"

        for keyword in keywords:
            fetched = 0
            cursor = None

            while fetched < max_results:
                current_limit = min(100, max_results - fetched)
                params = {"q": keyword, "limit": current_limit, "lang": 'it'}
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(
                    search_url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()

                posts = data.get("posts", [])
                if not posts:
                    break

                for post in posts:
                    record = post.get("record", {}) or {}
                    author = post.get("author", {}) or {}
                    viewer = post.get("viewer", {}) or {}
                    text_raw = record.get("text", "")

                    all_posts.append({
                        "id": post.get("uri"),
                        "cid": post.get("cid"),
                        "language": record.get("langs", [None])[0],
                        "created_at": record.get("createdAt"),
                        "indexed_at": post.get("indexedAt"),
                        "text_raw": text_raw,
                        "text_clean": self.preprocessor.clean_text(text_raw),
                        "hashtags": self.preprocessor.extract_hashtags(text_raw),
                        "keyword_source": keyword,
                        "author_did": author.get("did"),
                        "author_handle": author.get("handle"),
                        "author_display_name": author.get("displayName"),
                        "author_createdAt": author.get("createdAt"),
                        "like_count": post.get("likeCount", 0),
                        "repost_count": post.get("repostCount", 0),
                        "reply_count": post.get("replyCount", 0),
                        "quote_count": post.get("quoteCount", 0),
                        "bookmark_count": post.get("bookmarkCount", 0),
                        "viewer_like": viewer.get("like"),
                        "viewer_repost": viewer.get("repost"),
                        "viewer_pinned": viewer.get("pinned"),
                        "viewer_bookmarked": viewer.get("bookmarked")
                    })

                fetched += len(posts)
                cursor = data.get("cursor")

                if not cursor:
                    break

        print(f"Fetched {len(all_posts)} posts for keywords: {keywords}")
        return all_posts

    def fetch_time_series_by_keywords(self, keywords, max_results_per_range, start_time, end_time):
        self.authenticate()
        time_series_data = []
        search_url = f"{self.base_url}/app.bsky.feed.searchPosts"

        start_date = datetime.strptime(start_time, "%d-%m-%Y")
        end_date = datetime.strptime(end_time, "%d-%m-%Y")

        current_date = start_date
        while current_date <= end_date:
            since_iso = current_date.strftime("%Y-%m-%dT00:00:00Z")
            until_iso = (current_date + timedelta(days=1)
                         ).strftime("%Y-%m-%dT00:00:00Z")

            print(f"Scraping intermezzo temporale: {since_iso} -> {until_iso}")

            for keyword in keywords:
                fetched = 0
                cursor = None

                while fetched < max_results_per_range:
                    current_limit = min(100, max_results_per_range - fetched)
                    params = {
                        "q": keyword,
                        "limit": current_limit,
                        "lang": 'it',
                        "since": since_iso,
                        "until": until_iso,
                    }
                    if cursor:
                        params["cursor"] = cursor

                    try:
                        response = requests.get(
                            search_url, params=params, headers=self.headers)
                        response.raise_for_status()
                        data = response.json()
                    except requests.exceptions.HTTPError as e:
                        print(
                            f"Errore nella richiesta per la keyword '{keyword}' in data {since_iso}: {e}")
                        break

                    posts = data.get("posts", [])
                    if not posts:
                        break

                    for post in posts:
                        record = post.get("record", {}) or {}
                        author = post.get("author", {}) or {}
                        viewer = post.get("viewer", {}) or {}
                        text_raw = record.get("text", "")

                        time_series_data.append({
                            "id": post.get("uri"),
                            "cid": post.get("cid"),
                            "created_at": record.get("createdAt"),
                            "indexed_at": post.get("indexedAt"),
                            "text_raw": text_raw,
                            "text_clean": self.preprocessor.clean_text(text_raw),
                            "hashtags": self.preprocessor.extract_hashtags(text_raw),
                            "keyword_source": keyword,
                            "author_did": author.get("did"),
                            "author_handle": author.get("handle"),
                            "author_display_name": author.get("displayName"),
                            "author_createdAt": author.get("createdAt"),
                            "like_count": post.get("likeCount", 0),
                            "repost_count": post.get("repostCount", 0),
                            "reply_count": post.get("replyCount", 0),
                            "quote_count": post.get("quoteCount", 0),
                            "bookmark_count": post.get("bookmarkCount", 0),
                            "viewer_like": viewer.get("like"),
                            "viewer_repost": viewer.get("repost"),
                            "viewer_pinned": viewer.get("pinned"),
                            "viewer_bookmarked": viewer.get("bookmarked")
                        })

                    fetched += len(posts)
                    cursor = data.get("cursor")

                    if not cursor:
                        break

            current_date += timedelta(days=1)

        print(
            f"Estrazione completata. Raccolti {len(time_series_data)} post totali.")
        return time_series_data
