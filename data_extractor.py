from atproto import Client
from config import BLUESKY_HANDLE, BLUESKY_PASSWORD
from preprocessing import TextPreprocessor


class BlueskyDataExtractor:
    def __init__(self):
        self.client = Client()
        self.preprocessor = TextPreprocessor()

    def authenticate(self):
        self.client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)

    def fetch_posts_by_keywords(self, keywords, max_results):
        self.authenticate()
        all_posts = []

        for keyword in keywords:
            response = self.client.app.bsky.feed.search_posts(
                q=keyword, limit=max_results)
            for post in response.posts:
                text_raw = post.record.text
                all_posts.append({
                    "id": post.uri,
                    "created_at": post.record.created_at,
                    "text_raw": text_raw,
                    "text_clean": self.preprocessor.clean_text(text_raw),
                    "hashtags": self.preprocessor.extract_hashtags(text_raw),
                    "keyword_source": keyword
                })
        return all_posts
