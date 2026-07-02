from data_extractor import BlueskyDataExtractor
from config import *
import pandas as pd


if __name__ == "__main__":
    """extractor = BlueskyDataExtractor()
    keywords = ["israel"]

    data = extractor.fetch_posts_by_keywords(keywords, MAX_POSTS_PER_KEYWORD)

    pd.DataFrame(data).to_csv(
        r"datasets\bluesky_israel_1000_data.csv", index=False)

    keywords = ["palestine"]

    data = extractor.fetch_posts_by_keywords(keywords, MAX_POSTS_PER_KEYWORD)

    pd.DataFrame(data).to_csv(
        r"datasets\bluesky_palestine_1000_data.csv", index=False)
    """
    """
        import yake

        datasets = {
            "ISRAEL": israel_df,
            "PALESTINE": palestine_df
        }

        for name, df in datasets.items():
            print(f"\n==================== DATASET: {name} ====================")

            df['language'] = df['language'].fillna('unknown')
            df['text_clean'] = df['text_clean'].fillna('')

            for lang, group in df.groupby('language'):
                if lang == 'unknown' or len(group) < 3:
                    continue

                combined_text = " ".join(group['text_clean'].astype(str).tolist())
                yake_lang = lang if lang in [
                    'en', 'it', 'es', 'fr', 'de', 'pt'] else 'en'

                kw_extractor = yake.KeywordExtractor(
                    lan=yake_lang, n=1, dedupLim=0.85, top=15)
                keywords = kw_extractor.extract_keywords(combined_text)

                print(f"\nLingua: {lang.upper()} ({len(group)} post)")
                print(f"{'Keyword':<45} | {'Score'}")
                print("-" * 60)
                for kw, score in keywords:
                    print(f"{kw:<45} | {score:.4f}")
    """
    israel_df = pd.read_csv(r"datasets\bluesky_israel_1000_data.csv")
    palestine_df = pd.read_csv(
        r"C:\dev\nlp\bluesky_time_series_data_WITH_SENTIMENT.csv")

    from keybert import KeyBERT

    kw_model = KeyBERT(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    combined_text = " ".join(palestine_df['text_clean'].astype(str).tolist())

    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 2),
        top_n=50
    )

    print(f"\n ({len(palestine_df)} post)")
    print(f"{'Keyword':<45} | {'Score'}")
    print("-" * 60)
    for kw, score in keywords:
        print(f"{kw:<45} | {score:.4f}")
