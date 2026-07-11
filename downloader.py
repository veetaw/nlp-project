from data_extractor import BlueskyDataExtractor
from config import *
import pandas as pd
from preprocessing import TextPreprocessor
from keybert import KeyBERT

if __name__ == "__main__":
    df = pd.read_csv("bluesky_time_series_data_WITH_SENTIMENT.csv")

    tp = TextPreprocessor()
    df['text_clean'] = df['text_clean'].astype(str).apply(tp.clean_text)

    kw_model = KeyBERT(model="intfloat/multilingual-e5-large")

    combined_text = " ".join(df['text_clean'].astype(str).tolist())

    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 2),
        top_n=50
    )

    print(f"\n ({len(df)} post)")
    print(f"{'Keyword':<45} | {'Score'}")
    print("-" * 60)
    for kw, score in keywords:
        print(f"{kw:<45} | {score:.4f}")
