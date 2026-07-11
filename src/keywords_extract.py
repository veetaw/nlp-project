import pandas as pd
from keybert import KeyBERT
from preprocessing import TextPreprocessor
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

df = pd.read_csv("bluesky_time_series_data_WITH_SENTIMENT.csv")

tp = TextPreprocessor()
df['text_clean'] = df['text_clean'].astype(str).apply(tp.clean_text)

df = df[df['text_clean'].str.strip().str.len() > 3]

nltk.download('stopwords', quiet=True)
italian_stopwords = stopwords.words('italian')

kw_model = KeyBERT(model="intfloat/multilingual-e5-small")

posts_list = df['text_clean'].tolist()[:1000]

keywords = kw_model.extract_keywords(
    posts_list,
    keyphrase_ngram_range=(1, 2),
    top_n=5,
    stop_words=italian_stopwords
)

keyword_scores = defaultdict(float)

for post_kws in keywords:
    if isinstance(post_kws, list):
        for kw, score in post_kws:
            keyword_scores[kw] += score

global_top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:50]

print(f"\nAnalizzati ({len(posts_list)} post validi su Bluesky)")
print(f"{'Keyword':<45} | {'Punteggio Cumulativo'}")
print("-" * 65)
if not global_top_keywords:
    print("Nessuna keyword estratta.")
else:
    for kw, score in global_top_keywords:
        print(f"{kw:<45} | {score:.4f}")