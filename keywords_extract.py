import pandas as pd
from keybert import KeyBERT
from preprocessing import TextPreprocessor
from collections import defaultdict
import nltk

# 1. Caricamento dati
palestine_df = pd.read_csv(r"C:\dev\nlp\bluesky_time_series_data_WITH_SENTIMENT.csv")

# 2. Preprocessing
tp = TextPreprocessor()
palestine_df['text_clean'] = palestine_df['text_clean'].astype(str).apply(tp.clean_text)

# Rimuoviamo i post vuoti o troppo corti
palestine_df = palestine_df[palestine_df['text_clean'].str.strip().str.len() > 3]

# [AGGIORNAMENTO] Caricamento Stopwords ufficiali NLTK per l'italiano
nltk.download('stopwords', quiet=True) # quiet=True evita di intasare il terminale con i log di download
from nltk.corpus import stopwords
italian_stopwords = stopwords.words('italian')

# 3. Inizializzazione KeyBERT
kw_model = KeyBERT(model="intfloat/multilingual-e5-small")

# 4. Preparazione della lista dei post
posts_list = palestine_df['text_clean'].tolist()

posts_list = palestine_df['text_clean'].tolist()[:1000] # Prende solo i primi 1000

print(f"Avvio estrazione su {len(posts_list)} post... (Potrebbe richiedere del tempo)")

keywords = kw_model.extract_keywords(
    posts_list,
    keyphrase_ngram_range=(1, 2),
    top_n=5,                      # Estrae le top 5 migliori per ogni post
    stop_words=italian_stopwords  # <--- Passiamo la lista pulita di NLTK
)

# 5. Aggregazione dei risultati
keyword_scores = defaultdict(float)

for post_kws in keywords:
    if isinstance(post_kws, list):
        for kw, score in post_kws:
            keyword_scores[kw] += score

# Ordiniamo per la somma dei punteggi semantic score
global_top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:50]

# 6. Stampa dei risultati
print(f"\nAnalizzati ({len(posts_list)} post validi su Bluesky)")
print(f"{'Keyword':<45} | {'Punteggio Cumulativo'}")
print("-" * 65)
if not global_top_keywords:
    print("Nessuna keyword estratta.")
else:
    for kw, score in global_top_keywords:
        print(f"{kw:<45} | {score:.4f}")