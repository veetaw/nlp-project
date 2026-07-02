import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from config import SEARCH_KEYWORDS, MAX_POSTS_PER_KEYWORD, OUTPUT_DATA_PATH, OUTPUT_PLOT_DISTRIBUTION, OUTPUT_PLOT_CORRELATION, _DOWNLOAD
from data_extractor import BlueskyDataExtractor
from models import SentimentAnalyzer


def generate_visualizations(df):
    plt.figure(figsize=(10, 5))
    df_melted = df.melt(value_vars=[
                        "sentiment_multilingual", "sentiment_specific"], var_name="Modello", value_name="Sentiment")
    sns.countplot(data=df_melted, x="Sentiment", hue="Modello", palette="Set2")
    plt.title("Confronto Distribuzione Sentiment: Multilingua vs Specifico")
    plt.ylabel("Numero di Post")
    plt.savefig(OUTPUT_PLOT_DISTRIBUTION)
    plt.close()

    negative_posts = df[df["sentiment_specific"] == "negative"]
    all_neg_hashtags = [ht for sublist in negative_posts["hashtags"]
                        for ht in sublist if isinstance(sublist, list)]
    hashtag_counts = Counter(all_neg_hashtags).most_common(10)

    if hashtag_counts:
        plt.figure(figsize=(10, 5))
        ht_df = pd.DataFrame(hashtag_counts, columns=["Hashtag", "Frequenza"])
        sns.barplot(data=ht_df, x="Frequenza", y="Hashtag", palette="Reds_r")
        plt.title(
            "Hashtag Correlation: Primi 10 Tag nei Post con Sentiment Negativo")
        plt.savefig(OUTPUT_PLOT_CORRELATION)
        plt.close()


def main():
    extractor = BlueskyDataExtractor()
    if _DOWNLOAD:
        raw_data = extractor.fetch_posts_by_keywords(
            SEARCH_KEYWORDS, MAX_POSTS_PER_KEYWORD)
        df = pd.DataFrame(raw_data)
    else:
        df = pd.read_csv(r"C:\dev\nlp\bluesky_time_series_data.csv")

    df["text_clean"] = df["text_clean"].astype(str).str.strip()
    df = df[df["text_clean"] != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_dataset(df)

    df.to_csv(
        r"C:\dev\nlp\outputs\meloni_with_sentiment.csv", index=False)
    # generate_visualizations(df)


if __name__ == "__main__":
    main()

    # codice per scaricare la time series
    # data = BlueskyDataExtractor().fetch_time_series_by_keywords(
    #    SEARCH_KEYWORDS, MAX_POSTS_PER_KEYWORD, "01-01-2026", "18-06-2026")

    # pd.DataFrame(data).to_csv("bluesky_time_series_data.csv", index=False)
    # df = pd.read_csv(r"C:\dev\nlp\bluesky_time_series_data_WITH_SENTIMENT.csv")
