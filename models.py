from transformers import pipeline
from config import MULTILINGUAL_MODEL, SPECIFIC_MODEL


class SentimentAnalyzer:
    def __init__(self):
        self.multilingual_pipeline = pipeline(
            "sentiment-analysis",
            model=MULTILINGUAL_MODEL,
            tokenizer=MULTILINGUAL_MODEL
        )
        self.specific_pipeline = pipeline(
            "sentiment-analysis",
            model=SPECIFIC_MODEL,
            tokenizer=SPECIFIC_MODEL
        )

    def map_multilingual_label(self, label):
        label = label.lower()
        if "positive" in label or "pos" in label:
            return "positive"
        elif "negative" in label or "neg" in label:
            return "negative"
        return "neutral"

    def map_specific_label(self, label):
        label = label.lower()
        if "positive" in label:
            return "positive"
        elif "negative" in label:
            return "negative"
        return "neutral"

    def analyze_dataset(self, dataframe):
        texts = dataframe["text_clean"].tolist()

        multi_results = self.multilingual_pipeline(
            texts, truncation=True, max_length=512)
        spec_results = self.specific_pipeline(
            texts, truncation=True, max_length=512)

        dataframe["sentiment_multilingual"] = [
            self.map_multilingual_label(r["label"]) for r in multi_results]
        dataframe["sentiment_specific"] = [
            self.map_specific_label(r["label"]) for r in spec_results]

        return dataframe
