from transformers import pipeline
from config import MULTILINGUAL_MODEL, SPECIFIC_MODEL


class SentimentAnalyzer:
    def __init__(self):
        self.multilingual_pipeline = pipeline(
            "sentiment-analysis",
            model=MULTILINGUAL_MODEL,
            tokenizer=MULTILINGUAL_MODEL,
            top_k=None
        )
        self.specific_pipeline = pipeline(
            "sentiment-analysis",
            model=SPECIFIC_MODEL,
            tokenizer=SPECIFIC_MODEL,
            top_k=None
        )

    @staticmethod
    def compute_continuous_score(predictions_list):
        prob_pos = 0.0
        prob_neg = 0.0

        for pred in predictions_list:
            label = pred["label"].lower()
            if "pos" in label or "label_2" in label:
                prob_pos = pred["score"]
            elif "neg" in label or "label_0" in label:
                prob_neg = pred["score"]

        return 0.5 + (prob_pos - prob_neg) / 2

    @staticmethod
    def get_label_from_score(score):
        if score < 0.45:
            return "negative"
        elif score > 0.55:
            return "positive"
        else:
            return "neutral"

    def analyze_dataset(self, dataframe):
        texts = dataframe["text_clean"].tolist()
        total_texts = len(texts)

        if total_texts == 0:
            return dataframe

        batch_size = 32
        multi_results = []
        spec_results = []

        print(f"Inizio analisi su {total_texts} post...")

        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]

            multi_batch = self.multilingual_pipeline(
                batch_texts, truncation=True, max_length=512)
            spec_batch = self.specific_pipeline(
                batch_texts, truncation=True, max_length=512)

            multi_results.extend(multi_batch)
            spec_results.extend(spec_batch)

            processed_count = min(i + batch_size, total_texts)
            percentage = int((processed_count / total_texts) * 100)
            print(
                f"Progresso: {percentage}% completato ({processed_count}/{total_texts})")

        dataframe["sent_ml_score"] = [
            SentimentAnalyzer.compute_continuous_score(r) for r in multi_results]
        dataframe["sent_spec_score"] = [
            SentimentAnalyzer.compute_continuous_score(r) for r in spec_results]

        dataframe["sentiment_multilingual"] = [
            SentimentAnalyzer.get_label_from_score(s) for s in dataframe["sent_ml_score"]
        ]
        dataframe["sentiment_specific"] = [
            SentimentAnalyzer.get_label_from_score(s) for s in dataframe["sent_spec_score"]
        ]

        return dataframe

    def analyze_dataset_custom(self, dataframe, model_name, tokenizer=None, custom_name=None):
        if tokenizer is None:
            tokenizer = model_name
        if custom_name is None:
            custom_name = model_name

        custom_model_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=tokenizer,
            top_k=None
        )
        texts = dataframe["text_clean"].tolist()
        total_texts = len(texts)

        if total_texts == 0:
            return dataframe

        batch_size = 32
        res = []

        print(
            f"Inizio analisi su {total_texts} post, uso il modello: {model_name}...")

        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]

            batch = custom_model_pipeline(
                batch_texts, truncation=True, max_length=512)

            res.extend(batch)

            processed_count = min(i + batch_size, total_texts)
            percentage = int((processed_count / total_texts) * 100)
            print(
                f"Progresso: {percentage}% completato ({processed_count}/{total_texts})")

        dataframe[f"sent_{custom_name}_score"] = [
            SentimentAnalyzer.compute_continuous_score(r) for r in res]

        dataframe[f"sentiment_{custom_name}"] = [
            SentimentAnalyzer.get_label_from_score(s) for s in dataframe[f"sent_{custom_name}_score"]
        ]

        return dataframe
