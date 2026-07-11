import re


class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\S+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.clean_pattern = re.compile(r'[^a-zA-ZàèìòùÀÈÌÒÙáéíóúÁÉÍÓÚ\s]')

    def extract_hashtags(self, text):
        if not text:
            return []
        return self.hashtag_pattern.findall(text.lower())

    def clean_text(self, text):
        if not text:
            return ""
        text = self.url_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)
        text = self.hashtag_pattern.sub(r'\1', text)
        text = self.clean_pattern.sub("", text)
        text = " ".join(text.split())
        return text.lower()
