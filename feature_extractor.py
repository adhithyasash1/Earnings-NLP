"""Feature extraction"""
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from typing import List

from config import Config
from utils import setup_logging, calculate_readability
from models import Transcript


class FeatureExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.embedding_model = SentenceTransformer(config.embedding_model, device='cpu')
        self.sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def extract_features(self, transcripts: List[Transcript]) -> pd.DataFrame:
        features = []
        for t in tqdm(transcripts):
            feature_dict = {'ticker': t.ticker, 'date': t.date, 'filename': t.filename}
            embedding = self.embedding_model.encode(t.text)
            for i, val in enumerate(embedding):
                feature_dict[f'emb_{i}'] = val
            sentiment = self._extract_sentiment(t.text)
            feature_dict['sentiment'] = sentiment
            feature_dict['word_count'] = len(t.text.split())
            feature_dict['readability'] = calculate_readability(t.text)
            features.append(feature_dict)
        return pd.DataFrame(features)

    def _extract_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        result = self.sentiment_model(text[:512])[0]
        if result['label'] == 'positive':
            return result['score']
        elif result['label'] == 'negative':
            return -result['score']
        return 0.0