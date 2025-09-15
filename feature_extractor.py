# ============================================
# feature_extractor.py - Enhanced NLP Processing
# ============================================
"""Extract advanced features from text"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
from transformers import pipeline

# --- ADD THESE IMPORTS ---
from config import Config
from utils import setup_logging, cache_result, calculate_readability
from models import Transcript
# -------------------------


class FeatureExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.sentiment_model = pipeline("sentiment-analysis",
                                        model="ProsusAI/finbert")

    @cache_result("./cache", "features")
    def extract_features(self, transcripts: List[Transcript]) -> pd.DataFrame:
        """Extract comprehensive features from transcripts"""
        features = []

        for t in tqdm(transcripts, desc="Extracting features"):
            feature_dict = {
                'ticker': t.ticker,
                'date': t.date,
                'filename': t.filename,
            }

            # Full text embedding
            embedding = self.embedding_model.encode(t.text, convert_to_numpy=True)
            for i, val in enumerate(embedding):
                feature_dict[f'emb_{i}'] = val

            # Section-specific embeddings
            if t.sections:
                if t.sections.get('prepared_remarks'):
                    emb_remarks = self.embedding_model.encode(
                        t.sections['prepared_remarks'],
                        convert_to_numpy=True
                    )
                    for i, val in enumerate(emb_remarks):
                        feature_dict[f'emb_remarks_{i}'] = val

                if t.sections.get('qa'):
                    emb_qa = self.embedding_model.encode(
                        t.sections['qa'],
                        convert_to_numpy=True
                    )
                    for i, val in enumerate(emb_qa):
                        feature_dict[f'emb_qa_{i}'] = val

            # Sentiment analysis
            sentiment = self._extract_sentiment(t.text)
            feature_dict['sentiment'] = sentiment

            if t.sections:
                if t.sections.get('prepared_remarks'):
                    feature_dict['sentiment_remarks'] = self._extract_sentiment(
                        t.sections['prepared_remarks'][:512]  # Limit for BERT
                    )
                if t.sections.get('qa'):
                    feature_dict['sentiment_qa'] = self._extract_sentiment(
                        t.sections['qa'][:512]
                    )

            # Linguistic features
            feature_dict['word_count'] = len(t.text.split())
            feature_dict['readability'] = calculate_readability(t.text)

            features.append(feature_dict)

        return pd.DataFrame(features)

    def _extract_sentiment(self, text: str) -> float:
        """Extract sentiment using FinBERT"""
        if not text:
            return 0.0

        try:
            # Truncate text for the model
            result = self.sentiment_model(text[:512])[0]
            if result['label'] == 'positive':
                return result['score']
            elif result['label'] == 'negative':
                return -result['score']
            else:
                return 0.0
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0