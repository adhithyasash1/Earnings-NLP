# feature_extractor.py (Updated: Batch NMF)
"""Feature extraction"""
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
import os
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from config import Config
from utils import setup_logging, calculate_readability
from models import Transcript


class FeatureExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)

        # SSL/NLTK (unchanged)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('stopwords', quiet=False, halt_on_error=True)
        self.stop_words = set(stopwords.words('english')) | set(['company', 'quarter'])

        # VADER (unchanged)
        try:
            nltk.download('vader_lexicon', quiet=False, halt_on_error=True)
            self.vader = SentimentIntensityAnalyzer()
            self.logger.info("VADER initialized successfully")
        except Exception as e:
            self.logger.warning(f"NLTK download failed: {e}. Attempting manual lexicon load.")
            try:
                url = 'https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vaderSentiment/vader_lexicon.txt'
                response = requests.get(url)
                response.raise_for_status()
                lexicon_dict = {}
                for line in response.text.split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            lexicon_dict[parts[0]] = float(parts[1])
                self.vader = SentimentIntensityAnalyzer(lexicon=lexicon_dict)
                self.logger.info("Manual VADER lexicon loaded")
            except Exception as me:
                self.logger.error(f"Manual load failed: {me}. Disabling VADER.")
                self.vader = None

        self.embedding_model = SentenceTransformer('FinLang/finance-embeddings-investopedia', device='cpu')
        self.emb_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

        self.lm_dict = self._load_lm_dictionary()

        # For batch NMF
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')  # min_df=2 to avoid singletons
        self.nmf = NMF(n_components=5, random_state=42)

    def _load_lm_dictionary(self) -> pd.DataFrame:
        lm_path = 'lm_master_dictionary.csv'
        if not os.path.exists(lm_path):
            self.logger.info("Downloading Loughran-McDonald Master Dictionary...")
            url = "https://drive.google.com/uc?export=download&id=1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY"
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(lm_path, 'w') as f:
                    f.write(response.text)
            except Exception as e:
                self.logger.warning(f"Failed to download LM dictionary: {e}. Using simplified sets.")
                return pd.DataFrame()  # Fallback to simplified

        df = pd.read_csv(lm_path)
        df.columns = df.columns.str.upper()  # Normalize
        return df

    def extract_features(self, transcripts: List[Transcript]) -> pd.DataFrame:
        transcripts = sorted(transcripts, key=lambda t: (t.ticker, t.date))
        features = []
        prev_by_ticker = {}
        all_texts = []  # For batch NMF

        # First pass: Collect texts and compute non-topic features
        for t in tqdm(transcripts):
            feature_dict = {'ticker': t.ticker, 'date': t.date, 'filename': t.filename}

            prepared = t.sections.get('prepared_remarks', '')
            qa = t.sections.get('qa', '')
            full_text = t.text

            # Embeddings (unchanged)
            emb_prepared = self.embedding_model.encode(prepared) if prepared else np.zeros(self.emb_dim)
            emb_qa = self.embedding_model.encode(qa) if qa else np.zeros(self.emb_dim)
            emb_full = 0.3 * emb_prepared + 0.7 * emb_qa if qa else emb_prepared
            for i, val in enumerate(emb_full):
                feature_dict[f'emb_{i}'] = val

            # Sentiment (unchanged)
            sentiment_full = self._extract_sentiment(full_text)
            sentiment_qa = self._extract_sentiment(qa) if qa else 0.0
            feature_dict['sentiment_finbert'] = 0.5 * sentiment_full + 0.5 * sentiment_qa

            if self.vader:
                vader_full = self.vader.polarity_scores(full_text)['compound']
                vader_qa = self.vader.polarity_scores(qa)['compound'] if qa else 0.0
            else:
                vader_full = 0.0
                vader_qa = 0.0
            feature_dict['sentiment_vader'] = 0.5 * vader_full + 0.5 * vader_qa

            # LM (unchanged)
            words = [w.lower() for w in full_text.split() if w.lower() not in self.stop_words]
            total_words = max(len(words), 1)
            if not self.lm_dict.empty:
                lm_positive = self.lm_dict[self.lm_dict['POSITIVE'] > 0]['WORD'].str.lower().to_list()
                lm_negative = self.lm_dict[self.lm_dict['NEGATIVE'] > 0]['WORD'].str.lower().to_list()
                lm_uncertainty = self.lm_dict[self.lm_dict['UNCERTAINTY'] > 0]['WORD'].str.lower().to_list()
            else:
                # Fallback
                pass

            feature_dict['lm_positive'] = sum(1 for w in words if w in lm_positive) / total_words
            feature_dict['lm_negative'] = sum(1 for w in words if w in lm_negative) / total_words
            feature_dict['lm_uncertainty'] = sum(1 for w in words if w in lm_uncertainty) / total_words

            # Linguistic + Readability (unchanged)
            feature_dict['word_count'] = len(words)
            feature_dict['readability_gunning'] = textstat.gunning_fog(full_text)
            feature_dict['readability_flesch'] = textstat.flesch_reading_ease(full_text)
            feature_dict['readability_custom'] = calculate_readability(full_text)

            # Collect for batch NMF
            if len(words) > 0:
                all_texts.append(' '.join(words))
            else:
                all_texts.append('')  # Placeholder

            # Temp append without topics
            features.append(feature_dict)
            prev_by_ticker[t.ticker] = feature_dict.copy()

        # Batch NMF if texts
        if all_texts and any(t != '' for t in all_texts):
            try:
                tfidf = self.vectorizer.fit_transform(all_texts)
                W = self.nmf.fit_transform(tfidf)
                for idx, feature_dict in enumerate(features):
                    for i, prob in enumerate(W[idx]):
                        feature_dict[f'topic_{i}'] = prob
            except Exception as e:
                self.logger.warning(f"NMF failed: {e}. Setting topics to 0.")
                for feature_dict in features:
                    for i in range(5):
                        feature_dict[f'topic_{i}'] = 0.0
        else:
            for feature_dict in features:
                for i in range(5):
                    feature_dict[f'topic_{i}'] = 0.0

        # Second pass for deltas (now with topics)
        for t, feature_dict in zip(transcripts, features):
            prev = prev_by_ticker.get(t.ticker)
            if prev:
                for key in ['sentiment_finbert', 'sentiment_vader', 'lm_positive', 'lm_negative', 'lm_uncertainty',
                            'word_count'] + [f'topic_{i}' for i in range(5)]:
                    if key in feature_dict:  # Ensure key exists
                        feature_dict[f'{key}_delta'] = feature_dict[key] - prev[key]
            prev_by_ticker[t.ticker] = feature_dict.copy()

        return pd.DataFrame(features)

    def _extract_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
        scores = [self.sentiment_model(chunk)[0] for chunk in chunks]
        avg_score = sum(
            s['score'] if s['label'] == 'positive' else -s['score'] if s['label'] == 'negative' else 0 for s in
            scores) / len(scores)
        return avg_score