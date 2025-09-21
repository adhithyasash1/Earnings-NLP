# alpha_model.py (Updated: Vectorized labels, ternary option, add features to preprocessor)
"""Alpha signal generation"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

from config import Config
from utils import setup_logging


class AlphaModel:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.model_pipeline: Pipeline = None

    def _get_model(self):
        is_classification = self.config.prediction_target == "direction"
        if self.config.alpha_model == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_seed,
                class_weight='balanced' if is_classification else None
            )
        elif self.config.alpha_model == "xgboost":
            if is_classification:
                return xgb.XGBClassifier(
                    random_state=self.config.random_seed,
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    reg_alpha=1.0,
                    reg_lambda=1.0
                )
            else:
                return xgb.XGBRegressor(
                    random_state=self.config.random_seed,
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    reg_alpha=1.0,
                    reg_lambda=1.0
                )
        raise ValueError(f"Unknown model: {self.config.alpha_model}")

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        emb_cols = [c for c in X.columns if c.startswith('emb_')]
        linguistic_cols = [c for c in X.columns if
                           not c.startswith('emb_') and not c.startswith('topic_') and c in ['sentiment_finbert',
                                                                                             'sentiment_vader',
                                                                                             'word_count',
                                                                                             'readability_gunning',
                                                                                             'readability_flesch',
                                                                                             'readability_custom',
                                                                                             'lm_positive',
                                                                                             'lm_negative',
                                                                                             'lm_uncertainty']]
        topic_cols = [c for c in X.columns if c.startswith('topic_')]

        transformers = []

        if linguistic_cols:
            linguistic_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(
                    score_func=mutual_info_classif if self.config.prediction_target == "direction" else f_regression,
                    k=min(20, len(linguistic_cols))
                ))
            ])
            transformers.append(('linguistic', linguistic_pipeline, linguistic_cols))

        if topic_cols:
            topic_pipeline = Pipeline([
                ('scaler', StandardScaler()),
            ])
            transformers.append(('topics', topic_pipeline, topic_cols))  # No selector, keep all topics

        if emb_cols and self.config.use_pca:
            emb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self.config.pca_n_components, random_state=self.config.random_seed))
            ])
            transformers.append(('embeddings', emb_pipeline, emb_cols))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _create_labels(self, merged: pd.DataFrame, return_col: str) -> pd.DataFrame:
        returns = merged[return_col].copy()
        if self.config.labeling_strategy == "quantile":
            low = returns.quantile(0.3)
            high = returns.quantile(0.7)
            merged['label'] = np.where(returns >= high, 1, np.where(returns <= low, -1, 0))  # Changed to ternary: 1 buy, -1 sell, 0 hold
            # No dropna, keep holds for ternary classification
        else:  # median_split
            median = returns.median()
            merged['label'] = np.where(returns > median + 0.01, 1, np.where(returns < median - 0.01, -1, 0))  # Add thresholds for hold
        return merged

    def prepare_labels(self, features_df: pd.DataFrame, prices_df: pd.DataFrame,
                       benchmark_df: pd.DataFrame = None,
                       holding_period: int = 5) -> pd.DataFrame:
        features_df = features_df.copy()
        prices_df = prices_df.copy()
        features_df['date'] = pd.to_datetime(features_df['date'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

        features_df['entry_date'] = features_df['date'] + pd.Timedelta(days=2)
        features_df['exit_date'] = features_df['entry_date'] + pd.Timedelta(days=holding_period)

        # Vectorized: Merge and find entry/exit prices
        merged = pd.merge(features_df, prices_df, on='ticker', how='left')
        merged = merged[merged['timestamp'] >= merged['entry_date']]
        entry_prices = merged.groupby(['ticker', 'entry_date'])['close'].first().reset_index(name='entry_price')

        merged = pd.merge(features_df, prices_df, on='ticker', how='left')
        merged = merged[merged['timestamp'] <= merged['exit_date']]
        exit_prices = merged.groupby(['ticker', 'exit_date'])['close'].last().reset_index(name='exit_price')

        merged = features_df.merge(entry_prices, on=['ticker', 'entry_date'], how='left') \
            .merge(exit_prices, on=['ticker', 'exit_date'], how='left')
        merged = merged.dropna(subset=['entry_price', 'exit_price'])
        merged = merged[(merged['entry_price'] > 0) & (merged['exit_price'] > 0)]
        merged['future_return'] = (merged['exit_price'] - merged['entry_price']) / merged['entry_price']
        merged = merged[abs(merged['future_return']) <= 0.5]

        initial_count = len(features_df)
        after_merge = len(merged)
        self.logger.info(
            f"Labeled samples: {after_merge}/{initial_count} (dropped {initial_count - after_merge} due to missing/invalid prices)")

        # Optional: Next-day returns (config.next_day = True)
        if hasattr(self.config, 'next_day') and self.config.next_day:
            merged['exit_date'] = merged['entry_date'] + pd.Timedelta(days=1)

        if self.config.use_market_neutral and benchmark_df is not None:
            benchmark_df = benchmark_df.copy()
            benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
            bench_merged_entry = pd.merge(features_df[['entry_date']], benchmark_df, left_on='entry_date',
                                          right_on='timestamp', how='left')
            bench_entry = bench_merged_entry.groupby('entry_date')['close'].first().reset_index(name='bench_entry')

            bench_merged_exit = pd.merge(features_df[['exit_date']], benchmark_df, left_on='exit_date',
                                         right_on='timestamp', how='left')
            bench_exit = bench_merged_exit.groupby('exit_date')['close'].last().reset_index(name='bench_exit')

            merged = merged.merge(bench_entry, on='entry_date').merge(bench_exit, on='exit_date')
            merged = merged.dropna(subset=['bench_entry', 'bench_exit'])
            merged = merged[(merged['bench_entry'] > 0) & (merged['bench_exit'] > 0)]
            merged['benchmark_return'] = (merged['bench_exit'] - merged['bench_entry']) / merged['bench_entry']
            merged['relative_return'] = merged['future_return'] - merged['benchmark_return']
            return_col = 'relative_return'
        else:
            return_col = 'future_return'

        if self.config.prediction_target == "direction":
            merged = self._create_labels(merged, return_col)
        else:
            merged['label'] = merged[return_col]

        return merged.dropna(subset=['label'])

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        preprocessor = self._build_preprocessor(X)
        model = self._get_model()
        self.model_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

        tscv = TimeSeriesSplit(n_splits=5)
        param_grid = self.config.load_param_grid()
        if param_grid or self.config.use_hyperparam_tuning:  # Enabled if grid or flag
            tuner = GridSearchCV(self.model_pipeline, param_grid or {}, cv=tscv, scoring=self.config.tuner_scoring_metric)
            tuner.fit(X, y)
            self.model_pipeline = tuner.best_estimator_
            return tuner.best_score_

        self.model_pipeline.fit(X, y)
        return self._cross_validate(X, y, tscv)

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, cv) -> float:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model_pipeline, X, y, cv=cv, scoring=self.config.tuner_scoring_metric)
        return scores.mean()

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        is_classification = self.config.prediction_target == "direction"
        if is_classification:
            probs = self.model_pipeline.predict_proba(X)[:, 1]
            labels = (probs > 0.5).astype(int)
            return probs, labels
        else:
            predictions = self.model_pipeline.predict(X)
            return predictions, predictions

    def save(self, path: str = None):
        if path is None:
            path = Path(self.config.output_dir) / "model.pkl"
        joblib.dump(self.model_pipeline, path)

    @classmethod
    def load(cls, path: str):
        model_pipeline = joblib.load(path)
        instance = cls(Config())
        instance.model_pipeline = model_pipeline
        return instance