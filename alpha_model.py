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
        linguistic_cols = [c for c in X.columns if not c.startswith('emb_')]

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
            low_threshold = returns.quantile(0.3)
            high_threshold = returns.quantile(0.7)
            merged['label'] = np.where(returns >= high_threshold, 1,
                                       np.where(returns <= low_threshold, 0, np.nan))
            merged = merged.dropna(subset=['label'])
        else:  # median_split
            median = returns.median()
            merged['label'] = (returns >= median).astype(int)
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

        results = []
        for _, row in features_df.iterrows():
            ticker = row['ticker']
            entry_date = row['entry_date']
            exit_date = row['exit_date']

            ticker_prices = prices_df[prices_df['ticker'] == ticker].sort_values('timestamp')
            entry_prices = ticker_prices[ticker_prices['timestamp'] >= entry_date]
            if entry_prices.empty: continue
            entry_price = entry_prices.iloc[0]['close']

            exit_prices = ticker_prices[ticker_prices['timestamp'] <= exit_date]
            if exit_prices.empty: continue
            exit_price = exit_prices.iloc[-1]['close']

            if entry_price <= 0 or exit_price <= 0 or abs((exit_price - entry_price) / entry_price) > 0.5:
                continue

            future_return = (exit_price - entry_price) / entry_price
            result_row = row.to_dict()
            result_row['future_return'] = future_return
            results.append(result_row)

        merged = pd.DataFrame(results)

        if self.config.use_market_neutral and benchmark_df is not None:
            benchmark_df = benchmark_df.sort_values('timestamp')
            for idx, row in merged.iterrows():
                entry_date = row['entry_date']
                exit_date = row['exit_date']
                bench_entry = benchmark_df[benchmark_df['timestamp'] >= entry_date].iloc[0]['close'] if not benchmark_df[benchmark_df['timestamp'] >= entry_date].empty else None
                bench_exit = benchmark_df[benchmark_df['timestamp'] <= exit_date].iloc[-1]['close'] if not benchmark_df[benchmark_df['timestamp'] <= exit_date].empty else None
                if bench_entry is None or bench_exit is None or bench_entry <= 0 or bench_exit <= 0:
                    continue
                benchmark_return = (bench_exit - bench_entry) / bench_entry
                merged.at[idx, 'relative_return'] = row['future_return'] - benchmark_return
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
        if self.config.use_hyperparam_tuning:
            param_grid = self.config.load_param_grid()
            if param_grid:
                tuner = GridSearchCV(self.model_pipeline, param_grid, cv=tscv, scoring=self.config.tuner_scoring_metric)
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